use super::{
    compile::{Block, Error, compile},
    cpu::Cpu,
    decode::decode,
};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub struct Cache {
    entries: Vec<Entry>,
    index: HashMap<(u64, u64), Vec<usize>>,
}
struct Entry {
    bytes: Box<[u8]>,
    block: Block,
}
impl Cache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn get_or_compile(&mut self, pc: u64, bytes: &[u8]) -> Result<&Block, Error> {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hash);
        let key = (pc, hash.finish());
        if let Some(bucket) = self.index.get(&key) {
            for &at in bucket {
                if self.entries[at].bytes.as_ref() == bytes {
                    return Ok(&self.entries[at].block);
                }
            }
        }
        let block = compile(pc, bytes)?;
        let at = self.entries.len();
        self.entries.push(Entry {
            bytes: bytes.into(),
            block,
        });
        self.index.entry(key).or_default().push(at);
        Ok(&self.entries[at].block)
    }
    pub fn run_block<F>(&mut self, cpu: &mut Cpu, mut fetch: F) -> Result<u64, Error>
    where
        F: FnMut(&Cpu, u64, &mut [u8]) -> Result<usize, Error>,
    {
        let mut bytes = [0u8; 64 * 4];
        let mut count = 0usize;
        for _ in 0..64 {
            let at = cpu
                .pc
                .checked_add(count as u64)
                .ok_or(Error::AddressOverflow)?;
            let room = &mut bytes[count..count + 4];
            let length = match fetch(cpu, at, room) {
                Ok(l) => l,
                Err(_) if count > 0 => break,
                Err(e) => return Err(e),
            };
            if length != 4 {
                return Err(Error::InvalidInstructionLength);
            }
            count += 4;
            if decode(u32::from_le_bytes(room.try_into().unwrap()))?.terminates() {
                break;
            }
        }
        Ok(self.get_or_compile(cpu.pc, &bytes[..count])?.run(cpu))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_code_is_refetched_and_permission_checked() {
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        cpu.pc = 0x1000;
        cache
            .run_block(&mut cpu, |_, _, out| {
                out.copy_from_slice(&0xd4000001u32.to_le_bytes());
                Ok(4)
            })
            .unwrap();
        assert_eq!(cache.len(), 1);
        cpu.pc = 0x1000;
        let error = cache
            .run_block(&mut cpu, |_, at, _| {
                Err(Error::Memory(crate::memory::MemoryError::AccessFault {
                    address: at,
                    length: 4,
                }))
            })
            .unwrap_err();
        assert!(matches!(error, Error::Memory(_)));
        assert_eq!(cpu.pc, 0x1000);
        assert_eq!(cache.len(), 1);
    }
    #[test]
    fn mutated_code_gets_distinct_translation() {
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        let svc = 0xd4000001u32.to_le_bytes();
        let brk = 0xd4200020u32.to_le_bytes();
        cache.get_or_compile(0, &svc).unwrap().run(&mut cpu);
        assert_eq!(cpu.trap, super::super::cpu::Trap::Svc);
        cache.get_or_compile(0, &brk).unwrap().run(&mut cpu);
        assert_eq!(cpu.trap, super::super::cpu::Trap::Brk);
        assert_eq!(cache.len(), 2);
    }
    #[test]
    fn fetch_limit_counts_exactly_sixty_four_instructions() {
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        let mut calls = 0;
        let next = cache
            .run_block(&mut cpu, |_, at, out| {
                assert_eq!(at, calls * 4);
                calls += 1;
                out.copy_from_slice(&0xd503201fu32.to_le_bytes());
                Ok(4)
            })
            .unwrap();
        assert_eq!(calls, 64);
        assert_eq!(next, 256);
        assert_eq!(cpu.system.cntvct_el0, 64);
    }
}

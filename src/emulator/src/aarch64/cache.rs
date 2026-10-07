use super::{
    ahead::{Ahead, CodeReader},
    compile::{Block, Error, compile_with},
    cpu::Cpu,
    decode::decode,
};
use core::hash::{Hash, Hasher};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

pub(super) const MAX_INSNS: usize = 64;
const DIRECT_SET_BITS: u32 = 15;
const DIRECT_WAYS: usize = 4;
const NO_ENTRY: u32 = u32::MAX;
/// `Entry::epoch` value that matches no `Fetch::translation_epoch`.
const NO_EPOCH: u64 = u64::MAX;

pub struct Cache {
    entries: Vec<Entry>,
    index: HashMap<(u64, u64), Vec<usize>>,
    /// Set-associative `pc -> entry` table for the hit fast path, `DIRECT_WAYS`
    /// slots per set, newest first. Slots carry the pc so a probe never touches
    /// `entries` unless it matches.
    direct: Box<[DirectSlot]>,
    /// Where blocks come from on a local miss. Private to this cache unless
    /// built with `with_shared`.
    shared: Arc<SharedBlocks>,
}
impl Default for Cache {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
            direct: alloc_direct(),
            shared: Arc::new(SharedBlocks::new(false)),
        }
    }
}
#[derive(Clone, Copy)]
struct DirectSlot {
    pc: u64,
    at: u32,
}
fn alloc_direct() -> Box<[DirectSlot]> {
    let empty = DirectSlot {
        pc: 0,
        at: NO_ENTRY,
    };
    vec![empty; DIRECT_WAYS << DIRECT_SET_BITS].into_boxed_slice()
}
struct Entry {
    pc: u64,
    bytes: Box<[u8]>,
    block: Arc<Block>,
    /// True when the block is fully determined by its own bytes: it ended at a
    /// terminator or the instruction cap, and stays inside one 4 KiB page. A
    /// hit can then be validated by re-reading and comparing just those bytes,
    /// without decoding, and cannot differ from what a fresh scan would build.
    fast: bool,
    /// Physical page and write-tracker version the block was last validated
    /// against. Only set when the memory provides a `CodeTracker`.
    key: Option<(u64, u32)>,
    /// `Fetch::translation_epoch` at the time `key` was last confirmed. While
    /// it is unchanged `pc` still maps to `key`'s page, so a hit needs only the
    /// page version, not a fresh address translation.
    epoch: u64,
}
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Compiled blocks shared by every vCPU. A block takes the CPU as an argument
/// and holds no per-CPU state, so one compiled copy serves them all; sharing
/// means N vCPUs execute the same code without compiling it N times.
///
/// Blocks are keyed by `(pc, bytes)`, so changed code simply gets a new entry
/// and nothing ever needs invalidating here.
pub struct SharedBlocks {
    inline_memory: bool,
    map: Mutex<HashMap<(u64, u64), Vec<Arc<Slot>>>>,
    compiled: AtomicUsize,
    /// Background compilation of the blocks likely to run next, once enabled.
    ahead: OnceLock<Arc<Ahead>>,
}
impl Drop for SharedBlocks {
    fn drop(&mut self) {
        if let Some(ahead) = self.ahead.get() {
            ahead.shutdown();
        }
    }
}
struct Slot {
    bytes: Box<[u8]>,
    /// Empty until compiled. Holding this lock while compiling makes other
    /// vCPUs that want the same block wait for it instead of compiling it again,
    /// without blocking vCPUs that want different blocks.
    block: Mutex<Option<Arc<Block>>>,
}
impl SharedBlocks {
    pub fn new(inline_memory: bool) -> Self {
        Self {
            inline_memory,
            map: Mutex::new(HashMap::new()),
            compiled: AtomicUsize::new(0),
            ahead: OnceLock::new(),
        }
    }
    /// Compile the likely successors of every block on `threads` background
    /// threads, reading code through `reader`. Does nothing for 0 threads or when
    /// already enabled.
    pub fn enable_ahead(self: &Arc<Self>, reader: Arc<dyn CodeReader>, threads: usize) {
        if threads > 0 {
            let _ = self.ahead.set(Ahead::start(self, reader, threads));
        }
    }
    fn ahead(&self) -> Option<&Arc<Ahead>> {
        self.ahead.get()
    }
    /// Number of blocks compiled so far (not the number of lookups).
    pub fn compiled(&self) -> usize {
        self.compiled.load(Ordering::Relaxed)
    }
    pub fn inline_memory(&self) -> bool {
        self.inline_memory
    }
    pub(super) fn get(&self, pc: u64, bytes: &[u8]) -> Result<Arc<Block>, Error> {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hash);
        let slot = {
            let mut map = lock(&self.map);
            let bucket = map.entry((pc, hash.finish())).or_default();
            match bucket.iter().find(|slot| slot.bytes.as_ref() == bytes) {
                Some(slot) => slot.clone(),
                None => {
                    let slot = Arc::new(Slot {
                        bytes: bytes.into(),
                        block: Mutex::new(None),
                    });
                    bucket.push(slot.clone());
                    slot
                }
            }
        };
        let mut compiled = lock(&slot.block);
        if let Some(block) = &*compiled {
            return Ok(block.clone());
        }
        // A failed compile leaves the slot empty, so the next request retries.
        let block = Arc::new(compile_with(pc, bytes, self.inline_memory)?);
        self.compiled.fetch_add(1, Ordering::Relaxed);
        *compiled = Some(block.clone());
        Ok(block)
    }
}

/// Source of instruction bytes for the cache.
pub trait Fetch {
    /// Read `out.len()` bytes of code at virtual address `at`.
    fn fetch(&mut self, cpu: &Cpu, at: u64, out: &mut [u8]) -> Result<usize, Error>;
    /// Identify the physical code page behind `pc` for write tracking: the
    /// page number and its current version. With `watch`, start tracking the
    /// page first (call before fetching the bytes). `None` if untracked or if
    /// `pc` cannot be fetched; the cache then validates by content instead.
    fn code_key(&mut self, _cpu: &mut Cpu, _pc: u64, _watch: bool) -> Option<(u64, u32)> {
        None
    }
    /// A value that changes whenever some virtual-to-physical code mapping
    /// may have changed. A `code_key` taken under the same epoch still names
    /// the page `pc` fetches from. `None` (the default) means no such promise,
    /// so every hit re-derives its key through `code_key`.
    fn translation_epoch(&self) -> Option<u64> {
        None
    }
    /// Current write-tracker version of physical code page `page`.
    fn page_version(&self, _page: u64) -> Option<u32> {
        None
    }
}
struct Plain<F>(F);
impl<F: FnMut(&Cpu, u64, &mut [u8]) -> Result<usize, Error>> Fetch for Plain<F> {
    fn fetch(&mut self, cpu: &Cpu, at: u64, out: &mut [u8]) -> Result<usize, Error> {
        (self.0)(cpu, at, out)
    }
}
/// Index of the first slot of `pc`'s set.
fn set_base(pc: u64) -> usize {
    let word = pc >> 2;
    (((word ^ (word >> DIRECT_SET_BITS)) as usize) & ((1 << DIRECT_SET_BITS) - 1)) * DIRECT_WAYS
}
impl Cache {
    pub fn new() -> Self {
        Self::default()
    }
    /// Whether blocks from this cache read `Cpu::dtlb`.
    pub fn inline_memory(&self) -> bool {
        self.shared.inline_memory()
    }
    /// A cache whose blocks access memory through `Cpu::dtlb` when possible.
    /// Only valid when something fills `Cpu::dtlb` (the system machine does).
    pub fn with_inline_memory() -> Self {
        Self::with_shared(Arc::new(SharedBlocks::new(true)))
    }
    /// A per-vCPU cache backed by blocks shared with other vCPUs. Its hit paths
    /// stay local and lock-free; the shared map is used only on a local miss.
    pub fn with_shared(shared: Arc<SharedBlocks>) -> Self {
        Self {
            shared,
            ..Self::default()
        }
    }
    /// Start background compilation for this cache's blocks (see
    /// [`SharedBlocks::enable_ahead`]).
    pub fn start_ahead(&self, reader: Arc<dyn CodeReader>, threads: usize) {
        self.shared.enable_ahead(reader, threads);
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn get_or_compile(&mut self, pc: u64, bytes: &[u8]) -> Result<&Block, Error> {
        let at = self.lookup_or_compile(pc, bytes)?;
        Ok(&self.entries[at].block)
    }
    fn lookup_or_compile(&mut self, pc: u64, bytes: &[u8]) -> Result<usize, Error> {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hash);
        let key = (pc, hash.finish());
        if let Some(bucket) = self.index.get(&key) {
            for &at in bucket {
                if self.entries[at].bytes.as_ref() == bytes {
                    return Ok(at);
                }
            }
        }
        let block = self.shared.get(pc, bytes)?;
        // Another thread may have just written this block's code; make sure this
        // one fetches the new instructions.
        volt_target::native::instruction_barrier();
        let at = self.entries.len();
        self.entries.push(Entry {
            pc,
            bytes: bytes.into(),
            block,
            fast: false,
            key: None,
            epoch: NO_EPOCH,
        });
        self.index.entry(key).or_default().push(at);
        Ok(at)
    }
    fn find_direct(&self, pc: u64) -> u32 {
        let set = &self.direct[set_base(pc)..][..DIRECT_WAYS];
        for slot in set {
            if slot.pc == pc && slot.at != NO_ENTRY {
                return slot.at;
            }
        }
        NO_ENTRY
    }
    /// Make `at` the newest slot of `pc`'s set, replacing any slot already
    /// holding `pc` or else the oldest.
    fn insert_direct(&mut self, pc: u64, at: u32) {
        let set = &mut self.direct[set_base(pc)..][..DIRECT_WAYS];
        let way = set
            .iter()
            .position(|slot| slot.pc == pc && slot.at != NO_ENTRY)
            .unwrap_or(DIRECT_WAYS - 1);
        set.copy_within(..way, 1);
        set[0] = DirectSlot { pc, at };
    }
    pub fn run_block<F>(&mut self, cpu: &mut Cpu, fetch: F) -> Result<u64, Error>
    where
        F: FnMut(&Cpu, u64, &mut [u8]) -> Result<usize, Error>,
    {
        self.run_block_with(cpu, &mut Plain(fetch))
    }
    pub fn run_block_with<T: Fetch>(&mut self, cpu: &mut Cpu, fetch: &mut T) -> Result<u64, Error> {
        let pc = cpu.pc;
        // Fast path. Any failure or mismatch falls through to the full scan,
        // which reports faults and handles changed code exactly as before.
        let candidate = self.find_direct(pc);
        if candidate != NO_ENTRY {
            let at = candidate as usize;
            if self.entries[at].pc == pc && self.entries[at].fast {
                // 1. Tracked memory: the block is valid while its page's write
                //    version is unchanged. No fetch, no compare. Under an
                //    unchanged translation epoch even the address translation
                //    is skipped.
                let epoch = fetch.translation_epoch().unwrap_or(NO_EPOCH);
                if epoch != NO_EPOCH
                    && self.entries[at].epoch == epoch
                    && let Some((page, version)) = self.entries[at].key
                    && fetch.page_version(page) == Some(version)
                {
                    return Ok(self.entries[at].block.run(cpu));
                }
                if let Some(now) = fetch.code_key(cpu, pc, false)
                    && self.entries[at].key == Some(now)
                {
                    self.entries[at].epoch = epoch;
                    return Ok(self.entries[at].block.run(cpu));
                }
                // 2. Otherwise (untracked memory, or the page was written):
                //    refetch the stored length and compare the bytes. On a match
                //    the entry is still valid and gets the fresh version.
                let watched = fetch.code_key(cpu, pc, true);
                let len = self.entries[at].bytes.len();
                let mut bytes = [0u8; MAX_INSNS * 4];
                let room = &mut bytes[..len];
                if matches!(fetch.fetch(cpu, pc, room), Ok(n) if n == len)
                    && room == self.entries[at].bytes.as_ref()
                {
                    self.entries[at].key = watched;
                    self.entries[at].epoch = epoch;
                    return Ok(self.entries[at].block.run(cpu));
                }
            }
        }
        // Slow path. Watch the page BEFORE reading any bytes so a concurrent
        // write can only ever make the recorded version stale, never the bytes.
        let watched = fetch.code_key(cpu, pc, true);
        let mut bytes = [0u8; MAX_INSNS * 4];
        let mut count = 0usize;
        let mut complete = false;
        let mut last = None;
        for n in 0..MAX_INSNS {
            let at = cpu
                .pc
                .checked_add(count as u64)
                .ok_or(Error::AddressOverflow)?;
            let room = &mut bytes[count..count + 4];
            let length = match fetch.fetch(cpu, at, room) {
                Ok(l) => l,
                Err(_) if count > 0 => break,
                Err(e) => return Err(e),
            };
            if length != 4 {
                return Err(Error::InvalidInstructionLength);
            }
            count += 4;
            let word = u32::from_le_bytes(room.try_into().unwrap());
            let instruction = match decode(word) {
                Ok(insn) => insn,
                Err(_) => {
                    eprintln!("cache decode failure at pc {at:x}: word {word:08x}");
                    return Err(Error::Decode { word, pc: at });
                }
            };
            last = Some((instruction, at));
            if instruction.terminates_with(self.shared.inline_memory()) {
                complete = true;
                break;
            }
            // Never straddle a page: a block inside one page is validated by
            // that page's version alone, and the next page is translated and
            // checked when its own block starts.
            complete = n + 1 == MAX_INSNS || (pc + count as u64) & 0xfff == 0;
            if complete {
                break;
            }
        }
        let known = self.entries.len();
        let at = self.lookup_or_compile(pc, &bytes[..count])?;
        if complete && (pc & 0xfff) + count as u64 <= 0x1000 {
            self.entries[at].fast = true;
            self.entries[at].key = watched;
            self.entries[at].epoch = fetch.translation_epoch().unwrap_or(NO_EPOCH);
            self.insert_direct(pc, at as u32);
            // A block this cache has not seen: have spare cores build what can follow.
            if self.entries.len() > known
                && let (Some(ahead), Some((page, _)), Some((last, last_pc))) =
                    (self.shared.ahead(), watched, last)
            {
                ahead.offer_successors(last, last_pc, (page << 12) | (last_pc & 0xfff));
            }
        }
        Ok(self.entries[at].block.run(cpu))
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
    fn serve(
        code: &[u32],
        calls: &mut u32,
    ) -> impl FnMut(&Cpu, u64, &mut [u8]) -> Result<usize, Error> {
        let code = code.to_vec();
        let base = 0x1000u64;
        move |_, at, out| {
            *calls += 1;
            for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
                let word = code[((at - base) / 4) as usize + i];
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            Ok(out.len())
        }
    }
    #[test]
    fn hit_uses_one_fetch_and_detects_changed_code() {
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        let svc = 0xd4000001u32;
        let brk = 0xd4200020u32;
        let nop = 0xd503201fu32;
        let mut calls = 0;
        cpu.pc = 0x1000;
        cache
            .run_block(&mut cpu, serve(&[nop, nop, svc], &mut calls))
            .unwrap();
        assert_eq!(calls, 3);
        calls = 0;
        cpu.pc = 0x1000;
        cache
            .run_block(&mut cpu, serve(&[nop, nop, svc], &mut calls))
            .unwrap();
        assert_eq!(calls, 1, "hit should be a single validating fetch");
        assert_eq!(cache.len(), 1);
        // Same pc, changed bytes: must recompile, not reuse the stale block.
        calls = 0;
        cpu.pc = 0x1000;
        cache
            .run_block(&mut cpu, serve(&[nop, nop, brk], &mut calls))
            .unwrap();
        assert_eq!(cpu.trap, super::super::cpu::Trap::Brk);
        assert_eq!(cache.len(), 2);
    }
    #[test]
    fn blocks_end_at_the_page_boundary_and_hit_the_fast_path() {
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        let nop = 0xd503201fu32.to_le_bytes();
        let mut fetched = Vec::new();
        for _ in 0..2 {
            // Two nops reach the end of the page; the block must not read past it.
            cpu.pc = 0x1ff8;
            let next = cache
                .run_block(&mut cpu, |_, at, out| {
                    fetched.push(at);
                    out.chunks_exact_mut(4)
                        .for_each(|w| w.copy_from_slice(&nop));
                    Ok(out.len())
                })
                .unwrap();
            assert_eq!(next, 0x2000);
        }
        // First run scans two instructions; the second revalidates with one fetch.
        assert_eq!(fetched, [0x1ff8, 0x1ffc, 0x1ff8]);
        assert_eq!(cache.len(), 1);
    }
    /// Identity-mapped fetch over guest memory, counting byte fetches.
    struct Ram<M: crate::memory::GuestMemory> {
        memory: M,
        fetches: u32,
    }
    impl<M: crate::memory::GuestMemory> Fetch for Ram<M> {
        fn fetch(&mut self, _: &Cpu, at: u64, out: &mut [u8]) -> Result<usize, Error> {
            self.fetches += 1;
            self.memory.read(at, out).map_err(Error::Memory)?;
            Ok(out.len())
        }
        fn code_key(&mut self, _: &mut Cpu, pc: u64, watch: bool) -> Option<(u64, u32)> {
            let tracker = self.memory.code_tracker()?;
            let page = pc >> 12;
            Some((
                page,
                if watch {
                    tracker.watch(page)?
                } else {
                    tracker.version(page)?
                },
            ))
        }
    }
    fn ram() -> crate::memory::PhysicalMemory {
        use crate::memory::{PhysicalMemory, Region};
        PhysicalMemory::new(vec![Region::ram(0x1000, vec![0; 0x3000]).unwrap()]).unwrap()
    }
    fn words(memory: &mut impl crate::memory::GuestMemory, at: u64, code: &[u32]) {
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        memory.write(at, &bytes).unwrap();
    }
    const NOP: u32 = 0xd503201f;
    const SVC: u32 = 0xd4000001;
    const BRK: u32 = 0xd4200020;

    #[test]
    fn tracked_hit_does_no_fetch_and_code_writes_invalidate() {
        use crate::memory::GuestMemory;
        let mut source = Ram {
            memory: ram(),
            fetches: 0,
        };
        words(&mut source.memory, 0x1000, &[NOP, NOP, SVC]);
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        let run = |cache: &mut Cache, cpu: &mut Cpu, source: &mut Ram<_>| {
            cpu.pc = 0x1000;
            source.fetches = 0;
            cache.run_block_with(cpu, source).unwrap();
            source.fetches
        };
        assert_eq!(run(&mut cache, &mut cpu, &mut source), 3, "miss scans");
        assert_eq!(
            run(&mut cache, &mut cpu, &mut source),
            0,
            "hit does not fetch"
        );
        // Write to a different page: still no fetch.
        source.memory.write(0x2000, &[1]).unwrap();
        assert_eq!(run(&mut cache, &mut cpu, &mut source), 0);
        // Data write to the same page, code bytes unchanged: one revalidation
        // fetch, same block, then back to zero.
        source.memory.write(0x1800, &[1]).unwrap();
        assert_eq!(run(&mut cache, &mut cpu, &mut source), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(run(&mut cache, &mut cpu, &mut source), 0);
        // Code write: must run the new code.
        words(&mut source.memory, 0x1008, &[BRK]);
        run(&mut cache, &mut cpu, &mut source);
        assert_eq!(cpu.trap, super::super::cpu::Trap::Brk);
        assert_eq!(cache.len(), 2);
    }
    #[test]
    fn writes_through_another_shared_handle_invalidate() {
        use crate::memory::{GuestMemory, SharedMemory};
        let shared = SharedMemory::new(ram());
        let mut writer = shared.clone();
        let mut source = Ram {
            memory: shared,
            fetches: 0,
        };
        writer.write(0x1000, &SVC.to_le_bytes()).unwrap();
        let mut cache = Cache::new();
        let mut cpu = Cpu::default();
        cpu.pc = 0x1000;
        cache.run_block_with(&mut cpu, &mut source).unwrap();
        assert_eq!(cpu.trap, super::super::cpu::Trap::Svc);
        // Another vCPU rewrites the code.
        writer.write(0x1000, &BRK.to_le_bytes()).unwrap();
        cpu.pc = 0x1000;
        cache.run_block_with(&mut cpu, &mut source).unwrap();
        assert_eq!(cpu.trap, super::super::cpu::Trap::Brk);
    }
    #[test]
    fn caches_sharing_blocks_compile_each_block_once() {
        let shared = Arc::new(SharedBlocks::new(false));
        let mut a = Cache::with_shared(shared.clone());
        let mut b = Cache::with_shared(shared.clone());
        let svc = 0xd4000001u32.to_le_bytes();
        let brk = 0xd4200020u32.to_le_bytes();
        let mut cpu = Cpu::default();
        a.get_or_compile(0x1000, &svc).unwrap().run(&mut cpu);
        assert_eq!(cpu.trap, super::super::cpu::Trap::Svc);
        b.get_or_compile(0x1000, &svc).unwrap().run(&mut cpu);
        assert_eq!(
            shared.compiled(),
            1,
            "second cache reuses the first's block"
        );
        // Different bytes at the same pc are a different block.
        b.get_or_compile(0x1000, &brk).unwrap().run(&mut cpu);
        assert_eq!(cpu.trap, super::super::cpu::Trap::Brk);
        assert_eq!(shared.compiled(), 2);
        // Each cache still counts its own entries.
        assert_eq!((a.len(), b.len()), (1, 2));
    }
    #[test]
    fn concurrent_requests_for_one_block_compile_it_once() {
        let shared = Arc::new(SharedBlocks::new(false));
        let svc = 0xd4000001u32.to_le_bytes();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let shared = shared.clone();
                scope.spawn(move || {
                    let mut cache = Cache::with_shared(shared);
                    let mut cpu = Cpu::default();
                    cache.get_or_compile(0x2000, &svc).unwrap().run(&mut cpu);
                    assert_eq!(cpu.trap, super::super::cpu::Trap::Svc);
                });
            }
        });
        assert_eq!(shared.compiled(), 1);
    }
}

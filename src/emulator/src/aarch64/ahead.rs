//! Speculative background compilation.
//!
//! Translating a block costs far more than running it once, and a kernel boot
//! executes most of its code only a few times. While a vCPU runs a block it has
//! just built, spare cores can already be building the blocks control is most
//! likely to reach next: the statically known successors of each block,
//! followed transitively.
//!
//! Work is described by guest addresses only. A worker reads the code bytes
//! straight from guest RAM and splits blocks with the same rules as the
//! executing vCPU, then publishes the result through [`SharedBlocks`], which is
//! keyed by `(pc, bytes)`. A block built here is therefore exactly the block a
//! vCPU would have built for the same bytes: if the guest has changed the code
//! since, or the vCPU would have cut the block differently, the speculative
//! block is simply never looked up. It can never be wrong, only wasted.
//!
//! Only successors on the same 4 KiB page are followed. They share the page's
//! translation, so the physical address is known without walking page tables
//! on a worker thread.

use super::cache::{MAX_INSNS, SharedBlocks};
use super::host::{Dispatch, dispatch};
use alloc::collections::VecDeque;
use alloc::sync::{Arc, Weak};
use std::collections::HashSet;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use volt_target::aarch64::decode::Instruction;

/// Reads guest-physical bytes without any vCPU state, for worker threads.
pub trait CodeReader: Send + Sync {
    /// Fill `out` from guest physical memory at `physical`; `false` if the range
    /// is not plain RAM.
    fn read(&self, physical: u64, out: &mut [u8]) -> bool;
}

/// How many blocks deep a speculative chain may run before it stops. Bounds the
/// waste when the guess is wrong, such as a branch that is never taken.
const MAX_DEPTH: u32 = 32;
/// Most jobs waiting at once; further offers are dropped.
const MAX_QUEUED: usize = 4096;

struct Job {
    pc: u64,
    physical: u64,
    depth: u32,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    /// Physical addresses of blocks already offered, so loops and joins are
    /// built once.
    offered: HashSet<u64>,
    shutdown: bool,
}

/// A pool of compile-ahead worker threads.
pub struct Ahead {
    state: Mutex<State>,
    wake: Condvar,
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where control can go after `last`, a block's final instruction at `last_pc`.
/// Fixed targets only: an indirect branch or return has none to offer.
fn successors(last: Dispatch, last_pc: u64) -> ([u64; 2], usize) {
    let next = last_pc.wrapping_add(4);
    let at = |offset: i64| last_pc.wrapping_add(offset as u64);
    let instruction = match last {
        Dispatch::Guest(instruction) => instruction,
        // A host word falls through, as a block end that does not branch does.
        Dispatch::Host(_) => return ([next, 0], 1),
    };
    match instruction {
        Instruction::B(offset) => ([at(offset), 0], 1),
        Instruction::Call(call) if call.link => ([at(call.target), next], 2),
        Instruction::Call(call) => ([at(call.target), 0], 1),
        Instruction::BCond(branch) => ([at(branch.offset), next], 2),
        Instruction::TestBranch(test) => ([at(test.offset), next], 2),
        Instruction::Indirect(branch) if branch.link => ([next, 0], 1),
        Instruction::Indirect(_) | Instruction::Eret => ([0, 0], 0),
        // Every other block end, a trap or a cut at the instruction limit, resumes
        // at the next instruction.
        _ => ([next, 0], 1),
    }
}

impl Ahead {
    /// Start `threads` workers compiling into `shared`. They exit once `shared`
    /// is gone or [`Ahead::shutdown`] is called.
    pub fn start(
        shared: &Arc<SharedBlocks>,
        reader: Arc<dyn CodeReader>,
        threads: usize,
    ) -> Arc<Self> {
        let ahead = Arc::new(Self {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        for n in 0..threads {
            let (ahead, shared, reader) = (ahead.clone(), Arc::downgrade(shared), reader.clone());
            // A host that cannot spawn a thread just compiles on demand.
            let _ = thread::Builder::new()
                .name(format!("volt-jit-{n}"))
                .spawn(move || ahead.work(&shared, reader.as_ref()));
        }
        ahead
    }

    pub fn shutdown(&self) {
        lock(&self.state).shutdown = true;
        self.wake.notify_all();
    }

    /// Offer the blocks that can follow `last` (the final instruction of the
    /// block just built, at `last_pc`). `physical` is the physical address of
    /// `last_pc`; only successors on its page are taken.
    pub fn offer_successors(&self, last: Dispatch, last_pc: u64, physical: u64) {
        let (targets, count) = successors(last, last_pc);
        self.offer(&targets[..count], last_pc, physical, 0);
    }

    fn offer(&self, targets: &[u64], from_pc: u64, from_physical: u64, depth: u32) {
        let mut added = false;
        {
            let mut state = lock(&self.state);
            for &pc in targets {
                if pc & 3 != 0 || pc >> 12 != from_pc >> 12 || depth > MAX_DEPTH {
                    continue;
                }
                let physical = (from_physical & !0xfff) | (pc & 0xfff);
                if state.queue.len() < MAX_QUEUED && state.offered.insert(physical) {
                    state.queue.push_back(Job {
                        pc,
                        physical,
                        depth,
                    });
                    added = true;
                }
            }
        }
        if added {
            self.wake.notify_all();
        }
    }

    fn work(&self, shared: &Weak<SharedBlocks>, reader: &dyn CodeReader) {
        loop {
            let job = {
                let mut state = lock(&self.state);
                loop {
                    if state.shutdown {
                        return;
                    }
                    if let Some(job) = state.queue.pop_front() {
                        break job;
                    }
                    state = self
                        .wake
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            };
            let Some(shared) = shared.upgrade() else {
                return;
            };
            self.build(&shared, reader, &job);
        }
    }

    /// Build the block at `job`, then offer what follows it.
    fn build(&self, shared: &SharedBlocks, reader: &dyn CodeReader, job: &Job) {
        let room = 0x1000 - (job.physical & 0xfff) as usize;
        let mut page = [0u8; 0x1000];
        if !reader.read(job.physical, &mut page[..room]) {
            return;
        }
        // The same cut rules as `Cache::run_block_with`, so the bytes match.
        let mut count = 0usize;
        let mut last = None;
        let mut complete = false;
        for n in 0..MAX_INSNS {
            if count + 4 > room {
                return;
            }
            let word = u32::from_le_bytes(page[count..count + 4].try_into().unwrap());
            let Ok(instruction) = dispatch(word) else {
                return;
            };
            last = Some((instruction, job.pc + count as u64));
            count += 4;
            if instruction.terminates_with(shared.inline_memory())
                || n + 1 == MAX_INSNS
                || (job.pc + count as u64) & 0xfff == 0
            {
                complete = true;
                break;
            }
        }
        let (Some((last, last_pc)), true) = (last, complete) else {
            return;
        };
        // Offer what follows before compiling, so the chain of blocks is queued at
        // decode speed and several workers can build it at once rather than one
        // block behind another.
        let (targets, n) = successors(last, last_pc);
        self.offer(&targets[..n], job.pc, job.physical, job.depth + 1);
        // Speculative code may be data or an encoding the translator does not
        // expect. A panic there must not take the worker down with it.
        let bytes = &page[..count];
        let _ =
            std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| shared.get(job.pc, bytes)));
    }
}

/// Number of compile-ahead workers: `VOLT_JIT_THREADS` if set, otherwise a few
/// of the spare cores, leaving one for the vCPU.
pub fn default_threads() -> usize {
    match std::env::var("VOLT_JIT_THREADS") {
        Ok(value) => value.trim().parse().unwrap_or(0),
        Err(_) => std::thread::available_parallelism()
            .map_or(0, |n| n.get().saturating_sub(1))
            .min(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    struct Ram(Vec<u8>);
    impl CodeReader for Ram {
        fn read(&self, physical: u64, out: &mut [u8]) -> bool {
            self.0
                .get(physical as usize..physical as usize + out.len())
                .map(|bytes| out.copy_from_slice(bytes))
                .is_some()
        }
    }

    const NOP: u32 = 0xd503201f;
    const SVC: u32 = 0xd4000001;
    const MOVZ_X0_1: u32 = 0xd2800020;
    const B_SELF: u32 = 0x14000000;

    fn ram(words: &[(u64, u32)]) -> Arc<Ram> {
        let mut bytes = vec![0u8; 0x2000];
        for (at, word) in words {
            bytes[*at as usize..*at as usize + 4].copy_from_slice(&word.to_le_bytes());
        }
        Arc::new(Ram(bytes))
    }

    fn wait_for(shared: &SharedBlocks, compiled: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.compiled() < compiled && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn fixed_targets_follow_each_kind_of_block_end() {
        use volt_target::aarch64::decode::{Call, Test, Width};
        let at = 0x1000;
        let test = Instruction::TestBranch(Test {
            rt: 0,
            bit_index: None,
            negate: false,
            width: Width::X64,
            offset: -8,
        });
        let call = |link| Instruction::Call(Call { target: 0x40, link });
        let cases = [
            (Instruction::B(0x20), vec![0x1020]),
            (call(true), vec![0x1040, 0x1004]),
            (call(false), vec![0x1040]),
            (test, vec![0x0ff8, 0x1004]),
            (
                Instruction::Indirect(volt_target::aarch64::decode::Indirect {
                    rn: 1,
                    link: false,
                }),
                vec![],
            ),
            (
                Instruction::Indirect(volt_target::aarch64::decode::Indirect { rn: 1, link: true }),
                vec![0x1004],
            ),
            (Instruction::Eret, vec![]),
            (Instruction::Svc(0), vec![0x1004]),
            (Instruction::Nop, vec![0x1004]),
        ];
        for (last, expected) in cases {
            let (targets, n) = successors(Dispatch::Guest(last), at);
            assert_eq!(&targets[..n], expected.as_slice(), "{last:?}");
        }
    }

    #[test]
    fn successors_of_a_block_are_compiled_without_the_vcpu() {
        // 0x1004: b 0x1014. 0x1014: movz x0,#1; svc. 0x101c: b . (a loop on itself).
        let memory = ram(&[
            (0x1004, 0x14000004),
            (0x1014, MOVZ_X0_1),
            (0x1018, SVC),
            (0x101c, B_SELF),
            (0x1000, NOP),
        ]);
        let shared = Arc::new(SharedBlocks::new(false));
        let ahead = Ahead::start(&shared, memory, 2);
        ahead.offer_successors(Dispatch::Guest(Instruction::B(0x10)), 0x1004, 0x1004);
        wait_for(&shared, 2);
        // The block after the branch, and the one after its `svc`: nothing more,
        // even though the last one branches to itself.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(shared.compiled(), 2);
        // Offering again builds nothing new.
        ahead.offer_successors(Dispatch::Guest(Instruction::B(0x10)), 0x1004, 0x1004);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(shared.compiled(), 2);
        // The vCPU's own lookup of the same bytes finds the block already built.
        let words = [MOVZ_X0_1, SVC];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        shared.get(0x1014, &bytes).unwrap();
        assert_eq!(shared.compiled(), 2);
        ahead.shutdown();
    }

    #[test]
    fn successors_on_another_page_are_not_followed() {
        let memory = ram(&[(0x1ffc, B_SELF), (0x1000, NOP)]);
        let shared = Arc::new(SharedBlocks::new(false));
        let ahead = Ahead::start(&shared, memory, 1);
        // A branch from page 1 to page 2 must not be read on this page's translation.
        ahead.offer_successors(Dispatch::Guest(Instruction::B(0x1004)), 0x1ffc, 0x1ffc);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(shared.compiled(), 0);
        ahead.shutdown();
    }
}

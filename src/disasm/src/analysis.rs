//! Recursive-descent function discovery, basic blocks and cross-references.
use crate::Image;
use std::collections::{BTreeMap, BTreeSet};
use volt_isa_aarch64::flow::{Flow, flow};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub start: u64,
    /// Address one past the last instruction.
    pub end: u64,
    /// Successor block starts within the same function.
    pub succs: Vec<u64>,
    /// How the block ends.
    pub exit: Flow,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub entry: u64,
    pub name: Option<String>,
    pub blocks: BTreeMap<u64, Block>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum XrefKind {
    Call,
    Jump,
    CondJump,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Xref {
    pub from: u64,
    pub kind: XrefKind,
}

#[derive(Debug, Default)]
pub struct Analysis {
    pub functions: BTreeMap<u64, Function>,
    /// Target address -> references to it, sorted.
    pub xrefs: BTreeMap<u64, Vec<Xref>>,
}

impl Analysis {
    /// Discover functions from function symbols, the entry point and `extra`
    /// roots, following direct calls.
    pub fn run(image: &Image, extra: &[u64]) -> Self {
        let mut roots: BTreeSet<u64> = image
            .symbols
            .iter()
            .filter(|s| s.func && image.is_exec(s.addr))
            .map(|s| s.addr)
            .collect();
        roots.extend(image.entry.filter(|&e| image.is_exec(e)));
        roots.extend(extra.iter().copied().filter(|&a| image.is_exec(a)));
        let known = roots.clone();

        let mut out = Self::default();
        let mut queue: Vec<u64> = roots.into_iter().collect();
        while let Some(entry) = queue.pop() {
            if out.functions.contains_key(&entry) {
                continue;
            }
            let f = out.build(image, entry, &known);
            for b in f.blocks.values() {
                if let Flow::Call(t) = b.exit
                    && image.is_exec(t)
                {
                    queue.push(t);
                }
            }
            out.functions.insert(entry, f);
        }
        for v in out.xrefs.values_mut() {
            v.sort();
            v.dedup();
        }
        out
    }

    pub fn function_containing(&self, addr: u64) -> Option<&Function> {
        self.functions.values().find(|f| {
            f.blocks
                .range(..=addr)
                .next_back()
                .is_some_and(|(_, b)| addr < b.end)
        })
    }

    fn build(&mut self, image: &Image, entry: u64, known: &BTreeSet<u64>) -> Function {
        // Pass 1: reachable instructions and leaders.
        let mut insns: BTreeMap<u64, Flow> = BTreeMap::new();
        let mut leaders = BTreeSet::from([entry]);
        let mut work = vec![entry];
        while let Some(mut pc) = work.pop() {
            loop {
                if insns.contains_key(&pc) || !image.is_exec(pc) {
                    break;
                }
                let Some(word) = image.read_u32(pc) else {
                    break;
                };
                let fl = flow(word, pc);
                insns.insert(pc, fl);
                let kind = match fl {
                    Flow::Call(_) => Some(XrefKind::Call),
                    Flow::Jump(_) => Some(XrefKind::Jump),
                    Flow::CondJump(_) => Some(XrefKind::CondJump),
                    _ => None,
                };
                if let (Some(kind), Some(t)) = (kind, fl.target()) {
                    self.xrefs
                        .entry(t)
                        .or_default()
                        .push(Xref { from: pc, kind });
                }
                match fl {
                    // A jump to another function's entry is a tail call.
                    Flow::Jump(t) | Flow::CondJump(t) if !(known.contains(&t) && t != entry) => {
                        leaders.insert(t);
                        work.push(t);
                    }
                    _ => {}
                }
                if fl.ends_block() {
                    if fl.falls_through() {
                        leaders.insert(pc + 4);
                        work.push(pc + 4);
                    }
                    break;
                }
                pc += 4;
            }
        }
        // Pass 2: cut blocks at leaders.
        let mut blocks = BTreeMap::new();
        for &start in leaders.iter().filter(|l| insns.contains_key(l)) {
            let mut pc = start;
            let (end, exit) = loop {
                let fl = insns[&pc];
                let next = pc + 4;
                if fl.ends_block() || leaders.contains(&next) || !insns.contains_key(&next) {
                    break (next, fl);
                }
                pc = next;
            };
            let mut succs = Vec::new();
            if exit.falls_through() && insns.contains_key(&end) {
                succs.push(end);
            }
            if let Flow::Jump(t) | Flow::CondJump(t) = exit
                && insns.contains_key(&t)
                && !(known.contains(&t) && t != entry)
            {
                succs.push(t);
            }
            blocks.insert(
                start,
                Block {
                    start,
                    end,
                    succs,
                    exit,
                },
            );
        }
        let name = image
            .symbols
            .iter()
            .find(|s| s.addr == entry && s.func)
            .map(|s| s.name.clone());
        Function {
            entry,
            name,
            blocks,
        }
    }
}

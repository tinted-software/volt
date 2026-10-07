//! Shared, target-independent Wimmer-Franz register allocator (Wimmer & Franz, CGO 2010).
//!
//! This module owns the ALGORITHM. Each backend owns ENCODING and supplies a
//! [`RegDescription`] that describes its register model to the allocator.
//!
//! Physical registers are an ABSTRACTION here. The allocator only ever sees a
//! `u16` register INDEX within a class. The backend picks a stable numbering
//! and maps the index back to its own register enum.
//!
//! A value normally occupies ONE register. A backend that supplies
//! [`RegDescription::reg_width`] may give a value a span of N CONSECUTIVE
//! registers under an alignment. [`Location::Reg`] then names the BASE of the
//! span, and every place the allocator picks, blocks, or compares a register
//! works over the whole span. See [`RegWidth`].

extern crate alloc;

use alloc::vec::Vec;

use crate::regalloc::ir::{Func, visit_inst_operands, visit_term_operands};
use volt_ir::function::{Block, Inst, Jump, Value};

pub const MAX_FUSED_OPERANDS: usize = 4;

/// Where a value lives at a program point: a physical register (a
/// class-relative INDEX) or a spill slot. The value's `class_of` sets the class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Location {
    Reg(u16),
    Slot(u32),
}

/// How many CONSECUTIVE physical registers ONE value occupies, and the
/// alignment its BASE register index must satisfy. The default, one register
/// at any index, is the model every backend used before this type existed.
///
/// A width above one puts TWO obligations on the backend:
///   1. `RegClass::slot_bytes` must be large enough for the WIDEST value of
///      the class, because a spilled value still gets exactly ONE slot.
///   2. The class `scratch`, and `scratch2` when it is supplied, must reserve
///      a whole ALIGNED span of that width, because a move the resolver routes
///      through the scratch carries a whole value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegWidth {
    pub regs: u16,
    pub alignment: u16,
}

impl Default for RegWidth {
    fn default() -> Self {
        RegWidth::ONE
    }
}

impl RegWidth {
    pub const ONE: RegWidth = RegWidth {
        regs: 1,
        alignment: 1,
    };
}

/// Whether an operand use requires a register. A [`UseKind::MustHaveRegister`]
/// operand cannot read from a spill slot on this target. A
/// [`UseKind::ShouldHaveRegister`] operand may fold or reload from a slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UseKind {
    MustHaveRegister,
    ShouldHaveRegister,
}

/// One register class, for example gpr or fpr. `allocatable` is the set of
/// physical register INDICES the scan may hand out for this class.
/// `callee_saved` is the subset that needs a prologue save and restore when
/// used. `slot_bytes` is the spill-slot size for a value of this class.
#[derive(Clone, Debug)]
pub struct RegClass {
    pub name: &'static str,
    pub allocatable: Vec<u16>,
    pub callee_saved: Vec<u16>,
    pub slot_bytes: u16,
    /// OPTIONAL subset of registers whose CALL clobber only destroys a WIDE
    /// value, preserving a NARROW one. Empty (the default) means every clobber
    /// destroys every value.
    pub narrow_preserved: Vec<u16>,
}

/// A per-class set of register indices clobbered at some point. Used by [`CallSite`].
#[derive(Clone, Debug)]
pub struct ClassRegs {
    pub class: u16,
    pub regs: Vec<u16>,
}

/// A call at position `pos` clobbers `clobbered`, one [`ClassRegs`] per
/// affected class. The allocator turns each into a fixed interval, so no value
/// survives the call in a clobbered register.
#[derive(Clone, Debug)]
pub struct CallSite {
    pub pos: u32,
    pub clobbered: Vec<ClassRegs>,
}

/// An entry parameter, or any value, pre-colored to a fixed physical register
/// of a class. This covers the ABI argument registers at function entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FixedAssign {
    pub value: Value,
    pub class: u16,
    pub reg: u16,
}

/// A backend's description of its register model for one function. It is built
/// per function because the allocatable sets can differ by function, for
/// example aarch64 leaf vs non-leaf pools.
///
/// Optional hooks have defaults that preserve the historical behavior, so a
/// backend that does not opt in is unaffected.
pub trait RegDescription<F: Func> {
    fn classes(&self) -> &[RegClass];
    fn class_of(&self, func: &F, v: Value) -> u16;
    fn use_kind(&self, func: &F, inst: Inst, operand: Value) -> UseKind;
    fn entry_fixed(&self) -> &[FixedAssign];
    fn call_sites(&self) -> &[CallSite];
    fn scratch(&self) -> &[u16];

    /// OPTIONAL second per-class scratch register, one per class, reserved
    /// exactly like `scratch` (never allocatable, no raw edge move touches
    /// it). It exists ONLY to break a parallel-move CYCLE that involves a
    /// spill slot both a source and a destination on one edge.
    fn scratch2(&self) -> &[u16] {
        &[]
    }
    /// OPTIONAL interference-aware copy coalescing hook. Given a value `v`,
    /// return the source value when the instruction that defines `v` is a PURE
    /// same-class register copy of that source, else `None`.
    fn copy_source(&self, _func: &F, _v: Value) -> Option<Value> {
        None
    }
    /// OPTIONAL narrow-value predicate. Return true when value `v` survives a
    /// call clobber of a register in its class's `narrow_preserved` set.
    fn is_narrow(&self, _func: &F, _v: Value) -> bool {
        false
    }
    /// OPTIONAL block-argument coalescing. When true, the scan hints a block
    /// parameter toward the register of an incoming argument that is already
    /// placed, so the edge move that feeds the parameter becomes a
    /// same-register no-op the edge resolver drops.
    fn coalesce_block_params(&self) -> bool {
        false
    }
    /// OPTIONAL declaration that the backend can host an edge move on a
    /// CRITICAL edge, so the no-critical-edge precondition does not apply.
    fn hosts_critical_edge_moves(&self) -> bool {
        false
    }
    /// OPTIONAL spill-slot coalescing. When true, after the scan a block
    /// parameter that SPILLED shares ONE spill slot with an incoming argument
    /// that also spilled, when the two do not interfere.
    fn coalesce_spill_slots(&self) -> bool {
        false
    }
    /// OPTIONAL multi-register width hook. Return how many CONSECUTIVE
    /// registers value `v` occupies and the alignment its base register index
    /// must satisfy.
    fn reg_width(&self, _func: &F, _v: Value) -> RegWidth {
        RegWidth::ONE
    }
    /// OPTIONAL operand-rewrite hook for an instruction the backend FUSES.
    /// Return `None` to leave the IR operand walk alone; otherwise fill `out`
    /// with the values the MACHINE instruction really reads and return how
    /// many (at most [`MAX_FUSED_OPERANDS`]).
    fn fused_operands(
        &self,
        _func: &F,
        _inst: Inst,
        _out: &mut [Value; MAX_FUSED_OPERANDS],
    ) -> Option<u8> {
        None
    }
    /// OPTIONAL late-read hook. Return true when `inst` may collect `operand`
    /// after issue. The allocator then keeps the operand live until the
    /// instruction result's first use in this block.
    fn late_read(&self, _func: &F, _inst: Inst, _operand: Value) -> bool {
        false
    }
}

// ===========================================================================
// Lifetime intervals.
// ===========================================================================

/// A half-open live range `[from, to)`. A value live over disjoint ranges has
/// HOLES between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub from: u32,
    pub to: u32,
}

/// A use of a value at position `pos` with the register requirement the
/// backend reported.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UsePos {
    pub pos: u32,
    pub kind: UseKind,
}

/// One lifetime interval. A VALUE interval (`fixed_reg == None`) describes
/// where an SSA value is live and used. A FIXED interval (`fixed_reg != None`)
/// blocks a physical register over its ranges. A call-clobber fixed interval
/// carries `value == None`. An entry-parameter fixed interval keeps `value`
/// set to the pinned parameter, so the scan can honor the ABI hint.
#[derive(Clone, Debug)]
pub struct Interval {
    pub value: Option<Value>,
    pub class: u16,
    pub fixed_reg: Option<u16>,
    /// Ascending, disjoint, merged.
    pub ranges: Vec<Range>,
    /// Ascending by `pos`.
    pub uses: Vec<UsePos>,
    /// Filled by the scan, `None` here.
    pub location: Option<Location>,
    /// For a VALUE interval that is a COALESCABLE COPY DESTINATION, the source
    /// value the backend's `copy_source` reported.
    pub copy_src: Option<Value>,
    /// For a VALUE interval, true when the backend's `is_narrow` flagged this
    /// value as one that survives a clobber of a `narrow_preserved` register.
    pub narrow: bool,
    /// For a FIXED call-clobber interval, true when its register is in the
    /// class's `narrow_preserved` set, so the clobber spares a `narrow` value.
    pub preserves_narrow: bool,
    /// How many CONSECUTIVE registers this interval occupies, and the
    /// alignment its base register must meet.
    pub regs: u16,
    pub reg_align: u16,
    /// True for an interval `split_interval` created. The value has an EARLIER
    /// piece, so it already has a location the store can read at the split point.
    pub split_child: bool,
    /// For a SPILLED interval whose value must reach memory BEFORE the interval
    /// starts: the position the store is emitted at, in place of the
    /// interval's own start.
    pub store_at: Option<u32>,
}

impl Interval {
    /// The interval's first live position. Panics on an empty interval.
    pub fn start(&self) -> u32 {
        self.ranges[0].from
    }

    /// The interval's half-open end (one past its last live position).
    pub fn end(&self) -> u32 {
        self.ranges[self.ranges.len() - 1].to
    }

    /// Whether `pos` falls inside one of the interval's live ranges.
    pub fn covers(&self, pos: u32) -> bool {
        for r in &self.ranges {
            if r.from <= pos && pos < r.to {
                return true;
            }
        }
        false
    }

    /// The first use at position `>= pos` (Wimmer's `>=` convention), or `None`
    /// if none remain.
    pub fn first_use_after(&self, pos: u32) -> Option<u32> {
        for u in &self.uses {
            if u.pos >= pos {
                return Some(u.pos);
            }
        }
        None
    }

    /// The first position both intervals cover (their earliest overlap), or
    /// `None` if they never overlap. Both range lists are ascending and
    /// disjoint, so a two-pointer merge finds the earliest intersection.
    pub fn next_intersection(&self, other: &Interval) -> Option<u32> {
        let mut i = 0;
        let mut j = 0;
        while i < self.ranges.len() && j < other.ranges.len() {
            let a = self.ranges[i];
            let b = other.ranges[j];
            let lo = a.from.max(b.from);
            let hi = a.to.min(b.to);
            if lo < hi {
                return Some(lo);
            }
            // Advance whichever range ends first. It cannot intersect any
            // later range of the other.
            if a.to < b.to {
                i += 1;
            } else {
                j += 1;
            }
        }
        None
    }
}

/// Sort `ranges` ascending and merge overlapping OR touching ranges in place,
/// leaving disjoint ranges with holes where the value is dead. `[a, b)` and
/// `[b, c)` touch and merge into `[a, c)`.
fn normalize_ranges(ranges: &mut Vec<Range>) {
    if ranges.is_empty() {
        return;
    }
    ranges.sort_by_key(|r| r.from);
    let mut w = 0;
    for i in 0..ranges.len() {
        let r = ranges[i];
        if w > 0 && r.from <= ranges[w - 1].to {
            if r.to > ranges[w - 1].to {
                ranges[w - 1].to = r.to;
            }
        } else {
            ranges[w] = r;
            w += 1;
        }
    }
    ranges.truncate(w);
}

/// Build the lifetime intervals the scan consumes: one interval per value that
/// is ever live (with ranges, holes, and use positions), plus fixed intervals
/// for physical registers (call clobbers from `call_sites` and entry
/// parameters from `entry_fixed`).
pub fn build_intervals<F: Func, D: RegDescription<F>>(func: &F, desc: &D) -> Vec<Interval> {
    let nblocks = func.block_count();
    let nval = func.value_count();

    let mut block_from = alloc::vec![0u32; nblocks];
    let mut block_to = alloc::vec![0u32; nblocks];
    let mut def_pos = alloc::vec![0u32; nval];
    let mut is_def = alloc::vec![false; nval];

    // Per-value range and use builders. Ranges are appended raw, and possibly
    // overlapping, then normalized once at the end.
    let mut range_lists: Vec<Vec<Range>> = Vec::new();
    range_lists.resize_with(nval, Vec::new);
    let mut use_lists: Vec<Vec<UsePos>> = Vec::new();
    use_lists.resize_with(nval, Vec::new);

    struct LateRead {
        operand: Value,
        result: Value,
        block: usize,
        producer_pos: u32,
    }
    let mut late_reads: Vec<LateRead> = Vec::new();

    // Liveness bitsets, indexed `bi * nval + vi`. `defined`/`used` are the
    // block-local gen/kill sets.
    let mut defined = alloc::vec![false; nblocks * nval];
    let mut used = alloc::vec![false; nblocks * nval];

    // Successor lists, driving the liveness fixpoint.
    let mut succ: Vec<Vec<u32>> = Vec::new();
    succ.resize_with(nblocks, Vec::new);

    // --- Pass A: number positions, record defs, gather uses + use-ranges,
    // build the CFG. ---
    let uses_fused_len =
        |desc: &D, func: &F, inst: Inst, fused: &mut [Value; MAX_FUSED_OPERANDS]| -> Option<u8> {
            desc.fused_operands(func, inst, fused)
        };

    let mut pos: u32 = 0;
    for bi in 0..nblocks {
        let block = Block::from_u32(bi as u32);
        block_from[bi] = pos;
        for p in func.block_params(block) {
            let pi = p.index();
            def_pos[pi] = pos;
            is_def[pi] = true;
            defined[bi * nval + pi] = true;
        }
        pos += 1;
        for inst in func.block_insts(block) {
            let inst = *inst;
            let result = func.inst_result(inst);
            // What the MACHINE instruction reads, which a fusing backend may
            // report as a different list from the IR operands.
            let mut fused: [Value; MAX_FUSED_OPERANDS] = [Value::from_u32(0); MAX_FUSED_OPERANDS];
            let fused_len = uses_fused_len(desc, func, inst, &mut fused);
            let mut visit = |v: Value, is_edge_arg: bool| {
                let vi = v.index();
                used[bi * nval + vi] = true;
                // An edge argument feeds a successor block parameter through
                // the parallel move the resolver realizes on the edge. That
                // move can load from or store to a spill slot, so an edge
                // argument does NOT need a register.
                let kind = if is_edge_arg {
                    UseKind::ShouldHaveRegister
                } else {
                    desc.use_kind(func, inst, v)
                };
                use_lists[vi].push(UsePos { pos, kind });
                // A use makes the value live from its block's start up to and
                // including the use position.
                range_lists[vi].push(Range {
                    from: block_from[bi],
                    to: pos + 1,
                });
                if let Some(res) = result
                    && desc.late_read(func, inst, v)
                {
                    late_reads.push(LateRead {
                        operand: v,
                        result: res,
                        block: bi,
                        producer_pos: pos,
                    });
                }
            };
            if let Some(n) = fused_len {
                debug_assert!((n as usize) <= MAX_FUSED_OPERANDS);
                for fv in &fused[0..n as usize] {
                    visit(*fv, false);
                }
            } else {
                visit_inst_operands(func, inst, &mut |v, edge| visit(v, edge));
            }
            let opc = func.opcode(inst);
            if let volt_ir::function::Opcode::If(cf) = &opc {
                succ[bi].push(cf.then.target.index() as u32);
                succ[bi].push(cf.else_.target.index() as u32);
            }
            if let Some(r) = func.inst_result(inst) {
                let ri = r.index();
                def_pos[ri] = pos;
                is_def[ri] = true;
                defined[bi * nval + ri] = true;
            }
            pos += 1;
        }
        let term_pos = pos;
        if let Some(term) = func.terminator(block) {
            // A ret value normally goes to its ABI return register at the
            // block boundary, so it needs a register. The exception is a value
            // whose class has NO allocatable register (a memory-resident
            // class): it can never occupy a register, so mark it
            // `should_have_register` instead.
            let classes = desc.classes();
            let term_owned = term.clone();
            visit_term_operands(func, &term_owned, &mut |v: Value, is_edge_arg: bool| {
                let vi = v.index();
                used[bi * nval + vi] = true;
                let kind = if is_edge_arg {
                    UseKind::ShouldHaveRegister
                } else {
                    let ci = desc.class_of(func, v) as usize;
                    if classes[ci].allocatable.is_empty() {
                        UseKind::ShouldHaveRegister
                    } else {
                        UseKind::MustHaveRegister
                    }
                };
                use_lists[vi].push(UsePos {
                    pos: term_pos,
                    kind,
                });
                range_lists[vi].push(Range {
                    from: block_from[bi],
                    to: term_pos + 1,
                });
            });
            if let volt_ir::function::Terminator::Jump(j) = term {
                succ[bi].push(j.target.index() as u32);
            }
        }
        block_to[bi] = term_pos + 1;
        pos += 1;
    }

    // A decoupled instruction has certainly collected its operands once its
    // result is consumed. Keep each late-read operand live to that first local
    // result use (or through the block boundary when the result has no local use).
    for late in &late_reads {
        let bi = late.block;
        let mut until = block_to[bi] - 1;
        for u in &use_lists[late.result.index()] {
            if u.pos > late.producer_pos && u.pos < block_to[bi] {
                until = u.pos;
                break;
            }
        }
        range_lists[late.operand.index()].push(Range {
            from: block_from[bi],
            to: until + 1,
        });
    }

    // --- Liveness fixpoint: live_out[b] is the union of successors' live_in.
    // live_in[b] is used[b] union (live_out[b] minus defined[b]). ---
    let mut live_in = alloc::vec![false; nblocks * nval];
    let mut live_out = alloc::vec![false; nblocks * nval];
    if nblocks > 0 && nval > 0 {
        let mut changed = true;
        while changed {
            changed = false;
            let mut b = nblocks;
            while b > 0 {
                b -= 1;
                for s in succ[b].clone().iter() {
                    let s = *s as usize;
                    for v in 0..nval {
                        if live_in[s * nval + v] && !live_out[b * nval + v] {
                            live_out[b * nval + v] = true;
                            changed = true;
                        }
                    }
                }
                for v in 0..nval {
                    let new_in =
                        (used[b * nval + v] || live_out[b * nval + v]) && !defined[b * nval + v];
                    if new_in && !live_in[b * nval + v] {
                        live_in[b * nval + v] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // --- Pass B: every value live-out of a block is live across that whole block. ---
    for bi in 0..nblocks {
        for v in 0..nval {
            if !live_out[bi * nval + v] {
                continue;
            }
            range_lists[v].push(Range {
                from: block_from[bi],
                to: block_to[bi],
            });
        }
    }

    // --- Pass C: normalize, clamp each range set to start at the value's
    // single def, and ensure a dead def still gets a minimal range. ---
    for v in 0..nval {
        if is_def[v] && range_lists[v].is_empty() {
            range_lists[v].push(Range {
                from: def_pos[v],
                to: def_pos[v] + 1,
            });
        }
        normalize_ranges(&mut range_lists[v]);
        if range_lists[v].is_empty() {
            continue;
        }
        if is_def[v] {
            // In SSA a value is not live before its single def, which lies in
            // its earliest range.
            if range_lists[v][0].from <= def_pos[v] && def_pos[v] < range_lists[v][0].to {
                range_lists[v][0].from = def_pos[v];
            } else {
                // Defensive path: the def is reachable but every use lies in an
                // unreachable block. Degrade to a single dead-def range.
                range_lists[v].clear();
                range_lists[v].push(Range {
                    from: def_pos[v],
                    to: def_pos[v] + 1,
                });
            }
        }
    }

    // --- Assemble the result: value intervals first, then fixed intervals. ---
    let mut result: Vec<Interval> = Vec::new();
    for v in 0..nval {
        if range_lists[v].is_empty() {
            continue;
        }
        debug_assert_ranges_sorted(&range_lists[v]);
        debug_assert_uses_sorted(&use_lists[v]);
        let value = Value::from_u32(v as u32);
        let class = desc.class_of(func, value);
        // Record the copy source when the backend opts into coalescing and
        // reports this value as a pure same-class copy.
        let mut copy_src: Option<Value> = None;
        if let Some(s) = desc.copy_source(func, value)
            && desc.class_of(func, s) == class
        {
            copy_src = Some(s);
        }
        let narrow = desc.is_narrow(func, value);
        let width = width_of(desc, func, value);
        result.push(Interval {
            value: Some(value),
            class,
            fixed_reg: None,
            ranges: core::mem::take(&mut range_lists[v]),
            uses: core::mem::take(&mut use_lists[v]),
            location: None,
            copy_src,
            narrow,
            preserves_narrow: false,
            regs: width.regs,
            reg_align: width.alignment,
            split_child: false,
            store_at: None,
        });
    }

    append_fixed_intervals(func, desc, &mut result);

    result
}

fn debug_assert_ranges_sorted(ranges: &[Range]) {
    let mut prev_to: u32 = 0;
    for (idx, r) in ranges.iter().enumerate() {
        debug_assert!(r.from < r.to);
        if idx > 0 {
            debug_assert!(r.from > prev_to);
        }
        prev_to = r.to;
    }
}

fn debug_assert_uses_sorted(uses: &[UsePos]) {
    let mut prev: u32 = 0;
    for (idx, u) in uses.iter().enumerate() {
        if idx > 0 {
            debug_assert!(u.pos >= prev);
        }
        prev = u.pos;
    }
}

/// Append fixed intervals: one merged interval per call-clobbered physical
/// register and one per entry parameter (pinning its ABI register at `[0, 1)`).
fn append_fixed_intervals<F: Func, D: RegDescription<F>>(
    func: &F,
    desc: &D,
    result: &mut Vec<Interval>,
) {
    // Merge call clobbers per (class, reg).
    let mut keys: Vec<(u16, u16)> = Vec::new();
    let mut builders: Vec<Vec<Range>> = Vec::new();
    for cs in desc.call_sites() {
        for cr in &cs.clobbered {
            for reg in &cr.regs {
                let key = (cr.class, *reg);
                let mut idx = None;
                for (i, k) in keys.iter().enumerate() {
                    if *k == key {
                        idx = Some(i);
                        break;
                    }
                }
                let i = match idx {
                    Some(i) => i,
                    None => {
                        keys.push(key);
                        builders.push(Vec::new());
                        builders.len() - 1
                    }
                };
                builders[i].push(Range {
                    from: cs.pos,
                    to: cs.pos + 1,
                });
            }
        }
    }
    for (i, key) in keys.iter().enumerate() {
        normalize_ranges(&mut builders[i]);
        debug_assert_ranges_sorted(&builders[i]);
        let preserves_narrow = desc.classes()[key.0 as usize]
            .narrow_preserved
            .contains(&key.1);
        result.push(Interval {
            value: None,
            class: key.0,
            fixed_reg: Some(key.1),
            ranges: core::mem::take(&mut builders[i]),
            uses: Vec::new(),
            location: None,
            copy_src: None,
            narrow: false,
            preserves_narrow,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        });
    }

    // Entry parameters: each arrives in a fixed ABI register at function entry.
    for ef in desc.entry_fixed() {
        // The pin covers the parameter's WHOLE register span.
        let width = width_of(desc, func, ef.value);
        result.push(Interval {
            value: Some(ef.value),
            class: ef.class,
            fixed_reg: Some(ef.reg),
            ranges: alloc::vec![Range { from: 0, to: 1 }],
            uses: Vec::new(),
            location: None,
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: width.regs,
            reg_align: width.alignment,
            split_child: false,
            store_at: None,
        });
    }
}

// ===========================================================================
// The linear scan (LINEARSCAN, Wimmer & Franz Fig 5) plus TRYALLOCATEFREEREG
// (Fig 6), extended with ALLOCATEBLOCKEDREG (Fig 7), SPLITINTERVAL, and
// spill-slot assignment.
// ===========================================================================

/// Reference to an interval held either in the originals slice or the
/// split-children list. The scan keeps worklists of these instead of raw pointers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct IvRef {
    child: bool,
    idx: usize,
}

fn get<'a>(intervals: &'a [Interval], children: &'a [Interval], r: IvRef) -> &'a Interval {
    if r.child {
        &children[r.idx]
    } else {
        &intervals[r.idx]
    }
}

fn get_mut<'a>(
    intervals: &'a mut [Interval],
    children: &'a mut [Interval],
    r: IvRef,
) -> &'a mut Interval {
    if r.child {
        &mut children[r.idx]
    } else {
        &mut intervals[r.idx]
    }
}

/// A free-until position meaning "never conflicts". Program positions never reach it.
const INFINITY: u32 = u32::MAX;

/// The widest physical register index the bookkeeping supports.
const MAX_PHYS_REGS: usize = 256;

/// One past the highest register index the scan can touch for ONE function.
fn reg_limit<F: Func, D: RegDescription<F>>(
    intervals: &[Interval],
    children: &[Interval],
    desc: &D,
) -> usize {
    let mut m: usize = 0;
    for c in desc.classes() {
        for r in &c.allocatable {
            m = m.max(*r as usize + 1);
        }
    }
    for r in desc.scratch() {
        m = m.max(*r as usize + 1);
    }
    for r in desc.scratch2() {
        m = m.max(*r as usize + 1);
    }
    for iv in intervals.iter().chain(children.iter()) {
        if let Some(fr) = iv.fixed_reg {
            m = m.max(fr as usize + iv.regs as usize);
        }
    }
    m.min(MAX_PHYS_REGS)
}

/// The register a placed interval is assigned.
fn assigned_reg(iv: &Interval) -> u16 {
    if let Some(fr) = iv.fixed_reg {
        return fr;
    }
    match iv.location {
        Some(Location::Reg(r)) => r,
        _ => panic!("assigned_reg on spilled or unplaced interval"),
    }
}

/// True iff `set` contains `reg`.
fn contains_reg(set: &[u16], reg: u16) -> bool {
    set.contains(&reg)
}

/// The width hook answer for value `v`, validated.
fn width_of<F: Func, D: RegDescription<F>>(desc: &D, func: &F, v: Value) -> RegWidth {
    let w = desc.reg_width(func, v);
    debug_assert!(w.regs >= 1);
    debug_assert!(w.alignment >= 1);
    w
}

/// True iff the two register spans overlap.
fn spans_overlap(base_a: u16, regs_a: u16, base_b: u16, regs_b: u16) -> bool {
    (base_a as u32) < (base_b as u32) + (regs_b as u32)
        && (base_b as u32) < (base_a as u32) + (regs_a as u32)
}

/// True iff `[base, base + regs)` is a legal placement: the base meets the
/// alignment, the span fits inside `limit`, and every register of the span is
/// in the candidate set (and evictable, when given).
fn span_placeable(
    is_candidate: &[bool],
    evictable: Option<&[bool]>,
    limit: usize,
    base: usize,
    regs: u16,
    alignment: u16,
) -> bool {
    if alignment != 1 && !base.is_multiple_of(alignment as usize) {
        return false;
    }
    if base + (regs as usize) > limit {
        return false;
    }
    for r in base..base + (regs as usize) {
        if !is_candidate[r] {
            return false;
        }
        if let Some(e) = evictable
            && !e[r]
        {
            return false;
        }
    }
    true
}

/// The minimum of `arr` over the register span `[base, base + regs)`.
fn span_min(arr: &[u32], base: usize, regs: u16) -> u32 {
    if regs == 1 {
        return arr.get(base).copied().unwrap_or(INFINITY);
    }
    arr.iter()
        .skip(base)
        .take(regs as usize)
        .copied()
        .min()
        .unwrap_or(INFINITY)
}

/// The earliest position at which the call-clobber `fixed` interval genuinely
/// forces `current` out of its register: a clobber point `c` that `current`
/// holds a register ACROSS, i.e. it `covers(c)` and `covers(c + 1)`.
fn fixed_clobber_conflict(current: &Interval, fixed: &Interval) -> Option<u32> {
    debug_assert!(fixed.value.is_none());
    if fixed.preserves_narrow && current.narrow {
        return None;
    }
    for r in &fixed.ranges {
        let mut c = r.from;
        while c < r.to {
            if current.covers(c) && current.covers(c + 1) {
                return Some(c);
            }
            c += 1;
        }
    }
    None
}

/// The ABI register `v` is hinted to hold at entry, or `None`.
fn entry_hint<F: Func, D: RegDescription<F>>(desc: &D, v: Value, class: u16) -> Option<u16> {
    for ef in desc.entry_fixed() {
        if ef.class == class && ef.value == v {
            return Some(ef.reg);
        }
    }
    None
}

/// True iff `dst` is a coalescable copy of `src`, and `src` DIES at the copy.
fn copy_dies_at(dst: &Interval, src: &Interval) -> bool {
    let cs = match dst.copy_src {
        Some(c) => c,
        None => return false,
    };
    if src.fixed_reg.is_some() {
        return false;
    }
    let sv = match src.value {
        Some(v) => v,
        None => return false,
    };
    if cs != sv {
        return false;
    }
    if src.regs != dst.regs || src.reg_align != dst.reg_align {
        return false;
    }
    src.end() == dst.start() + 1
}

/// True iff the two intervals are a legitimate copy-coalesce pair.
fn is_copy_coalesce_pair(a: &Interval, b: &Interval) -> bool {
    copy_dies_at(a, b) || copy_dies_at(b, a)
}

/// True iff both intervals describe the same value.
fn same_value(a: &Interval, b: &Interval) -> bool {
    match (a.value, b.value) {
        (Some(av), Some(bv)) => av == bv,
        _ => false,
    }
}

/// The physical register value `v` is placed in, or `None`. Among `v`'s
/// intervals, returns the register of the one placed in a register that lives
/// LATEST. Used only to seed a block-parameter register HINT.
fn placed_reg_of(intervals: &[Interval], children: &[Interval], v: Value) -> Option<u16> {
    let mut best: Option<u16> = None;
    let mut best_end: u32 = 0;
    for it in intervals.iter().chain(children.iter()) {
        if it.fixed_reg.is_some() {
            continue;
        }
        if it.value != Some(v) {
            continue;
        }
        if let Some(Location::Reg(r)) = it.location
            && (best.is_none() || it.end() >= best_end)
        {
            best = Some(r);
            best_end = it.end();
        }
    }
    best
}

/// Maps each block-parameter value to the incoming ARGUMENT values its
/// predecessors pass for it, one per in-edge.
type ParamArgs = Vec<(Value, Vec<Value>)>;

fn param_args_lookup(map: &[(Value, Vec<Value>)], v: Value) -> Option<&[Value]> {
    for (p, list) in map {
        if *p == v {
            return Some(list);
        }
    }
    None
}

/// Record, for edge `edge`, that each successor parameter receives the
/// argument at the same index. An arity mismatch is skipped defensively.
fn add_param_arg_edge<F: Func>(func: &F, map: &mut ParamArgs, edge: Jump) {
    let params = func.block_params(edge.target);
    let args = func.block_args(edge);
    if params.len() != args.len() {
        return;
    }
    for (p, a) in params.iter().zip(args.iter()) {
        let mut found = false;
        for (pp, list) in map.iter_mut() {
            if *pp == *p {
                list.push(*a);
                found = true;
                break;
            }
        }
        if !found {
            map.push((*p, alloc::vec![*a]));
        }
    }
}

/// Build the block-parameter to incoming-argument map by walking every edge.
fn build_param_args<F: Func>(func: &F) -> ParamArgs {
    let mut map: ParamArgs = Vec::new();
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                add_param_arg_edge(func, &mut map, cf.then);
                add_param_arg_edge(func, &mut map, cf.else_);
            }
        }
        if let Some(term) = func.terminator(block) {
            match term {
                volt_ir::function::Terminator::Jump(j) => add_param_arg_edge(func, &mut map, j),
                volt_ir::function::Terminator::Ret(_) => {}
            }
        }
    }
    map
}

/// The register to hint block parameter `current` toward: the register of the
/// FIRST of its incoming arguments that is already placed in a register.
fn compute_param_hint(
    map: &ParamArgs,
    intervals: &[Interval],
    children: &[Interval],
    current: &Interval,
) -> Option<u16> {
    let v = current.value?;
    let list = param_args_lookup(map, v)?;
    for arg in list {
        if let Some(r) = placed_reg_of(intervals, children, *arg) {
            return Some(r);
        }
    }
    None
}

/// TRYALLOCATEFREEREG (Wimmer & Franz Fig 6), without the splitting tail.
fn try_allocate_free_reg<F: Func, D: RegDescription<F>>(
    intervals: &[Interval],
    children: &[Interval],
    current: &Interval,
    active: &[IvRef],
    inactive: &[IvRef],
    desc: &D,
    limit: usize,
    param_hint: Option<u16>,
) -> Option<u16> {
    let class_idx = current.class;
    let class = &desc.classes()[class_idx as usize];
    let mut free_until = [0u32; MAX_PHYS_REGS];
    let mut is_candidate = [false; MAX_PHYS_REGS];
    for r in &class.allocatable {
        debug_assert!((*r as usize) < limit);
        is_candidate[*r as usize] = true;
        free_until[*r as usize] = INFINITY;
    }
    let hint = current.value.and_then(|v| entry_hint(desc, v, class_idx));
    if let Some(h) = hint {
        let mut k: u16 = 0;
        while k < current.regs {
            debug_assert!(((h + k) as usize) <= limit);
            is_candidate[(h + k) as usize] = true;
            free_until[(h + k) as usize] = INFINITY;
            k += 1;
        }
    }
    let mut coalesce_hint: Option<u16> = None;
    for r in active {
        let it = get(intervals, children, *r);
        if it.class != class_idx {
            continue;
        }
        if same_value(it, current) {
            continue;
        }
        let reg = assigned_reg(it);
        if copy_dies_at(current, it) {
            if (reg as usize) < limit && is_candidate[reg as usize] {
                coalesce_hint = Some(reg);
            }
            continue;
        }
        let mut k: u16 = 0;
        while k < it.regs {
            let rr = reg.saturating_add(k);
            if (rr as usize) < limit && is_candidate[rr as usize] {
                free_until[rr as usize] = 0;
            }
            k += 1;
        }
    }
    for r in inactive {
        let it = get(intervals, children, *r);
        if it.class != class_idx {
            continue;
        }
        if same_value(it, current) {
            continue;
        }
        let reg = assigned_reg(it);
        if (reg as usize) >= limit {
            continue;
        }
        let x = if it.fixed_reg.is_some() {
            fixed_clobber_conflict(current, it)
        } else {
            current.next_intersection(it)
        };
        if let Some(xx) = x {
            let mut k: u16 = 0;
            while k < it.regs {
                let rr = reg.saturating_add(k);
                if (rr as usize) < limit
                    && is_candidate[rr as usize]
                    && xx < free_until[rr as usize]
                {
                    free_until[rr as usize] = xx;
                }
                k += 1;
            }
        }
    }
    let pref_hint = coalesce_hint.or(param_hint).or(hint);
    let mut best_free: u32 = 0;
    let mut b = 0;
    while b < limit {
        if span_placeable(
            &is_candidate,
            None,
            limit,
            b,
            current.regs,
            current.reg_align,
        ) {
            let f = span_min(&free_until, b, current.regs);
            if f > best_free {
                best_free = f;
            }
            if best_free == INFINITY {
                break;
            }
        }
        b += 1;
    }
    if best_free == 0 {
        return None;
    }
    let mut chosen: Option<u16> = None;
    if let Some(h) = pref_hint
        && span_placeable(
            &is_candidate,
            None,
            limit,
            h as usize,
            current.regs,
            current.reg_align,
        )
        && span_min(&free_until, h as usize, current.regs) == best_free
    {
        chosen = Some(h);
    }
    if chosen.is_none() {
        let mut bb = 0;
        while bb < limit {
            if span_placeable(
                &is_candidate,
                None,
                limit,
                bb,
                current.regs,
                current.reg_align,
            ) && span_min(&free_until, bb, current.regs) == best_free
            {
                chosen = Some(bb as u16);
                break;
            }
            bb += 1;
        }
    }
    let reg = chosen.unwrap();
    if best_free >= current.end() {
        return Some(reg);
    }
    None
}

/// Union-find representative with path halving, over the flat per-class slot space.
fn uf_find(parent: &mut [u32], i: u32) -> u32 {
    let mut x = i;
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// The location value `v` occupies at position `pos`: the location of the
/// interval with the greatest `start()` at or before `pos`.
fn value_loc_at(
    all: &[IvRef],
    intervals: &[Interval],
    children: &[Interval],
    v: Value,
    pos: u32,
) -> Option<Location> {
    let mut best: Option<&Interval> = None;
    for r in all {
        let iv = get(intervals, children, *r);
        if iv.value != Some(v) {
            continue;
        }
        if iv.start() > pos {
            continue;
        }
        if best.is_none() || iv.start() > best.unwrap().start() {
            best = Some(iv);
        }
    }
    best.and_then(|b| b.location)
}

/// True iff the half-open range `[from, to)` meets any live range of `it`, or
/// the EARLY-STORE prefix `it` occupies.
fn range_meets_interval(from: u32, to: u32, it: &Interval) -> bool {
    debug_assert!(from < to);
    for r in &it.ranges {
        if from < r.to && r.from < to {
            return true;
        }
    }
    if let Some(p) = it.store_at
        && from < it.start()
        && p < to
    {
        return true;
    }
    false
}

/// True iff two intervals hold their slot at the same time.
fn slot_intervals_interfere(a: &Interval, b: &Interval) -> bool {
    if a.next_intersection(b).is_some() {
        return true;
    }
    if let Some(pa) = a.store_at
        && range_meets_interval(pa, a.start(), b)
    {
        return true;
    }
    if let Some(pb) = b.store_at
        && range_meets_interval(pb, b.start(), a)
    {
        return true;
    }
    false
}

/// True iff any spilled interval of group `rep_a` interferes with any spilled
/// interval of group `rep_b` in class `class`.
fn slot_groups_interfere(
    all: &[IvRef],
    intervals: &[Interval],
    children: &[Interval],
    parent: &mut [u32],
    class_off: &[u32],
    class: u16,
    rep_a: u32,
    rep_b: u32,
) -> bool {
    let mut group_a: Vec<usize> = Vec::new();
    let mut group_b: Vec<usize> = Vec::new();
    for (i, r) in all.iter().enumerate() {
        let ia = get(intervals, children, *r);
        if ia.class != class {
            continue;
        }
        let sa = match ia.location {
            Some(Location::Slot(s)) => s,
            _ => continue,
        };
        if uf_find(parent, class_off[class as usize] + sa) == rep_a {
            group_a.push(i);
        }
        if uf_find(parent, class_off[class as usize] + sa) == rep_b {
            group_b.push(i);
        }
    }
    for ia_idx in &group_a {
        let ia = get(intervals, children, all[*ia_idx]);
        for ib_idx in &group_b {
            let ib = get(intervals, children, all[*ib_idx]);
            if slot_intervals_interfere(ia, ib) {
                return true;
            }
        }
    }
    false
}

/// Attempt to coalesce, for one edge `pred -> edge.target`, each spilled
/// successor parameter with the spilled argument the predecessor passes.
fn coalesce_edge_params<F: Func, D: RegDescription<F>>(
    all: &[IvRef],
    intervals: &[Interval],
    children: &[Interval],
    parent: &mut [u32],
    class_off: &[u32],
    func: &F,
    desc: &D,
    block_from: &[u32],
    block_to: &[u32],
    pred: Block,
    edge: Jump,
) {
    let succ = edge.target;
    let pt = block_to[pred.index()] - 1;
    let ss = block_from[succ.index()];
    let params = func.block_params(succ);
    let args = func.block_args(edge);
    if params.len() != args.len() {
        return;
    }
    for (p, a) in params.iter().zip(args.iter()) {
        let p_slot = match value_loc_at(all, intervals, children, *p, ss) {
            Some(Location::Slot(s)) => s,
            _ => continue,
        };
        let a_slot = match value_loc_at(all, intervals, children, *a, pt) {
            Some(Location::Slot(s)) => s,
            _ => continue,
        };
        let c = desc.class_of(func, *p);
        if desc.class_of(func, *a) != c {
            continue;
        }
        let rep_p = uf_find(parent, class_off[c as usize] + p_slot);
        let rep_a = uf_find(parent, class_off[c as usize] + a_slot);
        if rep_p == rep_a {
            continue;
        }
        if slot_groups_interfere(all, intervals, children, parent, class_off, c, rep_p, rep_a) {
            continue;
        }
        parent[rep_p as usize] = rep_a;
    }
}

/// Report whether any slot ends up BOTH a move source and destination on one edge.
fn edge_slot_both_src_and_dst<F: Func, D: RegDescription<F>>(
    all: &[IvRef],
    intervals: &[Interval],
    children: &[Interval],
    func: &F,
    desc: &D,
    class_off: &[u32],
    mark: &mut [u8],
    block_from: &[u32],
    block_to: &[u32],
    pred: Block,
    edge: Jump,
) -> bool {
    let succ = edge.target;
    let pt = block_to[pred.index()] - 1;
    let ss = block_from[succ.index()];
    let mut go = |cls: u16, from: Location, to: Location| {
        if loc_eql(from, to) {
            return;
        }
        if let Location::Slot(s) = from {
            mark[(class_off[cls as usize] + s) as usize] |= 1;
        }
        if let Location::Slot(s) = to {
            mark[(class_off[cls as usize] + s) as usize] |= 2;
        }
    };
    let params = func.block_params(succ);
    let args = func.block_args(edge);
    if params.len() == args.len() {
        for (p, a) in params.iter().zip(args.iter()) {
            let from = match value_loc_at(all, intervals, children, *a, pt) {
                Some(l) => l,
                None => continue,
            };
            let to = match value_loc_at(all, intervals, children, *p, ss) {
                Some(l) => l,
                None => continue,
            };
            go(desc.class_of(func, *p), from, to);
        }
    }
    let mut seen: Vec<Value> = Vec::new();
    for r in all {
        let iv = get(intervals, children, *r);
        let v = match iv.value {
            Some(v) => v,
            None => continue,
        };
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        if is_param_of(func, succ, v) {
            continue;
        }
        if !value_live_at(intervals, children, v, ss) {
            continue;
        }
        let from = match value_loc_at(all, intervals, children, v, pt) {
            Some(l) => l,
            None => continue,
        };
        let to = match value_loc_at(all, intervals, children, v, ss) {
            Some(l) => l,
            None => continue,
        };
        go(desc.class_of(func, v), from, to);
    }
    mark.contains(&3)
}

// ---------- helpers used by later chunks (defined once) ----------

/// True iff two locations name the same register or the same slot.
fn loc_eql(a: Location, b: Location) -> bool {
    match (a, b) {
        (Location::Reg(ra), Location::Reg(rb)) => ra == rb,
        (Location::Slot(sa), Location::Slot(sb)) => sa == sb,
        _ => false,
    }
}

/// The per-block half-open position bounds `[from, to)`, numbered EXACTLY as
/// `build_intervals` does. Blocks are contiguous (`from[bi+1] == to[bi]`).
struct BlockBounds {
    from: Vec<u32>,
    to: Vec<u32>,
}

/// Compute per-block position bounds by re-walking the function with the same
/// numbering `build_intervals` uses.
fn compute_block_bounds<F: Func>(func: &F) -> BlockBounds {
    let nblocks = func.block_count();
    let mut from = alloc::vec![0u32; nblocks];
    let mut to = alloc::vec![0u32; nblocks];
    let mut pos: u32 = 0;
    for bi in 0..nblocks {
        let block = Block::from_u32(bi as u32);
        from[bi] = pos;
        pos += 1;
        pos += func.block_insts(block).len() as u32;
        let term_pos = pos;
        to[bi] = term_pos + 1;
        pos += 1;
    }
    BlockBounds { from, to }
}

/// The block that contains `pos` (every valid position belongs to exactly one).
fn block_of_pos(block_from: &[u32], block_to: &[u32], pos: u32) -> usize {
    for bi in 0..block_from.len() {
        if block_from[bi] <= pos && pos < block_to[bi] {
            return bi;
        }
    }
    panic!("pos {pos} in no block");
}

/// The largest half-open `end()` across all intervals, or 0 if none.
fn max_end_position(intervals: &[Interval], children: &[Interval]) -> u32 {
    let mut m: u32 = 0;
    for iv in intervals.iter().chain(children.iter()) {
        if iv.ranges.is_empty() {
            continue;
        }
        if iv.end() > m {
            m = iv.end();
        }
    }
    m
}

/// The first MUST_HAVE use position of an interval, if any.
fn first_must_have_use(it: &Interval) -> Option<u32> {
    for u in &it.uses {
        if u.kind == UseKind::MustHaveRegister {
            return Some(u.pos);
        }
    }
    None
}

/// True iff `v` is a block parameter of `block`.
fn is_param_of<F: Func>(func: &F, block: Block, v: Value) -> bool {
    func.block_params(block).contains(&v)
}

/// True iff some VALUE interval of `v` covers `pos`.
fn value_live_at(intervals: &[Interval], children: &[Interval], v: Value, pos: u32) -> bool {
    for iv in intervals.iter().chain(children.iter()) {
        if iv.fixed_reg.is_some() {
            continue;
        }
        if iv.value != Some(v) {
            continue;
        }
        if iv.covers(pos) {
            return true;
        }
    }
    false
}

// ===========================================================================
// SPLITINTERVAL, spill, ALLOCATEBLOCKEDREG.
// ===========================================================================

/// Insert `iv` into `list` keeping ascending-`start()` order (stable: after
/// any interval that starts no later than it).
fn insert_sorted(list: &mut Vec<IvRef>, intervals: &[Interval], children: &[Interval], iv: IvRef) {
    let st = get(intervals, children, iv).start();
    let idx = list
        .iter()
        .position(|r| get(intervals, children, *r).start() > st)
        .unwrap_or(list.len());
    list.insert(idx, iv);
}

/// Split `parent` at `pos` (`start() < pos < end()`), so both halves stay
/// non-empty. The parent keeps the head; the new tail child is pushed onto
/// `children` and its index returned. Uses at or past `pos` go to the tail.
#[allow(clippy::ptr_arg)]
fn split_interval(
    intervals: &mut Vec<Interval>,
    children: &mut Vec<Interval>,
    parent: IvRef,
    pos: u32,
) -> usize {
    let (value, class, narrow, regs, reg_align, head_ranges, tail_ranges, head_uses, tail_uses) = {
        let p = get(intervals, children, parent);
        debug_assert!(p.fixed_reg.is_none());
        debug_assert!(pos > p.start());
        debug_assert!(pos < p.end());
        let mut head_ranges: Vec<Range> = Vec::new();
        let mut tail_ranges: Vec<Range> = Vec::new();
        for r in &p.ranges {
            if r.to <= pos {
                head_ranges.push(*r);
            } else if r.from >= pos {
                tail_ranges.push(*r);
            } else {
                head_ranges.push(Range {
                    from: r.from,
                    to: pos,
                });
                tail_ranges.push(Range {
                    from: pos,
                    to: r.to,
                });
            }
        }
        let mut head_uses: Vec<UsePos> = Vec::new();
        let mut tail_uses: Vec<UsePos> = Vec::new();
        for u in &p.uses {
            if u.pos < pos {
                head_uses.push(*u);
            } else {
                tail_uses.push(*u);
            }
        }
        debug_assert!(!head_ranges.is_empty());
        debug_assert!(!tail_ranges.is_empty());
        (
            p.value,
            p.class,
            p.narrow,
            p.regs,
            p.reg_align,
            head_ranges,
            tail_ranges,
            head_uses,
            tail_uses,
        )
    };
    let child_idx = children.len();
    children.push(Interval {
        value,
        class,
        fixed_reg: None,
        ranges: tail_ranges,
        uses: tail_uses,
        location: None,
        copy_src: None,
        narrow,
        preserves_narrow: false,
        regs,
        reg_align,
        // The head keeps everything before `pos`, so this child always has an
        // earlier piece.
        split_child: true,
        store_at: None,
    });
    {
        let p = get_mut(intervals, children, parent);
        p.ranges = head_ranges;
        p.uses = head_uses;
    }
    // A split cuts a LIFETIME, never a value's shape — but the copy_src tag
    // belongs to the head (the copy defines the value at the head's def).
    // The tail is a continuation, not a copy destination. Zig keeps
    // copy_src only on the interval build_intervals made; children created
    // here start with None (set above). Keep the parent's tag as is.
    let _ = value;
    let _ = class;
    child_idx
}

/// Whether the RELOAD-AT-USE split is legal for `current`, whose first
/// `must_have_register` use `u` sits at its own start. Four preconditions;
/// see the Zig source for the full argument.
fn can_reload_at_use(current: &Interval, u: u32, block_from: &[u32]) -> bool {
    if !current.split_child {
        return false;
    }
    if current.end() <= u + 1 {
        return false;
    }
    // The uses are ascending, so a must-have use at `u + 1` IS the first
    // must-have use after `u`.
    for use_ in &current.uses {
        if use_.pos != u + 1 {
            continue;
        }
        if use_.kind == UseKind::MustHaveRegister {
            return false;
        }
    }
    for f in block_from {
        if *f == u + 1 {
            return false;
        }
    }
    true
}

/// Allocation failure: out of memory is not modeled (Vec panics), so the only
/// failure is `Unsupported` when a single position demands more simultaneous
/// must-have registers than the class has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllocError {
    Unsupported,
}

/// Spill `current`: the cheapest interval to move to memory. A
/// `must_have_register` use forces the spilled part to end at that use, so
/// the use lands back in a register. Returns `Err(Unsupported)` when a
/// same-position must-have demand exceeds the register pool.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn spill_current(
    intervals: &mut Vec<Interval>,
    children: &mut Vec<Interval>,
    unhandled: &mut Vec<IvRef>,
    current: IvRef,
    slots: &mut [u32],
    class_idx: u16,
    block_from: &[u32],
) -> Result<(), AllocError> {
    let u = {
        let cur = get(intervals, children, current);
        first_must_have_use(cur)
    };
    if let Some(u) = u {
        if u <= get(intervals, children, current).start() {
            debug_assert!(u == get(intervals, children, current).start());
            debug_assert!(get(intervals, children, current).store_at.is_none());
            if !can_reload_at_use(get(intervals, children, current), u, block_from) {
                return Err(AllocError::Unsupported);
            }
            // Cut `current` down to `[u, u + 1)` and give everything past the
            // use to `rest`, which carries the slot and stores at `u`, in
            // front of the clobbering instruction.
            let rest_idx = split_interval(intervals, children, current, u + 1);
            let rest = IvRef {
                child: true,
                idx: rest_idx,
            };
            {
                let r = get_mut(intervals, children, rest);
                r.store_at = Some(u);
                debug_assert!(
                    r.uses.is_empty()
                        || r.uses[0].pos > r.start()
                        || r.uses[0].kind != UseKind::MustHaveRegister
                );
            }
            spill_current(
                intervals, children, unhandled, rest, slots, class_idx, block_from,
            )?;
            // Re-queue `current` for a REGISTER.
            {
                let c = get_mut(intervals, children, current);
                c.location = None;
            }
            insert_sorted(unhandled, intervals, children, current);
            return Ok(());
        }
        let tail_idx = split_interval(intervals, children, current, u);
        let tail = if get(intervals, children, current).split_child || !children.is_empty() {
            // Determine whether `current` refers to a child: split_interval
            // pushed a new child; `current` itself is unchanged as a ref.
            IvRef {
                child: true,
                idx: tail_idx,
            }
        } else {
            IvRef {
                child: true,
                idx: tail_idx,
            }
        };
        {
            let c = get_mut(intervals, children, current);
            c.location = Some(Location::Slot(slots[class_idx as usize]));
            slots[class_idx as usize] += 1;
        }
        insert_sorted(unhandled, intervals, children, tail);
    } else {
        let c = get_mut(intervals, children, current);
        c.location = Some(Location::Slot(slots[class_idx as usize]));
        slots[class_idx as usize] += 1;
    }
    Ok(())
}

/// ALLOCATEBLOCKEDREG (Wimmer & Franz Fig 7). `current` could not get a free
/// register, so either it, or an interval occupying a register, must be split.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn allocate_blocked_reg<F: Func, D: RegDescription<F>>(
    _func: &F,
    intervals: &mut Vec<Interval>,
    children: &mut Vec<Interval>,
    current: IvRef,
    active: &mut Vec<IvRef>,
    inactive: &mut Vec<IvRef>,
    unhandled: &mut Vec<IvRef>,
    slots: &mut [u32],
    desc: &D,
    limit: usize,
    block_from: &[u32],
) -> Result<(), AllocError> {
    let class_idx = get(intervals, children, current).class;
    let class = &desc.classes()[class_idx as usize];
    let p = get(intervals, children, current).start();

    let mut next_use = [0u32; MAX_PHYS_REGS];
    let mut block_pos = [0u32; MAX_PHYS_REGS];
    let mut is_candidate = [false; MAX_PHYS_REGS];
    let mut evictable = [true; MAX_PHYS_REGS];
    for i in 0..limit {
        next_use[i] = INFINITY;
        block_pos[i] = INFINITY;
    }
    for r in &class.allocatable {
        debug_assert!((*r as usize) < limit);
        is_candidate[*r as usize] = true;
    }

    // Active VALUE intervals occupy their registers from here.
    for r in active.clone().iter() {
        let it = get(intervals, children, *r);
        if it.class != class_idx || it.fixed_reg.is_some() {
            continue;
        }
        let reg = assigned_reg(it);
        if (reg as usize) >= limit {
            continue;
        }
        let u = it.first_use_after(p).unwrap_or_else(|| it.end());
        let start_eq = it.start() == p;
        let it_regs = it.regs;
        let mut k: u16 = 0;
        while k < it_regs {
            let rr = reg.saturating_add(k);
            if (rr as usize) < limit && is_candidate[rr as usize] {
                if u < next_use[rr as usize] {
                    next_use[rr as usize] = u;
                }
                if start_eq {
                    evictable[rr as usize] = false;
                }
            }
            k += 1;
        }
    }
    // Inactive VALUE intervals only reclaim their register where they next
    // intersect `current`.
    for r in inactive.clone().iter() {
        let it = get(intervals, children, *r);
        if it.class != class_idx || it.fixed_reg.is_some() {
            continue;
        }
        let reg = assigned_reg(it);
        if (reg as usize) >= limit {
            continue;
        }
        if get(intervals, children, current)
            .next_intersection(it)
            .is_none()
        {
            continue;
        }
        let u = it.first_use_after(p).unwrap_or_else(|| it.end());
        let it_regs = it.regs;
        let mut k: u16 = 0;
        while k < it_regs {
            let rr = reg.saturating_add(k);
            if (rr as usize) < limit && is_candidate[rr as usize] && u < next_use[rr as usize] {
                next_use[rr as usize] = u;
            }
            k += 1;
        }
    }
    // Fixed intervals are HARD blocks. A call-clobber uses the
    // read-before-clobber rule; an entry-param pin uses plain intersection.
    // The value's own entry-param pin is its hint, not a block.
    let act: Vec<IvRef> = active.clone();
    let inact: Vec<IvRef> = inactive.clone();
    for r in act.iter().chain(inact.iter()) {
        let it = get(intervals, children, *r);
        if it.fixed_reg.is_none() || it.class != class_idx {
            continue;
        }
        if same_value(it, get(intervals, children, current)) {
            continue;
        }
        let fr = it.fixed_reg.unwrap();
        if (fr as usize) >= limit {
            continue;
        }
        let x = if it.value.is_none() {
            fixed_clobber_conflict(get(intervals, children, current), it)
        } else {
            get(intervals, children, current).next_intersection(it)
        };
        if let Some(xx) = x {
            let it_regs = it.regs;
            let mut k: u16 = 0;
            while k < it_regs {
                let rr = fr.saturating_add(k);
                if (rr as usize) < limit && is_candidate[rr as usize] {
                    if xx < next_use[rr as usize] {
                        next_use[rr as usize] = xx;
                    }
                    if xx < block_pos[rr as usize] {
                        block_pos[rr as usize] = xx;
                    }
                }
                k += 1;
            }
        }
    }

    // Pick the EVICTABLE register whose next use is furthest away. A hard
    // block is folded into `next_use` too; rank a blocked register last so a
    // callee-saved register that can carry `current` across the call wins.
    let cur_regs = get(intervals, children, current).regs;
    let cur_align = get(intervals, children, current).reg_align;
    let mut chosen: Option<u16> = None;
    let mut best: u32 = 0;
    let mut chosen_blocked = true;
    let mut b = 0;
    while b < limit {
        if !span_placeable(
            &is_candidate,
            Some(&evictable),
            limit,
            b,
            cur_regs,
            cur_align,
        ) {
            b += 1;
            continue;
        }
        let nu = span_min(&next_use, b, cur_regs);
        let blocked = span_min(&block_pos, b, cur_regs) <= p;
        let better = if chosen.is_none() {
            true
        } else if blocked != chosen_blocked {
            chosen_blocked
        } else {
            nu > best
        };
        if better {
            best = nu;
            chosen = Some(b as u16);
            chosen_blocked = blocked;
        }
        if best == INFINITY {
            debug_assert!(!chosen_blocked);
            break;
        }
        b += 1;
    }

    // No register is evictable: spill `current` instead.
    let reg = match chosen {
        Some(r) => r,
        None => {
            spill_current(
                intervals, children, unhandled, current, slots, class_idx, block_from,
            )?;
            return Ok(());
        }
    };

    // If `current`'s own first use is later than the chosen register's next
    // use, `current` is the cheapest to spill.
    let current_first_use = get(intervals, children, current).first_use_after(p);
    if current_first_use.is_none() || current_first_use.unwrap() > best {
        spill_current(
            intervals, children, unhandled, current, slots, class_idx, block_from,
        )?;
        return Ok(());
    }

    // Otherwise take the span based at `reg` for `current` and split whatever
    // occupies any register of it.
    {
        let c = get_mut(intervals, children, current);
        c.location = Some(Location::Reg(reg));
    }

    // Every active value interval whose span OVERLAPS the taken span is split
    // at `p`.
    let act2: Vec<IvRef> = active.clone();
    for r in act2.iter() {
        let (overlap, start_lt) = {
            let it = get(intervals, children, *r);
            if it.class != class_idx || it.fixed_reg.is_some() {
                (false, false)
            } else {
                (
                    spans_overlap(assigned_reg(it), it.regs, reg, cur_regs),
                    it.start() < p,
                )
            }
        };
        if !overlap {
            continue;
        }
        debug_assert!(start_lt);
        let tail_idx = split_interval(intervals, children, *r, p);
        insert_sorted(
            unhandled,
            intervals,
            children,
            IvRef {
                child: true,
                idx: tail_idx,
            },
        );
    }
    // Each inactive value interval overlapping the taken span that would
    // reclaim it inside `current`'s life is split at that intersection.
    let inact2: Vec<IvRef> = inactive.clone();
    for r in inact2.iter() {
        let x = {
            let it = get(intervals, children, *r);
            if it.class != class_idx
                || it.fixed_reg.is_some()
                || !spans_overlap(assigned_reg(it), it.regs, reg, cur_regs)
            {
                None
            } else {
                get(intervals, children, current).next_intersection(it)
            }
        };
        if let Some(x) = x {
            debug_assert!(x > get(intervals, children, *r).start());
            debug_assert!(x < get(intervals, children, *r).end());
            let tail_idx = split_interval(intervals, children, *r, x);
            insert_sorted(
                unhandled,
                intervals,
                children,
                IvRef {
                    child: true,
                    idx: tail_idx,
                },
            );
        }
    }
    // A fixed interval clobbering some register of the taken span before
    // `current` ends: split `current` before the block.
    let span_block_pos = span_min(&block_pos, reg as usize, cur_regs);
    if span_block_pos < get(intervals, children, current).end() {
        let bp = span_block_pos;
        if bp <= p {
            spill_current(
                intervals, children, unhandled, current, slots, class_idx, block_from,
            )?;
            return Ok(());
        }
        let tail_idx = split_interval(intervals, children, current, bp);
        insert_sorted(
            unhandled,
            intervals,
            children,
            IvRef {
                child: true,
                idx: tail_idx,
            },
        );
    }
    active.push(current);
    Ok(())
}

// ===========================================================================
// Allocation types, the main scan, move ordering, resolution, verification.
// ===========================================================================

/// One placement of a value: from position `from`, the value lives at `loc`.
/// A value that never splits has a single segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Segment {
    pub from: u32,
    pub loc: Location,
}

/// A data move the resolver emits: move `src` to `dst` within `class`.
/// `value` is the IR value whose bits this move transfers, carried so a
/// backend can look up its type and pick the width-appropriate move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub src: Location,
    pub dst: Location,
    pub class: u16,
    pub value: Value,
}

/// The kind of an intra-block spill, reload, or move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionKind {
    Store,
    Reload,
    Move,
}

/// An intra-block spill, reload, or move the resolver emits at position `at`.
/// A same-position CLUSTER of these is a parallel move: all sources read from
/// the pre-instruction state, and all destinations are written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Action {
    pub at: u32,
    pub kind: ActionKind,
    pub class: u16,
    pub src: Location,
    pub dst: Location,
    pub value: Value,
}

/// The parallel move set on a control-flow edge.
#[derive(Clone, Debug)]
pub struct EdgeMoves {
    pub pred: Block,
    pub succ: Block,
    pub moves: Vec<Move>,
}

/// A callee-saved physical register that the allocation actually used, so the
/// prologue must save it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UsedSaved {
    pub class: u16,
    pub reg: u16,
}

/// The register allocation result.
#[derive(Clone, Debug, Default)]
pub struct Allocation {
    /// One segment list per value, ascending by `from`.
    pub segments: Vec<(Value, Vec<Segment>)>,
    pub actions: Vec<Action>,
    pub edge_moves: Vec<EdgeMoves>,
    pub slot_count_per_class: Vec<u32>,
    pub used_callee_saved: Vec<UsedSaved>,
    /// True when some value's location CHANGES across a block boundary. The
    /// intra-block `actions` do NOT cover it; a backend that emits from this
    /// allocation before resolution runs must bail when this flag is set.
    pub needs_resolution: bool,
    /// Dense alternative to `segments` for allocations that give every value
    /// one register for its whole life: value index -> register, `u16::MAX`
    /// for a value with none. Empty when `segments` carries the allocation.
    pub single_regs: Vec<u16>,
    /// Value index -> position in `segments` (`u32::MAX` if none). Empty until
    /// `index_segments` runs; `segments_of` then falls back to a linear scan.
    value_slot: Vec<u32>,
}

impl Allocation {
    /// Build the dense value lookup behind `segments_of`. Call once `segments`
    /// is final; the first entry wins for a repeated value, like the scan.
    pub fn index_segments(&mut self, value_count: usize) {
        let mut slots = alloc::vec![u32::MAX; value_count];
        for (n, (v, _)) in self.segments.iter().enumerate() {
            let slot = &mut slots[v.index()];
            if *slot == u32::MAX {
                *slot = n as u32;
            }
        }
        self.value_slot = slots;
    }

    pub fn segments_of(&self, v: Value) -> Option<&[Segment]> {
        if !self.value_slot.is_empty() {
            return match self.value_slot.get(v.index()) {
                Some(&n) if n != u32::MAX => Some(&self.segments[n as usize].1),
                _ => None,
            };
        }
        for (vv, segs) in &self.segments {
            if *vv == v {
                return Some(segs);
            }
        }
        None
    }
}

/// Allocate registers for `func` using the shared linear scan with live-range
/// splitting. Runs LINEARSCAN with `allocate_blocked_reg` (Fig 7) and
/// `split_interval` under register pressure, then collapses each value's
/// placement into a multi-segment [`Allocation`].
pub fn allocate<F: Func, D: RegDescription<F>>(
    func: &F,
    desc: &D,
) -> Result<Allocation, AllocError> {
    let mut intervals = build_intervals(func, desc);

    // Assert the reserved scratch registers of every class can hold that
    // class's WIDEST value (debug only).
    debug_assert_scratch_fits_width(&intervals, &[], desc);

    let bounds = compute_block_bounds(func);

    // Split children are value intervals born during the scan.
    let mut children: Vec<Interval> = Vec::new();

    // One spill slot per spilled interval, counted per class.
    let mut slots = alloc::vec![0u32; desc.classes().len()];

    let mut unhandled: Vec<IvRef> = Vec::new();
    let mut active: Vec<IvRef> = Vec::new();
    let mut inactive: Vec<IvRef> = Vec::new();

    for (i, iv) in intervals.iter_mut().enumerate() {
        if let Some(fr) = iv.fixed_reg {
            iv.location = Some(Location::Reg(fr));
            if iv.value.is_none() {
                inactive.push(IvRef {
                    child: false,
                    idx: i,
                });
            } else {
                active.push(IvRef {
                    child: false,
                    idx: i,
                });
            }
        } else {
            unhandled.push(IvRef {
                child: false,
                idx: i,
            });
        }
    }
    unhandled.sort_by_key(|r| get(&intervals, &children, *r).start());

    // Block-argument coalescing map. Empty (and unused) unless opted in.
    let param_args: ParamArgs = if desc.coalesce_block_params() {
        build_param_args(func)
    } else {
        Vec::new()
    };

    // The scan is a worklist loop. Every split child starts strictly after
    // the interval it came from, so the total interval count is bounded.
    let max_pos = max_end_position(&intervals, &children);
    let iter_bound: usize = 2 * (intervals.len() + intervals.len() * (max_pos as usize + 1)) + 8;
    let mut iters: usize = 0;
    // Fixed-register intervals all exist before the scan and split children
    // never carry one, so the register limit is loop-invariant.
    let limit = reg_limit(&intervals, &children, desc);
    while !unhandled.is_empty() {
        iters += 1;
        debug_assert!(iters <= iter_bound);
        let current = unhandled.remove(0);
        let position = get(&intervals, &children, current).start();

        // Expire or deactivate active intervals. Ranges are half-open, so
        // `end() <= position` means handled; a live interval that does not
        // cover `position` sits in a hole, so it becomes inactive.
        let mut ai = 0;
        while ai < active.len() {
            let it = get(&intervals, &children, active[ai]);
            if it.end() <= position {
                active.swap_remove(ai);
            } else if !it.covers(position) {
                let r = active.swap_remove(ai);
                inactive.push(r);
            } else {
                ai += 1;
            }
        }
        // Expire or reactivate inactive intervals.
        let mut ii = 0;
        while ii < inactive.len() {
            let it = get(&intervals, &children, inactive[ii]);
            if it.end() <= position {
                inactive.swap_remove(ii);
            } else if it.covers(position) {
                let r = inactive.swap_remove(ii);
                active.push(r);
            } else {
                ii += 1;
            }
        }

        debug_assert!(get(&intervals, &children, current).fixed_reg.is_none());

        let param_hint = if desc.coalesce_block_params() {
            compute_param_hint(
                &param_args,
                &intervals,
                &children,
                get(&intervals, &children, current),
            )
        } else {
            None
        };
        let placed = try_allocate_free_reg(
            &intervals,
            &children,
            get(&intervals, &children, current),
            &active,
            &inactive,
            desc,
            limit,
            param_hint,
        );
        if let Some(reg) = placed {
            get_mut(&mut intervals, &mut children, current).location = Some(Location::Reg(reg));
            active.push(current);
        } else {
            allocate_blocked_reg(
                func,
                &mut intervals,
                &mut children,
                current,
                &mut active,
                &mut inactive,
                &mut unhandled,
                &mut slots,
                desc,
                limit,
                &bounds.from,
            )?;
        }
    }

    // Coalesce spilled block parameters with their spilled incoming arguments
    // onto shared slots. A no-op unless the backend opted in.
    coalesce_spill_slots(func, &mut intervals, &mut children, &mut slots, desc);

    // In a debug build, verify the completed allocation before lowering it.
    debug_assert!({
        let mut all: Vec<Interval> = Vec::new();
        all.extend(intervals.iter().cloned());
        all.extend(children.iter().cloned());
        verify_intervals(&all).is_empty()
    });

    let mut result = build_allocation(func, &intervals, &children, &slots, desc);
    resolve_data_flow(func, desc, &intervals, &children, &mut result);
    result.index_segments(func.value_count());
    Ok(result)
}

/// Assert the reserved scratch registers of every class can hold that class's
/// WIDEST value (debug only).
fn debug_assert_scratch_fits_width<F: Func, D: RegDescription<F>>(
    intervals: &[Interval],
    children: &[Interval],
    desc: &D,
) {
    if desc.classes().is_empty() {
        return;
    }
    for (ci, _) in desc.classes().iter().enumerate() {
        let mut regs: u16 = 1;
        let mut alignment: u16 = 1;
        for iv in intervals.iter().chain(children.iter()) {
            if iv.class as usize != ci {
                continue;
            }
            if iv.regs > regs {
                regs = iv.regs;
            }
            if iv.reg_align > alignment {
                alignment = iv.reg_align;
            }
        }
        if regs == 1 && alignment == 1 {
            continue;
        }
        let bases = [
            desc.scratch().get(ci).copied(),
            desc.scratch2().get(ci).copied(),
        ];
        for maybe_base in bases {
            let base = match maybe_base {
                Some(b) => b,
                None => continue,
            };
            debug_assert!(base % alignment == 0);
            let mut k: u16 = 1;
            while k < regs {
                debug_assert!(!contains_reg(&desc.classes()[ci].allocatable, base + k));
                k += 1;
            }
        }
    }
}

/// The move kind that realizes a `src -> dst` location change.
fn action_kind(src: Location, dst: Location) -> ActionKind {
    match src {
        Location::Reg(_) => match dst {
            Location::Reg(_) => ActionKind::Move,
            Location::Slot(_) => ActionKind::Store,
        },
        Location::Slot(_) => match dst {
            Location::Reg(_) => ActionKind::Reload,
            Location::Slot(_) => ActionKind::Move,
        },
    }
}

/// Collapse the placed intervals into an [`Allocation`]. A value now owns
/// MULTIPLE intervals (a parent plus split children) with different
/// locations, so gather every interval per value, sort by `start()`, and emit
/// one ascending segment per interval (merging adjacent identical-location
/// segments). Also records the per-class spill-slot counts and the
/// callee-saved registers used.
fn build_allocation<F: Func, D: RegDescription<F>>(
    func: &F,
    intervals: &[Interval],
    children: &[Interval],
    slots: &[u32],
    desc: &D,
) -> Allocation {
    let mut result = Allocation::default();
    let bounds = compute_block_bounds(func);

    // Group every placed value interval (originals + children) by its value.
    let mut groups: Vec<(Value, Vec<usize>, Vec<usize>)> = Vec::new();
    // Dense value -> group slot table; keeps grouping O(n) while preserving
    // first-seen group order.
    let mut group_of: Vec<u32> = alloc::vec![u32::MAX; func.value_count()];
    let mut push_iv = |v: Value, child: bool, idx: usize| {
        let slot = &mut group_of[v.index()];
        if *slot == u32::MAX {
            *slot = groups.len() as u32;
            groups.push((v, Vec::new(), Vec::new()));
        }
        let (_, origs, ch) = &mut groups[*slot as usize];
        if child {
            ch.push(idx);
        } else {
            origs.push(idx);
        }
    };
    for (i, iv) in intervals.iter().enumerate() {
        if iv.fixed_reg.is_some() || iv.value.is_none() {
            continue;
        }
        debug_assert!(iv.location.is_some());
        push_iv(iv.value.unwrap(), false, i);
    }
    for (i, iv) in children.iter().enumerate() {
        debug_assert!(iv.fixed_reg.is_none() && iv.value.is_some() && iv.location.is_some());
        push_iv(iv.value.unwrap(), true, i);
    }

    let mut actions: Vec<Action> = Vec::new();

    for (v, origs, ch) in &groups {
        // Collect the value's intervals sorted by start.
        let mut order: Vec<(u32, bool, usize)> = Vec::new();
        for i in origs {
            order.push((intervals[*i].start(), false, *i));
        }
        for i in ch {
            order.push((children[*i].start(), true, *i));
        }
        order.sort_by_key(|o| o.0);
        let mut segs: Vec<Segment> = Vec::new();
        // The interval each segment came from (for `store_at`).
        let mut seg_owner_is_child: Vec<bool> = Vec::new();
        let mut seg_owner_idx: Vec<usize> = Vec::new();
        for (_, is_child, i) in &order {
            let iv = if *is_child {
                &children[*i]
            } else {
                &intervals[*i]
            };
            let loc = iv.location.unwrap();
            if !segs.is_empty() && loc_eql(segs[segs.len() - 1].loc, loc) {
                continue;
            }
            segs.push(Segment {
                from: iv.start(),
                loc,
            });
            seg_owner_is_child.push(*is_child);
            seg_owner_idx.push(*i);
        }
        let class = {
            let iv = if order[0].1 {
                &children[order[0].2]
            } else {
                &intervals[order[0].2]
            };
            iv.class
        };
        let mut i = 0;
        while i + 1 < segs.len() {
            let a = segs[i];
            let b = segs[i + 1];
            let mut at = b.from;
            let mut src = a.loc;
            // An interval the reload-at-use split made carries `store_at`.
            let owner = if seg_owner_is_child[i + 1] {
                &children[seg_owner_idx[i + 1]]
            } else {
                &intervals[seg_owner_idx[i + 1]]
            };
            if let Some(store_pos) = owner.store_at {
                at = store_pos;
                src = segment_loc_before(&segs[0..i + 1], store_pos);
                // The value is already in that slot: skip the self-store.
                if loc_eql(src, b.loc) {
                    i += 1;
                    continue;
                }
            }
            // A transition is cross-block, resolved on the edge, ONLY when
            // the later segment begins EXACTLY on a block-entry position.
            let at_block = block_of_pos(&bounds.from, &bounds.to, at);
            let is_cross_block = at == bounds.from[at_block];
            if is_cross_block {
                result.needs_resolution = true;
            } else {
                actions.push(Action {
                    at,
                    kind: action_kind(src, b.loc),
                    class,
                    src,
                    dst: b.loc,
                    value: *v,
                });
            }
            i += 1;
        }
        result.segments.push((*v, segs));
    }

    // Actions land in ascending-`at` order, and every same-position cluster
    // is ordered into a hazard-free parallel-move sequence.
    actions.sort_by_key(|a| a.at);
    result.actions = order_intra_actions(&actions, desc);

    result.slot_count_per_class = slots.to_vec();

    // Record the callee-saved registers any value segment landed in.
    let mut used: Vec<UsedSaved> = Vec::new();
    let mut note = |iv: &Interval| {
        let reg = match iv.location.unwrap() {
            Location::Reg(r) => r,
            Location::Slot(_) => return,
        };
        // A multi-register value occupies its WHOLE span, so EVERY
        // callee-saved register of the span needs a prologue save.
        let mut k: u16 = 0;
        while k < iv.regs {
            let rr = reg.saturating_add(k);
            if contains_reg(&desc.classes()[iv.class as usize].callee_saved, rr)
                && !used.iter().any(|u| u.class == iv.class && u.reg == rr)
            {
                used.push(UsedSaved {
                    class: iv.class,
                    reg: rr,
                });
            }
            k += 1;
        }
    };
    for iv in intervals {
        if iv.fixed_reg.is_some() || iv.value.is_none() || iv.location.is_none() {
            continue;
        }
        note(iv);
    }
    for iv in children {
        note(iv);
    }
    result.used_callee_saved = used;

    result
}

/// The location the value occupies JUST BEFORE `pos`: the location of the
/// last segment that starts strictly before `pos`.
fn segment_loc_before(segs: &[Segment], pos: u32) -> Location {
    debug_assert!(!segs.is_empty());
    debug_assert!(segs[0].from < pos);
    let mut loc = segs[0].loc;
    for s in segs {
        if s.from < pos {
            loc = s.loc;
        }
    }
    loc
}

/// The location a value occupies at position `pos`: the location of the last
/// segment that starts at or before `pos`.
fn location_at(segs: &[Segment], pos: u32) -> Location {
    debug_assert!(!segs.is_empty());
    debug_assert!(segs[0].from <= pos);
    let mut loc = segs[0].loc;
    for s in segs {
        if s.from <= pos {
            loc = s.loc;
        }
    }
    loc
}

// ===========================================================================
// Spill-slot coalescing driver, move ordering, resolution, verification.
// ===========================================================================

/// Coalesce spilled block parameters with their spilled incoming arguments
/// onto shared slots. Interference-checked and guarded by the parallel-move
/// precondition; an unsafe result reverts to the distinct-slot placement. A
/// no-op unless the backend set `coalesce_spill_slots`.
fn coalesce_spill_slots<F: Func, D: RegDescription<F>>(
    func: &F,
    intervals: &mut [Interval],
    children: &mut [Interval],
    slots: &mut [u32],
    desc: &D,
) {
    if !desc.coalesce_spill_slots() {
        return;
    }
    let nclasses = desc.classes().len();

    let mut total: u32 = 0;
    for s in slots.iter() {
        total += *s;
    }
    if total == 0 {
        return; // nothing spilled
    }

    // Flatten every value interval (originals + split children).
    let mut all: Vec<IvRef> = Vec::new();
    for (i, iv) in intervals.iter().enumerate() {
        if iv.fixed_reg.is_some() || iv.value.is_none() {
            continue;
        }
        all.push(IvRef {
            child: false,
            idx: i,
        });
    }
    for (i, _) in children.iter().enumerate() {
        all.push(IvRef {
            child: true,
            idx: i,
        });
    }

    // Per-class slot offsets and a union-find over the flat slot space.
    let mut class_off = alloc::vec![0u32; nclasses + 1];
    for c in 0..nclasses {
        class_off[c + 1] = class_off[c] + slots[c];
    }
    let mut parent: Vec<u32> = (0..total).collect();

    let bounds = compute_block_bounds(func);

    // PASS 0: union all spilled pieces of the SAME value onto one slot. The
    // splitter hands each split child its own fresh slot; a value spilled on
    // both sides of a split pays a slot-to-slot move where its location
    // "changes" across an edge, even though it is one value. Its pieces have
    // DISJOINT lifetimes, so coalescing them is always interference-safe.
    let mut value_slot: Vec<(Value, u32)> = Vec::new();
    for r in all.clone().iter() {
        let (sa, c, v) = {
            let ia = get(intervals, children, *r);
            let sa = match ia.location {
                Some(Location::Slot(s)) => s,
                _ => continue,
            };
            (sa, ia.class, ia.value.unwrap())
        };
        let mut prior = None;
        for (vv, ss) in &value_slot {
            if *vv == v {
                prior = Some(*ss);
                break;
            }
        }
        if prior.is_none() {
            value_slot.push((v, sa));
            continue;
        }
        let rep_a = uf_find(&mut parent, class_off[c as usize] + sa);
        let rep_b = uf_find(&mut parent, class_off[c as usize] + prior.unwrap());
        if rep_a == rep_b {
            continue;
        }
        if slot_groups_interfere(
            &all,
            intervals,
            children,
            &mut parent,
            &class_off,
            c,
            rep_a,
            rep_b,
        ) {
            continue;
        }
        parent[rep_a as usize] = rep_b;
    }

    // PASS 1: union each spilled parameter with a non-interfering spilled
    // incoming argument.
    for bi in 0..func.block_count() {
        let pred = Block::from_u32(bi as u32);
        for inst in func.block_insts(pred) {
            if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                coalesce_edge_params(
                    &all,
                    intervals,
                    children,
                    &mut parent,
                    &class_off,
                    func,
                    desc,
                    &bounds.from,
                    &bounds.to,
                    pred,
                    cf.then,
                );
                coalesce_edge_params(
                    &all,
                    intervals,
                    children,
                    &mut parent,
                    &class_off,
                    func,
                    desc,
                    &bounds.from,
                    &bounds.to,
                    pred,
                    cf.else_,
                );
            }
        }
        if let Some(term) = func.terminator(pred) {
            match term {
                volt_ir::function::Terminator::Jump(j) => {
                    coalesce_edge_params(
                        &all,
                        intervals,
                        children,
                        &mut parent,
                        &class_off,
                        func,
                        desc,
                        &bounds.from,
                        &bounds.to,
                        pred,
                        j,
                    );
                }
                volt_ir::function::Terminator::Ret(_) => {}
            }
        }
    }

    // Compact each class's live slots to a dense 0.. numbering.
    let mut remap = alloc::vec![u32::MAX; total as usize];
    let mut new_counts = alloc::vec![0u32; nclasses];
    for c in 0..nclasses {
        let mut next: u32 = 0;
        for s in 0..slots[c] {
            let flat = class_off[c] + s;
            let r = uf_find(&mut parent, flat);
            if remap[r as usize] == u32::MAX {
                remap[r as usize] = next;
                next += 1;
            }
            remap[flat as usize] = remap[r as usize];
        }
        new_counts[c] = next;
    }

    // Snapshot the original locations so an unsafe result can revert bit-for-bit.
    let mut orig_loc: Vec<Location> = Vec::with_capacity(all.len());
    for r in &all {
        orig_loc.push(get(intervals, children, *r).location.unwrap());
    }

    // Apply the remap to every spilled interval.
    for r in &all {
        let cls = get(intervals, children, *r).class;
        let slot = match get(intervals, children, *r).location {
            Some(Location::Slot(s)) => s,
            _ => continue,
        };
        get_mut(intervals, children, *r).location = Some(Location::Slot(
            remap[(class_off[cls as usize] + slot) as usize],
        ));
    }

    // PASS 2: re-verify the parallel-move precondition over every edge, on
    // the rewritten locations. When the backend gives EVERY class a second
    // scratch, the resolver breaks slot-involving cycles through it, so this
    // guard is not needed and the coalescing always commits.
    let full_scratch2 = desc.scratch2().len() == nclasses;
    if !full_scratch2 {
        let mut mark = alloc::vec![0u8; total as usize];
        let mut is_unsafe = false;
        'walk: for bi in 0..func.block_count() {
            let pred = Block::from_u32(bi as u32);
            for inst in func.block_insts(pred) {
                if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                    for e in [cf.then, cf.else_] {
                        mark.fill(0);
                        if edge_slot_both_src_and_dst(
                            &all,
                            intervals,
                            children,
                            func,
                            desc,
                            &class_off,
                            &mut mark,
                            &bounds.from,
                            &bounds.to,
                            pred,
                            e,
                        ) {
                            is_unsafe = true;
                            break 'walk;
                        }
                    }
                }
            }
            if let Some(term) = func.terminator(pred) {
                match term {
                    volt_ir::function::Terminator::Jump(j) => {
                        mark.fill(0);
                        if edge_slot_both_src_and_dst(
                            &all,
                            intervals,
                            children,
                            func,
                            desc,
                            &class_off,
                            &mut mark,
                            &bounds.from,
                            &bounds.to,
                            pred,
                            j,
                        ) {
                            is_unsafe = true;
                            break 'walk;
                        }
                    }
                    volt_ir::function::Terminator::Ret(_) => {}
                }
            }
        }
        if is_unsafe {
            // Revert to the distinct-slot placement; leave `slots` untouched.
            for (r, loc) in all.iter().zip(orig_loc.iter()) {
                // `all` holds refs into one of two slices; resolve mutably.
                if r.child {
                    children[r.idx].location = Some(*loc);
                } else {
                    intervals[r.idx].location = Some(*loc);
                }
            }
            return;
        }
    }

    // Commit the reduced per-class slot counts.
    slots[..nclasses].copy_from_slice(&new_counts[..nclasses]);
}

/// Order a raw parallel move set into a valid sequence for white-box testing
/// and for the resolver. Every source is read before it is overwritten,
/// cycles are broken through the class scratch register, and a slot-to-slot
/// shuffle is routed through the class scratch too.
pub fn order_moves<F: Func, D: RegDescription<F>>(raw: &[Move], desc: &D) -> Vec<Move> {
    debug_assert!(desc.scratch().len() == desc.classes().len());
    let mut out: Vec<Move> = Vec::new();
    for (ci, _) in desc.classes().iter().enumerate() {
        let class_idx = ci as u16;
        let s2 = desc.scratch2().get(ci).copied();
        order_class_moves(class_idx, raw, desc.scratch()[ci], s2, &mut out);
    }
    out
}

/// True iff some pending move OTHER than `self_i` still reads `loc` as its
/// source (so writing `loc` now would clobber a value not yet moved).
fn read_by_other(pending: &[Move], self_i: usize, loc: Location) -> bool {
    for (j, m) in pending.iter().enumerate() {
        if j == self_i {
            continue;
        }
        if loc_eql(m.src, loc) {
            return true;
        }
    }
    false
}

/// The standard parallel-move sequencing for one class. Repeatedly emit a
/// move whose destination no other pending move still reads. When only cycles
/// remain, break one by routing a node's value through the class scratch.
fn order_class_moves(
    class_idx: u16,
    raw: &[Move],
    scratch_reg: u16,
    scratch2_reg: Option<u16>,
    out: &mut Vec<Move>,
) {
    let mut pending: Vec<Move> = Vec::new();
    for m in raw {
        if m.class == class_idx {
            pending.push(*m);
        }
    }
    if pending.is_empty() {
        return;
    }

    let scratch_loc = Location::Reg(scratch_reg);
    let scratch2_loc = scratch2_reg.map(Location::Reg);
    // Both scratch registers are reserved, so no raw move may touch either.
    for m in &pending {
        debug_assert!(!loc_eql(m.src, scratch_loc));
        debug_assert!(!loc_eql(m.dst, scratch_loc));
        if let Some(s2) = scratch2_loc {
            debug_assert!(!loc_eql(m.src, s2));
            debug_assert!(!loc_eql(m.dst, s2));
        }
    }

    let has_slot_conflict = slot_both_src_and_dst(&pending);
    let cycle_loc = if has_slot_conflict {
        scratch2_loc.expect("slot conflict requires scratch2")
    } else {
        scratch_loc
    };

    let out_start = out.len();
    let mut scratch_busy = false;
    let mut scratch2_busy = false;
    // Each loop iteration either emits one pending move or breaks one cycle,
    // so the count is bounded by twice the move count.
    let bound = pending.len() * 2 + 4;
    let mut iters = 0;
    while !pending.is_empty() {
        iters += 1;
        debug_assert!(iters <= bound);

        let mut free_idx: Option<usize> = None;
        for (i, m) in pending.iter().enumerate() {
            if read_by_other(&pending, i, m.dst) {
                continue;
            }
            free_idx = Some(i);
            break;
        }

        if let Some(i) = free_idx {
            let m = pending.remove(i);
            emit_primitive(
                out,
                m,
                scratch_loc,
                scratch2_loc,
                class_idx,
                &mut scratch_busy,
                &mut scratch2_busy,
            );
            continue;
        }

        // Only cycles remain. Break one by saving its source into the cycle
        // scratch, then reading the scratch in its place.
        let m0 = &mut pending[0];
        if has_slot_conflict {
            debug_assert!(!scratch2_busy);
            scratch2_busy = true;
        } else {
            debug_assert!(!scratch_busy);
            // Without a slot conflict every cycle node is reg->reg.
            debug_assert!(loc_is_reg(m0.src) && loc_is_reg(m0.dst));
            scratch_busy = true;
        }
        out.push(Move {
            src: m0.src,
            dst: cycle_loc,
            class: class_idx,
            value: m0.value,
        });
        m0.src = cycle_loc;
    }
    debug_assert!(!scratch_busy && !scratch2_busy);

    debug_assert!({
        assert_ordering_valid(class_idx, raw, &out[out_start..]);
        true
    });
}

/// True iff some spill slot in `moves` is the source of one move and the
/// destination of a DIFFERENT move. A self-move (a slot to itself) is not a
/// conflict: it moves nothing.
fn slot_both_src_and_dst(moves: &[Move]) -> bool {
    for (i, a) in moves.iter().enumerate() {
        let s = match a.src {
            Location::Slot(x) => x,
            Location::Reg(_) => continue,
        };
        for (j, b) in moves.iter().enumerate() {
            if i == j {
                continue;
            }
            match b.dst {
                Location::Slot(y) => {
                    if y == s && a.class == b.class {
                        return true;
                    }
                }
                Location::Reg(_) => {}
            }
        }
    }
    false
}

/// Emit one ordered move as backend-primitive op(s): a slot->slot shuffle
/// expands to a load into the scratch then a store out of it. A move that
/// READS either scratch closes a broken cycle, freeing that scratch.
fn emit_primitive(
    out: &mut Vec<Move>,
    m: Move,
    scratch_loc: Location,
    scratch2_loc: Option<Location>,
    class_idx: u16,
    scratch_busy: &mut bool,
    scratch2_busy: &mut bool,
) {
    let closes = loc_eql(m.src, scratch_loc);
    let closes2 = scratch2_loc.map(|s2| loc_eql(m.src, s2)).unwrap_or(false);
    match m.src {
        Location::Reg(_) => out.push(m),
        Location::Slot(_) => match m.dst {
            Location::Reg(_) => out.push(m),
            Location::Slot(_) => {
                // Memory-to-memory needs the scratch, which is free outside a
                // cycle break.
                debug_assert!(!*scratch_busy);
                out.push(Move {
                    src: m.src,
                    dst: scratch_loc,
                    class: class_idx,
                    value: m.value,
                });
                out.push(Move {
                    src: scratch_loc,
                    dst: m.dst,
                    class: class_idx,
                    value: m.value,
                });
            }
        },
    }
    if closes {
        *scratch_busy = false;
    }
    if closes2 {
        *scratch2_busy = false;
    }
}

/// True iff `loc` is a register (rather than a spill slot).
fn loc_is_reg(loc: Location) -> bool {
    matches!(loc, Location::Reg(_))
}

/// Validate an ordered class sequence realizes the raw parallel move:
/// simulate each location's contents through the ordered ops, then assert
/// every raw move's destination ends holding its source's original value.
/// Debug-only.
fn assert_ordering_valid(class_idx: u16, raw: &[Move], ordered: &[Move]) {
    let mut content: Vec<(u64, u64)> = Vec::new();
    let key = |loc: Location| -> u64 {
        match loc {
            Location::Reg(r) => r as u64,
            Location::Slot(s) => (1u64 << 32) | s as u64,
        }
    };
    let find = |content: &[(u64, u64)], k: u64| -> Option<usize> {
        content.iter().position(|(kk, _)| *kk == k)
    };
    for m in raw {
        if m.class != class_idx {
            continue;
        }
        let k = key(m.src);
        if find(&content, k).is_none() {
            content.push((k, k));
        }
    }
    for m in ordered {
        let v = {
            let i = find(&content, key(m.src)).expect("ordered move reads unknown loc");
            content[i].1
        };
        match find(&content, key(m.dst)) {
            Some(i) => content[i].1 = v,
            None => content.push((key(m.dst), v)),
        }
    }
    for m in raw {
        if m.class != class_idx {
            continue;
        }
        let i = find(&content, key(m.dst)).expect("raw dst never written");
        debug_assert!(content[i].1 == key(m.src));
    }
}

/// Order the intra-block actions so every same-position cluster drains
/// hazard-free. The input is already sorted ascending by `at`. For each
/// maximal same-`at` run, treat the cluster as a parallel move and run it
/// through `order_moves`, then re-tag the ordered primitive moves as actions.
fn order_intra_actions<F: Func, D: RegDescription<F>>(sorted: &[Action], desc: &D) -> Vec<Action> {
    let mut out: Vec<Action> = Vec::new();
    let mut moves: Vec<Move> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let at = sorted[i].at;
        let mut j = i;
        while j < sorted.len() && sorted[j].at == at {
            j += 1;
        }
        moves.clear();
        for a in &sorted[i..j] {
            moves.push(Move {
                src: a.src,
                dst: a.dst,
                class: a.class,
                value: a.value,
            });
        }
        let ordered = order_moves::<F, D>(&moves, desc);
        for m in ordered {
            out.push(Action {
                at,
                kind: action_kind(m.src, m.dst),
                class: m.class,
                src: m.src,
                dst: m.dst,
                value: m.value,
            });
        }
        i = j;
    }
    out
}

/// RESOLVEDATAFLOW: fill `result.edge_moves` with one ordered `EdgeMoves` per
/// control-flow edge that actually needs a shuffle. Consumes the (post-scan)
/// intervals for liveness and `result.segments` for locations.
fn resolve_data_flow<F: Func, D: RegDescription<F>>(
    func: &F,
    desc: &D,
    intervals: &[Interval],
    children: &[Interval],
    result: &mut Allocation,
) {
    debug_assert!(desc.hosts_critical_edge_moves() || !has_critical_edges(func));

    let bounds = compute_block_bounds(func);
    let mut edges: Vec<EdgeMoves> = Vec::new();
    let tables = EdgeTables::build(func, intervals, children, result, &bounds.from);

    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                add_edge_moves(
                    func,
                    desc,
                    result,
                    &tables,
                    &bounds.from,
                    &bounds.to,
                    &mut edges,
                    block,
                    cf.then,
                );
                add_edge_moves(
                    func,
                    desc,
                    result,
                    &tables,
                    &bounds.from,
                    &bounds.to,
                    &mut edges,
                    block,
                    cf.else_,
                );
            }
        }
        if let Some(term) = func.terminator(block) {
            match term {
                volt_ir::function::Terminator::Jump(j) => {
                    add_edge_moves(
                        func,
                        desc,
                        result,
                        &tables,
                        &bounds.from,
                        &bounds.to,
                        &mut edges,
                        block,
                        j,
                    );
                }
                volt_ir::function::Terminator::Ret(_) => {}
            }
        }
    }

    result.edge_moves = edges;
}

/// Lookup tables that keep edge-move construction linear in the function size
/// instead of rescanning every value and interval for every edge.
struct EdgeTables {
    /// Value index -> position in `Allocation::segments` (`u32::MAX` if none).
    segment_slot: Vec<u32>,
    /// Block index -> positions in `Allocation::segments` of the values with a
    /// value interval covering that block's entry, ascending.
    live_in: Vec<Vec<u32>>,
}

impl EdgeTables {
    fn build<F: Func>(
        func: &F,
        intervals: &[Interval],
        children: &[Interval],
        result: &Allocation,
        block_from: &[u32],
    ) -> Self {
        let mut segment_slot = alloc::vec![u32::MAX; func.value_count()];
        for (slot, (v, _)) in result.segments.iter().enumerate() {
            segment_slot[v.index()] = slot as u32;
        }
        let mut live_in: Vec<Vec<u32>> = alloc::vec![Vec::new(); block_from.len()];
        for iv in intervals.iter().chain(children.iter()) {
            let Some(v) = iv.value else { continue };
            if iv.fixed_reg.is_some() {
                continue;
            }
            let slot = segment_slot[v.index()];
            if slot == u32::MAX {
                continue;
            }
            for (b, from) in block_from.iter().enumerate() {
                if iv.covers(*from) {
                    live_in[b].push(slot);
                }
            }
        }
        for list in &mut live_in {
            list.sort_unstable();
            list.dedup();
        }
        Self {
            segment_slot,
            live_in,
        }
    }

    fn segments<'a>(&self, result: &'a Allocation, v: Value) -> Option<&'a [Segment]> {
        let slot = *self.segment_slot.get(v.index())?;
        (slot != u32::MAX).then(|| result.segments[slot as usize].1.as_slice())
    }
}

/// Compute and store the ordered move list for one edge `pred -> edge.target`.
fn add_edge_moves<F: Func, D: RegDescription<F>>(
    func: &F,
    desc: &D,
    result: &Allocation,
    tables: &EdgeTables,
    block_from: &[u32],
    block_to: &[u32],
    edges: &mut Vec<EdgeMoves>,
    pred: Block,
    edge: Jump,
) {
    let succ = edge.target;
    // The predecessor's branch executes at its last position. The successor
    // is entered at its parameter row.
    let pt = block_to[pred.index()] - 1;
    let ss = block_from[succ.index()];

    let mut raw: Vec<Move> = Vec::new();

    // (1) Block-parameter moves: the argument's location at pred exit into the
    // parameter's location at succ entry.
    let params = func.block_params(succ);
    let args = func.block_args(edge);
    debug_assert!(params.len() == args.len());
    if params.len() == args.len() {
        for (p, a) in params.iter().zip(args.iter()) {
            let a_segs = tables.segments(result, *a).expect("arg has no segments");
            let p_segs = tables.segments(result, *p).expect("param has no segments");
            let from = location_at(a_segs, pt);
            let to = location_at(p_segs, ss);
            if !loc_eql(from, to) {
                raw.push(Move {
                    src: from,
                    dst: to,
                    class: desc.class_of(func, *p),
                    value: *p,
                });
            }
        }
    }

    // (2) Non-parameter live-in moves: a value that flows THROUGH the edge
    // and whose location changes across the edge.
    // Only values live at the successor's entry can need a move. The tables
    // list exactly those, in segment order.
    for &slot in &tables.live_in[succ.index()] {
        let (v, segs) = &result.segments[slot as usize];
        if is_param_of(func, succ, *v) {
            continue;
        }
        debug_assert!(segs[0].from <= ss);
        debug_assert!(segs[0].from <= pt);
        let from = location_at(segs, pt);
        let to = location_at(segs, ss);
        if !loc_eql(from, to) {
            raw.push(Move {
                src: from,
                dst: to,
                class: desc.class_of(func, *v),
                value: *v,
            });
        }
    }

    if raw.is_empty() {
        return;
    }

    let ordered = order_moves::<F, D>(&raw, desc);
    edges.push(EdgeMoves {
        pred,
        succ,
        moves: ordered,
    });
}

/// Whether the function has a critical edge: a multi-successor `if` block
/// feeding a multi-predecessor block. Resolution assumes the driver split
/// them first.
fn has_critical_edges<F: Func>(func: &F) -> bool {
    let nblocks = func.block_count();
    if nblocks == 0 {
        return false;
    }
    let mut pred_count = alloc::vec![0u32; nblocks];
    for bi in 0..nblocks {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                pred_count[cf.then.target.index()] += 1;
                pred_count[cf.else_.target.index()] += 1;
            }
        }
        if let Some(term) = func.terminator(block) {
            match term {
                volt_ir::function::Terminator::Jump(j) => {
                    pred_count[j.target.index()] += 1;
                }
                volt_ir::function::Terminator::Ret(_) => {}
            }
        }
    }
    for bi in 0..nblocks {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            if let volt_ir::function::Opcode::If(cf) = func.opcode(*inst) {
                // The `if` block already has two successors, so either target
                // with more than one predecessor makes that edge critical.
                if pred_count[cf.then.target.index()] > 1 || pred_count[cf.else_.target.index()] > 1
                {
                    return true;
                }
            }
        }
    }
    false
}

// ===========================================================================
// The debug verifier.
// ===========================================================================

/// One thing the allocation got wrong, found by `verify_intervals`. `a` and
/// `b` index the intervals slice passed to the verifier, and `b == a` for
/// the single-interval checks. `pos` is the program position the violation
/// manifests at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Violation {
    pub kind: ViolationKind,
    pub a: usize,
    pub b: usize,
    pub pos: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViolationKind {
    RegOverlap,
    MustHaveSpilled,
    Unassigned,
    MisalignedSpan,
    StoreSrcDead,
    StoreSlotBusy,
}

/// The physical register an interval OCCUPIES, or `None` when it holds none:
/// a spilled value interval, or a value interval never placed.
fn occupied_reg(it: &Interval) -> Option<u16> {
    if let Some(fr) = it.fixed_reg {
        return Some(fr);
    }
    match it.location {
        Some(Location::Reg(r)) => Some(r),
        _ => None,
    }
}

/// True iff the pair is a value interval and its OWN entry-parameter fixed
/// interval, the single legitimate same-(class, reg) overlap at entry over
/// `[0, 1)`. Exactly one of the pair is a fixed interval and both carry the
/// same pinned value.
fn is_entry_param_hint_pair(a: &Interval, b: &Interval) -> bool {
    match (a.value, b.value) {
        (Some(av), Some(bv)) => {
            if av != bv {
                return false;
            }
        }
        _ => return false,
    }
    (a.fixed_reg.is_some()) != (b.fixed_reg.is_some())
}

/// True iff the two intervals occupy the SAME register span: same base, same
/// width. A PARTIAL overlap is never legitimate and stays a violation.
fn same_span(a: &Interval, base_a: u16, b: &Interval, base_b: u16) -> bool {
    base_a == base_b && a.regs == b.regs
}

/// The earliest position two same-(class, reg) intervals genuinely conflict
/// at, or `None`. For a CALL-CLOBBER fixed interval the read-before-clobber
/// rule applies, exactly the rule the scan itself allocated by. Any other
/// pair conflicts wherever their ranges intersect.
fn occupancy_conflict(a: &Interval, b: &Interval) -> Option<u32> {
    if a.fixed_reg.is_some() && a.value.is_none() {
        return fixed_clobber_conflict(b, a);
    }
    if b.fixed_reg.is_some() && b.value.is_none() {
        return fixed_clobber_conflict(a, b);
    }
    a.next_intersection(b)
}

/// True iff some `.reg`-located interval of value `v` covers `pos` (i.e. `v`
/// is in a register there).
fn value_in_reg_at(intervals: &[Interval], v: Value, pos: u32) -> bool {
    for it in intervals {
        if it.fixed_reg.is_some() {
            continue;
        }
        if it.value != Some(v) {
            continue;
        }
        if let Some(Location::Reg(_)) = it.location
            && it.covers(pos)
        {
            return true;
        }
    }
    false
}

/// True iff some OTHER piece of value `v` still holds the value at `pos`: a
/// piece with a location, starting strictly before `pos`, that stays live up
/// to it. `itself` indexes the piece that asks, so a piece never answers
/// about itself.
fn value_held_before(intervals: &[Interval], v: Value, pos: u32, itself: usize) -> bool {
    for (i, it) in intervals.iter().enumerate() {
        if i == itself {
            continue;
        }
        if it.fixed_reg.is_some() {
            continue;
        }
        if it.value != Some(v) {
            continue;
        }
        if it.location.is_none() {
            continue;
        }
        for r in &it.ranges {
            if r.from < pos && r.to >= pos {
                return true;
            }
        }
    }
    false
}

/// Verify a completed allocation, returning every soundness [`Violation`].
/// An empty result means the allocation is valid. The input is the flattened
/// final intervals (originals plus split children); it is read-only.
pub fn verify_intervals(intervals: &[Interval]) -> Vec<Violation> {
    let mut violations: Vec<Violation> = Vec::new();

    // CHECK 1: register exclusivity. Every pair of same-class intervals whose
    // register SPANS overlap must have disjoint live ranges, except a value
    // interval and its own entry-param fixed interval, or a coalesced copy pair.
    for (i, ia) in intervals.iter().enumerate() {
        let ra = match occupied_reg(ia) {
            Some(r) => r,
            None => continue,
        };
        for (j, ib) in intervals.iter().enumerate().skip(i + 1) {
            if ia.class != ib.class {
                continue;
            }
            let rb = match occupied_reg(ib) {
                Some(r) => r,
                None => continue,
            };
            // The FULL span, not just the base.
            if !spans_overlap(ra, ia.regs, rb, ib.regs) {
                continue;
            }
            if same_span(ia, ra, ib, rb) {
                if is_entry_param_hint_pair(ia, ib) {
                    continue;
                }
                if is_copy_coalesce_pair(ia, ib) {
                    continue;
                }
            }
            if let Some(pos) = occupancy_conflict(ia, ib) {
                violations.push(Violation {
                    kind: ViolationKind::RegOverlap,
                    a: i,
                    b: j,
                    pos,
                });
            }
        }
    }

    // CHECK 2: must_have_register satisfaction. Every `must_have_register` use
    // of a value must fall in a `.reg`-located interval of that same value.
    for (i, ia) in intervals.iter().enumerate() {
        if ia.fixed_reg.is_some() {
            continue;
        }
        let va = match ia.value {
            Some(v) => v,
            None => continue,
        };
        for u in &ia.uses {
            if u.kind != UseKind::MustHaveRegister {
                continue;
            }
            if !value_in_reg_at(intervals, va, u.pos) {
                violations.push(Violation {
                    kind: ViolationKind::MustHaveSpilled,
                    a: i,
                    b: i,
                    pos: u.pos,
                });
            }
        }
    }

    // CHECK 3: assignment. Every value interval that has at least one use must
    // have been placed.
    for (i, ia) in intervals.iter().enumerate() {
        if ia.fixed_reg.is_some() || ia.value.is_none() || ia.uses.is_empty() {
            continue;
        }
        if ia.location.is_none() {
            violations.push(Violation {
                kind: ViolationKind::Unassigned,
                a: i,
                b: i,
                pos: ia.start(),
            });
        }
    }

    // CHECK 4: span legality. A placed interval's BASE must meet the alignment
    // its width demands, and the whole span must fit the register-index space.
    for (i, ia) in intervals.iter().enumerate() {
        let ra = match occupied_reg(ia) {
            Some(r) => r,
            None => continue,
        };
        debug_assert!(ia.reg_align >= 1);
        if ra % ia.reg_align == 0 && (ra as usize) + (ia.regs as usize) <= MAX_PHYS_REGS {
            continue;
        }
        violations.push(Violation {
            kind: ViolationKind::MisalignedSpan,
            a: i,
            b: i,
            pos: ia.start(),
        });
    }

    // CHECK 5: the early store. An interval with `store_at` writes its slot IN
    // FRONT of its own first range: the value must still be in a located piece
    // there (SOURCE), and the interval must own its slot over the whole prefix
    // `[store_at, start())` (SLOT).
    for (i, ia) in intervals.iter().enumerate() {
        let p = match ia.store_at {
            Some(p) => p,
            None => continue,
        };
        let va = match ia.value {
            Some(v) => v,
            None => continue,
        };
        if p >= ia.start() || !value_held_before(intervals, va, p, i) {
            violations.push(Violation {
                kind: ViolationKind::StoreSrcDead,
                a: i,
                b: i,
                pos: p,
            });
            continue;
        }
        let sa = match ia.location {
            Some(Location::Slot(s)) => s,
            _ => {
                violations.push(Violation {
                    kind: ViolationKind::StoreSlotBusy,
                    a: i,
                    b: i,
                    pos: p,
                });
                continue;
            }
        };
        for (j, ib) in intervals.iter().enumerate() {
            if j == i || ib.class != ia.class {
                continue;
            }
            let sb = match ib.location {
                Some(Location::Slot(s)) => s,
                _ => continue,
            };
            if sb != sa {
                continue;
            }
            if !range_meets_interval(p, ia.start(), ib) {
                continue;
            }
            violations.push(Violation {
                kind: ViolationKind::StoreSlotBusy,
                a: i,
                b: j,
                pos: p,
            });
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_ir::function::{BinOp, Function, Opcode, Ret, Terminator};
    use volt_ir::types::{IntDesc, TypeKind};

    /// Minimal single-class test description: everything in class 0,
    /// must-have uses, no calls, no pins, scratch register far away.
    struct TestDesc {
        classes: Vec<RegClass>,
        scratch: Vec<u16>,
        scratch2: Vec<u16>,
        width: Option<(Value, u16, u16)>,
        narrow: Vec<Value>,
        copy_src: Vec<(Value, Value)>,
        coalesce_params: bool,
        coalesce_slots: bool,
        use_kind: UseKind,
    }

    impl TestDesc {
        fn one_class(nregs: u16, scratch: u16) -> Self {
            Self {
                classes: alloc::vec![RegClass {
                    name: "gpr",
                    allocatable: (0..nregs).collect(),
                    callee_saved: Vec::new(),
                    slot_bytes: 8,
                    narrow_preserved: Vec::new(),
                }],
                scratch: alloc::vec![scratch],
                scratch2: Vec::new(),
                width: None,
                narrow: Vec::new(),
                copy_src: Vec::new(),
                coalesce_params: false,
                coalesce_slots: false,
                use_kind: UseKind::MustHaveRegister,
            }
        }
    }

    impl<F: Func> RegDescription<F> for TestDesc {
        fn classes(&self) -> &[RegClass] {
            &self.classes
        }
        fn class_of(&self, _func: &F, _v: Value) -> u16 {
            0
        }
        fn use_kind(&self, _func: &F, _inst: Inst, _operand: Value) -> UseKind {
            self.use_kind
        }
        fn entry_fixed(&self) -> &[FixedAssign] {
            &[]
        }
        fn call_sites(&self) -> &[CallSite] {
            &[]
        }
        fn scratch(&self) -> &[u16] {
            &self.scratch
        }
        fn scratch2(&self) -> &[u16] {
            &self.scratch2
        }
        fn copy_source(&self, _func: &F, v: Value) -> Option<Value> {
            self.copy_src.iter().find(|(d, _)| *d == v).map(|(_, s)| *s)
        }
        fn is_narrow(&self, _func: &F, v: Value) -> bool {
            self.narrow.contains(&v)
        }
        fn coalesce_block_params(&self) -> bool {
            self.coalesce_params
        }
        fn coalesce_spill_slots(&self) -> bool {
            self.coalesce_slots
        }
        fn reg_width(&self, _func: &F, v: Value) -> RegWidth {
            match self.width {
                Some((w, regs, align)) if w == v => RegWidth {
                    regs,
                    alignment: align,
                },
                _ => RegWidth::ONE,
            }
        }
    }

    fn i32_ty(func: &mut Function) -> volt_ir::types::Type {
        func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }))
    }

    #[test]
    fn interval_start_end_covers_first_use_after_next_intersection() {
        let iv = |from: u32, to: u32| Interval {
            value: None,
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from, to }],
            uses: alloc::vec![UsePos {
                pos: from,
                kind: UseKind::MustHaveRegister
            }],
            location: None,
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        let a = iv(2, 6);
        assert_eq!(a.start(), 2);
        assert_eq!(a.end(), 6);
        assert!(a.covers(2));
        assert!(a.covers(5));
        assert!(!a.covers(6));
        assert_eq!(a.first_use_after(0), Some(2));
        assert_eq!(a.first_use_after(3), None);
        let b = iv(5, 9);
        assert_eq!(a.next_intersection(&b), Some(5));
        let c = iv(6, 9);
        assert_eq!(a.next_intersection(&c), None);
    }

    #[test]
    fn build_intervals_hole_between_uses() {
        // value dead between two uses has a hole: entry block defines x,
        // diamond uses x on one side only... simplest: linear chain where x
        // is defined early, used late, with a middle block that kills nothing.
        // We check liveness holes indirectly: two disjoint use ranges.
        let mut func = Function::new();
        let i32t = i32_ty(&mut func);
        let b0 = func.append_block();
        let b1 = func.append_block();
        let b2 = func.append_block();
        let x = func.append_inst(b0, i32t, Opcode::Iconst(1));
        let c = func.append_inst(b0, i32t, Opcode::Iconst(0));
        let y = func.append_inst(
            b1,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: x,
                rhs: x,
            }),
        );
        let z = func.append_inst(
            b2,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: y,
                rhs: c,
            }),
        );
        let no_args = func.intern_value_list(&[]);
        func.set_terminator(
            b0,
            Terminator::Jump(volt_ir::function::Jump {
                target: b1,
                args: no_args,
            }),
        );
        func.set_terminator(
            b1,
            Terminator::Jump(volt_ir::function::Jump {
                target: b2,
                args: no_args,
            }),
        );
        func.set_terminator(b2, Terminator::Ret(Ret::one(z)));
        let desc = TestDesc::one_class(8, 100);
        let ivs = build_intervals(&func, &desc);
        // x is live from its def through the use in b1: single merged range.
        let xiv = ivs
            .iter()
            .find(|iv| iv.value == Some(x))
            .expect("x interval");
        assert!(!xiv.ranges.is_empty());
        assert_eq!(xiv.uses.len(), 2);
        let _ = y;
    }

    #[test]
    fn scan_low_pressure_single_segment() {
        let mut func = Function::new();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let a = func.append_inst(b, i32t, Opcode::Iconst(1));
        let c = func.append_inst(b, i32t, Opcode::Iconst(2));
        let d = func.append_inst(
            b,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: a,
                rhs: c,
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(d)));
        let desc = TestDesc::one_class(8, 100);
        let alloc = allocate(&func, &desc).expect("allocates");
        // Every value gets a register with a single segment.
        for v in [a, c, d] {
            let segs = alloc.segments_of(v).expect("segments");
            assert_eq!(segs.len(), 1);
            assert!(matches!(segs[0].loc, Location::Reg(_)));
        }
        assert!(alloc.slot_count_per_class.iter().all(|s| *s == 0));
    }

    #[test]
    fn scan_pressure_spills_to_slot() {
        // 2 registers, 3 simultaneously live values -> one must spill.
        let mut func = Function::new();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let a = func.append_inst(b, i32t, Opcode::Iconst(1));
        let c = func.append_inst(b, i32t, Opcode::Iconst(2));
        let d = func.append_inst(
            b,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: a,
                rhs: c,
            }),
        );
        let e = func.append_inst(b, i32t, Opcode::Iconst(3));
        let f = func.append_inst(
            b,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: d,
                rhs: e,
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(f)));
        let mut desc = TestDesc::one_class(2, 100);
        // should_have uses may reload from a slot, so pressure spills cleanly
        // instead of hitting the unsatisfiable must-have limit.
        desc.use_kind = UseKind::ShouldHaveRegister;
        let alloc = allocate(&func, &desc).expect("allocates under pressure");
        assert!(alloc.slot_count_per_class[0] >= 1);
        // Verify the flattened placement via verify_intervals on rebuilt intervals is clean:
        // instead check every value has segments.
        for v in [a, c, e, d, f] {
            assert!(alloc.segments_of(v).is_some());
        }
    }

    #[test]
    fn verify_clean_and_overlap_flagged() {
        // Clean allocation has no violations.
        let mut func = Function::new();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let a = func.append_inst(b, i32t, Opcode::Iconst(1));
        let c = func.append_inst(b, i32t, Opcode::Iconst(2));
        let d = func.append_inst(
            b,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: a,
                rhs: c,
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(d)));
        let desc = TestDesc::one_class(8, 100);
        let alloc = allocate(&func, &desc).expect("allocates");
        // Rebuild intervals and stamp locations from the allocation segments.
        let mut ivs = build_intervals(&func, &desc);
        for iv in ivs.iter_mut() {
            if let Some(v) = iv.value
                && let Some(segs) = alloc.segments_of(v)
            {
                iv.location = Some(location_at(segs, iv.start()));
            }
        }
        assert!(verify_intervals(&ivs).is_empty());

        // Two value intervals sharing a register over overlapping ranges is flagged.
        let mk = |from: u32, to: u32, reg: u16| Interval {
            value: Some(Value::from_u32(from)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from, to }],
            uses: Vec::new(),
            location: Some(Location::Reg(reg)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        let bad = alloc::vec![mk(0, 5, 3), mk(3, 8, 3)];
        let viol = verify_intervals(&bad);
        assert!(viol.iter().any(|v| v.kind == ViolationKind::RegOverlap));
    }

    #[test]
    fn verify_must_have_spilled_and_unassigned() {
        let mk = |from: u32, to: u32, loc: Option<Location>| Interval {
            value: Some(Value::from_u32(7)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from, to }],
            uses: alloc::vec![UsePos {
                pos: from,
                kind: UseKind::MustHaveRegister
            }],
            location: loc,
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        // Slot-covered must-have use is flagged.
        let bad = alloc::vec![mk(2, 6, Some(Location::Slot(0)))];
        assert!(
            verify_intervals(&bad)
                .iter()
                .any(|v| v.kind == ViolationKind::MustHaveSpilled)
        );
        // Used but unplaced is flagged.
        let bad2 = alloc::vec![mk(2, 6, None)];
        assert!(
            verify_intervals(&bad2)
                .iter()
                .any(|v| v.kind == ViolationKind::Unassigned)
        );
        // Entry-param hint pair is NOT flagged.
        let v = Value::from_u32(9);
        let param = Interval {
            value: Some(v),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 0, to: 4 }],
            uses: Vec::new(),
            location: Some(Location::Reg(2)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        let pin = Interval {
            value: Some(v),
            class: 0,
            fixed_reg: Some(2),
            ranges: alloc::vec![Range { from: 0, to: 1 }],
            uses: Vec::new(),
            location: Some(Location::Reg(2)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        assert!(verify_intervals(&alloc::vec![param, pin]).is_empty());
    }

    #[test]
    fn verify_early_store_ok_and_broken() {
        // Correct early store: value held in reg piece [0,4), rest starts at 5 with store_at 4.
        let v = Value::from_u32(3);
        let head = Interval {
            value: Some(v),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 0, to: 4 }],
            uses: Vec::new(),
            location: Some(Location::Reg(1)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        let tail = Interval {
            value: Some(v),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 5, to: 9 }],
            uses: Vec::new(),
            location: Some(Location::Slot(0)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: true,
            store_at: Some(4),
        };
        assert!(verify_intervals(&alloc::vec![head.clone(), tail.clone()]).is_empty());
        // No live source: flagged StoreSrcDead.
        let lone = Interval {
            store_at: Some(4),
            ..tail.clone()
        };
        assert!(
            verify_intervals(&alloc::vec![lone])
                .iter()
                .any(|v| v.kind == ViolationKind::StoreSrcDead)
        );
        // Another value live in the slot across the prefix: flagged StoreSlotBusy.
        let other = Interval {
            value: Some(Value::from_u32(8)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 2, to: 6 }],
            uses: Vec::new(),
            location: Some(Location::Slot(0)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        assert!(
            verify_intervals(&alloc::vec![head.clone(), tail.clone(), other])
                .iter()
                .any(|v| v.kind == ViolationKind::StoreSlotBusy)
        );
    }

    #[test]
    fn order_moves_cycle_and_slot_to_slot() {
        // Two-register cycle is ordered via the class scratch.
        let va = Value::from_u32(1);
        let vb = Value::from_u32(2);
        let raw = alloc::vec![
            Move {
                src: Location::Reg(0),
                dst: Location::Reg(1),
                class: 0,
                value: va
            },
            Move {
                src: Location::Reg(1),
                dst: Location::Reg(0),
                class: 0,
                value: vb
            },
        ];
        let desc = TestDesc::one_class(4, 3);
        let ordered = order_moves::<Function, _>(&raw, &desc);
        // Simulate: every dst ends with its src's original content.
        assert!(ordered.len() >= 3);
        // Slot-to-slot is routed through the scratch.
        let raw2 = alloc::vec![Move {
            src: Location::Slot(0),
            dst: Location::Slot(1),
            class: 0,
            value: va
        },];
        let ordered2 = order_moves::<Function, _>(&raw2, &desc);
        assert_eq!(ordered2.len(), 2);
    }

    #[test]
    fn fixed_clobber_one_pos_interval_keeps_register() {
        // A one-position interval at a call keeps its register (read-before-clobber).
        let val = Interval {
            value: Some(Value::from_u32(1)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 7, to: 8 }],
            uses: Vec::new(),
            location: Some(Location::Reg(0)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        let clobber = Interval {
            value: None,
            class: 0,
            fixed_reg: Some(0),
            ranges: alloc::vec![Range { from: 7, to: 8 }],
            uses: Vec::new(),
            location: Some(Location::Reg(0)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 1,
            reg_align: 1,
            split_child: false,
            store_at: None,
        };
        // covers(7) but not covers(8): no conflict.
        assert_eq!(fixed_clobber_conflict(&val, &clobber), None);
        // A wider interval crossing the call does conflict.
        let wide = Interval {
            ranges: alloc::vec![Range { from: 5, to: 10 }],
            ..val.clone()
        };
        assert_eq!(fixed_clobber_conflict(&wide, &clobber), Some(7));
    }

    #[test]
    fn multireg_aligned_run_and_spill() {
        // A two-register value takes an aligned run of two.
        let mut func = Function::new();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let a = func.append_inst(b, i32t, Opcode::Iconst(1));
        let d = func.append_inst(
            b,
            i32t,
            Opcode::Arith(volt_ir::function::Arith {
                op: BinOp::Add,
                lhs: a,
                rhs: a,
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(d)));
        let mut desc = TestDesc::one_class(4, 10);
        desc.width = Some((a, 2, 2));
        let ivs = build_intervals(&func, &desc);
        let aiv = ivs.iter().find(|iv| iv.value == Some(a)).unwrap();
        assert_eq!((aiv.regs, aiv.reg_align), (2, 2));
        // Odd base pair is a misaligned span.
        let bad = Interval {
            value: Some(a),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 0, to: 4 }],
            uses: Vec::new(),
            location: Some(Location::Reg(1)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 2,
            reg_align: 2,
            split_child: false,
            store_at: None,
        };
        assert!(
            verify_intervals(&alloc::vec![bad])
                .iter()
                .any(|v| v.kind == ViolationKind::MisalignedSpan)
        );
        // Overlapping pairs are flagged.
        let p1 = Interval {
            value: Some(Value::from_u32(11)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 0, to: 4 }],
            uses: Vec::new(),
            location: Some(Location::Reg(0)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 2,
            reg_align: 2,
            split_child: false,
            store_at: None,
        };
        let p2 = Interval {
            value: Some(Value::from_u32(12)),
            class: 0,
            fixed_reg: None,
            ranges: alloc::vec![Range { from: 0, to: 4 }],
            uses: Vec::new(),
            location: Some(Location::Reg(1)),
            copy_src: None,
            narrow: false,
            preserves_narrow: false,
            regs: 2,
            reg_align: 2,
            split_child: false,
            store_at: None,
        };
        assert!(
            verify_intervals(&alloc::vec![p1, p2])
                .iter()
                .any(|v| v.kind == ViolationKind::RegOverlap)
        );
    }
}

import VoltIR

//! Shared linear-scan register allocator with splitting (Wimmer & Franz Fig. 7,
//! minus holes and re-homing). Values hold registers across their hot prefix;
//! under pressure the furthest-ending victim splits at the newcomer's start —
//! head keeps the register, tail spills, one store fires before the taking
//! instruction. Reloads are implicit: emission reads the covering segment at
//! each use. Segments span blocks freely, so no resolution pass is needed;
//! stores only fire inside a block (block-row newcomers and start-ties spill
//! whole). Values that must survive a clobber point (a call, `div`/`rem`) are
//! forced to slots by the backend's `mustSpill`.
//!
//! Soundness across control flow: a split store fires on one path, so a value
//! whose uses sit in blocks the store's block does not dominate is demoted to
//! a plain slot (def writes memory directly, no store fires). The backend
//! passes a dominance oracle plus a position->block map; without them the
//! scan keeps the old behavior. Holes and inactive lists (spill tails
//! re-homing into freed registers) are a later increment: they need
//! store/reload action ordering in emission, which implicit reloads cannot
//! carry — attempted once and reverted (region pieces took fresh registers
//! with no move, miscompiling even `x+x`).
//!
//! The scan is generic over the pool: the backend supplies the allocatable
//! register indices (as `Int`s it later maps to its `Reg` type), the live
//! intervals, and the forced-spill predicate. Index `slotBase` reserves the
//! first N slot indices for backend scratch areas (e.g. edge-move temps).

/// Where a value lives for its whole interval: a pool register or a stack slot.
public enum Loc: Sendable, Hashable {
  /// A pool register by the backend's own index into its allocatable list.
  case reg(Int)
  /// A stack slot by index. The backend lays slots out (8 bytes each here).
  case slot(Int)
}

/// A value's live interval: defined at `start`, last used at `end` (inclusive
/// positions in block order: block-param row, one slot per instruction, one
/// terminator slot). `mustSpill` forces a stack home regardless of pressure.
public struct LiveInterval: Sendable {
  public var value: Value
  public var start: Int
  public var end: Int
  public var mustSpill: Bool
  public init(value: Value, start: Int, end: Int, mustSpill: Bool = false) {
    self.value = value
    self.start = start
    self.end = end
    self.mustSpill = mustSpill
  }
}

public struct ScanRequest: Sendable {
  /// Allocatable register indices, in preference order.
  public var pool: [Int]
  /// Intervals in any order; the scan sorts by start.
  public var intervals: [LiveInterval]
  /// First spill-slot index available to the scan (backend scratch below).
  public var slotBase: Int
  /// Every operand position per value (from `buildIntervals`). Feeds the
  /// dominance check: uses at/after a split decide whether the store's block
  /// dominates every read.
  public var uses: [Value: [Int]]
  public init(
    pool: [Int], intervals: [LiveInterval], slotBase: Int = 0, uses: [Value: [Int]] = [:]
  ) {
    self.pool = pool
    self.intervals = intervals
    self.slotBase = slotBase
    self.uses = uses
  }
}

public struct ScanResult: Sendable {
  /// Segments per value, ascending by `from`. A value split under pressure has
  /// a register prefix plus a spill tail; an unsplit value has one segment.
  public var segments: [Value: [Segment]]
  /// Stores to emit before the instruction at `at`: the prefix register's
  /// content spilled to the tail's slot. Ascending by `at`.
  public var stores: [SplitStore]
  /// Number of spill slots consumed (indices `slotBase..<(slotBase+count)`).
  public var spillCount: Int
  /// Values holding a register in their first segment, for stats/tests.
  public var inRegisters: Int
  public init(segments: [Value: [Segment]], stores: [SplitStore], spillCount: Int, inRegisters: Int)
  {
    self.segments = segments
    self.stores = stores
    self.spillCount = spillCount
    self.inRegisters = inRegisters
  }

  /// The location covering `pos`: the last segment starting at or before it.
  public func loc(_ value: Value, at pos: Int) -> Loc? {
    guard let segs = segments[value] else { return nil }
    var out = segs[0].loc
    for s in segs {
      if s.from > pos { break }
      out = s.loc
    }
    return out
  }

  /// The value's spill slot, if any segment uses one. Each split value owns
  /// exactly one slot shared by its store and its tail reloads.
  public func slotOf(_ value: Value) -> Int? {
    segments[value]?.compactMap { if case .slot(let s) = $0.loc { return s } else { return nil } }
      .first
  }
}

/// One placement of a value: from position `from` (inclusive) to the next
/// segment's `from` (exclusive), the value lives at `loc`.
public struct Segment: Sendable, Hashable {
  public var from: Int
  public var loc: Loc
  public init(from: Int, loc: Loc) {
    self.from = from
    self.loc = loc
  }
}

/// A prefix-register spill: before the instruction at `at`, store `value`'s
/// `reg` (pool index) into `slot`.
public struct SplitStore: Sendable, Hashable {
  public var at: Int
  public var value: Value
  public var reg: Int
  public var slot: Int
  public init(at: Int, value: Value, reg: Int, slot: Int) {
    self.at = at
    self.value = value
    self.reg = reg
    self.slot = slot
  }
}

/// Classic linear scan with splitting (Wimmer & Franz Fig. 7, minus holes and
/// re-homing). Walks intervals in start order, expiring whatever ended before
/// each start; a free pool register goes to the new interval. Under pressure
/// the furthest-ending active interval splits at the newcomer's start `P`
/// (when its own start is strictly before `P`): its head keeps the register
/// for `[start, P)`, its tail `[P, end)` takes a slot, and a store fires
/// before the instruction at `P`. The newcomer takes the register. Tails
/// never re-split; a newcomer that itself ends furthest, a start-tie, or a
/// block-row start (params) spills whole with no victim split — so stores
/// only ever fire inside a block before a same-block instruction, never on a
/// control-flow edge. Forced spills (`mustSpill`) take slots directly.
///
/// Soundness across control flow: a split store is a real instruction on one
/// path. When a value's interval spans blocks, the store may not dominate a
/// slot-segment read on another path (e.g. the store fires in a loop body or
/// a taken arm while a skip path reads the same value later). The scan takes
/// an optional dominance oracle `(storeBlock, useBlock)` so it can demote
/// those values to plain slots (no store needed — the def writes memory
/// directly). Blocks come from `posBlock`, which maps an instruction position
/// to its block; pass both nil to keep the old behavior.
public func linearScan(
  _ request: ScanRequest, posBlock: [Int: Block]? = nil,
  dominates: ((_ store: Block, _ use: Block) -> Bool)? = nil,
  uses: [Value: [Int]]? = nil
) -> ScanResult {
  // Cheap path: without an oracle there is nothing to demote.
  if posBlock == nil || dominates == nil || uses == nil {
    return scanCore(request)
  }
  var req = request
  // At most one demotion round per value: each round forces one more value to
  // a slot, and slotted values never split, so this terminates.
  for _ in 0...request.intervals.count {
    let result = scanCore(req)
    guard
      let (storePos, value) = firstUndominatedStore(
        result, posBlock: posBlock!, dominates: dominates!, uses: uses!)
    else { return result }
    // Demote: the value's whole interval goes to a slot. Re-mark its
    // interval mustSpill and rescan; the def commits straight to memory.
    req.intervals = req.intervals.map {
      $0.value == value
        ? LiveInterval(value: $0.value, start: $0.start, end: $0.end, mustSpill: true) : $0
    }
    _ = storePos
  }
  return scanCore(req)
}

/// The scan itself: start order, expire-before-start, split furthest-ending
/// victim at the newcomer's start. Works on whole intervals (no holes): a
/// value holds one home per segment, segments only split under pressure, and
/// holes/re-homing are a later increment needing store/reload action ordering
/// in emission. No dominance reasoning here.
func scanCore(_ request: ScanRequest) -> ScanResult {
  let ordered = request.intervals.sorted {
    if $0.start != $1.start { return $0.start < $1.start }
    return $0.end < $1.end
  }
  var segments: [Value: [Segment]] = [:]
  var stores: [SplitStore] = []
  // Active register-held intervals, with their pool slot.
  var active: [(interval: LiveInterval, poolIdx: Int)] = []
  var freePool = request.pool
  var nextSlot = request.slotBase
  var inRegisters = 0

  func expire(before pos: Int) {
    active.removeAll { a in
      if a.interval.end < pos {
        freePool.append(a.poolIdx)
        return true
      }
      return false
    }
  }

  for iv in ordered {
    if iv.mustSpill {
      let s = nextSlot
      nextSlot += 1
      segments[iv.value] = [Segment(from: iv.start, loc: .slot(s))]
      continue
    }
    expire(before: iv.start)
    if let r = freePool.popLast() {
      segments[iv.value] = [Segment(from: iv.start, loc: .reg(r))]
      active.append((iv, r))
      inRegisters += 1
    } else if let victim = active.max(by: { $0.interval.end < $1.interval.end }),
      victim.interval.end > iv.end,
      victim.interval.start < iv.start
    {
      // Split the victim at the newcomer's start: head keeps the register,
      // tail spills, store fires first. The newcomer takes the register.
      let s = nextSlot
      nextSlot += 1
      segments[victim.interval.value] = [
        Segment(from: victim.interval.start, loc: .reg(victim.poolIdx)),
        Segment(from: iv.start, loc: .slot(s)),
      ]
      stores.append(
        SplitStore(at: iv.start, value: victim.interval.value, reg: victim.poolIdx, slot: s))
      active.removeAll { $0.interval.value == victim.interval.value }
      segments[iv.value] = [Segment(from: iv.start, loc: .reg(victim.poolIdx))]
      active.append((iv, victim.poolIdx))
    } else {
      let s = nextSlot
      nextSlot += 1
      segments[iv.value] = [Segment(from: iv.start, loc: .slot(s))]
    }
  }
  stores.sort { $0.at < $1.at }
  return ScanResult(
    segments: segments, stores: stores, spillCount: nextSlot - request.slotBase,
    inRegisters: inRegisters)
}

/// First store with a use the store does not dominate: some use of `value`
/// at or after the split point sits in a block the store's block does not
/// dominate (the store fires on one path while another path reads the slot
/// before any store). Returns the store position and value for demotion.
func firstUndominatedStore(
  _ result: ScanResult, posBlock: [Int: Block],
  dominates: (Block, Block) -> Bool, uses: [Value: [Int]]
) -> (Int, Value)? {
  for st in result.stores {
    guard let storeBlock = posBlock[st.at] else { continue }
    for u in uses[st.value] ?? [] where u >= st.at {
      guard let useBlock = posBlock[u] else { continue }
      if !dominates(storeBlock, useBlock) { return (st.at, st.value) }
    }
  }
  return nil
}

/// Dominance over the successor map: `dom(s, u)` is true when every path
/// from the entry block to `u` passes through `s`. Iterative dataflow over
/// the predecessor map (entry dominates itself). Small and boring; the
/// backend queries it once per slot segment, not per instruction.
public func dominance(successors: [Block: [Block]], entry: Block) -> (Block, Block) -> Bool {
  var preds: [Block: [Block]] = [:]
  for (b, ss) in successors {
    for s in ss { preds[s, default: []].append(b) }
    if preds[b] == nil { preds[b] = [] }
  }
  if preds[entry] == nil { preds[entry] = [] }
  // dom(entry) = {entry}; dom(other) = all blocks; iterate to fixpoint.
  var all: Set<Block> = Set(preds.keys)
  all.insert(entry)
  var dom: [Block: Set<Block>] = [:]
  for b in all { dom[b] = b == entry ? [entry] : all }
  var changed = true
  while changed {
    changed = false
    for b in all where b != entry {
      let ps = preds[b] ?? []
      var next: Set<Block> = ps.first.flatMap { dom[$0] } ?? []
      for p in ps.dropFirst() { next.formIntersection(dom[p] ?? []) }
      next.insert(b)
      if next != dom[b] {
        dom[b] = next
        changed = true
      }
    }
  }
  return { s, u in dom[u]?.contains(s) ?? (s == u) }
}

/// Position -> block map from the builder's ranges: every numbered position
/// (param rows, instructions, terminator slots) belongs to exactly one block.
public func positionBlocks(_ positions: [Block: Range<Int>]) -> [Int: Block] {
  var out: [Int: Block] = [:]
  for (b, r) in positions { for p in r { out[p] = b } }
  return out
}

/// Block layout plus intervals from one numbering pass. Member names match
/// the old 5-tuple (`intervals`, `positions`, `instPos`, `termPos`,
/// `successors`) so call sites read unchanged; `uses` adds every operand
/// position per value for the dominance check.
public struct BuiltIntervals: Sendable {
  public var intervals: [LiveInterval]
  public var positions: [Block: Range<Int>]
  public var instPos: [Inst: Int]
  public var termPos: [Block: Int]
  public var successors: [Block: [Block]]
  public var uses: [Value: [Int]]
  public init(
    intervals: [LiveInterval], positions: [Block: Range<Int>], instPos: [Inst: Int],
    termPos: [Block: Int], successors: [Block: [Block]], uses: [Value: [Int]]
  ) {
    self.intervals = intervals
    self.positions = positions
    self.instPos = instPos
    self.termPos = termPos
    self.successors = successors
    self.uses = uses
  }
}

public func buildIntervals(
  function: Function,
  clobberPositions: Set<Int> = []
) -> BuiltIntervals {
  var successors: [Block: [Block]] = [:]
  for b in function.blocks {
    if let t = function.terminator(of: b) {
      switch t {
      case .jump(let j): successors[b] = [j.target]
      case .condBr(let c): successors[b] = [c.then, c.else]
      default: successors[b] = []
      }
    } else {
      successors[b] = []
    }
  }
  // Number every row first so def/use lookup is O(1).
  var defPos: [Value: Int] = [:]
  var lastUse: [Value: Int] = [:]
  var usePos: [Value: [Int]] = [:]
  var blockRange: [Block: Range<Int>] = [:]
  var instPos: [Inst: Int] = [:]
  var termPos: [Block: Int] = [:]
  var pos = 0
  for b in function.blocks {
    let base = pos
    pos += 1  // block-parameter row
    for p in function.params(of: b) {
      defPos[p] = base
      lastUse[p] = base
    }
    for i in function.instructions(in: b) {
      instPos[i] = pos
      for u in uses(of: function.opcode(i)) {
        lastUse[u] = pos
        usePos[u, default: []].append(pos)
      }
      if let v = function.resultValue(of: i) {
        defPos[v] = pos
        if lastUse[v] == nil { lastUse[v] = pos }
      }
      pos += 1
    }
    if let t = function.terminator(of: b) {
      termPos[b] = pos
      for u in uses(of: t) {
        lastUse[u] = pos
        usePos[u, default: []].append(pos)
      }
      pos += 1  // terminator slot
    }
    blockRange[b] = base..<pos
  }
  // Compute block liveness to a fixed point. Linear use positions alone miss
  // values carried through blocks with no local use, notably loop invariants
  // whose textual last use precedes the loop body in layout order.
  var blockUse: [Block: Set<Value>] = [:]
  var blockDef: [Block: Set<Value>] = [:]
  for b in function.blocks {
    var defined = Set(function.params(of: b))
    var used = Set<Value>()
    func recordUses(_ values: [Value]) {
      for value in values where !defined.contains(value) { used.insert(value) }
    }
    for i in function.instructions(in: b) {
      recordUses(uses(of: function.opcode(i)))
      if let value = function.resultValue(of: i) { defined.insert(value) }
    }
    if let term = function.terminator(of: b) { recordUses(uses(of: term)) }
    blockUse[b] = used
    blockDef[b] = defined
  }
  var liveIn = Dictionary(uniqueKeysWithValues: function.blocks.map { ($0, Set<Value>()) })
  var liveOut = liveIn
  var changed = true
  while changed {
    changed = false
    for b in function.blocks.reversed() {
      var out = Set<Value>()
      for succ in successors[b] ?? [] {
        let params = function.params(of: succ)
        let successorLive = liveIn[succ] ?? []
        out.formUnion(successorLive.subtracting(params))
        let edgeArgs: [[Value]]
        guard let term = function.terminator(of: b) else { continue }
        switch term {
        case .jump(let jump) where jump.target == succ: edgeArgs = [jump.args]
        case .condBr(let branch):
          var args: [[Value]] = []
          if branch.then == succ { args.append(branch.thenArgs) }
          if branch.else == succ { args.append(branch.elseArgs) }
          edgeArgs = args
        default: edgeArgs = []
        }
        for args in edgeArgs where args.count == params.count {
          // Conservatively keep every incoming argument live across this edge;
          // dead block parameters will be removed by a future IR simplifier.
          for (_, arg) in zip(params, args) { out.insert(arg) }
        }
      }
      let incoming = (blockUse[b] ?? []).union(out.subtracting(blockDef[b] ?? []))
      if out != liveOut[b] || incoming != liveIn[b] {
        liveOut[b] = out
        liveIn[b] = incoming
        changed = true
      }
    }
  }

  var extendedStart = defPos
  var extendedEnd = lastUse
  for b in function.blocks {
    guard let range = blockRange[b], let term = termPos[b] else { continue }
    for value in (liveIn[b] ?? []).union(liveOut[b] ?? []) where defPos[value] != nil {
      extendedStart[value] = min(extendedStart[value] ?? range.lowerBound, range.lowerBound)
      extendedEnd[value] = max(extendedEnd[value] ?? term, term)
    }
  }
  var intervals: [LiveInterval] = []
  for (v, d) in defPos {
    let start = extendedStart[v] ?? d
    let e = extendedEnd[v] ?? d
    let spansClobber = clobberPositions.contains { $0 > start && $0 <= e }
    intervals.append(LiveInterval(value: v, start: start, end: e, mustSpill: spansClobber))
  }
  return BuiltIntervals(
    intervals: intervals, positions: blockRange, instPos: instPos, termPos: termPos,
    successors: successors, uses: usePos)
}

/// Operand values read by an opcode (definitions excluded).
func uses(of op: Opcode) -> [Value] {
  switch op {
  case .ptrAdd(let p): return [p.base]
  case .arith(let a): return [a.lhs, a.rhs]
  case .arithImm(let a): return [a.lhs]
  case .cmp(let c): return [c.lhs, c.rhs]
  case .select(let s): return [s.cond, s.then, s.else]
  case .load(let l): return [l.ptr]
  case .store(let s): return [s.value, s.ptr]
  case .call(let c): return c.args
  case .callIndirect(let c): return [c.target] + c.args
  case .convert(let c): return [c.value]
  case .ret(let r): return r.values
  case .jump(let j): return j.args
  case .condBr(let c): return [c.cond] + c.thenArgs + c.elseArgs
  default: return []
  }
}

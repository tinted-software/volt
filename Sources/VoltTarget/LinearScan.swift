import VoltIR

//! Shared linear-scan register allocator with splitting (Wimmer & Franz Fig. 7,
//! minus holes and re-homing). Values hold registers across their hot prefix;
//! under pressure the furthest-ending victim splits at the newcomer's start —
//! head keeps the register, tail spills, one store fires before the taking
//! instruction. Reloads are implicit: emission reads the covering segment at
//! each use. Segments span blocks freely, so no resolution pass is needed;
//! stores only fire inside a block (block-row newcomers and start-ties spill
//! whole). Values that must survive a clobber point (a call, `div`/`rem`) are
//! forced to slots by the backend's `mustSpill`. Next increment: holes and
//! inactive lists so spill tails can re-home into freed registers.
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
  public init(pool: [Int], intervals: [LiveInterval], slotBase: Int = 0) {
    self.pool = pool
    self.intervals = intervals
    self.slotBase = slotBase
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
public func linearScan(_ request: ScanRequest) -> ScanResult {
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

/// Build block-order intervals for every value in `function`: position = a
/// per-block row (param row, one slot per instruction, one terminator slot).
/// `defPos`/`usePos` number those rows. `mustSpill` marks values whose interval
/// covers a clobber position (the backend passes those positions in).
public func buildIntervals(
  function: Function,
  clobberPositions: Set<Int> = []
) -> (
  intervals: [LiveInterval],
  positions: [Block: Range<Int>],
  instPos: [Inst: Int],
  termPos: [Block: Int]
) {
  // Number every row first so def/use lookup is O(1).
  var defPos: [Value: Int] = [:]
  var lastUse: [Value: Int] = [:]
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
      }
      if let v = function.resultValue(of: i) {
        defPos[v] = pos
        if lastUse[v] == nil { lastUse[v] = pos }
      }
      pos += 1
    }
    if let t = function.terminator(of: b) {
      termPos[b] = pos
      for u in uses(of: t) { lastUse[u] = pos }
      pos += 1  // terminator slot
    }
    blockRange[b] = base..<pos
  }
  var intervals: [LiveInterval] = []
  for (v, d) in defPos {
    let e = lastUse[v] ?? d
    let spansClobber = clobberPositions.contains { $0 > d && $0 <= e }
    intervals.append(LiveInterval(value: v, start: d, end: e, mustSpill: spansClobber))
  }
  return (intervals, blockRange, instPos, termPos)
}

/// Operand values read by an opcode (definitions excluded).
func uses(of op: Opcode) -> [Value] {
  switch op {
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

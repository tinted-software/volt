//! x86_64 instruction selection: lowers an IR `Function` to machine code.
//! System V AMD64 ABI: integer arguments in RDI, RSI, RDX, RCX, R8, R9, result
//! in RAX. Modeled on `libs/vulcan-target/x86_64/isel.zig`, simplified for the
//! v1 integer subset.
//!
//! Allocation model: a classic no-split linear scan (`VoltTarget/linearScan`)
//! gives every value one home — a pool register or a stack slot — for its whole
//! interval. Register-held values never touch memory between def and last use.
//! The pool holds caller-saved argument/data registers plus callee-saved
//! registers (saved/restored in prologue/epilogue when used); the fixed temps
//! (rax, rcx, rdx, r10, r11) stay out of the pool so operand shuffling never
//! collides with a live home. Values live across a `call` or `div`/`rem` are
//! forced to slots (the ABI clobbers caller-saved registers; `div` needs
//! rax/rdx). x86 arithmetic is two-operand, so `c = a op b` becomes
//! `mov c, a` then `op c, b`.
//!
//! Supported v1 subset: iconst, arith/arithImm (add, sub, mul, and, or, xor,
//! shl, shr, div, rem via idiv), cmp, select (via cmov), ret/jump/condBr with
//! block-parameter edge moves, call (direct, intra-module rel32 + ABI arg
//! shuffling). Anything else throws `CodegenError.unsupported`.

import VoltIR
import VoltTarget

let argRegs: [Reg] = [.rdi, .rsi, .rdx, .rcx, .r8, .r9]
let retReg: Reg = .rax
let scratch1: Reg = .r10  // operand scratch: never in the pool, never a home
let scratch2: Reg = .r11  // second operand scratch: same reservation

/// Allocatable pool, least-preferred first (`linearScan` pops from the end, so
/// argument registers come out first — a param homed in its ABI register needs
/// no entry move). Callee-saved registers sit deepest; the prologue saves only
/// the ones actually used.
let poolRegs: [Reg] = [.r15, .r14, .r13, .r12, .rbx, .r9, .r8, .rsi, .rdi]
let calleeSavedRegs: Set<Reg> = [.rbx, .r12, .r13, .r14, .r15]

/// Lower `function` to machine code plus its call relocations.
/// - Precondition: every block ends in a terminator; block params are i64-width
///   integers (the v1 ABI carries all values as 64-bit slots).
public func compileFunction(_ function: Function) throws -> CompiledCode {
  guard function.blockCount() > 0 else {
    throw CodegenError.invalidFunction("no blocks")
  }
  var ctx = ISelContext(function: function)
  try ctx.run()
  return CompiledCode(bytes: ctx.code, relocs: ctx.relocs)
}

/// Run the allocator over `function` and return the raw scan (homes, spill
/// count, register count) for tests and tools. `compileFunction` uses this.
public func allocateFunction(_ function: Function) throws -> ScanResult {
  guard function.blockCount() > 0 else {
    throw CodegenError.invalidFunction("no blocks")
  }
  let clobbers = clobberPositions(function)
  let built = buildIntervals(function: function, clobberPositions: clobbers)
  let pool = Array(0..<poolRegs.count)
  let dom = dominance(successors: built.successors, entry: function.blocks[0])
  return linearScan(
    ScanRequest(pool: pool, intervals: built.intervals, slotBase: 0, uses: built.uses),
    posBlock: positionBlocks(built.positions), dominates: dom, uses: built.uses)
}

/// Instruction positions (in `buildIntervals` numbering) that clobber
/// caller-saved registers: calls clobber them all; `div`/`rem` need rax/rdx.
func clobberPositions(_ function: Function) -> Set<Int> {
  var out = Set<Int>()
  var pos = 0
  for b in function.blocks {
    pos += 1  // block-parameter row
    for i in function.instructions(in: b) {
      switch function.opcode(i) {
      case .call, .callIndirect:
        out.insert(pos)
      case .arith(let a) where a.op == .div || a.op == .rem:
        out.insert(pos)
      case .arithImm(let a) where a.op == .div || a.op == .rem:
        out.insert(pos)
      default:
        break
      }
      pos += 1
    }
    pos += 1  // terminator slot
  }
  return out
}

/// The largest parallel-move width in `function`: block edges, the entry
/// shuffle, and call-arg shuffles (for temp-area sizing).
func maxMoveWidth(_ function: Function) -> Int {
  var m = 1
  if !function.blocks.isEmpty { m = max(m, function.params(of: function.blocks[0]).count) }
  for b in function.blocks {
    for i in function.instructions(in: b) {
      if case .call(let c) = function.opcode(i) { m = max(m, c.args.count) }
    }
    guard let t = function.terminator(of: b) else { continue }
    switch t {
    case .jump(let j): m = max(m, j.args.count)
    case .condBr(let c): m = max(m, c.thenArgs.count, c.elseArgs.count)
    default: break
    }
  }
  return m
}

struct ISelContext {
  let function: Function
  var code: [UInt8] = []
  var relocs: [Reloc] = []
  /// Allocation: segments per value plus prefix-spill stores.
  var scan = ScanResult(segments: [:], stores: [], spillCount: 0, inRegisters: 0)
  var spillSlots: Int = 0
  /// 8-byte temp slots after the value slots, for parallel-move routing.
  var tempSlots: Int = 1
  /// Callee-saved pool regs actually used (pushed in prologue, popped per ret).
  var saved: [Reg] = []
  /// Byte offset of each block's label (filled as blocks are emitted).
  var label: [Block: Int] = [:]
  /// Fixups: (site, target block) for jumps patched after layout.
  var fixups: [(site: Int, block: Block)] = []
  var frameSize: Int = 0
  /// Position maps from interval building (shared numbering with the scan).
  var instPos: [Inst: Int] = [:]
  var termPos: [Block: Int] = [:]
  /// Position currently being emitted; segment lookups read through it.
  var curPos: Int = 0
  /// Next pending prefix-spill store in `scan.stores` (ascending by `at`).
  var storeIdx: Int = 0

  init(function: Function) { self.function = function }

  /// Pool register for a pool index.
  func poolReg(_ i: Int) -> Reg { poolRegs[i] }
  /// Home register covering the current position, else nil.
  func regAt(_ v: Value) -> Reg? {
    if case .reg(let i) = scan.loc(v, at: curPos) { return poolReg(i) }
    return nil
  }
  /// The value's spill slot (every spilled value owns exactly one).
  func slotOf(_ v: Value) -> Int {
    guard let s = scan.slotOf(v) else { fatalError("value \(v) has no slot") }
    return s
  }
  /// RSP-relative byte offset of spill slot `s`.
  func slotOff(_ s: Int) -> Int32 { Int32(s * 8) }
  /// RSP-relative byte offset of temp slot `t`.
  func tempOff(_ t: Int) -> Int32 { Int32((spillSlots + t) * 8) }
  mutating func run() throws {
    let params = function.params(of: function.blocks[0])
    guard params.count <= argRegs.count else {
      throw CodegenError.unsupported("more than \(argRegs.count) integer args")
    }
    let clobbers = clobberPositions(function)
    let built = buildIntervals(function: function, clobberPositions: clobbers)
    let pool = Array(0..<poolRegs.count)
    // Dominance oracle for split stores: a store in block S covers a slot
    // read in block U only when S dominates U. Undominated values demote to
    // plain slots (def writes memory directly, no store fires).
    let dom = dominance(successors: built.successors, entry: function.blocks[0])
    let posBlock = positionBlocks(built.positions)
    scan = linearScan(
      ScanRequest(pool: pool, intervals: built.intervals, slotBase: 0, uses: built.uses),
      posBlock: posBlock, dominates: dom, uses: built.uses)
    instPos = built.instPos
    termPos = built.termPos
    spillSlots = scan.spillCount
    tempSlots = maxMoveWidth(function)

    // Callee-saved pool regs with any register segment need save/restore.
    var used = Set<Int>()
    for segs in scan.segments.values {
      for s in segs { if case .reg(let i) = s.loc { used.insert(i) } }
    }
    saved = poolRegs.enumerated().compactMap {
      used.contains($0.offset) && calleeSavedRegs.contains($0.element) ? $0.element : nil
    }

    // Frame: value slots + temp area, sized so (pushes + sub) keeps the
    // 16-alignment a `call` needs. Entry rsp%16==8; k pushes move it by 8k.
    let need = (spillSlots + tempSlots) * 8
    let k = saved.count
    var frame = need
    while (8 * k + frame) % 16 != 8 { frame += 8 }
    if frame < 8 { frame += 16 }
    frameSize = frame

    emitPrologue()
    // Move incoming ABI args into their homes (parallel: an ABI reg may be
    // another param's home). Entry moves write each param's first segment.
    curPos = 0
    var srcs: [MoveSrc] = []
    var dsts: [MoveLoc] = []
    for (i, p) in params.enumerated() {
      srcs.append(.abi(argRegs[i]))
      dsts.append(firstLoc(p))
    }
    try parallelMoves(srcs: srcs, dsts: dsts)
    for b in function.blocks {
      label[b] = code.count
      try emitBlock(b)
    }
    try patchFixups()
  }

  /// A value's first-segment location: entry/edge moves land before any split.
  func firstLoc(_ v: Value) -> MoveLoc {
    guard let first = scan.segments[v]?.first else { fatalError("value \(v) has no segments") }
    switch first.loc {
    case .reg(let i): return .reg(poolReg(i))
    case .slot(let s): return .slot(s)
    }
  }

  /// The value produced by `inst`. Uses the function's def table directly.
  func valueOf(inst: Inst) -> Value {
    guard let v = function.resultValue(of: inst) else {
      fatalError("no value for inst \(inst)")
    }
    return v
  }

  mutating func emitPrologue() {
    for r in saved { code += pushReg(r) }
    // sub rsp, frameSize — encoded as 48 81 EC imm32.
    let u = UInt32(bitPattern: Int32(frameSize))
    code += [
      0x48, 0x81, 0xEC, UInt8(u & 0xFF), UInt8((u >> 8) & 0xFF), UInt8((u >> 16) & 0xFF),
      UInt8((u >> 24) & 0xFF),
    ]
  }

  mutating func emitEpilogue() {
    // add rsp, frameSize; pop saved (reverse); ret.
    let u = UInt32(bitPattern: Int32(frameSize))
    code += [
      0x48, 0x81, 0xC4, UInt8(u & 0xFF), UInt8((u >> 8) & 0xFF), UInt8((u >> 16) & 0xFF),
      UInt8((u >> 24) & 0xFF),
    ]
    for r in saved.reversed() { code += popReg(r) }
    code += x86Ret()
  }

  /// Fire prefix-spill stores due at `pos`, before its instruction.
  mutating func fireStores(at pos: Int) {
    while storeIdx < scan.stores.count, scan.stores[storeIdx].at == pos {
      let st = scan.stores[storeIdx]
      code += movToStack(slotOff(st.slot), poolReg(st.reg))
      storeIdx += 1
    }
  }

  /// Materialize value `v`'s covering segment in a register: its home reg, or
  /// `scratch` loaded from its slot. The caller must not hold another live
  /// value in `scratch`.
  mutating func operandReg(_ v: Value, scratch: Reg) -> Reg {
    guard let loc = scan.loc(v, at: curPos) else {
      fatalError("value \(v) has no segment at \(curPos)")
    }
    switch loc {
    case .reg(let i): return poolReg(i)
    case .slot(let s):
      code += movFromStack(scratch, slotOff(s))
      return scratch
    }
  }

  /// The register a fresh definition goes to: its covering home reg, or
  /// `fallback` (a reserved temp) when the covering segment is a slot.
  func defReg(_ v: Value, fallback: Reg) -> Reg {
    regAt(v) ?? fallback
  }

  /// Commit register `r` to value `v`'s covering segment (no-op when home).
  mutating func commitReg(_ r: Reg, _ v: Value) {
    guard let loc = scan.loc(v, at: curPos) else {
      fatalError("value \(v) has no segment at \(curPos)")
    }
    switch loc {
    case .reg(let i):
      if poolReg(i) != r { code += movReg(poolReg(i), r) }
    case .slot(let s):
      code += movToStack(slotOff(s), r)
    }
  }

  /// The result value of `inst`.
  func resultOf(_ inst: Inst) -> Value { valueOf(inst: inst) }

  mutating func emitBlock(_ b: Block) throws {
    for inst in function.instructions(in: b) {
      guard let p = instPos[inst] else {
        throw CodegenError.invalidFunction("inst has no position")
      }
      curPos = p
      fireStores(at: p)
      try emitInst(inst)
    }
    guard let term = function.terminator(of: b) else {
      throw CodegenError.invalidFunction("block \(b.raw) has no terminator")
    }
    guard let tp = termPos[b] else {
      throw CodegenError.invalidFunction("block \(b.raw) has no terminator slot")
    }
    curPos = tp
    try emitTerminator(term, from: b)
  }

  mutating func emitInst(_ inst: Inst) throws {
    let op = function.opcode(inst)
    switch op {
    case .iconst(let imm):
      let v = resultOf(inst)
      let d = defReg(v, fallback: .rax)
      if imm >= Int64(Int32.min), imm <= Int64(Int32.max) {
        code += movImm(d, Int32(imm), true)
      } else {
        // movImm64 needs the full 64-bit form; only rax is guaranteed
        // scratch-safe here, so go through rax when d is a home reg.
        if d == .rax {
          code += movImm64(d, UInt64(bitPattern: imm))
        } else {
          code += movImm64(.rax, UInt64(bitPattern: imm))
          code += movReg(d, .rax)
        }
      }
      commitReg(d, v)

    case .arith(let a):
      try emitArith(op: a.op, lhs: a.lhs, rhs: a.rhs, dst: resultOf(inst))

    case .arithImm(let a):
      try emitArithImm(op: a.op, lhs: a.lhs, imm: a.imm, dst: resultOf(inst))

    case .cmp(let c):
      try emitCmp(op: c.op, lhs: c.lhs, rhs: c.rhs, dst: resultOf(inst))

    case .select(let s):
      // cond -> rax (test); then/else -> regs or scratch; cmov on test.
      let c = operandReg(s.cond, scratch: .rax)
      if c != .rax { code += movReg(.rax, c) }
      let t = operandReg(s.then, scratch: scratch1)
      // `else` must not share the destination: route through scratch2 when
      // it would collide with then's register.
      var e: Reg
      if let er = regAt(s.else), er == t {
        code += movReg(scratch2, er)
        e = scratch2
      } else {
        e = operandReg(s.else, scratch: scratch2)
        if e == t {
          code += movReg(scratch2, e)
          e = scratch2
        }
      }
      code += testRR(.rax, .rax, true)
      code += cmovcc(t, e, .e)  // if cond==0, dst = else
      commitReg(t, resultOf(inst))

    case .call(let c):
      try emitCall(symbolIndex: c.symbol, args: c.args, dst: resultOfOrNil(inst))

    case .callIndirect:
      throw CodegenError.unsupported("call_indirect")

    case .store(let s):
      let val = operandReg(s.value, scratch: scratch1)
      // The pointer must not share val's register: route via scratch2.
      var ptr: Reg
      if let pr = regAt(s.ptr), pr == val {
        code += movReg(scratch2, pr)
        ptr = scratch2
      } else {
        ptr = operandReg(s.ptr, scratch: scratch2)
        if ptr == val {
          code += movReg(scratch2, ptr)
          ptr = scratch2
        }
      }
      code += movToMem(ptr, 0, val)

    case .load(let l):
      let ptr = operandReg(l.ptr, scratch: scratch1)
      let v = resultOf(inst)
      let d = defReg(v, fallback: scratch1)
      if d == ptr {
        // `mov d, [d]`: the ModRM form reads [d] before writing d, so
        // same-register base+dest is exact.
        code += movFromMem(d, d, 0)
      } else {
        code += movFromMem(d, ptr, 0)
      }
      commitReg(d, v)

    case .ret, .jump, .condBr:
      throw CodegenError.invalidFunction("terminator in body")

    default:
      throw CodegenError.unsupported(describe(op))
    }
  }

  func resultOfOrNil(_ inst: Inst) -> Value? { function.resultValue(of: inst) }

  /// Prepare a binary op: destination register `d`, left `l`, right `r`.
  /// The right side is routed through scratch2 when it would share `d`
  /// (a non-commutative op would otherwise lose it to the `mov d, l`).
  mutating func prepareBinary(lhs: Value, rhs: Value, dst: Value) -> (d: Reg, l: Reg, r: Reg) {
    let d = defReg(dst, fallback: scratch1)
    var l = operandReg(lhs, scratch: scratch1)
    // If lhs came from a slot it now sits in scratch1, which may BE d.
    // That's fine: the mov below becomes a no-op and we compute in place.
    if l == scratch1, d == scratch1 { l = scratch1 }
    var r: Reg
    if let rr = regAt(rhs), rr == d {
      code += movReg(scratch2, rr)
      r = scratch2
    } else {
      r = operandReg(rhs, scratch: scratch2)
      if r == d {
        code += movReg(scratch2, r)
        r = scratch2
      }
    }
    return (d, l, r)
  }

  mutating func emitArith(op: BinOp, lhs: Value, rhs: Value, dst: Value) throws {
    switch op {
    case .div, .rem:
      // Signed division via idiv: dividend rax, divisor in rcx (not rax/rdx).
      // Values spanning a div are forced to slots, so rax/rcx/rdx are free:
      // none of them can be a live home here.
      let a = operandReg(lhs, scratch: .rax)
      if a != .rax { code += movReg(.rax, a) }
      let b = operandReg(rhs, scratch: .rcx)
      if b != .rcx { code += movReg(.rcx, b) }
      code += cqo()
      code += idiv(.rcx, true)
      // quotient in rax, remainder in rdx.
      commitReg(op == .div ? .rax : .rdx, dst)
    case .mul:
      let (d, l, r) = prepareBinary(lhs: lhs, rhs: rhs, dst: dst)
      if d != l { code += movReg(d, l) }
      code += imulRR(d, r, true)
      commitReg(d, dst)
    case .add, .sub, .bitAnd, .bitOr, .bitXor:
      let (d, l, r) = prepareBinary(lhs: lhs, rhs: rhs, dst: dst)
      if d != l { code += movReg(d, l) }
      switch op {
      case .add: code += addRR(d, r, true)
      case .sub: code += subRR(d, r, true)
      case .bitAnd: code += andRR(d, r, true)
      case .bitOr: code += orRR(d, r, true)
      default: code += xorRR(d, r, true)
      }
      commitReg(d, dst)
    case .shl, .shr:
      // Shift count in cl (rcx is reserved, never a home).
      let l = operandReg(lhs, scratch: scratch1)
      let k = operandReg(rhs, scratch: .rcx)
      if k != .rcx { code += movReg(.rcx, k) }
      let d = defReg(dst, fallback: scratch1)
      if d != l { code += movReg(d, l) }
      code += shiftCl(op == .shl ? 4 : 5, d, true)
      commitReg(d, dst)
    case .mulH:
      throw CodegenError.unsupported("mulh")
    }
  }

  mutating func emitArithImm(op: BinOp, lhs: Value, imm: Int64, dst: Value) throws {
    let l = operandReg(lhs, scratch: scratch1)
    let d = defReg(dst, fallback: scratch1)
    if d != l { code += movReg(d, l) }
    guard imm >= Int64(Int32.min), imm <= Int64(Int32.max) else {
      // Wide immediate: materialize in scratch2, fall back to reg-reg.
      code += movImm64(scratch2, UInt64(bitPattern: imm))
      switch op {
      case .add: code += addRR(d, scratch2, true)
      case .sub: code += subRR(d, scratch2, true)
      case .bitAnd: code += andRR(d, scratch2, true)
      case .bitOr: code += orRR(d, scratch2, true)
      case .bitXor: code += xorRR(d, scratch2, true)
      case .mul: code += imulRR(d, scratch2, true)
      default:
        throw CodegenError.unsupported("arith-imm \(op)")
      }
      commitReg(d, dst)
      return
    }
    let i32imm = Int32(imm)
    switch op {
    case .add: code += aluImm(.add, d, i32imm, true)
    case .sub: code += aluImm(.sub, d, i32imm, true)
    case .bitAnd: code += aluImm(.and, d, i32imm, true)
    case .bitOr: code += aluImm(.orr, d, i32imm, true)
    case .bitXor: code += aluImm(.xor, d, i32imm, true)
    case .mul: code += imulImm(d, d, i32imm, true)
    case .shl where imm >= 0 && imm < 64:
      code += shiftImm(.shl, d, UInt8(imm), true)
    case .shr where imm >= 0 && imm < 64:
      code += shiftImm(.shr, d, UInt8(imm), true)
    default:
      throw CodegenError.unsupported("arith-imm \(op)")
    }
    commitReg(d, dst)
  }

  mutating func emitCmp(op: CmpOp, lhs: Value, rhs: Value, dst: Value) throws {
    // `cmp r, r` (same home both sides) is exact — equality with itself —
    // so no routing is needed even when both operands share a register.
    let l = operandReg(lhs, scratch: scratch1)
    let r = operandReg(rhs, scratch: scratch2)
    code += cmpRR(l, r, true)
    let cond: Cond
    switch op {
    case .eq: cond = .e
    case .ne: cond = .ne
    case .lt: cond = .l
    case .le: cond = .le
    case .gt: cond = .g
    case .ge: cond = .ge
    }
    code += setcc(.rax, cond)
    code += movzxByte(.rax, .rax)
    commitReg(.rax, dst)
  }

  mutating func emitCall(symbolIndex: UInt32, args: [Value], dst: Value?) throws {
    guard args.count <= argRegs.count else {
      throw CodegenError.unsupported("more than \(argRegs.count) call args")
    }
    // Shuffle args into ABI registers through the parallel-move router:
    // an arg's home may itself be an ABI register another arg needs.
    // Call-spanning values are forced to slots, so no live home dies here.
    // Sources read at the call's position (a split may have moved them).
    try parallelMoves(
      srcs: args.map { .loc(locOf($0)) }, dsts: Array(argRegs.prefix(args.count)).map { .reg($0) })
    // call rel32 with a zero placeholder; the linker resolves it.
    let name = externName(symbolIndex)
    relocs.append(Reloc(offset: code.count + 1, symbol: name, kind: .call))
    code += callRel(0)
    if let dst {
      commitReg(retReg, dst)
    }
  }

  /// v1 symbol table: indices map to `extern<N>` unless the driver overrides.
  /// `Linker` replaces these with real names via `compileModule(names:)`.
  var externNames: [UInt32: String] = [:]
  func externName(_ i: UInt32) -> String { externNames[i] ?? "extern\(i)" }

  mutating func emitTerminator(_ term: Opcode, from: Block) throws {
    switch term {
    case .ret(let r):
      if let v = r.values.first {
        let r0 = operandReg(v, scratch: retReg)
        if r0 != retReg { code += movReg(retReg, r0) }
      }
      emitEpilogue()
    case .jump(let j):
      try emitEdgeMoves(args: j.args, to: j.target)
      // jmp rel32 placeholder; patched after layout.
      fixups.append((site: code.count + 1, block: j.target))
      code += jmp(0)
    case .condBr(let c):
      //   test cond,cond
      //   jz elseMoves
      //   <then moves>; jmp thenBlock
      //   elseMoves: <else moves>; jmp elseBlock
      let cr = operandReg(c.cond, scratch: .rax)
      if cr != .rax { code += movReg(.rax, cr) }
      code += testRR(.rax, .rax, true)
      let jzSite = code.count + 2  // 0F 84 rel32: disp starts 2 bytes in
      code += jcc(.e, 0)  // placeholder -> elseMoves
      try emitEdgeMoves(args: c.thenArgs, to: c.then)
      let jmpThenSite = code.count + 1
      code += jmp(0)  // placeholder -> thenBlock
      let elseMovesAt = code.count
      try emitEdgeMoves(args: c.elseArgs, to: c.else)
      let jmpElseSite = code.count + 1
      code += jmp(0)  // placeholder -> elseBlock
      // Patch the jz to elseMoves now (position known).
      let relElse = Int32(elseMovesAt - (jzSite + 4))
      patchRel32(at: jzSite, rel: relElse)
      fixups.append((site: jmpThenSite, block: c.then))
      fixups.append((site: jmpElseSite, block: c.else))
    default:
      throw CodegenError.invalidFunction("not a terminator")
    }
  }

  /// Copy edge `args` into the successor's parameter homes. Sources read at
  /// the predecessor's terminator slot; destinations are the params' first
  /// segments (moves land before any split inside the successor).
  mutating func emitEdgeMoves(args: [Value], to target: Block) throws {
    let params = function.params(of: target)
    guard args.count == params.count else {
      throw CodegenError.invalidFunction("arity mismatch on edge to block \(target.raw)")
    }
    try parallelMoves(srcs: args.map { .loc(locOf($0)) }, dsts: params.map { firstLoc($0) })
  }

  mutating func patchRel32(at site: Int, rel: Int32) {
    let u = UInt32(bitPattern: rel)
    code[site] = UInt8(u & 0xFF)
    code[site + 1] = UInt8((u >> 8) & 0xFF)
    code[site + 2] = UInt8((u >> 16) & 0xFF)
    code[site + 3] = UInt8((u >> 24) & 0xFF)
  }

  mutating func patchFixups() throws {
    for f in fixups {
      guard let target = label[f.block] else {
        throw CodegenError.invalidFunction("branch to unemitted block \(f.block.raw)")
      }
      let rel = Int32(target - (f.site + 4))
      patchRel32(at: f.site, rel: rel)
    }
  }
}

/// A parallel-move source: a concrete register/slot location, or a fixed ABI
/// register (call args, entry params) that is not any value's home.
enum MoveSrc {
  case loc(MoveLoc)
  case abi(Reg)
}

enum MoveLoc: Equatable {
  case reg(Reg)
  case slot(Int)
}

/// Read every source into the temp area (in order), then write every temp to
/// its destination. All sources are captured before any destination is
/// written, so swaps and cycles are exact.
extension ISelContext {
  mutating func parallelMoves(srcs: [MoveSrc], dsts: [MoveLoc]) throws {
    guard srcs.count == dsts.count else {
      throw CodegenError.invalidFunction("parallel move arity mismatch")
    }
    guard srcs.count <= tempSlots else {
      throw CodegenError.unsupported("move with more than \(tempSlots) lanes")
    }
    // Fast path: identical endpoints need no code. An ABI source is never any
    // value's home, so ABI->dst always moves.
    var work: [(src: MoveSrc, dst: MoveLoc)] = []
    for (s, d) in zip(srcs, dsts) {
      if case .loc(let l) = s, l == d { continue }
      work.append((s, d))
    }
    if work.isEmpty { return }
    for (idx, w) in work.enumerated() {
      switch w.src {
      case .loc(.reg(let r)):
        code += movToStack(tempOff(idx), r)
      case .loc(.slot(let s)):
        code += movFromStack(scratch1, slotOff(s))
        code += movToStack(tempOff(idx), scratch1)
      case .abi(let reg):
        code += movToStack(tempOff(idx), reg)
      }
    }
    for (idx, w) in work.enumerated() {
      switch w.dst {
      case .reg(let r):
        code += movFromStack(r, tempOff(idx))
      case .slot(let s):
        code += movFromStack(scratch1, tempOff(idx))
        code += movToStack(slotOff(s), scratch1)
      }
    }
  }

  /// A value's covering-segment location at the current position.
  func locOf(_ v: Value) -> MoveLoc {
    guard let loc = scan.loc(v, at: curPos) else {
      fatalError("value \(v) has no segment at \(curPos)")
    }
    switch loc {
    case .reg(let i): return .reg(poolReg(i))
    case .slot(let s): return .slot(s)
    }
  }
}

func describe(_ op: Opcode) -> String {
  switch op {
  case .arith(let a): return "arith.\(a.op)"
  case .arithImm(let a): return "arithImm.\(a.op)"
  case .cmp(let c): return "cmp.\(c.op)"
  case .iconst: return "iconst"
  case .select: return "select"
  case .load: return "load"
  case .store: return "store"
  case .call: return "call"
  case .callIndirect: return "call_indirect"
  case .convert: return "convert"
  case .unary(let u): return "unary.\(u.op)"
  case .globalAddr: return "global_addr"
  default: return "other"
  }
}

#if X86_64
//! End-to-end tests for the x86_64 JIT: build IR, compile, map executable,
//! and call the result on the host. These mirror the Zig backend's execution
//! tests (arithmetic, branches, calls across linked functions).

import Testing
import VoltIR
import VoltTarget

typealias JitFn1 = @convention(c) (Int64) -> Int64
typealias JitFn2 = @convention(c) (Int64, Int64) -> Int64

func buildTestContext() -> (fn: Function, i64: Type, bool: Type) {
  let fn = Function()
  let i64 = fn.types.intern(.int(signedness: true, bits: 64))
  let bool = fn.types.intern(.bool)
  return (fn, i64, bool)
}

@Suite("x86_64 JIT")
struct X86_64JITTests {
  func run(_ mod: JITModule, _ name: String) throws -> JittedModule {
    try jitCompileModule(mod)
  }

  @Test("return a constant")
  func returnConstant() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let b = Builder(function: fn, block: entry)
    b.ret(b.iconst(i64, 42))
    let j = try run(JITModule(functions: [NamedFunction(name: "f", function: fn)]), "f")
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "f")!
    #expect(f(0) == 42)
  }

  @Test("double an argument")
  func doubleArg() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    b.ret(b.arith(.add, x, x))
    let j = try run(JITModule(functions: [NamedFunction(name: "f", function: fn)]), "f")
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "f")!
    #expect(f(21) == 42)
    #expect(f(-3) == -6)
  }

  @Test("integer arithmetic chain")
  func arithChain() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let y = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    // (x + y) * (x - y)
    let s = b.arith(.add, x, y)
    let d = b.arith(.sub, x, y)
    b.ret(b.arith(.mul, s, d))
    let j = try run(JITModule(functions: [NamedFunction(name: "f", function: fn)]), "f")
    let f: JitFn2 = j.entry(as: JitFn2.self, name: "f")!
    #expect(f(5, 3) == 16)
    #expect(f(10, 4) == 84)
  }

  @Test("immediate arithmetic and bitwise ops")
  func immAndBitwise() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    let a = b.arithImm(.add, x, 100)
    let m = b.arithImm(.mul, a, 3)
    let o = b.arith(.bitOr, m, x)
    let v = b.arith(.bitAnd, o, b.iconst(i64, 0xFF))
    b.ret(v)
    let j = try run(JITModule(functions: [NamedFunction(name: "f", function: fn)]), "f")
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "f")!
    #expect(f(1) == ((1 + 100) * 3 | 1) & 0xFF)
    #expect(f(7) == ((7 + 100) * 3 | 7) & 0xFF)
  }

  @Test("division and remainder")
  func divRem() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let y = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    b.ret(b.arith(.div, x, y))
    let j = try run(JITModule(functions: [NamedFunction(name: "div", function: fn)]), "div")
    let f: JitFn2 = j.entry(as: JitFn2.self, name: "div")!
    #expect(f(20, 3) == 6)
    #expect(f(-20, 3) == -6)

    let (fn2, i64b, _) = buildTestContext()
    let e2 = fn2.appendBlock()
    let a2 = fn2.appendBlockParam(e2, i64b)
    let b2 = fn2.appendBlockParam(e2, i64b)
    let bb = Builder(function: fn2, block: e2)
    bb.ret(bb.arith(.rem, a2, b2))
    let j2 = try run(JITModule(functions: [NamedFunction(name: "rem", function: fn2)]), "rem")
    let g: JitFn2 = j2.entry(as: JitFn2.self, name: "rem")!
    #expect(g(20, 3) == 2)
  }

  @Test("conditional branch picks a side")
  func condBranch() throws {
    // max(x, y): if x > y ret x else ret y
    let (fn, i64, bool) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let y = fn.appendBlockParam(entry, i64)
    let thenB = fn.appendBlock()
    let elseB = fn.appendBlock()
    let b = Builder(function: fn, block: entry)
    let c = b.icmp(.gt, x, y, boolTy: bool)
    b.condBr(c, then: thenB, else: elseB)
    Builder(function: fn, block: thenB).ret(x)
    Builder(function: fn, block: elseB).ret(y)
    let j = try run(JITModule(functions: [NamedFunction(name: "max", function: fn)]), "max")
    let f: JitFn2 = j.entry(as: JitFn2.self, name: "max")!
    #expect(f(5, 3) == 5)
    #expect(f(3, 5) == 5)
    #expect(f(-1, -1) == -1)
  }

  @Test("loop with block parameters counts down")
  func loopCountdown() throws {
    // sum(n): i = n; acc = 0; loop: if i == 0 ret acc else { acc += i; i -= 1 }
    let (fn, i64, bool) = buildTestContext()
    let entry = fn.appendBlock()
    let n = fn.appendBlockParam(entry, i64)
    let head = fn.appendBlock()
    let body = fn.appendBlock()
    let done = fn.appendBlock()
    let hi = fn.appendBlockParam(head, i64)
    let ha = fn.appendBlockParam(head, i64)
    let da = fn.appendBlockParam(done, i64)
    let bi = fn.appendBlockParam(body, i64)
    let ba = fn.appendBlockParam(body, i64)
    let b = Builder(function: fn, block: entry)
    let zero = b.iconst(i64, 0)
    b.jump(head, [n, zero])

    let hb = Builder(function: fn, block: head)
    let c = hb.icmp(.eq, hi, hb.iconst(i64, 0), boolTy: bool)
    hb.condBr(c, then: done, [ha], else: body, [hi, ha])

    let bb = Builder(function: fn, block: body)
    let acc2 = bb.arith(.add, ba, bi)
    let i2 = bb.arithImm(.sub, bi, 1)
    bb.jump(head, [i2, acc2])

    Builder(function: fn, block: done).ret(da)

    let j = try run(JITModule(functions: [NamedFunction(name: "sum", function: fn)]), "sum")
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "sum")!
    #expect(f(0) == 0)
    #expect(f(10) == 55)
    #expect(f(100) == 5050)
  }

  @Test("linked call resolves across functions")
  func linkedCall() throws {
    // helper(x) = x * 2; main(x) = helper(x) + 1
    let (helper, hi64, _) = buildTestContext()
    let he = helper.appendBlock()
    let hx = helper.appendBlockParam(he, hi64)
    let hb = Builder(function: helper, block: he)
    hb.ret(hb.arith(.add, hx, hx))

    let (main, mi64, _) = buildTestContext()
    let me = main.appendBlock()
    let mx = main.appendBlockParam(me, mi64)
    let mb = Builder(function: main, block: me)
    // call symbol index 0 -> "helper" via names table
    let callInst = main.appendInst(me, mi64, .call(Call(symbol: 0, args: [mx])))
    let one = mb.iconst(mi64, 1)
    mb.ret(mb.arith(.add, callInst, one))

    let mod = JITModule(functions: [
      NamedFunction(name: "main", function: main),
      NamedFunction(name: "helper", function: helper),
    ])
    let linked = try compileModule(mod, names: ["helper"])
    #expect(linked.addressOf("main") != nil)
    #expect(linked.addressOf("helper") != linked.addressOf("main"))
    let j = try JittedModule(linked: linked)
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "main")!
    #expect(f(20) == 41)
  }

  @Test("select becomes a cmov")
  func selectCmov() throws {
    // abs(x): select x >= 0 ? x : 0 - x
    let (fn, i64, bool) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    let c = b.icmp(.ge, x, b.iconst(i64, 0), boolTy: bool)
    let nx = b.arith(.sub, b.iconst(i64, 0), x)
    b.ret(b.select(c, x, nx))
    let j = try run(JITModule(functions: [NamedFunction(name: "abs", function: fn)]), "abs")
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "abs")!
    #expect(f(5) == 5)
    #expect(f(-5) == 5)
    #expect(f(0) == 0)
  }

  @Test("shifts")
  func shifts() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let k = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    b.ret(b.arith(.shl, x, k))
    let j = try run(JITModule(functions: [NamedFunction(name: "shl", function: fn)]), "shl")
    let f: JitFn2 = j.entry(as: JitFn2.self, name: "shl")!
    #expect(f(1, 10) == 1024)
    #expect(f(3, 4) == 48)
  }

  @Test("encoder known vectors")
  func encoderVectors() throws {
    #expect(movImm(.rax, 42, true) == [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00])
    #expect(movReg(.rax, .rbx) == [0x48, 0x89, 0xD8])
    #expect(addRR(.rax, .rbx, true) == [0x48, 0x01, 0xD8])
    #expect(subRR(.rax, .rbx, true) == [0x48, 0x29, 0xD8])
    #expect(imulRR(.rax, .rbx, true) == [0x48, 0x0F, 0xAF, 0xC3])
    #expect(movImm(.rax, 42, false) == [0xB8, 0x2A, 0x00, 0x00, 0x00])
    #expect(addRR(.rax, .rbx, false) == [0x01, 0xD8])
    #expect(movImm(.r8, 42, false) == [0x41, 0xB8, 0x2A, 0x00, 0x00, 0x00])
    #expect(x86Ret() == [0xC3])
    #expect(pushReg(.rbx) == [0x53])
    #expect(pushReg(.r12) == [0x41, 0x54])
    #expect(popReg(.rbx) == [0x5B])
    #expect(callRel(0) == [0xE8, 0x00, 0x00, 0x00, 0x00])
    #expect(jmp(0) == [0xE9, 0x00, 0x00, 0x00, 0x00])
  }
}

@Suite("linear-scan allocator")
struct AllocatorTests {
  @Test("low pressure keeps everything in registers")
  func noSpillSmall() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    b.ret(b.arith(.add, x, x))
    let scan = try allocateFunction(fn)
    #expect(scan.spillCount == 0)
    #expect(scan.inRegisters == 2)  // x and the add result
  }

  @Test("high pressure spills and still executes exactly")
  func spillUnderPressure() throws {
    // 12 live iconsts summed pairwise: 9 pool regs cannot hold them all.
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let b = Builder(function: fn, block: entry)
    var vs: [Value] = []
    for i in 0..<12 { vs.append(b.iconst(i64, Int64(i + 1))) }
    var acc = vs[0]
    for v in vs[1...] { acc = b.arith(.add, acc, v) }
    b.ret(acc)
    let scan = try allocateFunction(fn)
    #expect(scan.spillCount > 0)
    let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "sum12", function: fn)]))
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "sum12")!
    #expect(f(0) == 78)  // 1+...+12
  }

  @Test("values spanning a call are forced to slots")
  func callSpanningSlots() throws {
    // v defined before the call, used after: must be a slot (ABI clobbers).
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    let v = b.arith(.add, x, x)  // live across the call below
    let callInst = fn.appendInst(entry, i64, .call(Call(symbol: 0, args: [x])))
    b.ret(b.arith(.add, v, callInst))
    let scan = try allocateFunction(fn)
    // Forced-spill values hold slot segments for the whole interval.
    guard
      scan.segments[v]?.allSatisfy({ if case .slot = $0.loc { return true } else { return false } })
        == true
    else {
      Issue.record(
        "call-spanning value not forced to a slot: \(String(describing: scan.segments[v]))")
      return
    }
    // Execution: helper(x)=x+1; main(x)=(x+x)+helper(x).
    let (helper, hi64, _) = buildTestContext()
    let he = helper.appendBlock()
    let hx = helper.appendBlockParam(he, hi64)
    let hb = Builder(function: helper, block: he)
    hb.ret(hb.arithImm(.add, hx, 1))
    let mod = JITModule(functions: [
      NamedFunction(name: "main", function: fn),
      NamedFunction(name: "helper", function: helper),
    ])
    let j = try jitCompileModule(mod, names: ["helper"])
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "main")!
    #expect(f(10) == 31)  // (10+10) + (10+1)
  }

  @Test("values spanning div are forced to slots and execute exactly")
  func divSpanningSlots() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let y = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    let v = b.arith(.add, x, y)  // live across the div below
    let q = b.arith(.div, x, y)
    b.ret(b.arith(.add, v, q))
    let scan = try allocateFunction(fn)
    guard
      scan.segments[v]?.allSatisfy({ if case .slot = $0.loc { return true } else { return false } })
        == true
    else {
      Issue.record(
        "div-spanning value not forced to a slot: \(String(describing: scan.segments[v]))")
      return
    }
    let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "f", function: fn)]))
    let f: JitFn2 = j.entry(as: JitFn2.self, name: "f")!
    #expect(f(20, 3) == 29)  // (20+3) + (20/3)
  }

  @Test("callee-saved regs survive calls")
  func calleeSavedPreserved() throws {
    // acc = 36 live across the call: forced to a slot, reloaded after.
    // main returns 36 + (36+100) = 172.
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let b = Builder(function: fn, block: entry)
    var vs: [Value] = []
    for i in 0..<8 { vs.append(b.iconst(i64, Int64(i + 1))) }
    var acc = vs[0]
    for v in vs[1...] { acc = b.arith(.add, acc, v) }
    let callInst = fn.appendInst(entry, i64, .call(Call(symbol: 0, args: [acc])))
    b.ret(b.arith(.add, acc, callInst))
    let (helper, hi64, _) = buildTestContext()
    let he = helper.appendBlock()
    let hx = helper.appendBlockParam(he, hi64)
    let hb = Builder(function: helper, block: he)
    hb.ret(hb.arithImm(.add, hx, 100))
    let mod = JITModule(functions: [
      NamedFunction(name: "main", function: fn),
      NamedFunction(name: "helper", function: helper),
    ])
    let j = try jitCompileModule(mod, names: ["helper"])
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "main")!
    #expect(f(0) == 172)
  }
}

@Suite("interval splitting")
struct SplitTests {
  @Test("pressure splits the furthest-ending victim with a store")
  func splitFires() throws {
    // 12 live iconsts: the first-defined long-lived values split (reg prefix
    // + spill tail) instead of spilling whole.
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let b = Builder(function: fn, block: entry)
    var vs: [Value] = []
    for i in 0..<12 { vs.append(b.iconst(i64, Int64(i + 1))) }
    var acc = vs[0]
    for v in vs[1...] { acc = b.arith(.add, acc, v) }
    b.ret(acc)
    let scan = try allocateFunction(fn)
    #expect(scan.spillCount > 0)
    // At least one value has a reg-prefix + slot-tail pair.
    func isReg(_ l: Loc) -> Bool { if case .reg = l { return true } else { return false } }
    func isSlot(_ l: Loc) -> Bool { if case .slot = l { return true } else { return false } }
    let split = scan.segments.values.contains {
      $0.count == 2 && isReg($0[0].loc) && isSlot($0[1].loc)
    }
    #expect(split, "expected a split value, segments: \(scan.segments)")
    // Every store names a real split: its value's tail slot at its position.
    for st in scan.stores {
      let tail = scan.segments[st.value]?.last
      #expect(tail?.loc == .slot(st.slot))
      #expect(tail.map { st.at >= $0.from } == true)
    }
    #expect(!scan.stores.isEmpty)
    // Execution exact with splits live.
    let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "sum12", function: fn)]))
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "sum12")!
    #expect(f(0) == 78)  // 1+...+12
  }

  @Test("split prefix reads the register, tail reloads from the slot")
  func splitPrefixAndTail() throws {
    // 9 live 2x values across a sum chain; each use after its split must see
    // the stored value, each use before it the register. f(x) = 9*2x + 45.
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    var lane: [Value] = []
    for _ in 0..<9 { lane.append(b.arith(.add, x, x)) }  // 9 live 2x values
    var acc = b.iconst(i64, 0)
    for v in lane { acc = b.arith(.add, acc, v) }
    b.ret(b.arithImm(.add, acc, 45))
    let scan = try allocateFunction(fn)
    let splits = scan.segments.values.filter { $0.count > 1 }.count
    #expect(splits > 0, "expected splits under 10-live pressure")
    let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "f", function: fn)]))
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "f")!
    #expect(f(1) == 63)  // 9*2 + 45
    #expect(f(10) == 225)  // 9*20 + 45
  }

  @Test("no splits without pressure, no stores recorded")
  func noSplitSmall() throws {
    let (fn, i64, _) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let y = fn.appendBlockParam(entry, i64)
    let b = Builder(function: fn, block: entry)
    b.ret(b.arith(.add, x, y))
    let scan = try allocateFunction(fn)
    #expect(scan.stores.isEmpty)
    #expect(scan.segments.values.allSatisfy { $0.count == 1 })
  }
}

@Suite("split-store dominance")
struct SplitDominanceTests {
  @Test("split store inside a skipped body still reaches the use")
  func skippedBodySplitStore() throws {
    // Layout order: entry, body, skip, done. The body runs 9-deep pressure
    // that splits x's interval; the skip arm jumps straight to done with x.
    // x's store fires in the body, but the skip path must still read x's
    // register value at done. If the scan puts done's read on a slot segment
    // whose store only fires on the body path, f(0) miscompiles.
    let (fn, i64, bool) = buildTestContext()
    let entry = fn.appendBlock()
    let x = fn.appendBlockParam(entry, i64)
    let body = fn.appendBlock()
    let skip = fn.appendBlock()
    let done = fn.appendBlock()
    let d = fn.appendBlockParam(done, i64)
    let b = Builder(function: fn, block: entry)
    let c = b.icmp(.eq, x, b.iconst(i64, 0), boolTy: bool)
    b.condBr(c, then: skip, [], else: body, [])
    let bb = Builder(function: fn, block: body)
    var lane: [Value] = []
    for _ in 0..<9 { lane.append(bb.arith(.add, x, x)) }
    var acc = bb.iconst(i64, 0)
    for v in lane { acc = bb.arith(.add, acc, v) }
    bb.jump(done, [bb.arith(.add, acc, x)])
    Builder(function: fn, block: skip).jump(done, [x])
    Builder(function: fn, block: done).ret(d)
    let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "f", function: fn)]))
    let f: JitFn1 = j.entry(as: JitFn1.self, name: "f")!
    #expect(f(0) == 0)  // skipped path: x untouched
    #expect(f(1) == 19)  // 9*2 + 1
  }
}

@Test("pointer addition addresses a nonzero field offset")
func pointerOffsetLoadStore() throws {
  let fn = Function()
  let i64 = fn.types.intern(.int(signedness: true, bits: 64))
  let ptr = fn.types.ptrGlobal()
  let entry = fn.appendBlock()
  let base = fn.appendBlockParam(entry, ptr)
  let b = Builder(function: fn, block: entry)
  let field = b.ptrAdd(base, byteOffset: 8)
  b.store(b.iconst(i64, 42), field)
  b.ret(b.load(i64, field))
  let j = try jitCompileModule(JITModule(functions: [NamedFunction(name: "field", function: fn)]))
  typealias FieldFn = @convention(c) (UnsafeMutablePointer<Int64>) -> Int64
  let f: FieldFn = j.entry(as: FieldFn.self, name: "field")!
  var data: [Int64] = [0, 0]
  let result = data.withUnsafeMutableBufferPointer { f($0.baseAddress!) }
  #expect(result == 42)
  #expect(data[1] == 42)
}

#endif

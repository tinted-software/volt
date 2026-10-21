import Testing
import VoltIR

@Suite("IR verifier")
struct IRVerifierTests {
  func makeTypes(_ function: Function) -> (i64: Type, bool: Type, ptr: Type) {
    (
      function.types.intern(.int(signedness: true, bits: 64)),
      function.types.intern(.bool),
      function.types.ptrGlobal()
    )
  }

  @Test("accepts a well-formed function with block parameters")
  func acceptsValidControlFlow() throws {
    let function = Function()
    let types = makeTypes(function)
    let entry = function.appendBlock()
    let body = function.appendBlock()
    let done = function.appendBlock()

    let builder = Builder(function: function, block: entry)
    let one = builder.iconst(types.i64, 1)
    builder.jump(body, [one])

    let bodyBuilder = Builder(function: function, block: body)
    let value = function.appendBlockParam(body, types.i64)
    let next = bodyBuilder.arith(.add, value, one)
    bodyBuilder.jump(done, [next])

    let doneValue = function.appendBlockParam(done, types.i64)
    Builder(function: function, block: done).ret(doneValue)
    try function.verify()
  }

  @Test("rejects blocks without terminators")
  func rejectsMissingTerminator() {
    let function = Function()
    _ = makeTypes(function)
    _ = function.appendBlock()
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects invalid branch targets")
  func rejectsInvalidBranchTarget() {
    let function = Function()
    _ = makeTypes(function)
    let block = function.appendBlock()
    Builder(function: function, block: block).jump(Block(raw: 7))
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects block argument arity mismatches")
  func rejectsBlockArgumentArityMismatch() {
    let function = Function()
    let types = makeTypes(function)
    let entry = function.appendBlock()
    let target = function.appendBlock()
    _ = function.appendBlockParam(target, types.i64)
    Builder(function: function, block: entry).jump(target)
    Builder(function: function, block: target).ret()
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects stores through non-pointer values")
  func rejectsInvalidStorePointer() {
    let function = Function()
    let types = makeTypes(function)
    let block = function.appendBlock()
    let builder = Builder(function: function, block: block)
    let value = builder.iconst(types.i64, 1)
    builder.store(value, value)
    builder.ret()
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects invalid value references without trapping")
  func rejectsInvalidValue() {
    let function = Function()
    let types = makeTypes(function)
    let block = function.appendBlock()
    let builder = Builder(function: function, block: block)
    let lhs = builder.iconst(types.i64, 1)
    let rhs = builder.iconst(types.i64, 2)
    let valid = builder.arith(.add, lhs, rhs)
    var opcode = function.opcode(function.instForValue(valid)!)
    if case .arith(var arithmetic) = opcode {
      arithmetic.lhs = Value(raw: 99)
      arithmetic.rhs = Value(raw: 99)
      opcode = .arith(arithmetic)
    }
    let malformed = function.appendInst(block, types.i64, opcode)
    builder.ret(malformed)
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects uses before their definitions")
  func rejectsUseBeforeDefinition() {
    let function = Function()
    let types = makeTypes(function)
    let block = function.appendBlock()
    let builder = Builder(function: function, block: block)
    let lhs = builder.iconst(types.i64, 1)
    let rhs = builder.iconst(types.i64, 2)
    let valid = builder.arith(.add, lhs, rhs)
    var opcode = function.opcode(function.instForValue(valid)!)
    if case .arith(var arithmetic) = opcode {
      let forward = Value(raw: UInt32(function.valueTypes.count))
      arithmetic.lhs = forward
      arithmetic.rhs = lhs
      opcode = .arith(arithmetic)
    }
    let malformed = function.appendInst(block, types.i64, opcode)
    builder.ret(malformed)
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects definitions that do not dominate a use")
  func rejectsNonDominatingDefinition() {
    let function = Function()
    let types = makeTypes(function)
    let entry = function.appendBlock()
    let left = function.appendBlock()
    let right = function.appendBlock()
    let builder = Builder(function: function, block: entry)
    let condition = builder.iconst(types.bool, 1)
    builder.condBr(condition, then: left, else: right)

    let leftValue = Builder(function: function, block: left).iconst(types.i64, 4)
    Builder(function: function, block: left).ret(leftValue)
    let rightBuilder = Builder(function: function, block: right)
    let result = rightBuilder.arith(.add, leftValue, leftValue)
    rightBuilder.ret(result)
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects type-invalid arithmetic")
  func rejectsMismatchedResultType() {
    let function = Function()
    let types = makeTypes(function)
    let block = function.appendBlock()
    let builder = Builder(function: function, block: block)
    let lhs = builder.iconst(types.i64, 2)
    let rhs = builder.iconst(types.i64, 3)
    let valid = builder.arith(.add, lhs, rhs)
    let opcode = function.opcode(function.instForValue(valid)!)
    let result = function.appendInst(block, types.bool, opcode)
    builder.ret(result)
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }

  @Test("rejects a type from a different function type table")
  func rejectsForeignType() {
    let foreign = Function()
    let foreignType = foreign.types.intern(.bool)
    let function = Function()
    let block = function.appendBlock()
    let value = function.appendInst(block, foreignType, .iconst(1))
    Builder(function: function, block: block).ret(value)
    #expect(throws: IRVerificationError.self) { try function.verify() }
  }
}

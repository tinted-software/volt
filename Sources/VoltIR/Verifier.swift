//! Structural, control-flow, SSA, and type verification for Volt IR.

/// A malformed IR function and a human-readable reason for rejecting it.
public struct IRVerificationError: Error, Sendable, CustomStringConvertible {
  public let message: String

  public init(_ message: String) { self.message = message }
  public var description: String { "invalid IR: \(message)" }
}

private struct ValueDefinition {
  let block: Int
  let instructionPosition: Int?
}

extension Function {
  /// Verify storage consistency, CFG edges, value definitions, SSA dominance,
  /// and the type contracts expressible without a module symbol table.
  ///
  /// Direct-call signatures cannot be checked here because `Function` does not
  /// carry a symbol table. Backends should run this before lowering.
  public func verify() throws {
    func require(_ condition: Bool, _ message: @autoclosure () -> String) throws {
      guard condition else { throw IRVerificationError(message()) }
    }

    let blockCount = blocks.count
    let instCount = instOpcodes.count
    let valueCount = valueTypes.count

    try require(blockCount > 0, "function has no blocks")
    try require(blockParams.count == blockCount, "block parameter table count mismatch")
    try require(blockParamTypes.count == blockCount, "block parameter type table count mismatch")
    try require(blockInsts.count == blockCount, "block instruction table count mismatch")
    try require(terminators.count == blockCount, "terminator table count mismatch")
    try require(instResults.count == instCount, "instruction result table count mismatch")
    try require(instResultTypes.count == instResults.compactMap { $0 }.count,
      "instruction result type table count mismatch")
    try require(valueDefs.count == valueCount, "value definition table count mismatch")

    for (index, block) in blocks.enumerated() {
      try require(block.raw == UInt32(index), "block IDs are not dense at index \(index)")
      try require(blockParams[index].count == blockParamTypes[index].count,
        "block \(index) parameter/type count mismatch")
    }

    func validType(_ type: Type) -> Bool {
      guard types.contains(type) else { return false }
      switch type.kind {
      case .vector(let vector): return vector.len > 0 && validType(vector.elem)
      case .indirectStruct(let fields): return fields.allSatisfy(validType)
      default: return true
      }
    }
    for (index, type) in valueTypes.enumerated() {
      try require(validType(type), "value \(index) has a type not interned in this function")
    }
    for (index, type) in instResultTypes.enumerated() {
      try require(validType(type), "instruction \(index) has a result type not interned in this function")
    }
    for block in 0..<blockCount {
      for (paramIndex, type) in blockParamTypes[block].enumerated() {
        try require(validType(type), "block \(block) parameter \(paramIndex) has an invalid type")
      }
    }

    var instructionOwner = Array<Int?>(repeating: nil, count: instCount)
    var instructionPosition = Array<Int?>(repeating: nil, count: instCount)
    for block in 0..<blockCount {
      for (position, inst) in blockInsts[block].enumerated() {
        let index = Int(inst.raw)
        try require(index < instCount, "block \(block) references invalid instruction \(inst.raw)")
        try require(instructionOwner[index] == nil,
          "instruction \(index) appears in multiple block instruction lists")
        instructionOwner[index] = block
        instructionPosition[index] = position
      }
    }
    for index in 0..<instCount {
      try require(instructionOwner[index] != nil, "instruction \(index) is not owned by a block")
    }

    var definitions = Array<ValueDefinition?>(repeating: nil, count: valueCount)
    for block in 0..<blockCount {
      for (paramIndex, value) in blockParams[block].enumerated() {
        let index = Int(value.raw)
        try require(index < valueCount, "block \(block) references invalid parameter value \(value.raw)")
        try require(definitions[index] == nil, "value \(index) has multiple definitions")
        try require(valueDefs[index] == nil, "block parameter \(index) has an instruction definition")
        try require(valueTypes[index] == blockParamTypes[block][paramIndex],
          "block \(block) parameter \(index) type table mismatch")
        definitions[index] = ValueDefinition(block: block, instructionPosition: nil)
      }
    }
    var resultTypeIndex = 0
    for index in 0..<instCount {
      let owner = instructionOwner[index]!
      let position = instructionPosition[index]!
      let opcode = instOpcodes[index]
      let result = instResults[index]
      if case .store = opcode {
        try require(result == nil, "store instruction \(index) unexpectedly has a result")
      } else {
        try require(result != nil, "instruction \(index) is missing its result value")
      }
      if let result {
        let valueIndex = Int(result.raw)
        try require(valueIndex < valueCount, "instruction \(index) references invalid result value \(result.raw)")
        try require(definitions[valueIndex] == nil, "value \(valueIndex) has multiple definitions")
        try require(valueDefs[valueIndex] == Inst(raw: UInt32(index)),
          "instruction \(index) result definition table mismatch")
        try require(valueTypes[valueIndex] == instResultTypes[resultTypeIndex],
          "instruction \(index) result type table mismatch")
        resultTypeIndex += 1
        definitions[valueIndex] = ValueDefinition(block: owner, instructionPosition: position)
      }
      switch opcode {
      case .ret, .jump, .condBr:
        throw IRVerificationError("terminator stored in instruction list at instruction \(index)")
      default: break
      }
    }
    for index in 0..<valueCount {
      try require(definitions[index] != nil, "value \(index) has no definition")
      if let definingInst = valueDefs[index] {
        let instIndex = Int(definingInst.raw)
        try require(instIndex < instCount, "value \(index) references invalid defining instruction \(definingInst.raw)")
        try require(instResults[instIndex] == Value(raw: UInt32(index)),
          "value \(index) and instruction result tables disagree")
      }
    }

    var successors = Array(repeating: [Int](), count: blockCount)
    var predecessors = Array(repeating: [Int](), count: blockCount)
    for block in 0..<blockCount {
      guard let terminator = terminators[block] else {
        throw IRVerificationError("block \(block) has no terminator")
      }
      func addEdge(_ target: Block, _ args: [Value]) throws {
        let targetIndex = Int(target.raw)
        try require(targetIndex < blockCount, "block \(block) branches to invalid block \(target.raw)")
        let params = blockParams[targetIndex]
        try require(args.count == params.count,
          "edge \(block) -> \(targetIndex) passes \(args.count) arguments for \(params.count) parameters")
        for (argument, parameter) in zip(args, params) {
          try require(valueIndexIsValid(argument), "edge \(block) -> \(targetIndex) uses invalid value \(argument.raw)")
          try require(valueTypes[Int(argument.raw)] == valueTypes[Int(parameter.raw)],
            "edge \(block) -> \(targetIndex) argument type mismatch")
        }
        successors[block].append(targetIndex)
        predecessors[targetIndex].append(block)
      }
      switch terminator {
      case .ret(let ret):
        try require(ret.values.count <= 4, "block \(block) returns more than four values")
        for value in ret.values {
          try require(valueIndexIsValid(value), "block \(block) returns invalid value \(value.raw)")
        }
      case .jump(let jump):
        try addEdge(jump.target, jump.args)
      case .condBr(let branch):
        try require(valueIndexIsValid(branch.cond), "block \(block) branches on invalid value \(branch.cond.raw)")
        try require(isBool(valueTypes[Int(branch.cond.raw)]), "block \(block) conditional branch condition is not bool")
        try addEdge(branch.then, branch.thenArgs)
        try addEdge(branch.else, branch.elseArgs)
      default:
        throw IRVerificationError("block \(block) has a non-terminator opcode")
      }
    }

    // Compute dominators for reachable blocks. Unreachable blocks are treated
    // as independent roots, so a definition there cannot accidentally appear
    // to dominate a use in another disconnected region.
    var reachable = Set<Int>()
    var worklist = [0]
    while let block = worklist.popLast() {
      if reachable.insert(block).inserted { worklist.append(contentsOf: successors[block]) }
    }
    var dominators = Array(repeating: Set<Int>(), count: blockCount)
    for block in 0..<blockCount {
      dominators[block] = block == 0 || !reachable.contains(block) ? [block] : reachable
    }
    var changed = true
    while changed {
      changed = false
      for block in 1..<blockCount where reachable.contains(block) {
        let preds = predecessors[block].filter { reachable.contains($0) }
        var next = preds.first.map { dominators[$0] } ?? Set<Int>()
        for predecessor in preds.dropFirst() { next.formIntersection(dominators[predecessor]) }
        next.insert(block)
        if next != dominators[block] { dominators[block] = next; changed = true }
      }
    }

    func checkUse(_ value: Value, from block: Int, before position: Int, context: String) throws {
      try require(valueIndexIsValid(value), "\(context) references invalid value \(value.raw)")
      let definition = definitions[Int(value.raw)]!
      if definition.block == block {
        if let definingPosition = definition.instructionPosition {
          try require(definingPosition < position,
            "\(context) uses value \(value.raw) before its definition")
        }
      } else {
        try require(dominators[block].contains(definition.block),
          "\(context) uses value \(value.raw) that does not dominate this block")
      }
    }

    for block in 0..<blockCount {
      for (position, inst) in blockInsts[block].enumerated() {
        let index = Int(inst.raw)
        let opcode = instOpcodes[index]
        for value in operands(of: opcode) {
          try checkUse(value, from: block, before: position, context: "instruction \(index)")
        }
        switch opcode {
        case .alloca(let op):
          try require(validType(op.elem), "instruction \(index) has an alloca element type not interned in this function")
        case .vaArg(let op):
          try require(validType(op.ty), "instruction \(index) has a va_arg type not interned in this function")
        default: break
        }
        try verifyTypes(
          opcode, result: instResults[index].map { valueTypes[Int($0.raw)] },
          typeOf: { valueTypes[Int($0.raw)] }, context: "instruction \(index)")
      }
      let termPosition = blockInsts[block].count
      for value in operands(of: terminators[block]!) {
        try checkUse(value, from: block, before: termPosition, context: "terminator in block \(block)")
      }
    }

    func valueIndexIsValid(_ value: Value) -> Bool { Int(value.raw) < valueCount }
    func isBool(_ type: Type) -> Bool { if case .bool = type.kind { return true }; return false }
  }
}

private func operands(of opcode: Opcode) -> [Value] {
  switch opcode {
  case .ptrAdd(let op): return [op.base]
  case .arith(let op): return [op.lhs, op.rhs]
  case .arithImm(let op): return [op.lhs]
  case .cmp(let op): return [op.lhs, op.rhs]
  case .select(let op): return [op.cond, op.then, op.else]
  case .structNew(let op): return op.fields
  case .extract(let op): return [op.aggregate]
  case .convert(let op): return [op.value]
  case .unary(let op): return [op.value]
  case .load(let op): return [op.ptr]
  case .store(let op): return [op.value, op.ptr]
  case .prefetch(let op): return [op.ptr]
  case .call(let op): return op.args + (op.retDest.map { [$0] } ?? [])
  case .callIndirect(let op): return [op.target] + op.args + (op.retDest.map { [$0] } ?? [])
  case .dot(let op): return [op.acc, op.a, op.b]
  case .reduce(let op): return [op.vector]
  case .splat(let op): return [op.scalar]
  case .vaStart(let op): return [op.list]
  case .vaEnd(let op): return [op.list]
  case .vaArg(let op): return [op.list]
  case .ret(let op): return op.values
  case .jump(let op): return op.args
  case .condBr(let op): return [op.cond] + op.thenArgs + op.elseArgs
  case .compare, .alloca, .globalAddr, .iconst: return []
  }
}

private func verifyTypes(
  _ opcode: Opcode, result: Type?, typeOf: (Value) -> Type, context: String
) throws {
  func fail(_ reason: String) -> IRVerificationError { IRVerificationError("\(context): \(reason)") }
  guard let result else {
    guard case .store(let op) = opcode else { throw fail("missing result") }
    guard isPointer(typeOf(op.ptr)) else { throw fail("store address must be a pointer") }
    return
  }
  func same(_ lhs: Type, _ rhs: Type) -> Bool { lhs == rhs }
  func isInteger(_ type: Type) -> Bool { if case .int = type.kind { return true }; return false }
  func isFloat(_ type: Type) -> Bool { if case .float = type.kind { return true }; return false }
  func isPointer(_ type: Type) -> Bool { if case .ptr = type.kind { return true }; return false }
  func isBool(_ type: Type) -> Bool { if case .bool = type.kind { return true }; return false }
  func arithmeticType(_ type: Type, op: BinOp) -> Bool {
    let integerOp = op == .add || op == .sub || op == .mul || op == .div || op == .rem
      || op == .bitAnd || op == .bitOr || op == .bitXor || op == .shl || op == .shr || op == .mulH
    let floatingOp = op == .add || op == .sub || op == .mul || op == .div || op == .rem
    if isInteger(type) { return integerOp }
    if isFloat(type) { return floatingOp }
    if case .vector(let vector) = type.kind {
      return (isInteger(vector.elem) && integerOp) || (isFloat(vector.elem) && floatingOp)
    }
    return false
  }
  switch opcode {
  case .ptrAdd(let op):
    guard same(result, typeOf(op.base)), isPointer(result) else { throw fail("ptrAdd requires matching pointer operand and result types") }
  case .arith(let op):
    guard same(typeOf(op.lhs), typeOf(op.rhs)), same(result, typeOf(op.lhs)), arithmeticType(result, op: op.op) else {
      throw fail("arithmetic operands and result must have matching numeric types")
    }
  case .arithImm(let op):
    guard same(result, typeOf(op.lhs)), isInteger(result) else { throw fail("immediate arithmetic requires a matching integer result") }
  case .compare:
    guard isBool(result) else { throw fail("comparison result must be bool") }
  case .cmp(let op):
    let operandType = typeOf(op.lhs)
    let comparable: Bool
    if isInteger(operandType) || isFloat(operandType) {
      comparable = true
    } else if isBool(operandType) {
      comparable = op.op == .eq || op.op == .ne
    } else if isPointer(operandType) {
      comparable = op.op == .eq || op.op == .ne
    } else {
      comparable = false
    }
    guard same(operandType, typeOf(op.rhs)), comparable, isBool(result) else {
      throw fail("comparison requires comparable matching operands and bool result")
    }
  case .select(let op):
    guard isBool(typeOf(op.cond)), same(result, typeOf(op.then)), same(result, typeOf(op.else)) else {
      throw fail("select requires a bool condition and matching arm/result types")
    }
  case .structNew(let op):
    guard case .indirectStruct(let fields) = result.kind, fields.count == op.fields.count,
      zip(fields, op.fields).allSatisfy({ $0.0 == typeOf($0.1) })
    else { throw fail("struct construction fields do not match result type") }
  case .extract(let op):
    let aggregate = typeOf(op.aggregate)
    switch aggregate.kind {
    case .indirectStruct(let fields):
      guard Int(op.index) < fields.count, result == fields[Int(op.index)] else { throw fail("invalid struct field extraction") }
    case .vector(let vector):
      guard op.index < vector.len, result == vector.elem else { throw fail("invalid vector lane extraction") }
    default: throw fail("extract requires a struct or vector operand")
    }
  case .convert:
    break
  case .unary(let op):
    if op.op != .reinterpret {
      guard isFloat(result), same(result, typeOf(op.value)) else { throw fail("floating unary operation requires matching float types") }
    }
  case .alloca:
    guard isPointer(result) else { throw fail("alloca result must be a pointer") }
  case .load(let op):
    guard isPointer(typeOf(op.ptr)) else { throw fail("load address must be a pointer") }
  case .store(let op):
    guard isPointer(typeOf(op.ptr)) else { throw fail("store address must be a pointer") }
  case .prefetch(let op):
    guard isPointer(typeOf(op.ptr)) else { throw fail("prefetch address must be a pointer") }
  case .call(let op):
    guard Int(op.numFixed) <= op.args.count, op.retPieces.count <= 4, op.retRegs <= 4 else {
      throw fail("invalid call ABI metadata")
    }
  case .callIndirect(let op):
    guard isPointer(typeOf(op.target)), Int(op.numFixed) <= op.args.count, op.retPieces.count <= 4, op.retRegs <= 4 else {
      throw fail("invalid indirect call target or ABI metadata")
    }
  case .globalAddr:
    guard isPointer(result) else { throw fail("global address result must be a pointer") }
  case .dot(let op):
    guard same(result, typeOf(op.acc)) else { throw fail("dot result type must match accumulator") }
  case .reduce(let op):
    guard case .vector(let vector) = typeOf(op.vector).kind, result == vector.elem else {
      throw fail("reduce result must match vector element type")
    }
  case .splat(let op):
    guard case .vector(let vector) = result.kind, vector.elem == typeOf(op.scalar) else {
      throw fail("splat result must be a vector of the scalar type")
    }
  case .vaStart(let op):
    guard isPointer(typeOf(op.list)) else { throw fail("variadic list must be a pointer") }
  case .vaEnd(let op):
    guard isPointer(typeOf(op.list)) else { throw fail("variadic list must be a pointer") }
  case .vaArg(let op):
    guard isPointer(typeOf(op.list)), result == op.ty else { throw fail("va_arg list or result type is invalid") }
  case .iconst(let value):
    if isInteger(result) { break }
    guard isBool(result), value == 0 || value == 1 else {
      throw fail("constant must have an integer type, or be 0/1 for bool")
    }
  case .ret, .jump, .condBr:
    throw fail("terminator stored as an instruction")
  }
}


//! Constructs IR through a cursor: wraps a `Function` and tracks a current
//! insertion block, so callers append instructions without threading the block
//! through every call. Result types derive from operands where possible.
//! Mirrors `libs/vulcan-ir/builder.zig`.

public final class Builder {
  public let function: Function
  public var block: Block

  public init(function: Function, block: Block) {
    self.function = function
    self.block = block
  }

  /// Redirect subsequent instructions to a different block.
  public func switchTo(_ block: Block) { self.block = block }

  /// An integer constant of type `ty`.
  @discardableResult
  public func iconst(_ ty: Type, _ value: Int64) -> Value {
    function.appendInst(block, ty, .iconst(value))
  }

  /// A binary arithmetic/bitwise op. The result type is taken from `lhs`.
  @discardableResult
  public func arith(_ op: BinOp, _ lhs: Value, _ rhs: Value) -> Value {
    function.appendInst(
      block, function.valueType(of: lhs), .arith(Arith(op: op, lhs: lhs, rhs: rhs)))
  }

  /// Integer addition (shorthand).
  @discardableResult
  public func iadd(_ lhs: Value, _ rhs: Value) -> Value { arith(.add, lhs, rhs) }

  /// Arithmetic against a constant: `lhs <op> imm`.
  @discardableResult
  public func arithImm(_ op: BinOp, _ lhs: Value, _ imm: Int64) -> Value {
    function.appendInst(
      block, function.valueType(of: lhs), .arithImm(ArithImm(op: op, lhs: lhs, imm: imm)))
  }

  /// Integer comparison. Produces `boolTy`; signedness comes from the operands.
  @discardableResult
  public func icmp(_ op: CmpOp, _ lhs: Value, _ rhs: Value, boolTy: Type) -> Value {
    function.appendInst(block, boolTy, .cmp(Compare(op: op, lhs: lhs, rhs: rhs)))
  }

  /// Value-producing conditional. The result type is taken from `thenValue`.
  @discardableResult
  public func select(_ cond: Value, _ thenValue: Value, _ elseValue: Value) -> Value {
    function.appendInst(
      block, function.valueType(of: thenValue),
      .select(Select(cond: cond, then: thenValue, else: elseValue)))
  }

  /// Add a signed byte offset to a pointer, preserving its pointer type.
  @discardableResult
  public func ptrAdd(_ base: Value, byteOffset: Int64) -> Value {
    let type = function.valueType(of: base)
    return function.appendInst(block, type, .ptrAdd(PtrAdd(base: base, byteOffset: byteOffset)))
  }

  /// Load a value of type `ty` from `ptr`.
  @discardableResult
  public func load(_ ty: Type, _ ptr: Value) -> Value {
    function.appendInst(block, ty, .load(Load(ptr: ptr, isVolatile: false)))
  }

  /// Store `value` to `ptr`.
  public func store(_ value: Value, _ ptr: Value) {
    function.appendStore(block, value: value, ptr: ptr)
  }

  /// Return from the function, optionally with a value.
  public func ret(_ value: Value? = nil) {
    function.setTerminator(block, .ret(Ret(values: value.map { [$0] } ?? [])))
  }

  /// Jump unconditionally to `target`, passing `args` to its parameters.
  public func jump(_ target: Block, _ args: [Value] = []) {
    function.setTerminator(block, .jump(Jump(target: target, args: args)))
  }

  /// Append a conditional branch to `thenBlock` / `elseBlock`.
  public func condBr(
    _ cond: Value, then thenBlock: Block, _ thenArgs: [Value] = [],
    else elseBlock: Block, _ elseArgs: [Value] = []
  ) {
    function.setTerminator(
      block,
      .condBr(
        CondBr(cond: cond, then: thenBlock, thenArgs: thenArgs, else: elseBlock, elseArgs: elseArgs)
      ))
  }
}

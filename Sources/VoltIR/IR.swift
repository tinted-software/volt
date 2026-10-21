//! The Volt IR: interned types, SSA values, basic blocks, and instructions.
//! Modeled on Volt's existing Zig IR (`libs/vulcan-ir`), rewritten in Swift.
//!
//! SSA values cross block boundaries as block parameters (not phi nodes). The
//! control-flow graph is derived from terminators, never stored separately.

public struct Type: Sendable, Hashable, CustomStringConvertible {
  public indirect enum Kind: Sendable, Hashable {
    case bool
    case int(signedness: Bool, bits: UInt16)  // false = unsigned
    case float(FloatKind)
    case vector(Vector)
    case indirectStruct([Type])  // boxed to keep the enum fixed-size
    case ptr(AddressSpace)
  }
  public enum FloatKind: Sendable, Hashable { case f16, f32, f64, f128 }
  public struct Vector: Sendable, Hashable {
    public var len: UInt32
    public var elem: Type
  }
  public enum AddressSpace: Sendable, Hashable { case global, shared, `private`, constant }

  /// Stable index into the owning `TypeTable`'s `kinds` array.
  public let index: Int
  public let kind: Kind

  init(_ kind: Kind, index: Int) {
    self.kind = kind
    self.index = index
  }

  public var description: String {
    switch kind {
    case .bool: return "bool"
    case .int(let s, let b): return "\(s ? "i" : "u")\(b)"
    case .float(let f):
      switch f {
      case .f16: return "f16"
      case .f32: return "f32"
      case .f64: return "f64"
      case .f128: return "f128"
      }
    case .indirectStruct(let fields):
      return "{\(fields.map { $0.description }.joined(separator: ","))}"
    case .ptr: return "ptr"
    case .vector(let v): return "<\(v.len) x \(v.elem)>"
    }
  }
}

public struct TypeTable: Sendable {
  public init() {
    self.kinds = []
    self.dedup = [:]
  }
  fileprivate var kinds: [Type.Kind]
  fileprivate var dedup: [Type.Kind: Type]

  public mutating func intern(_ kind: Type.Kind) -> Type {
    if let existing = dedup[kind] { return existing }
    let idx = kinds.count
    let t = Type(kind, index: idx)
    kinds.append(kind)
    dedup[kind] = t
    return t
  }

  public mutating func ptrGlobal() -> Type { intern(.ptr(.global)) }

  public func kind(of type: Type) -> Type.Kind { kinds[type.index] }
  public var count: Int { kinds.count }
}

/// A handle to a basic block within a function.
public struct Block: Sendable, Hashable, Comparable {
  public let raw: UInt32
  public init(raw: UInt32) { self.raw = raw }
  public static func < (lhs: Block, rhs: Block) -> Bool { lhs.raw < rhs.raw }
}

/// A handle to an SSA value: a block parameter or an instruction result.
public struct Value: Sendable, Hashable, Comparable {
  public let raw: UInt32
  public init(raw: UInt32) { self.raw = raw }
  public static func < (lhs: Value, rhs: Value) -> Bool { lhs.raw < rhs.raw }
}

/// A handle to an instruction within a function.
public struct Inst: Sendable, Hashable {
  public let raw: UInt32
  public init(raw: UInt32) { self.raw = raw }
}

// Binary arithmetic/bitwise relations.
public enum BinOp: Sendable, Hashable, CaseIterable {
  case add, sub, mul, div, rem, bitAnd, bitOr, bitXor, shl, shr, mulH
}

// Integer comparison relations.
public enum CmpOp: Sendable, Hashable, CaseIterable {
  case eq, ne, lt, le, gt, ge
}

// Unary floating-point / reinterpret ops.
public enum UnaryOp: Sendable, Hashable, CaseIterable {
  case reinterpret, sqrt, ceil, floor, trunc, nearest
}

// A binary arithmetic instruction.
public struct Arith: Sendable, Hashable {
  public var op: BinOp
  public var lhs: Value
  public var rhs: Value
}
public struct ArithImm: Sendable, Hashable {
  public var op: BinOp
  public var lhs: Value
  public var imm: Int64
}
public struct Compare: Sendable, Hashable {
  public var op: CmpOp
  public var lhs: Value
  public var rhs: Value
}
public struct Select: Sendable, Hashable {
  public var cond: Value
  public var then: Value
  public var `else`: Value
}
public struct StructNew: Sendable, Hashable { public var fields: [Value] }
public struct Extract: Sendable, Hashable {
  public var aggregate: Value
  public var index: UInt32
}
public struct Convert: Sendable, Hashable { public var value: Value }
public struct Unary: Sendable, Hashable {
  public var op: UnaryOp
  public var value: Value
}
public struct Alloca: Sendable, Hashable { public var elem: Type }
public struct Load: Sendable, Hashable {
  public var ptr: Value
  public var isVolatile: Bool
}
public struct Store: Sendable, Hashable {
  public var value: Value
  public var ptr: Value
  public var isVolatile: Bool
}
public struct Prefetch: Sendable, Hashable { public var ptr: Value }

// Call a function named by symbol index.
public struct Call: Sendable, Hashable {
  public var symbol: UInt32
  public var args: [Value]
  public var isVariadic: Bool = false
  public var numFixed: UInt32 = 0
  public var retDest: Value? = nil
  public var retRegs: UInt8 = 0
  public var retPieces: [RetPiece] = Array(repeating: .init(), count: 4)
  public var sret: Bool = false
  public init(symbol: UInt32, args: [Value]) {
    self.symbol = symbol
    self.args = args
  }
}
public struct RetPiece: Sendable, Hashable {
  public var fp: Bool = false
  public var offset: UInt8 = 0
  public var bytes: UInt8 = 8
}

/// Call the function whose address is `target` (a `ptr`).
public struct CallIndirect: Sendable, Hashable {
  public var target: Value
  public var args: [Value]
  public var isVariadic: Bool = false
  public var numFixed: UInt32 = 0
  public var retDest: Value? = nil
  public var retRegs: UInt8 = 0
  public var retPieces: [RetPiece] = Array(repeating: .init(), count: 4)
  public var sret: Bool = false
}

/// The address of a named global symbol.
public struct GlobalAddr: Sendable, Hashable {
  public var symbol: UInt32
  public var viaGOT: Bool = false
}

public struct Dot: Sendable, Hashable {
  public var acc: Value
  public var a: Value
  public var b: Value
}
public struct Reduce: Sendable, Hashable {
  public var vector: Value
  public var op: BinOp
}
public struct Splat: Sendable, Hashable { public var scalar: Value }
public struct VaStart: Sendable, Hashable { public var list: Value }
public struct VaArg: Sendable, Hashable {
  public var list: Value
  public var ty: Type
}
public struct VaEnd: Sendable, Hashable { public var list: Value }

//! The opcode carried by an `Inst`.
public struct PtrAdd: Sendable, Hashable {
  public var base: Value
  public var byteOffset: Int64
}

public enum Opcode: Sendable, Hashable {
  case ptrAdd(PtrAdd)
  case arith(Arith)
  case arithImm(ArithImm)
  case compare(CmpOp)
  case cmp(Compare)
  case select(Select)
  case structNew(StructNew)
  case extract(Extract)
  case convert(Convert)
  case unary(Unary)
  case alloca(Alloca)
  case load(Load)
  case store(Store)
  case prefetch(Prefetch)
  case call(Call)
  case callIndirect(CallIndirect)
  case globalAddr(GlobalAddr)
  case dot(Dot)
  case reduce(Reduce)
  case splat(Splat)
  case vaStart(VaStart)
  case vaArg(VaArg)
  case vaEnd(VaEnd)
  case iconst(Int64)
  case ret(Ret)
  case jump(Jump)
  case condBr(CondBr)
}

/// A return terminator: zero or up to four values.
public struct Ret: Sendable, Hashable { public var values: [Value] }
public struct Jump: Sendable, Hashable {
  public var target: Block
  public var args: [Value]
}
public struct CondBr: Sendable, Hashable {
  public var cond: Value
  public var then: Block
  public var thenArgs: [Value]
  public var `else`: Block
  public var elseArgs: [Value]
}

/// A function: blocks, instructions, values, types.
/// Storage is dense arrays indexed by handle id, so every lookup is O(1).
public final class Function {
  public var types = TypeTable()
  public private(set) var blocks: [Block] = []
  public private(set) var blockParams: [[Value]] = []
  public private(set) var blockInsts: [[Inst]] = []
  public private(set) var blockParamTypes: [[Type]] = []
  // Terminator per block, in block order; nil until `setTerminator`.
  public private(set) var terminators: [Opcode?] = []
  // Per-inst opcode and result type, in inst order.
  public private(set) var instOpcodes: [Opcode] = []
  public private(set) var instResultTypes: [Type] = []
  // Per-value type and defining inst (nil = block parameter), in value order.
  public private(set) var valueTypes: [Type] = []
  public private(set) var valueDefs: [Inst?] = []
  // Result value by instruction id; nil for instructions without results.
  public private(set) var instResults: [Value?] = []

  public init() {}

  public func appendBlock() -> Block {
    let b = Block(raw: UInt32(blocks.count))
    blocks.append(b)
    blockParams.append([])
    blockInsts.append([])
    blockParamTypes.append([])
    terminators.append(nil)
    return b
  }

  public func appendBlockParam(_ block: Block, _ ty: Type) -> Value {
    let v = Value(raw: UInt32(valueTypes.count))
    valueTypes.append(ty)
    valueDefs.append(nil)
    blockParams[Int(block.raw)].append(v)
    blockParamTypes[Int(block.raw)].append(ty)
    return v
  }

  /// Append a value-producing instruction; returns its result value.
  @discardableResult
  public func appendInst(_ block: Block, _ ty: Type, _ op: Opcode) -> Value {
    let i = Inst(raw: UInt32(instOpcodes.count))
    instOpcodes.append(op)
    instResultTypes.append(ty)
    let v = Value(raw: UInt32(valueTypes.count))
    instResults.append(v)
    valueTypes.append(ty)
    valueDefs.append(i)
    blockInsts[Int(block.raw)].append(i)
    return v
  }

  /// The instruction that defines `value` (nil for block parameters).
  public func instForValue(_ value: Value) -> Inst? { valueDefs[Int(value.raw)] }

  /// The result value produced by `inst` (nil for stores, which produce none).
  public func resultValue(of inst: Inst) -> Value? { instResults[Int(inst.raw)] }

  /// Append a store (no result).
  public func appendStore(_ block: Block, value: Value, ptr: Value) {
    let i = Inst(raw: UInt32(instOpcodes.count))
    instOpcodes.append(.store(Store(value: value, ptr: ptr, isVolatile: false)))
    instResults.append(nil)
    blockInsts[Int(block.raw)].append(i)
  }

  public func setTerminator(_ block: Block, _ op: Opcode) {
    switch op {
    case .ret, .jump, .condBr: terminators[Int(block.raw)] = op
    default: preconditionFailure("not a terminator")
    }
  }

  public func opcode(_ inst: Inst) -> Opcode { instOpcodes[Int(inst.raw)] }
  public func valueType(of value: Value) -> Type { valueTypes[Int(value.raw)] }
  public func definingInst(of value: Value) -> Inst? { valueDefs[Int(value.raw)] }
  public func instructions(in block: Block) -> [Inst] { blockInsts[Int(block.raw)] }
  public func params(of block: Block) -> [Value] { blockParams[Int(block.raw)] }
  public func terminator(of block: Block) -> Opcode? { terminators[Int(block.raw)] }
  public func blockCount() -> Int { blocks.count }
}

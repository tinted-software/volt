#if X86_64
  import Testing
  import VoltIR
  import VoltTarget

  private enum IRInterpreterError: Error {
    case unsupportedOpcode(Opcode)
    case unsupportedOperation(BinOp)
    case missingTerminator(Block)
    case executionLimit
    case argumentCountMismatch
  }

  /// A small test oracle for the integer/control-flow subset used in differential tests.
  private func interpret(_ function: Function, arguments: [Int64]) throws -> Int64 {
    var values = Array(repeating: UInt64(0), count: function.valueTypes.count)
    let entry = Block(raw: 0)
    let entryParams = function.params(of: entry)
    guard entryParams.count == arguments.count else {
      throw IRInterpreterError.argumentCountMismatch
    }

    func value(_ handle: Value) -> UInt64 { values[Int(handle.raw)] }
    func binary(_ op: BinOp, _ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
      switch op {
      case .add: return lhs &+ rhs
      case .sub: return lhs &- rhs
      case .mul: return lhs &* rhs
      case .bitAnd: return lhs & rhs
      case .bitOr: return lhs | rhs
      case .bitXor: return lhs ^ rhs
      case .shl: return lhs &<< (rhs & 63)
      // Volt's `shr` is logical (the backend emits SHR, not SAR), regardless of
      // the signedness used by comparisons or division.
      case .shr: return lhs >> Int(rhs & 63)
      default: throw IRInterpreterError.unsupportedOperation(op)
      }
    }
    func compare(_ op: CmpOp, _ lhs: UInt64, _ rhs: UInt64) -> Bool {
      let left = Int64(bitPattern: lhs)
      let right = Int64(bitPattern: rhs)
      switch op {
      case .eq: return lhs == rhs
      case .ne: return lhs != rhs
      case .lt: return left < right
      case .le: return left <= right
      case .gt: return left > right
      case .ge: return left >= right
      }
    }

    var block = entry
    var incoming = arguments.map { UInt64(bitPattern: $0) }
    for _ in 0..<100_000 {
      let params = function.params(of: block)
      guard params.count == incoming.count else { throw IRInterpreterError.argumentCountMismatch }
      for (parameter, argument) in zip(params, incoming) {
        values[Int(parameter.raw)] = argument
      }

      for inst in function.instructions(in: block) {
        let opcode = function.opcode(inst)
        let result: UInt64
        switch opcode {
        case .iconst(let immediate): result = UInt64(bitPattern: immediate)
        case .arith(let op): result = try binary(op.op, value(op.lhs), value(op.rhs))
        case .arithImm(let op):
          result = try binary(op.op, value(op.lhs), UInt64(bitPattern: op.imm))
        case .cmp(let op): result = compare(op.op, value(op.lhs), value(op.rhs)) ? 1 : 0
        case .select(let op): result = value(op.cond) != 0 ? value(op.then) : value(op.else)
        default: throw IRInterpreterError.unsupportedOpcode(opcode)
        }
        if let resultValue = function.resultValue(of: inst) {
          values[Int(resultValue.raw)] = result
        }
      }

      guard let terminator = function.terminator(of: block) else {
        throw IRInterpreterError.missingTerminator(block)
      }
      switch terminator {
      case .ret(let ret):
        guard let result = ret.values.first else { return 0 }
        return Int64(bitPattern: value(result))
      case .jump(let jump):
        incoming = jump.args.map(value)
        block = jump.target
      case .condBr(let branch):
        let takeThen = value(branch.cond) != 0
        if takeThen {
          incoming = branch.thenArgs.map(value)
          block = branch.then
        } else {
          incoming = branch.elseArgs.map(value)
          block = branch.else
        }
      default: throw IRInterpreterError.unsupportedOpcode(terminator)
      }
    }
    throw IRInterpreterError.executionLimit
  }

  private struct DeterministicGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return state
    }
  }

  @Suite("differential JIT")
  struct DifferentialJITTests {
    @Test("randomized integer expressions agree with the IR interpreter")
    func randomizedExpressions() throws {
      let operations: [BinOp] = [.add, .sub, .mul, .bitAnd, .bitOr, .bitXor, .shl, .shr]
      let immediateOperations: [BinOp] = [.add, .sub, .mul, .bitAnd, .bitOr, .bitXor]
      var generator = DeterministicGenerator(state: 0x564F_4C54)
      var functions: [(String, Function)] = []

      for functionIndex in 0..<12 {
        let function = Function()
        let i64 = function.types.intern(.int(signedness: true, bits: 64))
        let entry = function.appendBlock()
        let x = function.appendBlockParam(entry, i64)
        let y = function.appendBlockParam(entry, i64)
        let builder = Builder(function: function, block: entry)
        var values = [x, y]

        for _ in 0..<20 {
          let lhs = values[Int(generator.next() % UInt64(values.count))]
          if generator.next() % 4 == 0 {
            let op = immediateOperations[Int(generator.next() % UInt64(immediateOperations.count))]
            let immediate = Int64(generator.next() % 257) - 128
            values.append(builder.arithImm(op, lhs, immediate))
          } else {
            let rhs = values[Int(generator.next() % UInt64(values.count))]
            let op = operations[Int(generator.next() % UInt64(operations.count))]
            values.append(builder.arith(op, lhs, rhs))
          }
        }

        var result = values[values.count - 1]
        for value in values.dropFirst(2).dropLast().reversed() {
          result = builder.arith(.bitXor, result, value)
        }
        builder.ret(result)
        functions.append(("generated_\(functionIndex)", function))
      }

      let module = JITModule(functions: functions.map { NamedFunction(name: $0.0, function: $0.1) })
      let jitted = try jitCompileModule(module)
      let inputs: [(Int64, Int64)] = [
        (0, 0), (1, -1), (Int64.max, 3), (Int64.min, -7),
        (0x1234_5678, -0x7654_321), (-91, 17),
      ]

      for (name, function) in functions {
        let compiled = try #require(jitted.int64BinaryFunction(named: name))
        for (x, y) in inputs {
          let expected = try interpret(function, arguments: [x, y])
          #expect(compiled(x, y) == expected, "\(name)(\(x), \(y))")
        }
      }
    }

    @Test("randomized branch and block-parameter results agree with the interpreter")
    func randomizedBranches() throws {
      let function = Function()
      let i64 = function.types.intern(.int(signedness: true, bits: 64))
      let bool = function.types.intern(.bool)
      let entry = function.appendBlock()
      let left = function.appendBlock()
      let right = function.appendBlock()
      let join = function.appendBlock()
      let x = function.appendBlockParam(entry, i64)
      let y = function.appendBlockParam(entry, i64)
      let result = function.appendBlockParam(join, i64)

      let builder = Builder(function: function, block: entry)
      let condition = builder.icmp(.lt, x, y, boolTy: bool)
      builder.condBr(condition, then: left, else: right)
      Builder(function: function, block: left).jump(join, [x])
      Builder(function: function, block: right).jump(join, [y])
      Builder(function: function, block: join).ret(result)

      let module = JITModule(functions: [NamedFunction(name: "choose", function: function)])
      let jitted = try jitCompileModule(module)
      let compiled = try #require(jitted.int64BinaryFunction(named: "choose"))
      let cases: [(Int64, Int64)] = [(-4, 9), (9, -4), (0, 0), (Int64.min, Int64.max)]
      var generator = DeterministicGenerator(state: 0x4252_414E_4348)
      let randomCases = (0..<64).map { _ in
        (Int64(bitPattern: generator.next()), Int64(bitPattern: generator.next()))
      }
      for (x, y) in cases + randomCases {
        #expect(compiled(x, y) == (try interpret(function, arguments: [x, y])))
      }
    }

    @Test("randomized loop results agree with the IR interpreter")
    func randomizedLoops() throws {
      let function = Function()
      let i64 = function.types.intern(.int(signedness: true, bits: 64))
      let bool = function.types.intern(.bool)
      let entry = function.appendBlock()
      let head = function.appendBlock()
      let body = function.appendBlock()
      let done = function.appendBlock()
      let n = function.appendBlockParam(entry, i64)
      let index = function.appendBlockParam(head, i64)
      let accumulator = function.appendBlockParam(head, i64)
      let bodyIndex = function.appendBlockParam(body, i64)
      let bodyAccumulator = function.appendBlockParam(body, i64)
      let result = function.appendBlockParam(done, i64)

      let entryBuilder = Builder(function: function, block: entry)
      entryBuilder.jump(head, [n, entryBuilder.iconst(i64, 0)])
      let headBuilder = Builder(function: function, block: head)
      let finished = headBuilder.icmp(.eq, index, headBuilder.iconst(i64, 0), boolTy: bool)
      headBuilder.condBr(finished, then: done, [accumulator], else: body, [index, accumulator])
      let bodyBuilder = Builder(function: function, block: body)
      let nextAccumulator = bodyBuilder.arith(.add, bodyAccumulator, bodyIndex)
      let nextIndex = bodyBuilder.arithImm(.sub, bodyIndex, 1)
      bodyBuilder.jump(head, [nextIndex, nextAccumulator])
      Builder(function: function, block: done).ret(result)

      let module = JITModule(functions: [NamedFunction(name: "countdown", function: function)])
      let jitted = try jitCompileModule(module)
      let compiled: @convention(c) (Int64) -> Int64 = try #require(
        jitted.entry(as: (@convention(c) (Int64) -> Int64).self, name: "countdown"))
      for n in 0...64 {
        #expect(compiled(Int64(n)) == (try interpret(function, arguments: [Int64(n)])))
      }
    }
  }
#endif

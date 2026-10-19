//! In-memory linker + JIT for x86-64. Compiles named functions, lays them out
//! consecutively in one blob, and resolves each `call` relocation (a rel32
//! displacement) to the target symbol's offset. The result is a
//! position-independent blob entered at any function's offset.
//! Modeled on `libs/vulcan-target/x86_64/link.zig` + `aarch64/jit.zig`.
//!
//! v1 carries only `.call` relocs; `.pcrel_lea` (data) and `.got_pcrel` (which
//! needs a dynamic GOT) fail closed with `CodegenError.unsupportedReloc`.

import VoltIR
import VoltTarget

/// Compile and link `module`'s functions into one blob, 16-byte aligned starts,
/// NOP-padded between functions. Intra-module `.call` relocs resolve to
/// `rel32 = target - (site + 4)`.
public func compileModule(_ module: JITModule, names: [String]? = nil) throws -> LinkedModule {
  var compiled: [CompiledCode] = []
  for (i, e) in module.functions.enumerated() {
    var c = try compileFunction(e.function)
    // v1 symbol threading: `Call` carries an index; the driver supplies names.
    if let names, !names.isEmpty {
      c.relocs = c.relocs.map { r in
        var r = r
        if r.symbol.hasPrefix("extern"), let idx = UInt32(r.symbol.dropFirst(6)) {
          if Int(idx) < names.count { r.symbol = names[Int(idx)] }
        }
        _ = i
        return r
      }
    }
    compiled.append(c)
  }

  var offsets: [Int] = []
  var total = 0
  for c in compiled {
    offsets.append(total)
    total += (c.bytes.count + 15) & ~15
  }

  var symbols: [FuncSymbol] = []
  for (i, e) in module.functions.enumerated() {
    symbols.append(FuncSymbol(name: e.name, offset: offsets[i]))
  }

  var code = [UInt8](repeating: 0x90, count: total)  // pad with NOPs
  for (i, c) in compiled.enumerated() {
    code.replaceSubrange(offsets[i]..<(offsets[i] + c.bytes.count), with: c.bytes)
  }

  for (i, c) in compiled.enumerated() {
    for r in c.relocs {
      let site = offsets[i] + r.offset
      switch r.kind {
      case .call:
        guard let target = symbols.first(where: { $0.name == r.symbol })?.offset else {
          throw CodegenError.undefinedSymbol(r.symbol)
        }
        let rel = Int32(target - (site + 4))
        let u = UInt32(bitPattern: rel)
        code[site] = UInt8(u & 0xFF)
        code[site + 1] = UInt8((u >> 8) & 0xFF)
        code[site + 2] = UInt8((u >> 16) & 0xFF)
        code[site + 3] = UInt8((u >> 24) & 0xFF)
      case .pcrelLea:
        throw CodegenError.unsupportedReloc("pcrel_lea needs map-time data addresses")
      case .gotPCRel:
        throw CodegenError.unsupportedReloc("got_pcrel needs a dynamic GOT")
      }
    }
  }
  return LinkedModule(code: code, symbols: symbols)
}

/// A JIT-compiled module: its live executable buffer plus each function's byte
/// offset into it.
public final class JittedModule {
  public let memory: ExecutableMemory
  public let linked: LinkedModule

  public init(linked: LinkedModule) throws {
    self.linked = linked
    self.memory = try ExecutableMemory(byteCount: linked.code.count)
    try memory.write(linked.code)
    try memory.protectExecutable()
  }

  /// A typed callable pointer to the function named `name`, or nil if absent.
  public func entry<T>(as type: T.Type, name: String) -> T? {
    guard let off = linked.addressOf(name) else { return nil }
    return memory.entry(as: type, offset: off)
  }
}

/// JIT-compile a module: link it into one position-independent image and map it
/// into executable memory.
public func jitCompileModule(_ module: JITModule, names: [String]? = nil) throws -> JittedModule {
  try JittedModule(linked: compileModule(module, names: names))
}

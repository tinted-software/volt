//! The Volt target seam: shared value types and protocols that every backend
//! (AArch64, x86_64, ...) implements so linkers, JIT drivers, and tools stay
//! arch-agnostic. Modeled on Volt's existing Zig target seam
//! (`libs/vulcan-target`): per-target `disasm` / `encode` / `isel` / `link` /
//! `jit` units behind one small set of shared contracts.
//!
//! Scope note: this rewrite currently ships an AArch64 *disassembler* and an
//! x86_64 *code generator + JIT*. The seam is deliberately wider than that so
//! future targets (and the reverse directions) slot in without renegotiation:
//! `RelocKind` already names the GOT/pc-relative shapes, and `ExecutableMemory`
//! carries an icache-sync hook that is a no-op on x86_64.

import VoltIR

#if canImport(Glibc)
  import Glibc
#elseif canImport(Darwin)
  import Darwin
#endif

// ---------------------------------------------------------------
// Machine-code representation
// ---------------------------------------------------------------

/// A compiled chunk of machine code plus its metadata.
public struct CompiledCode: Sendable {
  /// Emitted bytes. Fixed-width backends (AArch64) store little-endian words here.
  public var bytes: [UInt8]
  /// Relocations emitted by codegen, unresolved until link/map time.
  public var relocs: [Reloc]
  /// Source-line rows, keyed by byte offset within `bytes`.
  public var lines: [LineEntry]

  public init(bytes: [UInt8], relocs: [Reloc] = [], lines: [LineEntry] = []) {
    self.bytes = bytes
    self.relocs = relocs
    self.lines = lines
  }
}

/// A source-line row: the byte offset where a source line's code begins.
public struct LineEntry: Sendable, Hashable {
  public var offset: UInt32
  public var line: UInt32
  public init(offset: UInt32, line: UInt32) {
    self.offset = offset
    self.line = line
  }
}

/// A relocation shape a codegen backend can emit.
public enum RelocKind: Sendable, Hashable {
  /// A direct-call displacement (`call rel32` / `bl`), resolved intra-module
  /// at link time against `LinkedModule.symbols`.
  case call
  /// A `global_addr` data reference (`lea rd, [rip+disp32]` on x86_64),
  /// patched once the symbol's runtime address is known at map time.
  case pcrelLea
  /// A GOT-indirect `global_addr` (`via_got`); needs a dynamic GOT, so the
  /// in-memory JIT linker rejects it fail-closed.
  case gotPCRel
}

/// A relocation targeting `symbol`, whose disp32 field starts at byte `offset`.
public struct Reloc: Sendable, Hashable {
  public var offset: Int
  public var symbol: String
  public var kind: RelocKind
  public init(offset: Int, symbol: String, kind: RelocKind = .call) {
    self.offset = offset
    self.symbol = symbol
    self.kind = kind
  }
}

/// A function's location in a linked image: its name and byte offset.
public struct FuncSymbol: Sendable, Hashable {
  public var name: String
  public var offset: Int
  public init(name: String, offset: Int) {
    self.name = name
    self.offset = offset
  }
}

/// A linked image: one flat code blob plus every function's byte offset.
public struct LinkedModule: Sendable {
  public var code: [UInt8]
  public var symbols: [FuncSymbol]
  public init(code: [UInt8], symbols: [FuncSymbol]) {
    self.code = code
    self.symbols = symbols
  }
  /// A function's byte offset, or nil if not defined.
  public func addressOf(_ name: String) -> Int? {
    symbols.first(where: { $0.name == name })?.offset
  }
}

/// A named function to link into a module. `Function` is a reference type, so
/// this (and `JITModule`) is intentionally *not* Sendable.
public struct NamedFunction {
  public var name: String
  public var function: Function
  public init(name: String, function: Function) {
    self.name = name
    self.function = function
  }
}

/// A module of named functions to link into one image.
public struct JITModule {
  public var functions: [NamedFunction]
  public init(functions: [NamedFunction] = []) {
    self.functions = functions
  }
}

/// Codegen failure modes. Backends throw these instead of inventing their own
/// error types, so drivers never map backend-specific tags.
public enum CodegenError: Error, Sendable, CustomStringConvertible {
  case unsupported(String)
  case invalidFunction(String)
  case undefinedSymbol(String)
  case unsupportedReloc(String)

  public var description: String {
    switch self {
    case .unsupported(let what): return "unsupported: \(what)"
    case .invalidFunction(let what): return "invalid function: \(what)"
    case .undefinedSymbol(let name): return "undefined symbol: \(name)"
    case .unsupportedReloc(let what): return "unsupported reloc: \(what)"
    }
  }
}

// ---------------------------------------------------------------
// Target seam protocols
// ---------------------------------------------------------------

/// A target's disassembler: decodes machine code into a readable listing.
/// Anything outside the decoded subset renders as a raw directive
/// (`.word 0x<hex>` / `.byte 0x<hex>`) so nothing is silently mis-decoded.
public protocol Disassembler {
  /// Decode one instruction at `code[offset]`; nil when truncated.
  /// Unknown encodings return a raw directive with the consumed length.
  static func disassembleOne(code: [UInt8], at offset: Int) -> (text: String, length: Int)?
  /// Render a whole code buffer as `addr: hex  text` lines.
  static func format(code: [UInt8]) -> String
  /// Render a linked image with `name:` labels and call-target annotations.
  static func formatModule(code: [UInt8], symbols: [FuncSymbol]) -> String
}

/// A target's instruction selector: lowers an IR `Function` to machine code.
public protocol InstructionSelector {
  static func compileFunction(_ function: Function) throws -> CompiledCode
}

/// A target's in-memory linker: compiles named functions into one blob and
/// resolves intra-module relocations.
public protocol InMemoryLinker {
  static func compileModule(_ module: JITModule) throws -> LinkedModule
}

// ---------------------------------------------------------------
// Executable memory: pluggable W^X layer
// ---------------------------------------------------------------

public enum JITMemoryError: Error, Sendable {
  case mapFailed(errno: Int32)
  case protectFailed(errno: Int32)
  case outOfBounds
  case empty
}

func systemPageSize() -> Int {
  let ps = sysconf(Int32(_SC_PAGESIZE))
  return ps > 0 ? Int(ps) : 4096
}

@inline(never)
func syncInstructionCache(_ base: UnsafeRawPointer, _ byteCount: Int) {
  // x86_64 has a coherent instruction cache: no flush needed after writing
  // code. AArch64 will need a D-cache clean + I-cache invalidate here; the
  // hook exists so that backend slots in without touching callers.
  _ = base
  _ = byteCount
}

/// A W^X executable buffer: mapped read+write, flipped to read+execute on
/// `protectExecutable()`. Owns its mapping; `deinit` unmaps.
public final class ExecutableMemory {
  public let base: UnsafeMutableRawPointer
  /// Usable byte count (mapping itself is page-rounded; see `mappedSize`).
  public let byteCount: Int
  private let mappedSize: Int
  private var isExecutable = false

  public init(byteCount: Int) throws {
    guard byteCount > 0 else { throw JITMemoryError.empty }
    let page = systemPageSize()
    guard byteCount <= Int.max - (page - 1) else { throw JITMemoryError.outOfBounds }
    let mapped = (byteCount + page - 1) & ~(page - 1)
    let prot = PROT_READ | PROT_WRITE
    let flags = MAP_PRIVATE | MAP_ANONYMOUS
    guard let ptr = mmap(nil, mapped, prot, flags, -1, 0),
      ptr != MAP_FAILED
    else { throw JITMemoryError.mapFailed(errno: errno) }
    self.base = ptr
    self.byteCount = byteCount
    self.mappedSize = mapped
  }

  /// Write `bytes` at `offset`. Traps in debug builds after `protectExecutable`.
  public func write(_ bytes: [UInt8], at offset: Int = 0) throws {
    precondition(offset >= 0, "negative write offset")
    guard offset <= byteCount, bytes.count <= byteCount - offset else {
      throw JITMemoryError.outOfBounds
    }
    guard !bytes.isEmpty else { return }
    precondition(!isExecutable, "writing to finalized executable buffer")
    bytes.withUnsafeBufferPointer { src in
      (base + offset).copyMemory(from: src.baseAddress!, byteCount: bytes.count)
    }
  }

  deinit {
    _ = munmap(base, mappedSize)
  }

  /// Flip the mapping to read+execute and synchronize the instruction cache.
  public func protectExecutable() throws {
    precondition(!isExecutable, "already executable")
    guard mprotect(base, mappedSize, PROT_READ | PROT_EXEC) == 0 else {
      throw JITMemoryError.protectFailed(errno: errno)
    }
    syncInstructionCache(UnsafeRawPointer(base), byteCount)
    isExecutable = true
  }
  /// A typed callable pointer at `offset` (e.g. `@convention(c) (Int64) -> Int64`).
  /// Reinterprets the address itself — it does NOT load a pointer from code.
  /// The offset must identify a byte inside this mapping; the returned callable
  /// pointer must not outlive this `ExecutableMemory` instance.
  public func entry<T>(as type: T.Type, offset: Int = 0) -> T {
    precondition(offset >= 0 && offset < byteCount, "entry offset outside executable mapping")
    precondition(isExecutable, "mapping must be executable before obtaining an entry")
    return unsafeBitCast(base + offset, to: T.self)
  }
}

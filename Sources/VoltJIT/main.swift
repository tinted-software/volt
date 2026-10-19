import VoltIR
import VoltTarget
import VoltTargetX86_64

// double(x) = x * 2
let fn = Function()
var types = fn.types
let i64 = types.intern(.int(signedness: true, bits: 64))
let entry = fn.appendBlock()
let x = fn.appendBlockParam(entry, i64)
let b = Builder(function: fn, block: entry)
let two = b.iconst(i64, 2)
let r = b.arith(.mul, x, two)
b.ret(r)

let mod = JITModule(functions: [NamedFunction(name: "double", function: fn)])
do {
  let jitted = try jitCompileModule(mod)
  typealias Fn = @convention(c) (Int64) -> Int64
  let f: Fn = jitted.entry(as: Fn.self, name: "double")!
  print("double(21) = \(f(21))")
} catch {
  print("jit failed: \(error)")
}

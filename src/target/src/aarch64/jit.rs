//! In-process AArch64 linked-image JIT with owned W^X code and data mappings.
//! Cache maintenance and global relocations are handled by the shared native mapper.
//! Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).

use super::link::Linked;
use crate::native::{JittedModule, ModuleError};

/// A linked module and its executable mapping. Data/BSS and relocated global
/// pointers remain alive for exactly as long as the generated code.
pub struct Compiled {
    pub linked: Linked,
    mapping: JittedModule,
}

impl Compiled {
    pub fn new(linked: Linked) -> Result<Self, ModuleError> {
        let mapping = crate::native::map_linked_aarch64(&linked)?;
        Ok(Self { linked, mapping })
    }

    pub fn offset_of(&self, name: &str) -> Option<usize> {
        self.linked.address_of(name)
    }

    pub fn code(&self) -> &[u8] {
        self.mapping.code()
    }

    pub fn data_addr(&self, name: &str) -> Option<*mut u8> {
        self.mapping.data_addr(name)
    }

    /// Return the callable entry for a linked function.
    ///
    /// # Safety
    /// T must be the generated function's exact AAPCS64 signature, the host
    /// must be AArch64, and the pointer must not outlive this module.
    pub unsafe fn entry<T: Copy>(&self, name: &str) -> Option<T> {
        let offset = self.offset_of(name)?;
        // SAFETY: the caller establishes the function signature and ISA.
        unsafe { self.mapping.entry_at(offset) }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::aarch64::{isel::ModelCaps, link::compile_module};
    use volt_ir::function::{
        ArithImm, BinOp, Function, GlobalAddr, MemFlags, Opcode, Ret, Terminator,
    };
    use volt_ir::module::{DataReloc, Module};

    #[test]
    fn linked_module_maps_calls_globals_and_data_relocations() {
        // inc(x) = x + 1
        let mut inc = Function::new();
        let u64_t = inc.types.parse_type("i64").unwrap();
        let b = inc.append_block();
        let x = inc.append_block_param(b, u64_t);
        let r = inc.append_inst(
            b,
            u64_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: x,
                imm: 1,
            }),
        );
        inc.set_terminator(b, Terminator::Ret(Ret::one(r)));
        // twice(x) = inc(inc(x))
        let mut twice = Function::new();
        let u64_t = twice.types.parse_type("i64").unwrap();
        let b = twice.append_block();
        let x = twice.append_block_param(b, u64_t);
        let once = twice.append_call(b, u64_t, "inc", &[x]);
        let r = twice.append_call(b, u64_t, "inc", &[once]);
        twice.set_terminator(b, Terminator::Ret(Ret::one(r)));
        // seed() = *seed_value
        let mut seed = Function::new();
        let u64_t = seed.types.parse_type("i64").unwrap();
        let ptr_t = seed.types.parse_type("ptr").unwrap();
        let b = seed.append_block();
        let symbol = seed.intern_symbol("seed_value");
        let addr = seed.append_inst(
            b,
            ptr_t,
            Opcode::GlobalAddr(GlobalAddr {
                symbol,
                via_got: false,
            }),
        );
        let v = seed.append_load(b, u64_t, addr, MemFlags::new());
        seed.set_terminator(b, Terminator::Ret(Ret::one(v)));

        let mut module = Module::new();
        module.add_function("twice", twice);
        module.add_function("inc", inc);
        module.add_function("seed", seed);
        module.add_writable("seed_value", 40u64.to_le_bytes().to_vec());
        module.add_data_relocs(
            "ptrs",
            vec![0; 8],
            vec![DataReloc {
                offset: 0,
                symbol: "inc".into(),
            }],
        );
        let linked = compile_module(&module, &ModelCaps::default()).unwrap();
        let compiled = Compiled::new(linked).unwrap();
        // SAFETY: signatures match the IR above and the host is AArch64.
        unsafe {
            let twice: extern "C" fn(u64) -> u64 = compiled.entry("twice").unwrap();
            assert_eq!(twice(5), 7);
            let seed: extern "C" fn() -> u64 = compiled.entry("seed").unwrap();
            assert_eq!(seed(), 40);
            let inc: extern "C" fn(u64) -> u64 = compiled.entry("inc").unwrap();
            let slot = compiled
                .data_addr("ptrs")
                .unwrap()
                .cast::<u64>()
                .read_unaligned();
            assert_eq!(slot, inc as usize as u64);
        }
        assert_eq!(compiled.offset_of("twice"), Some(0));
        assert!(!compiled.code().is_empty());
    }
}

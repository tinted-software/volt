extern crate alloc;

#[path = "src/shader_ir.rs"]
mod shader_ir;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/shader_ir.rs");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let processor = "gfx1032";

    let vs_ir = shader_ir::build_vertex_shader();
    let vs_blob = shader_ir::compile_to_blob(&vs_ir, processor, false)
        .expect("build-time VS compilation failed");
    fs::write(out_dir.join("vs.vshd"), vs_blob.encode()).expect("write vs.vshd");

    let mut table = String::from(
        "pub static VS_BLOB: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/vs.vshd\"));\n\n\
         pub fn lookup_ps_blob(fog_mode: u32, alpha_func: u32) -> Option<&'static [u8]> {\n\
         \x20   match (fog_mode, alpha_func) {\n",
    );

    for fog_mode in 0..4u32 {
        for &alpha_func in &shader_ir::ALPHA_FUNCS {
            let key = shader_ir::PsKey {
                alpha_func,
                fog_mode,
            };
            let ps_ir = shader_ir::build_pixel_shader(&key);
            let ps_blob = shader_ir::compile_to_blob(&ps_ir, processor, key.uses_kill())
                .unwrap_or_else(|e| panic!("build-time PS ({key:?}) compilation failed: {e}"));
            let filename = format!("ps_{fog_mode}_{alpha_func:#x}.vshd");
            fs::write(out_dir.join(&filename), ps_blob.encode())
                .unwrap_or_else(|e| panic!("write {filename}: {e}"));
            table.push_str(&format!(
                "        ({fog_mode}, {alpha_func:#x}) => Some(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{filename}\"))),\n"
            ));
        }
    }

    table.push_str("        _ => None,\n    }\n}\n");
    fs::write(out_dir.join("shaders_gen.rs"), table).expect("write shaders_gen.rs");
}

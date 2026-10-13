use volt_disasm::{Analysis, Image};
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump <binary> [function]");
    let want = std::env::args().nth(2);
    let img = Image::parse(&std::fs::read(path).unwrap()).unwrap();
    let a = Analysis::run(&img, &[]);
    let blocks: usize = a.functions.values().map(|f| f.blocks.len()).sum();
    eprintln!(
        "{} symbols, {} functions, {} blocks, {} xref targets",
        img.symbols.len(),
        a.functions.len(),
        blocks,
        a.xrefs.len()
    );
    let Some(want) = want else { return };
    let f = a
        .functions
        .values()
        .find(|f| f.name.as_deref() == Some(&want))
        .expect("no such function");
    for b in f.blocks.values() {
        println!("block {:#x} -> {:x?}  [{:?}]", b.start, b.succs, b.exit);
        for i in img.disassemble(b.start, ((b.end - b.start) / 4) as usize) {
            println!("  {:#x}: {:08x}  {}", i.addr, i.word, i.text);
        }
    }
}

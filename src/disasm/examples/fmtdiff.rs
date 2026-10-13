//! Compare `volt_isa_aarch64::format` against GNU `objdump` over an AArch64 ELF.
//!
//! ```text
//! cargo run --release -p volt-disasm --example fmtdiff -- <elf> [mnemonic] [limit]
//! ```
//!
//! For every instruction `objdump -d` lists it decodes the word with
//! `volt_isa_aarch64::decode`, renders it at its address, and compares the text
//! after normalising what the two spell differently on purpose (trailing `// ...`
//! comments, `<symbol+off>` annotations, and the `0x` on branch targets). It
//! reports totals, then mismatches and undecodable words grouped by `objdump`'s
//! mnemonic, most frequent first, with one example of each. A mnemonic argument
//! restricts the report to that mnemonic and prints up to `limit` examples.

use std::collections::HashMap;
use std::process::Command;
use volt_isa_aarch64::decode::decode;

/// `objdump`'s own spelling, reduced to what must agree.
fn normalize(text: &str) -> String {
    let text = text.split("//").next().unwrap_or("");
    let text = match text.find('<') {
        Some(at) => &text[..at],
        None => text,
    };
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = text.replace(", ", ",");
    // A branch target is the last operand, written bare by objdump and `0x...` here.
    match text.rsplit_once(',').or_else(|| text.rsplit_once(' ')) {
        Some((head, last))
            if last.len() >= 4
                && !last.starts_with('#')
                && last
                    .trim_start_matches("0x")
                    .chars()
                    .all(|c| c.is_ascii_hexdigit())
                && (last.starts_with("0x") || head.contains(' ') || !head.contains(',')) =>
        {
            let sep = if text.contains(',') { "," } else { " " };
            format!("{head}{sep}{}", last.trim_start_matches("0x"))
        }
        _ => text,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: fmtdiff <elf> [mnemonic] [limit]");
    let only = args.next();
    let limit: usize = args.next().map_or(3, |n| n.parse().unwrap());
    let listing = Command::new("objdump")
        .args(["-d", "--no-show-raw-insn", "-w", &path])
        .output()
        .expect("objdump");
    let raw = Command::new("objdump")
        .args(["-d", "-w", &path])
        .output()
        .expect("objdump");
    let listing = String::from_utf8_lossy(&listing.stdout).into_owned();
    let raw = String::from_utf8_lossy(&raw.stdout).into_owned();

    // address -> objdump text, and address -> word.
    let mut text: Vec<(u64, String)> = Vec::new();
    for line in listing.lines() {
        let Some((addr, rest)) = line.trim().split_once(":\t") else {
            continue;
        };
        if let Ok(addr) = u64::from_str_radix(addr, 16) {
            text.push((addr, rest.trim().to_string()));
        }
    }
    let mut words: HashMap<u64, u32> = HashMap::new();
    for line in raw.lines() {
        let mut parts = line.trim().splitn(3, '\t');
        let (Some(addr), Some(word)) = (parts.next(), parts.next()) else {
            continue;
        };
        if let (Ok(addr), Ok(word)) = (
            u64::from_str_radix(addr.trim_end_matches(':'), 16),
            u32::from_str_radix(word.trim(), 16),
        ) {
            words.insert(addr, word);
        }
    }

    struct Group {
        count: usize,
        examples: Vec<(u32, String, String)>,
    }
    let mut mismatch: HashMap<String, Group> = HashMap::new();
    let mut undecoded: HashMap<String, Group> = HashMap::new();
    let (mut total, mut matched) = (0usize, 0usize);
    for (addr, expected) in &text {
        let Some(&word) = words.get(addr) else {
            continue;
        };
        // Data in the text section: `.word`, `.inst`, `udf`.
        let mnemonic = expected.split_whitespace().next().unwrap_or("").to_string();
        if mnemonic.starts_with('.') {
            continue;
        }
        total += 1;
        let (group, ours) = match decode(word) {
            Ok(instruction) => {
                let ours = instruction.display_at(*addr).to_string();
                if normalize(&ours) == normalize(expected) {
                    matched += 1;
                    continue;
                }
                (&mut mismatch, ours)
            }
            Err(_) => (&mut undecoded, String::new()),
        };
        if only.as_ref().is_some_and(|m| *m != mnemonic) {
            let _ = ours;
            group
                .entry(mnemonic)
                .or_insert(Group {
                    count: 0,
                    examples: Vec::new(),
                })
                .count += 1;
            continue;
        }
        let entry = group.entry(mnemonic).or_insert(Group {
            count: 0,
            examples: Vec::new(),
        });
        entry.count += 1;
        if entry.examples.len() < limit {
            entry.examples.push((word, expected.clone(), ours));
        }
    }
    println!(
        "total {total}  matched {matched}  mismatched {}  undecoded {}",
        mismatch.values().map(|g| g.count).sum::<usize>(),
        undecoded.values().map(|g| g.count).sum::<usize>()
    );
    for (title, groups) in [("MISMATCH", &mismatch), ("UNDECODED", &undecoded)] {
        let mut groups: Vec<_> = groups.iter().collect();
        groups.sort_by_key(|(name, g)| (std::cmp::Reverse(g.count), (*name).clone()));
        for (name, group) in groups
            .into_iter()
            .filter(|(name, _)| only.as_ref().is_none_or(|m| m == *name))
            .take(if only.is_some() { usize::MAX } else { 40 })
        {
            println!("{title} {:>6} {name}", group.count);
            for (word, expected, ours) in &group.examples {
                println!("         {word:08x}  objdump: {expected}");
                if title == "MISMATCH" {
                    println!("                    ours:    {ours}");
                }
            }
        }
    }
}

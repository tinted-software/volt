//! Flattened device tree patching for a device tree supplied by the user.
//!
//! The kernel reads `/chosen/bootargs` for its command line and `/memory` `reg`
//! for the RAM it may use, so both must describe what the emulator provides. The
//! blob is parsed into an owned tree of nodes and properties, keeping their order
//! and the memory reservation map, the two properties are edited there, and the
//! tree is serialized into a fresh blob. Nothing else in the tree is touched.

use alloc::vec::Vec;
use core::fmt;

const MAGIC: u32 = 0xd00d_feed;
const HEADER_LEN: usize = 40;
/// Newest format version this reader and writer speak.
const VERSION: u32 = 17;
const LAST_COMP_VERSION: u32 = 16;
const RESERVE_ENTRY_LEN: usize = 16;

const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;
const FDT_NOP: u32 = 4;
const FDT_END: u32 = 9;

/// Cell counts the spec assumes when the root node omits them.
const DEFAULT_ADDRESS_CELLS: u32 = 2;
const DEFAULT_SIZE_CELLS: u32 = 1;

const TRUNCATED_STRUCT: Error = Error::Malformed("structure block ends inside a token");
const PROP_PAST_STRUCT: Error = Error::Malformed("property runs past structure block");

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The blob is shorter than its header, or than the `totalsize` it declares.
    Truncated {
        len: usize,
        needed: usize,
    },
    BadMagic(u32),
    UnsupportedVersion(u32),
    /// The blob violates the flattened format; the message names the first problem found.
    Malformed(&'static str),
    /// No node named `memory*` with `device_type = "memory"` exists to receive RAM.
    NoMemoryNode,
    /// `#address-cells` or `#size-cells` is not 1 or 2, so `reg` cannot be encoded.
    CellCount(u32),
    /// The RAM base or size does not fit in the single cell the root node declares.
    ValueTooLarge {
        value: u64,
        cells: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated { len, needed } => {
                write!(f, "device tree is {len} bytes, needs {needed}")
            }
            Error::BadMagic(magic) => write!(f, "bad device tree magic {magic:#010x}"),
            Error::UnsupportedVersion(version) => {
                write!(
                    f,
                    "unsupported device tree version {version}, need 17 or newer"
                )
            }
            Error::Malformed(what) => write!(f, "malformed device tree: {what}"),
            Error::NoMemoryNode => {
                write!(
                    f,
                    "device tree has no memory node with device_type \"memory\""
                )
            }
            Error::CellCount(cells) => {
                write!(f, "root cell count {cells} is not 1 or 2")
            }
            Error::ValueTooLarge { value, cells } => {
                write!(f, "{value:#x} does not fit in {cells} cell(s)")
            }
        }
    }
}

impl core::error::Error for Error {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    name: Vec<u8>,
    props: Vec<Prop>,
    children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Prop {
    name: Vec<u8>,
    value: Vec<u8>,
}

impl Node {
    fn named(name: &[u8]) -> Self {
        Node {
            name: name.to_vec(),
            props: Vec::new(),
            children: Vec::new(),
        }
    }
}

struct Parsed {
    reservations: Vec<(u64, u64)>,
    boot_cpuid_phys: u32,
    root: Node,
}

/// Return a new blob equal to `blob` except: if `bootargs` is `Some`, `/chosen`
/// gets the `bootargs` string property (replacing any existing one, creating
/// `/chosen` if absent); if `memory` is `Some((base, size))`, the first node
/// (depth-first) whose name starts with `memory` and whose `device_type` is
/// `memory` gets `reg` encoded with the root's `#address-cells` and `#size-cells`.
/// Every other node and property is preserved in order.
pub fn patch(
    blob: &[u8],
    bootargs: Option<&str>,
    memory: Option<(u64, u64)>,
) -> Result<Vec<u8>, Error> {
    let Parsed {
        reservations,
        boot_cpuid_phys,
        mut root,
    } = parse(blob)?;
    if let Some(args) = bootargs {
        let mut value = args.as_bytes().to_vec();
        value.push(0);
        set_prop(chosen_node(&mut root), b"bootargs", value);
    }
    if let Some((base, size)) = memory {
        let address_cells = cell_count(&root, b"#address-cells", DEFAULT_ADDRESS_CELLS)?;
        let size_cells = cell_count(&root, b"#size-cells", DEFAULT_SIZE_CELLS)?;
        let mut reg = Vec::new();
        push_cells(&mut reg, base, address_cells)?;
        push_cells(&mut reg, size, size_cells)?;
        let node = memory_node_mut(&mut root).ok_or(Error::NoMemoryNode)?;
        set_prop(node, b"reg", reg);
    }
    serialize(&root, &reservations, boot_cpuid_phys)
}

fn chosen_node(root: &mut Node) -> &mut Node {
    let index = match root.children.iter().position(|c| c.name == b"chosen") {
        Some(index) => index,
        None => {
            root.children.push(Node::named(b"chosen"));
            root.children.len() - 1
        }
    };
    &mut root.children[index]
}

fn memory_node_mut(node: &mut Node) -> Option<&mut Node> {
    if is_memory_device(node) {
        return Some(node);
    }
    node.children.iter_mut().find_map(memory_node_mut)
}

fn is_memory_device(node: &Node) -> bool {
    node.name.starts_with(b"memory")
        && prop_value(node, b"device_type")
            .is_some_and(|value| value.strip_suffix(b"\0").unwrap_or(value) == b"memory")
}

fn prop_value<'a>(node: &'a Node, name: &[u8]) -> Option<&'a [u8]> {
    node.props
        .iter()
        .find(|p| p.name == name)
        .map(|p| p.value.as_slice())
}

fn set_prop(node: &mut Node, name: &[u8], value: Vec<u8>) {
    match node.props.iter_mut().find(|p| p.name == name) {
        Some(prop) => prop.value = value,
        None => node.props.push(Prop {
            name: name.to_vec(),
            value,
        }),
    }
}

fn cell_count(root: &Node, name: &[u8], default: u32) -> Result<u32, Error> {
    match prop_value(root, name) {
        None => Ok(default),
        Some(value) => <[u8; 4]>::try_from(value)
            .map(u32::from_be_bytes)
            .map_err(|_| Error::Malformed("cell count property is not one u32")),
    }
}

/// Appends `value` as `cells` big-endian u32s; only 1 and 2 cells are valid.
fn push_cells(out: &mut Vec<u8>, value: u64, cells: u32) -> Result<(), Error> {
    match cells {
        1 => {
            let word = u32::try_from(value).map_err(|_| Error::ValueTooLarge { value, cells })?;
            out.extend_from_slice(&word.to_be_bytes());
            Ok(())
        }
        2 => {
            out.extend_from_slice(&value.to_be_bytes());
            Ok(())
        }
        _ => Err(Error::CellCount(cells)),
    }
}

fn parse(blob: &[u8]) -> Result<Parsed, Error> {
    let header = parse_header(blob)?;
    let data = blob
        .get(..header.totalsize)
        .ok_or(Error::Malformed("totalsize exceeds blob"))?;
    let reservations = parse_reservations(data, header.off_mem_rsvmap)?;
    let structs = block(
        data,
        header.off_dt_struct,
        header.size_dt_struct,
        "structure block lies outside totalsize",
    )?;
    let strings = block(
        data,
        header.off_dt_strings,
        header.size_dt_strings,
        "strings block lies outside totalsize",
    )?;
    if header.off_dt_struct % 4 != 0 {
        return Err(Error::Malformed("structure block is not 4-byte aligned"));
    }
    let root = parse_struct(structs, strings)?;
    Ok(Parsed {
        reservations,
        boot_cpuid_phys: header.boot_cpuid_phys,
        root,
    })
}

struct Header {
    totalsize: usize,
    off_dt_struct: u32,
    off_dt_strings: u32,
    off_mem_rsvmap: u32,
    size_dt_strings: u32,
    size_dt_struct: u32,
    boot_cpuid_phys: u32,
}

fn parse_header(blob: &[u8]) -> Result<Header, Error> {
    if blob.len() < HEADER_LEN {
        return Err(Error::Truncated {
            len: blob.len(),
            needed: HEADER_LEN,
        });
    }
    let word = |at: usize| read_u32(blob, at).ok_or(Error::Malformed("header is truncated"));
    let magic = word(0)?;
    if magic != MAGIC {
        return Err(Error::BadMagic(magic));
    }
    let totalsize = word(4)? as usize;
    let version = word(20)?;
    if version < VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    if totalsize < HEADER_LEN {
        return Err(Error::Malformed("totalsize is smaller than the header"));
    }
    if totalsize > blob.len() {
        return Err(Error::Truncated {
            len: blob.len(),
            needed: totalsize,
        });
    }
    Ok(Header {
        totalsize,
        off_dt_struct: word(8)?,
        off_dt_strings: word(12)?,
        off_mem_rsvmap: word(16)?,
        boot_cpuid_phys: word(28)?,
        size_dt_strings: word(32)?,
        size_dt_struct: word(36)?,
    })
}

/// Reads (address, size) pairs up to the all-zero terminator.
fn parse_reservations(data: &[u8], off: u32) -> Result<Vec<(u64, u64)>, Error> {
    const PAST_END: Error = Error::Malformed("memory reservation block runs past totalsize");
    let mut entries = Vec::new();
    let mut pos = off as usize;
    loop {
        let address = read_u64(data, pos).ok_or(PAST_END)?;
        let size = read_u64(data, pos + 8).ok_or(PAST_END)?;
        pos += RESERVE_ENTRY_LEN;
        if address == 0 && size == 0 {
            return Ok(entries);
        }
        entries.push((address, size));
    }
}

fn parse_struct(structs: &[u8], strings: &[u8]) -> Result<Node, Error> {
    let mut stack: Vec<Node> = Vec::new();
    let mut root: Option<Node> = None;
    let mut pos = 0usize;
    loop {
        let token = read_u32(structs, pos).ok_or(TRUNCATED_STRUCT)?;
        pos += 4;
        match token {
            FDT_BEGIN_NODE => {
                if stack.is_empty() && root.is_some() {
                    return Err(Error::Malformed("more than one root node"));
                }
                let (name, next) = read_padded_name(structs, pos)?;
                pos = next;
                stack.push(Node {
                    name,
                    props: Vec::new(),
                    children: Vec::new(),
                });
            }
            FDT_END_NODE => {
                let node = stack
                    .pop()
                    .ok_or(Error::Malformed("FDT_END_NODE with no open node"))?;
                match stack.last_mut() {
                    Some(parent) => parent.children.push(node),
                    None => root = Some(node),
                }
            }
            FDT_PROP => {
                let len = read_u32(structs, pos).ok_or(TRUNCATED_STRUCT)? as usize;
                let name_off = read_u32(structs, pos + 4).ok_or(TRUNCATED_STRUCT)?;
                pos += 8;
                let end = pos.checked_add(len).ok_or(PROP_PAST_STRUCT)?;
                let value = structs.get(pos..end).ok_or(PROP_PAST_STRUCT)?.to_vec();
                pos = align4(end);
                if pos > structs.len() {
                    return Err(PROP_PAST_STRUCT);
                }
                let name = string_at(strings, name_off)?;
                let parent = stack
                    .last_mut()
                    .ok_or(Error::Malformed("property outside any node"))?;
                parent.props.push(Prop { name, value });
            }
            FDT_NOP => {}
            FDT_END => {
                if !stack.is_empty() {
                    return Err(Error::Malformed("FDT_END inside an open node"));
                }
                return root.ok_or(Error::Malformed("no root node before FDT_END"));
            }
            _ => return Err(Error::Malformed("unknown structure token")),
        }
    }
}

/// Reads a NUL-terminated node name at `at`; returns it with the next token offset.
fn read_padded_name(structs: &[u8], at: usize) -> Result<(Vec<u8>, usize), Error> {
    let rest = structs
        .get(at..)
        .ok_or(Error::Malformed("node name starts outside structure block"))?;
    let nul = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or(Error::Malformed("unterminated node name"))?;
    let next = align4(at + nul + 1);
    if next > structs.len() {
        return Err(Error::Malformed(
            "node name padding runs past structure block",
        ));
    }
    Ok((rest[..nul].to_vec(), next))
}

fn string_at(strings: &[u8], at: u32) -> Result<Vec<u8>, Error> {
    let rest = strings.get(at as usize..).ok_or(Error::Malformed(
        "property name offset outside strings block",
    ))?;
    let nul = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or(Error::Malformed("unterminated property name"))?;
    Ok(rest[..nul].to_vec())
}

/// Serializes `root` into a version 17 blob. Property names are interned into the
/// strings block, so each distinct name is stored once.
fn serialize(
    root: &Node,
    reservations: &[(u64, u64)],
    boot_cpuid_phys: u32,
) -> Result<Vec<u8>, Error> {
    let mut writer = Writer {
        structs: Vec::new(),
        strings: Vec::new(),
        names: Vec::new(),
    };
    writer.node(root)?;
    push_u32(&mut writer.structs, FDT_END);

    let off_mem_rsvmap = HEADER_LEN;
    let rsvmap_len = (reservations.len() + 1) * RESERVE_ENTRY_LEN;
    let off_dt_struct = off_mem_rsvmap + rsvmap_len;
    let off_dt_strings = off_dt_struct + writer.structs.len();
    let totalsize = off_dt_strings + writer.strings.len();

    let to_u32 =
        |value: usize| u32::try_from(value).map_err(|_| Error::Malformed("blob exceeds 4 GiB"));
    let mut out = Vec::with_capacity(totalsize);
    push_u32(&mut out, MAGIC);
    push_u32(&mut out, to_u32(totalsize)?);
    push_u32(&mut out, to_u32(off_dt_struct)?);
    push_u32(&mut out, to_u32(off_dt_strings)?);
    push_u32(&mut out, to_u32(off_mem_rsvmap)?);
    push_u32(&mut out, VERSION);
    push_u32(&mut out, LAST_COMP_VERSION);
    push_u32(&mut out, boot_cpuid_phys);
    push_u32(&mut out, to_u32(writer.strings.len())?);
    push_u32(&mut out, to_u32(writer.structs.len())?);
    for &(address, size) in reservations {
        out.extend_from_slice(&address.to_be_bytes());
        out.extend_from_slice(&size.to_be_bytes());
    }
    out.extend_from_slice(&[0; RESERVE_ENTRY_LEN]);
    out.extend_from_slice(&writer.structs);
    out.extend_from_slice(&writer.strings);
    Ok(out)
}

struct Writer {
    structs: Vec<u8>,
    strings: Vec<u8>,
    names: Vec<(Vec<u8>, u32)>,
}

impl Writer {
    fn name_offset(&mut self, name: &[u8]) -> Result<u32, Error> {
        if let Some((_, offset)) = self.names.iter().find(|(n, _)| n.as_slice() == name) {
            return Ok(*offset);
        }
        let offset = u32::try_from(self.strings.len())
            .map_err(|_| Error::Malformed("strings block exceeds 4 GiB"))?;
        self.strings.extend_from_slice(name);
        self.strings.push(0);
        self.names.push((name.to_vec(), offset));
        Ok(offset)
    }

    fn node(&mut self, node: &Node) -> Result<(), Error> {
        push_u32(&mut self.structs, FDT_BEGIN_NODE);
        self.structs.extend_from_slice(&node.name);
        self.structs.push(0);
        pad4(&mut self.structs);
        for prop in &node.props {
            let name_off = self.name_offset(&prop.name)?;
            let len = u32::try_from(prop.value.len())
                .map_err(|_| Error::Malformed("property value exceeds 4 GiB"))?;
            push_u32(&mut self.structs, FDT_PROP);
            push_u32(&mut self.structs, len);
            push_u32(&mut self.structs, name_off);
            self.structs.extend_from_slice(&prop.value);
            pad4(&mut self.structs);
        }
        for child in &node.children {
            self.node(child)?;
        }
        push_u32(&mut self.structs, FDT_END_NODE);
        Ok(())
    }
}

fn block<'a>(data: &'a [u8], off: u32, size: u32, what: &'static str) -> Result<&'a [u8], Error> {
    let start = off as usize;
    let end = start
        .checked_add(size as usize)
        .ok_or(Error::Malformed(what))?;
    data.get(start..end).ok_or(Error::Malformed(what))
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let word: [u8; 4] = bytes.get(at..end)?.try_into().ok()?;
    Some(u32::from_be_bytes(word))
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    let word: [u8; 8] = bytes.get(at..end)?.try_into().ok()?;
    Some(u64::from_be_bytes(word))
}

fn align4(value: usize) -> usize {
    (value + 3) & !3
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn pad4(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn prop(name: &str, value: &[u8]) -> Prop {
        Prop {
            name: name.as_bytes().to_vec(),
            value: value.to_vec(),
        }
    }

    fn cstr(s: &str) -> Vec<u8> {
        let mut v = s.as_bytes().to_vec();
        v.push(0);
        v
    }

    fn be32(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_be_bytes()).collect()
    }

    fn node(name: &str, props: Vec<Prop>, children: Vec<Node>) -> Node {
        Node {
            name: name.as_bytes().to_vec(),
            props,
            children,
        }
    }

    /// Root with a memory-controller (not a memory device), memory, chosen and a serial.
    fn fixture_with_cells(address_cells: u32, size_cells: u32) -> Node {
        let mut root = Node::named(b"");
        root.props = vec![
            prop("#address-cells", &be32(&[address_cells])),
            prop("#size-cells", &be32(&[size_cells])),
            prop("compatible", &cstr("apple,j274")),
        ];
        root.children = vec![
            node(
                "memory-controller@20e000000",
                vec![prop("compatible", &cstr("apple,memory-controller"))],
                vec![],
            ),
            node(
                "memory@800000000",
                vec![
                    prop("device_type", &cstr("memory")),
                    prop("reg", &be32(&[8, 0, 2, 0])),
                ],
                vec![],
            ),
            node(
                "chosen",
                vec![prop("stdout-path", &cstr("/serial@235200000"))],
                vec![],
            ),
            node(
                "serial@235200000",
                vec![
                    prop("compatible", &cstr("ns16550")),
                    prop("reg", &be32(&[2, 0x3520_0000, 0, 0x4000])),
                ],
                vec![],
            ),
        ];
        root
    }

    fn fixture() -> Node {
        fixture_with_cells(2, 2)
    }

    fn blob_of(root: &Node) -> Vec<u8> {
        serialize(root, &[], 0).unwrap()
    }

    fn tree_of(blob: &[u8]) -> Node {
        parse(blob).unwrap().root
    }

    fn child<'a>(node: &'a Node, name: &str) -> &'a Node {
        node.children
            .iter()
            .find(|c| c.name == name.as_bytes())
            .unwrap()
    }

    fn prop_of<'a>(node: &'a Node, name: &str) -> &'a [u8] {
        prop_value(node, name.as_bytes()).unwrap()
    }

    fn set_u32(blob: &mut [u8], at: usize, value: u32) {
        blob[at..at + 4].copy_from_slice(&value.to_be_bytes());
    }

    #[test]
    fn round_trip_keeps_tree_order_and_reservations() {
        let original = fixture();
        let reservations = [(0x4000_0000u64, 0x0010_0000u64)];
        let blob = serialize(&original, &reservations, 0).unwrap();
        let out = patch(&blob, None, None).unwrap();
        let parsed = parse(&out).unwrap();
        assert_eq!(parsed.root, original);
        assert_eq!(parsed.reservations, reservations);
    }

    #[test]
    fn bootargs_is_appended_to_chosen_when_absent() {
        let out = patch(
            &blob_of(&fixture()),
            Some("console=ttyS0 root=/dev/vda"),
            None,
        )
        .unwrap();
        let out = tree_of(&out);
        let chosen = child(&out, "chosen");
        let names: Vec<&[u8]> = chosen.props.iter().map(|p| p.name.as_slice()).collect();
        assert_eq!(names, [&b"stdout-path"[..], &b"bootargs"[..]]);
        assert_eq!(
            prop_of(chosen, "bootargs"),
            b"console=ttyS0 root=/dev/vda\0"
        );
    }

    #[test]
    fn bootargs_replaces_existing_property_in_place() {
        let mut root = fixture();
        let chosen = root
            .children
            .iter_mut()
            .find(|c| c.name == b"chosen")
            .unwrap();
        chosen.props.insert(0, prop("bootargs", &cstr("old")));
        let out = tree_of(&patch(&blob_of(&root), Some("new args"), None).unwrap());
        let chosen = child(&out, "chosen");
        let names: Vec<&[u8]> = chosen.props.iter().map(|p| p.name.as_slice()).collect();
        assert_eq!(names, [&b"bootargs"[..], &b"stdout-path"[..]]);
        assert_eq!(prop_of(chosen, "bootargs"), b"new args\0");
    }

    #[test]
    fn chosen_is_created_at_root_when_absent() {
        let mut root = fixture();
        root.children.retain(|c| c.name != b"chosen");
        let out = tree_of(&patch(&blob_of(&root), Some("quiet"), None).unwrap());
        assert_eq!(out.children.len(), root.children.len() + 1);
        assert_eq!(&out.children[..root.children.len()], &root.children[..]);
        let chosen = out.children.last().unwrap();
        assert_eq!(chosen.name, b"chosen");
        assert_eq!(chosen.props, vec![prop("bootargs", b"quiet\0")]);
    }

    #[test]
    fn memory_reg_uses_two_two_cells() {
        let out = patch(
            &blob_of(&fixture()),
            None,
            Some((0x1_4000_0000, 0x2_0000_0000)),
        )
        .unwrap();
        let out = tree_of(&out);
        let mem = child(&out, "memory@800000000");
        assert_eq!(prop_of(mem, "reg"), be32(&[1, 0x4000_0000, 2, 0]));
    }

    #[test]
    fn memory_reg_uses_one_one_cells() {
        let out = patch(
            &blob_of(&fixture_with_cells(1, 1)),
            None,
            Some((0x4000_0000, 0x0800_0000)),
        )
        .unwrap();
        let out = tree_of(&out);
        let mem = child(&out, "memory@800000000");
        assert_eq!(prop_of(mem, "reg"), be32(&[0x4000_0000, 0x0800_0000]));
    }

    #[test]
    fn patch_touches_only_chosen_and_memory_reg() {
        let original = fixture();
        let out = tree_of(
            &patch(
                &blob_of(&original),
                Some("quiet"),
                Some((0x1_4000_0000, 0x2_0000_0000)),
            )
            .unwrap(),
        );
        assert_eq!(out.props, original.props);
        assert_eq!(out.name, original.name);
        assert_eq!(
            child(&out, "memory-controller@20e000000"),
            child(&original, "memory-controller@20e000000")
        );
        assert_eq!(
            child(&out, "serial@235200000"),
            child(&original, "serial@235200000")
        );
        assert_eq!(out.children.len(), original.children.len());
        assert_eq!(
            prop_of(child(&out, "memory@800000000"), "device_type"),
            b"memory\0"
        );
        assert_eq!(
            prop_of(child(&out, "chosen"), "stdout-path"),
            b"/serial@235200000\0"
        );
    }

    #[test]
    fn memory_value_too_large_for_one_cell_is_rejected() {
        let blob = blob_of(&fixture_with_cells(1, 1));
        assert_eq!(
            patch(&blob, None, Some((1 << 32, 0x1000))),
            Err(Error::ValueTooLarge {
                value: 1 << 32,
                cells: 1
            })
        );
    }

    #[test]
    fn unsupported_cell_count_is_rejected() {
        let blob = blob_of(&fixture_with_cells(3, 2));
        assert_eq!(
            patch(&blob, None, Some((0x4000_0000, 0x1000))),
            Err(Error::CellCount(3))
        );
    }

    #[test]
    fn memory_patch_needs_a_memory_device_node() {
        let mut root = fixture();
        root.children.retain(|c| !c.name.starts_with(b"memory@"));
        assert_eq!(
            patch(&blob_of(&root), None, Some((0x4000_0000, 0x1000))),
            Err(Error::NoMemoryNode)
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut blob = blob_of(&fixture());
        blob[0] ^= 0xff;
        assert!(matches!(patch(&blob, None, None), Err(Error::BadMagic(_))));
    }

    #[test]
    fn truncated_blob_is_rejected() {
        let blob = blob_of(&fixture());
        assert!(matches!(
            patch(&blob[..blob.len() - 1], None, None),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            patch(&blob[..HEADER_LEN - 1], None, None),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn offsets_past_end_are_rejected() {
        let blob = blob_of(&fixture());
        let total = blob.len() as u32;

        let mut bad_struct = blob.clone();
        set_u32(&mut bad_struct, 8, total + 4);
        assert!(matches!(
            patch(&bad_struct, None, None),
            Err(Error::Malformed(_))
        ));

        let mut bad_strings = blob;
        set_u32(&mut bad_strings, 12, u32::MAX);
        assert!(matches!(
            patch(&bad_strings, None, None),
            Err(Error::Malformed(_))
        ));
    }
}

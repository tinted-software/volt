//! Target-neutral machine intermediate representation.
//!
//! A [`MachineFunction`] is a linear instruction list plus a table of values.
//! Each value occupies a contiguous block of 32-bit registers in one register
//! class. Fixed blocks represent registers assigned by the hardware ABI.

use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RegisterClass {
    /// Uniform 32-bit scalar registers.
    Scalar,
    /// Per-lane 32-bit vector registers.
    Vector,
    /// Hardware predicate registers, such as VCC_LO.
    Predicate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Operand {
    Value {
        value: ValueId,
        element: u8,
    },
    Immediate(u32),
    Physical {
        register_class: RegisterClass,
        register: u8,
    },
}

impl Operand {
    pub fn value(self) -> Option<ValueId> {
        match self {
            Self::Value { value, .. } => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatBinaryOperation {
    Add,
    Subtract,
    Multiply,
}

/// Floating comparison predicates. "Unordered" predicates also match NaNs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatComparisonKind {
    OrderedLessThan,
    OrderedGreaterThan,
    OrderedLessThanOrEqual,
    OrderedGreaterThanOrEqual,
    OrderedEqual,
    OrderedNotEqual,
    UnorderedNotEqual,
    UnorderedLessThanOrEqual,
    UnorderedLessThan,
    UnorderedGreaterThanOrEqual,
    UnorderedGreaterThan,
    UnorderedEqual,
    AlwaysTrue,
    AlwaysFalse,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValuePurpose {
    Parameter(u32),
    InterpolationCoordinates,
    Temporary,
}

#[derive(Clone, Copy, Debug)]
pub struct ValueDescription {
    pub register_class: RegisterClass,
    /// Length of this value's contiguous 32-bit register block.
    pub register_count: u8,
    /// Physical base register assigned by the target ABI, if any.
    pub fixed_register: Option<u8>,
    pub purpose: ValuePurpose,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Constant {
        destination: ValueId,
        bits: u32,
    },
    Copy {
        destination: ValueId,
        source: Operand,
    },
    FloatBinary {
        operation: FloatBinaryOperation,
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    /// IEEE-correct division, expanded by the target before allocation.
    FloatDivide {
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    FusedMultiplyAdd {
        destination: ValueId,
        first: Operand,
        second: Operand,
        third: Operand,
        negate_first: bool,
    },
    /// The destination is both read and overwritten by this instruction.
    FusedMultiplyAccumulate {
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    Reciprocal {
        destination: ValueId,
        source: Operand,
    },
    /// Exponential pseudo-instruction, expanded by the target.
    ExponentialBaseTwo {
        destination: ValueId,
        source: Operand,
    },
    /// Single hardware exponential instruction after target expansion.
    HardwareExponentialBaseTwo {
        destination: ValueId,
        source: Operand,
    },
    LoadExponent {
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    DivideScale {
        destination: ValueId,
        predicate: Option<ValueId>,
        first: Operand,
        second: Operand,
        third: Operand,
    },
    /// Reads the carry predicate produced by the preceding divide scale.
    DivideFusedMultiplyAdd {
        destination: ValueId,
        first: Operand,
        second: Operand,
        third: Operand,
        predicate: Operand,
    },
    DivideFixup {
        destination: ValueId,
        first: Operand,
        second: Operand,
        third: Operand,
    },
    FloatCompare {
        comparison: FloatComparisonKind,
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    Select {
        destination: ValueId,
        condition: Operand,
        when_true: Operand,
        when_false: Operand,
    },
    IntegerMultiply {
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    /// Sign-extending arithmetic shift right of a 32-bit value (`>> 31`
    /// semantics: 0 for non-negative, -1 for negative).
    IntegerShiftRight32 {
        destination: ValueId,
        first: Operand,
        second: Operand,
    },
    /// Define one element of a 64-bit address and optionally its carry predicate.
    AddressLow {
        destination: ValueId,
        destination_element: u8,
        carry: Option<ValueId>,
        first: Operand,
        second: Operand,
    },
    AddressHigh {
        destination: ValueId,
        destination_element: u8,
        first: Operand,
        second: Operand,
        carry: Operand,
    },
    LoadGlobal {
        destination: ValueId,
        register_count: u8,
        address: Operand,
        offset: u32,
    },
    StoreGlobal {
        register_count: u8,
        address: Operand,
        offset: u32,
        source: Operand,
    },
    /// The second interpolation step reads the result of the first step.
    Interpolate {
        second_step: bool,
        destination: ValueId,
        coordinates: Operand,
        previous_value: Option<Operand>,
        /// ABI-supplied primitive mask copied into M0 before interpolation.
        primitive_mask: Operand,
        attribute: u32,
        channel: u32,
    },
    Export {
        target: u32,
        values: [Operand; 4],
        mask: u8,
        done: bool,
        valid_mask: bool,
    },
    Kill {
        condition: Operand,
    },
    WaitForLoads,
    ZeroExecutionMask,
    AndExecutionMaskWithPredicate {
        predicate: Operand,
    },
    BranchIfNoActiveLanes {
        label: u32,
    },
    Branch {
        label: u32,
    },
    BranchIfPredicate {
        predicate: Operand,
        label: u32,
    },
    Label {
        id: u32,
    },
    Return,
}

#[derive(Clone, Debug, Default)]
pub struct MachineFunction {
    pub values: Vec<ValueDescription>,
    pub instructions: Vec<Instruction>,
}

impl MachineFunction {
    pub fn add_value(
        &mut self,
        register_class: RegisterClass,
        register_count: u8,
        fixed_register: Option<u8>,
        purpose: ValuePurpose,
    ) -> ValueId {
        let value = ValueId(self.values.len() as u32);
        self.values.push(ValueDescription {
            register_class,
            register_count,
            fixed_register,
            purpose,
        });
        value
    }

    pub fn description(&self, value: ValueId) -> &ValueDescription {
        &self.values[value.index()]
    }

    /// All value elements defined by an instruction, including secondary
    /// predicates and carries. A global load defines its entire register block.
    pub fn definitions(instruction: &Instruction) -> Vec<(ValueId, u8)> {
        let mut definitions = Vec::new();
        Self::visit_definitions(instruction, |value, element| {
            definitions.push((value, element))
        });
        definitions
    }

    /// Visit definitions without allocating a temporary list.
    pub fn visit_definitions(instruction: &Instruction, mut visit: impl FnMut(ValueId, u8)) {
        match instruction {
            Instruction::Constant { destination, .. }
            | Instruction::Copy { destination, .. }
            | Instruction::FloatBinary { destination, .. }
            | Instruction::FloatDivide { destination, .. }
            | Instruction::FusedMultiplyAdd { destination, .. }
            | Instruction::FusedMultiplyAccumulate { destination, .. }
            | Instruction::Reciprocal { destination, .. }
            | Instruction::ExponentialBaseTwo { destination, .. }
            | Instruction::HardwareExponentialBaseTwo { destination, .. }
            | Instruction::LoadExponent { destination, .. }
            | Instruction::DivideFusedMultiplyAdd { destination, .. }
            | Instruction::DivideFixup { destination, .. }
            | Instruction::FloatCompare { destination, .. }
            | Instruction::Select { destination, .. }
            | Instruction::IntegerMultiply { destination, .. }
            | Instruction::IntegerShiftRight32 { destination, .. }
            | Instruction::Interpolate { destination, .. } => visit(*destination, 0),
            Instruction::DivideScale {
                destination,
                predicate,
                ..
            } => {
                visit(*destination, 0);
                if let Some(predicate) = predicate {
                    visit(*predicate, 0);
                }
            }
            Instruction::AddressLow {
                destination,
                destination_element,
                carry,
                ..
            } => {
                visit(*destination, *destination_element);
                if let Some(carry) = carry {
                    visit(*carry, 0);
                }
            }
            Instruction::AddressHigh {
                destination,
                destination_element,
                ..
            } => {
                visit(*destination, *destination_element);
            }
            Instruction::LoadGlobal {
                destination,
                register_count,
                ..
            } => {
                for element in 0..*register_count {
                    visit(*destination, element);
                }
            }
            Instruction::StoreGlobal { .. }
            | Instruction::Export { .. }
            | Instruction::Kill { .. }
            | Instruction::WaitForLoads
            | Instruction::ZeroExecutionMask
            | Instruction::AndExecutionMaskWithPredicate { .. }
            | Instruction::BranchIfNoActiveLanes { .. }
            | Instruction::Branch { .. }
            | Instruction::BranchIfPredicate { .. }
            | Instruction::Label { .. }
            | Instruction::Return => {}
        }
    }

    /// Register operands read by an instruction, excluding immediates. A tied
    /// accumulator is read before its destination is overwritten.
    pub fn uses(instruction: &Instruction) -> Vec<Operand> {
        let mut uses = Vec::new();
        Self::visit_uses(instruction, |operand| uses.push(operand));
        uses
    }

    /// Visit register uses without allocating a temporary list.
    pub fn visit_uses(instruction: &Instruction, mut visit: impl FnMut(Operand)) {
        let mut add = |operand: Operand| {
            if !matches!(operand, Operand::Immediate(_)) {
                visit(operand);
            }
        };
        match instruction {
            Instruction::Constant { .. } => {}
            Instruction::Copy { source, .. }
            | Instruction::Reciprocal { source, .. }
            | Instruction::ExponentialBaseTwo { source, .. }
            | Instruction::HardwareExponentialBaseTwo { source, .. } => add(*source),
            Instruction::FloatBinary { first, second, .. }
            | Instruction::FloatDivide { first, second, .. }
            | Instruction::LoadExponent { first, second, .. }
            | Instruction::FloatCompare { first, second, .. }
            | Instruction::IntegerMultiply { first, second, .. }
            | Instruction::IntegerShiftRight32 { first, second, .. }
            | Instruction::AddressLow { first, second, .. } => {
                add(*first);
                add(*second);
            }
            Instruction::FusedMultiplyAdd {
                first,
                second,
                third,
                ..
            }
            | Instruction::DivideScale {
                first,
                second,
                third,
                ..
            }
            | Instruction::DivideFixup {
                first,
                second,
                third,
                ..
            } => {
                add(*first);
                add(*second);
                add(*third);
            }
            Instruction::DivideFusedMultiplyAdd {
                first,
                second,
                third,
                predicate,
                ..
            } => {
                add(*first);
                add(*second);
                add(*third);
                add(*predicate);
            }
            Instruction::FusedMultiplyAccumulate {
                destination,
                first,
                second,
            } => {
                add(Operand::Value {
                    value: *destination,
                    element: 0,
                });
                add(*first);
                add(*second);
            }
            Instruction::Select {
                condition,
                when_true,
                when_false,
                ..
            } => {
                add(*condition);
                add(*when_true);
                add(*when_false);
            }
            Instruction::AddressHigh {
                first,
                second,
                carry,
                ..
            } => {
                add(*first);
                add(*second);
                add(*carry);
            }
            Instruction::LoadGlobal { address, .. } => add(*address),
            Instruction::StoreGlobal {
                address, source, ..
            } => {
                add(*address);
                add(*source);
            }
            Instruction::Interpolate {
                coordinates,
                previous_value,
                primitive_mask,
                ..
            } => {
                add(*coordinates);
                add(*primitive_mask);
                if let Some(previous_value) = previous_value {
                    add(*previous_value);
                }
            }
            Instruction::Export { values, mask, .. } => {
                for (channel, value) in values.iter().enumerate() {
                    if mask & (1 << channel) != 0 {
                        add(*value);
                    }
                }
            }
            Instruction::Kill { condition } => add(*condition),
            Instruction::AndExecutionMaskWithPredicate { predicate }
            | Instruction::BranchIfPredicate { predicate, .. } => add(*predicate),
            Instruction::WaitForLoads
            | Instruction::ZeroExecutionMask
            | Instruction::BranchIfNoActiveLanes { .. }
            | Instruction::Branch { .. }
            | Instruction::Label { .. }
            | Instruction::Return => {}
        }
    }
}

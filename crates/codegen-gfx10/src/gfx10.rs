//! GFX10.3 (RDNA2, wave32) ABI and machine-code encoder.
//! Export operands are read asynchronously; emission waits before reusing their
//! VGPRs or changing EXEC, and drains exports at control-flow boundaries.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use volt_codegen::machine::{
    FloatBinaryOperation, FloatComparisonKind, Instruction, MachineFunction, Operand,
    RegisterClass, ValueId, ValuePurpose,
};
use volt_codegen::targets::{
    Allocation, InterpolationMode, ParameterLocation, ShaderStage, Signature, Summary, Target,
};

#[derive(Debug, PartialEq, Eq)]
pub enum Gfx10Error {
    Unsupported(&'static str),
    MultipleLiterals,
    ImmediateRange,
    Malformed(&'static str),
}

impl core::fmt::Display for Gfx10Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported GFX10 construct: {msg}"),
            Self::MultipleLiterals => {
                f.write_str("instruction requires multiple literal constants")
            }
            Self::ImmediateRange => f.write_str("immediate out of range"),
            Self::Malformed(msg) => write!(f, "malformed machine instruction: {msg}"),
        }
    }
}
pub const VCC_LO: u8 = 106;
pub const M0: u8 = 124;
pub const NULL: u8 = 125;
pub const EXEC_LO: u8 = 126;
pub const EXP_NULL: u32 = 9;
pub const EXP_POS0: u32 = 12;

const SPI_SHADER_PGM_RSRC1_VS: u32 = 0xb128;
const SPI_SHADER_PGM_RSRC1_PS: u32 = 0xb028;
const SPI_SHADER_PGM_RSRC2_PS: u32 = 0xb02c;
const SPI_TMPRING_SIZE: u32 = 0x286e8;
const SPI_PS_INPUT_ENA: u32 = 0x286cc;
const SPI_PS_INPUT_ADDR: u32 = 0x286d0;

#[derive(Clone, Copy, Debug)]
pub struct Gfx10 {
    processor_name: &'static str,
}

pub fn gfx103_processor(ip_discovery_version: u32) -> &'static str {
    match ip_discovery_version & 0xff {
        0 => "gfx1030",
        1 => "gfx1033",
        2 => "gfx1031",
        3 => "gfx1035",
        4 => "gfx1032",
        5 => "gfx1034",
        6 => "gfx1036",
        7 => "gfx1037",
        _ => "gfx1030",
    }
}

pub fn for_processor(processor: &str) -> Option<Gfx10> {
    let processor_name = match processor {
        "gfx1030" => "gfx1030",
        "gfx1031" => "gfx1031",
        "gfx1032" => "gfx1032",
        "gfx1033" => "gfx1033",
        "gfx1034" => "gfx1034",
        "gfx1035" => "gfx1035",
        "gfx1036" => "gfx1036",
        "gfx1037" => "gfx1037",
        _ => return None,
    };
    Some(Gfx10 { processor_name })
}

fn value_operand(value: ValueId) -> Operand {
    Operand::Value { value, element: 0 }
}

fn float_operand(value: f32) -> Operand {
    Operand::Immediate(value.to_bits())
}

impl Gfx10 {
    fn source(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        operand: Operand,
    ) -> Result<Source, Gfx10Error> {
        match operand {
            Operand::Immediate(bits) => Ok(Source::Immediate(bits)),
            Operand::Physical {
                register_class,
                register,
            } => Ok(match register_class {
                RegisterClass::Vector => Source::Vector(register),
                RegisterClass::Scalar | RegisterClass::Predicate => Source::Scalar(register),
            }),
            Operand::Value { value, element } => {
                let description = function
                    .values
                    .get(value.index())
                    .ok_or(Gfx10Error::Malformed("unknown value"))?;
                if element >= description.register_count {
                    return Err(Gfx10Error::Malformed("value element out of range"));
                }
                let register = *allocation
                    .bases
                    .get(value.index())
                    .ok_or(Gfx10Error::Malformed("unallocated value"))?;
                let register = register
                    .checked_add(element)
                    .ok_or(Gfx10Error::Malformed("register block overflows"))?;
                Ok(match description.register_class {
                    RegisterClass::Vector => Source::Vector(register),
                    RegisterClass::Scalar | RegisterClass::Predicate => Source::Scalar(register),
                })
            }
        }
    }

    fn vector(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        operand: Operand,
    ) -> Result<u8, Gfx10Error> {
        match self.source(function, allocation, operand)? {
            Source::Vector(register) => Ok(register),
            _ => Err(Gfx10Error::Unsupported("expected a vector register")),
        }
    }

    fn scalar(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        operand: Operand,
    ) -> Result<u8, Gfx10Error> {
        match self.source(function, allocation, operand)? {
            Source::Scalar(register) => Ok(register),
            _ => Err(Gfx10Error::Unsupported(
                "expected a scalar or predicate register",
            )),
        }
    }

    fn destination(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        value: ValueId,
        element: u8,
        register_class: RegisterClass,
    ) -> Result<u8, Gfx10Error> {
        let description = function
            .values
            .get(value.index())
            .ok_or(Gfx10Error::Malformed("unknown destination"))?;
        if description.register_class != register_class || element >= description.register_count {
            return Err(Gfx10Error::Unsupported(
                "destination register class or element",
            ));
        }
        let register = *allocation
            .bases
            .get(value.index())
            .ok_or(Gfx10Error::Malformed("unallocated destination"))?;
        register
            .checked_add(element)
            .ok_or(Gfx10Error::ImmediateRange)
    }

    fn vector_destination(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        value: ValueId,
    ) -> Result<u8, Gfx10Error> {
        self.destination(function, allocation, value, 0, RegisterClass::Vector)
    }

    fn predicate_destination(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        value: ValueId,
    ) -> Result<u8, Gfx10Error> {
        let register =
            self.destination(function, allocation, value, 0, RegisterClass::Predicate)?;
        if register != VCC_LO {
            return Err(Gfx10Error::Unsupported("predicate must occupy vcc_lo"));
        }
        Ok(register)
    }

    fn expand_division(
        &self,
        function: &mut MachineFunction,
        output: &mut Vec<Instruction>,
        destination: ValueId,
        numerator: Operand,
        denominator: Operand,
    ) {
        let mut temporary =
            || function.add_value(RegisterClass::Vector, 1, None, ValuePurpose::Temporary);
        let scaled_denominator = temporary();
        let scaled_numerator = temporary();
        let reciprocal = temporary();
        let error = temporary();
        let quotient = temporary();
        let second_error = temporary();
        let final_error = temporary();
        let predicate = function.add_value(
            RegisterClass::Predicate,
            1,
            Some(VCC_LO),
            ValuePurpose::Temporary,
        );
        output.push(Instruction::DivideScale {
            destination: scaled_denominator,
            predicate: None,
            first: denominator,
            second: denominator,
            third: numerator,
        });
        output.push(Instruction::Reciprocal {
            destination: reciprocal,
            source: value_operand(scaled_denominator),
        });
        output.push(Instruction::FusedMultiplyAdd {
            destination: error,
            first: value_operand(scaled_denominator),
            second: value_operand(reciprocal),
            third: float_operand(1.0),
            negate_first: true,
        });
        output.push(Instruction::FusedMultiplyAccumulate {
            destination: reciprocal,
            first: value_operand(error),
            second: value_operand(reciprocal),
        });
        output.push(Instruction::DivideScale {
            destination: scaled_numerator,
            predicate: Some(predicate),
            first: numerator,
            second: denominator,
            third: numerator,
        });
        output.push(Instruction::FloatBinary {
            operation: FloatBinaryOperation::Multiply,
            destination: quotient,
            first: value_operand(scaled_numerator),
            second: value_operand(reciprocal),
        });
        output.push(Instruction::FusedMultiplyAdd {
            destination: second_error,
            first: value_operand(scaled_denominator),
            second: value_operand(quotient),
            third: value_operand(scaled_numerator),
            negate_first: true,
        });
        output.push(Instruction::FusedMultiplyAccumulate {
            destination: quotient,
            first: value_operand(second_error),
            second: value_operand(reciprocal),
        });
        output.push(Instruction::FusedMultiplyAdd {
            destination: final_error,
            first: value_operand(scaled_denominator),
            second: value_operand(quotient),
            third: value_operand(scaled_numerator),
            negate_first: true,
        });
        output.push(Instruction::DivideFusedMultiplyAdd {
            destination: final_error,
            first: value_operand(final_error),
            second: value_operand(reciprocal),
            third: value_operand(quotient),
            predicate: value_operand(predicate),
        });
        output.push(Instruction::DivideFixup {
            destination,
            first: value_operand(final_error),
            second: denominator,
            third: numerator,
        });
    }

    fn expand_exponential(
        &self,
        function: &mut MachineFunction,
        output: &mut Vec<Instruction>,
        destination: ValueId,
        source: Operand,
    ) {
        let predicate = function.add_value(
            RegisterClass::Predicate,
            1,
            Some(VCC_LO),
            ValuePurpose::Temporary,
        );
        let scale = function.add_value(RegisterClass::Vector, 1, None, ValuePurpose::Temporary);
        let inverse_scale =
            function.add_value(RegisterClass::Vector, 1, None, ValuePurpose::Temporary);
        output.push(Instruction::FloatCompare {
            comparison: FloatComparisonKind::OrderedGreaterThan,
            destination: predicate,
            first: float_operand(-126.0),
            second: source,
        });
        output.push(Instruction::Select {
            destination: scale,
            condition: value_operand(predicate),
            when_true: float_operand(64.0),
            when_false: Operand::Immediate(0),
        });
        output.push(Instruction::FloatBinary {
            operation: FloatBinaryOperation::Add,
            destination,
            first: source,
            second: value_operand(scale),
        });
        output.push(Instruction::Select {
            destination: inverse_scale,
            condition: value_operand(predicate),
            when_true: Operand::Immediate((-64i32) as u32),
            when_false: Operand::Immediate(0),
        });
        output.push(Instruction::HardwareExponentialBaseTwo {
            destination,
            source: value_operand(destination),
        });
        output.push(Instruction::LoadExponent {
            destination,
            first: value_operand(destination),
            second: value_operand(inverse_scale),
        });
    }
}

impl Target for Gfx10 {
    type Error = Gfx10Error;
    fn processor_name(&self) -> &'static str {
        self.processor_name
    }
    fn wave_size(&self) -> u32 {
        32
    }
    fn maximum_vector_registers(&self) -> u32 {
        256
    }
    fn maximum_scalar_registers(&self) -> u32 {
        106
    }
    fn maximum_predicate_registers(&self) -> u32 {
        1
    }
    fn predicate_register_base(&self) -> u8 {
        VCC_LO
    }

    fn parameter_location(&self, signature: &Signature, index: u32) -> ParameterLocation {
        if index as usize >= signature.parameters.len() {
            return ParameterLocation::Unavailable;
        }
        if index < signature.number_of_scalar_parameters {
            let base = signature
                .parameters
                .iter()
                .take(index as usize)
                .fold(0u32, |sum, parameter| sum + u32::from(parameter.size));
            return u8::try_from(base).map_or(ParameterLocation::Unavailable, |base| {
                ParameterLocation::Fixed(RegisterClass::Scalar, base)
            });
        }
        match signature.stage {
            ShaderStage::Vertex => {
                // The vertex index is an integer, but arrives in v0 just like
                // every other non-inreg VS parameter.
                let base = signature
                    .parameters
                    .iter()
                    .skip(signature.number_of_scalar_parameters as usize)
                    .take((index - signature.number_of_scalar_parameters) as usize)
                    .fold(0u32, |sum, parameter| sum + u32::from(parameter.size));
                u8::try_from(base).map_or(ParameterLocation::Unavailable, |base| {
                    ParameterLocation::Fixed(RegisterClass::Vector, base)
                })
            }
            ShaderStage::Pixel => match index - signature.number_of_scalar_parameters {
                1 => ParameterLocation::Fixed(RegisterClass::Vector, 0),
                _ => ParameterLocation::Unavailable,
            },
        }
    }

    fn interpolation_coordinates(&self, mode: InterpolationMode) -> Option<(u8, u8)> {
        match mode {
            InterpolationMode::Center => Some((0, 1)),
            _ => None,
        }
    }
    fn perspective_mode(&self) -> InterpolationMode {
        InterpolationMode::Center
    }

    fn expand_pseudo_instructions(&self, function: &mut MachineFunction) -> Result<(), Gfx10Error> {
        let discard_label = function
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Label { id } => Some(*id),
                Instruction::Branch { label }
                | Instruction::BranchIfPredicate { label, .. }
                | Instruction::BranchIfNoActiveLanes { label } => Some(*label),
                _ => None,
            })
            .max()
            .map_or(Some(0), |id| id.checked_add(1))
            .ok_or(Gfx10Error::Malformed(
                "no free label for discard completion",
            ))?;
        let instructions = core::mem::take(&mut function.instructions);
        let mut expanded = Vec::with_capacity(instructions.len());
        let mut needs_discard_completion = false;
        for instruction in instructions {
            match instruction {
                Instruction::FloatDivide {
                    destination,
                    first,
                    second,
                } => {
                    self.expand_division(function, &mut expanded, destination, first, second);
                }
                Instruction::ExponentialBaseTwo {
                    destination,
                    source,
                } => {
                    self.expand_exponential(function, &mut expanded, destination, source);
                }
                Instruction::Kill {
                    condition: Operand::Immediate(0),
                }
                | Instruction::ZeroExecutionMask => {
                    expanded.push(Instruction::ZeroExecutionMask);
                    expanded.push(Instruction::BranchIfNoActiveLanes {
                        label: discard_label,
                    });
                    needs_discard_completion = true;
                }
                Instruction::Kill {
                    condition: Operand::Immediate(_),
                } => {}
                Instruction::Kill { condition }
                | Instruction::AndExecutionMaskWithPredicate {
                    predicate: condition,
                } => {
                    expanded.push(Instruction::AndExecutionMaskWithPredicate {
                        predicate: condition,
                    });
                    expanded.push(Instruction::BranchIfNoActiveLanes {
                        label: discard_label,
                    });
                    needs_discard_completion = true;
                }
                other => expanded.push(other),
            }
        }
        if needs_discard_completion {
            if !matches!(expanded.last(), Some(Instruction::Return)) {
                return Err(Gfx10Error::Malformed("discard shader must end with return"));
            }
            expanded.push(Instruction::Label { id: discard_label });
            expanded.push(Instruction::Export {
                target: EXP_NULL,
                values: [Operand::Immediate(0); 4],
                mask: 0,
                done: true,
                valid_mask: true,
            });
            expanded.push(Instruction::Return);
        }
        function.instructions = expanded;
        Ok(())
    }

    fn encode_function(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
    ) -> Result<Vec<u8>, Gfx10Error> {
        let mut encoder = Encoder::new();
        let mut labels = Vec::new();
        let mut branches = Vec::new();
        let mut current_interpolation_mask = None;
        let mut export_registers = [0u64; 4];
        let mut pending_exports = false;
        for instruction in &function.instructions {
            if pending_exports {
                let mut overwrites_export = false;
                let mut error = None;
                MachineFunction::visit_definitions(instruction, |value, element| {
                    match self.source(function, allocation, Operand::Value { value, element }) {
                        Ok(Source::Vector(register)) => {
                            let register = usize::from(register);
                            overwrites_export |=
                                export_registers[register / 64] & (1u64 << (register % 64)) != 0;
                        }
                        Ok(_) => {}
                        Err(failure) => error = Some(failure),
                    }
                });
                if let Some(error) = error {
                    return Err(error);
                }
                // EXP reads both its source VGPRs and EXEC after it is issued.
                // Keep this post-allocation: distinct values can share a VGPR.
                // Draining at branches and joins keeps this local scoreboard
                // correct for every path, including backward edges.
                if overwrites_export
                    || matches!(
                        instruction,
                        Instruction::ZeroExecutionMask
                            | Instruction::AndExecutionMaskWithPredicate { .. }
                            | Instruction::Branch { .. }
                            | Instruction::BranchIfPredicate { .. }
                            | Instruction::BranchIfNoActiveLanes { .. }
                            | Instruction::Label { .. }
                    )
                {
                    encoder.sopp(0x0c, 0xff0fu16 as i16); // s_waitcnt expcnt(0)
                    export_registers.fill(0);
                    pending_exports = false;
                }
            }
            match instruction {
                Instruction::Label { id } => {
                    if labels.iter().any(|(previous, _)| previous == id) {
                        return Err(Gfx10Error::Malformed("duplicate label"));
                    }
                    labels.push((*id, encoder.pc()));
                }
                Instruction::Branch { label } => {
                    branches.push((encoder.pc(), *label));
                    encoder.sopp(2, 0);
                }
                Instruction::BranchIfNoActiveLanes { label } => {
                    branches.push((encoder.pc(), *label));
                    encoder.sopp(8, 0);
                }
                Instruction::BranchIfPredicate { predicate, label } => {
                    let register = self.scalar(function, allocation, *predicate)?;
                    encoder.sop2(
                        0x0e,
                        NULL,
                        Source::Scalar(EXEC_LO),
                        Source::Scalar(register),
                    )?;
                    branches.push((encoder.pc(), *label));
                    encoder.sopp(5, 0);
                }
                Instruction::Constant { destination, bits } => {
                    let description = function.description(*destination);
                    match description.register_class {
                        RegisterClass::Vector => encoder.vop1(
                            1,
                            self.vector_destination(function, allocation, *destination)?,
                            Source::Immediate(*bits),
                        )?,
                        RegisterClass::Scalar => encoder.sop1(
                            3,
                            self.destination(
                                function,
                                allocation,
                                *destination,
                                0,
                                RegisterClass::Scalar,
                            )?,
                            Source::Immediate(*bits),
                        )?,
                        RegisterClass::Predicate => {
                            return Err(Gfx10Error::Unsupported("predicate constant"));
                        }
                    }
                    if description.register_class == RegisterClass::Scalar {
                        current_interpolation_mask = None;
                    }
                }
                Instruction::Copy {
                    destination,
                    source,
                } => {
                    let source = self.source(function, allocation, *source)?;
                    match function.description(*destination).register_class {
                        RegisterClass::Vector => encoder.vop1(
                            1,
                            self.vector_destination(function, allocation, *destination)?,
                            source,
                        )?,
                        RegisterClass::Scalar => encoder.sop1(
                            3,
                            self.destination(
                                function,
                                allocation,
                                *destination,
                                0,
                                RegisterClass::Scalar,
                            )?,
                            source,
                        )?,
                        RegisterClass::Predicate => encoder.sop1(
                            3,
                            self.predicate_destination(function, allocation, *destination)?,
                            source,
                        )?,
                    }
                    if function.description(*destination).register_class == RegisterClass::Scalar {
                        current_interpolation_mask = None;
                    }
                }
                Instruction::FloatBinary {
                    operation,
                    destination,
                    first,
                    second,
                } => {
                    let first = self.source(function, allocation, *first)?;
                    let second = self.source(function, allocation, *second)?;
                    let opcode = match operation {
                        FloatBinaryOperation::Add => 0x103,
                        FloatBinaryOperation::Subtract => 0x104,
                        FloatBinaryOperation::Multiply => 0x108,
                    };
                    encoder.vop3a(
                        opcode,
                        self.vector_destination(function, allocation, *destination)?,
                        [first, second, Source::Scalar(0)],
                        0,
                        0,
                    )?;
                }
                Instruction::FusedMultiplyAdd {
                    destination,
                    first,
                    second,
                    third,
                    negate_first,
                } => {
                    encoder.vop3a(
                        0x14b,
                        self.vector_destination(function, allocation, *destination)?,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            self.source(function, allocation, *third)?,
                        ],
                        0,
                        u8::from(*negate_first),
                    )?;
                }
                Instruction::FusedMultiplyAccumulate {
                    destination,
                    first,
                    second,
                } => {
                    let destination =
                        self.vector_destination(function, allocation, *destination)?;
                    encoder.vop3a(
                        0x12b,
                        destination,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            Source::Scalar(0),
                        ],
                        0,
                        0,
                    )?;
                }
                Instruction::Reciprocal {
                    destination,
                    source,
                } => encoder.vop1(
                    0x2a,
                    self.vector_destination(function, allocation, *destination)?,
                    self.source(function, allocation, *source)?,
                )?,
                Instruction::HardwareExponentialBaseTwo {
                    destination,
                    source,
                } => encoder.vop1(
                    0x25,
                    self.vector_destination(function, allocation, *destination)?,
                    self.source(function, allocation, *source)?,
                )?,
                Instruction::LoadExponent {
                    destination,
                    first,
                    second,
                } => encoder.vop3a(
                    0x362,
                    self.vector_destination(function, allocation, *destination)?,
                    [
                        self.source(function, allocation, *first)?,
                        self.source(function, allocation, *second)?,
                        Source::Scalar(0),
                    ],
                    0,
                    0,
                )?,
                Instruction::DivideScale {
                    destination,
                    predicate,
                    first,
                    second,
                    third,
                } => {
                    let scalar_destination = match predicate {
                        Some(predicate) => {
                            self.predicate_destination(function, allocation, *predicate)?
                        }
                        None => NULL,
                    };
                    encoder.vop3b(
                        0x16d,
                        self.vector_destination(function, allocation, *destination)?,
                        scalar_destination,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            self.source(function, allocation, *third)?,
                        ],
                    )?;
                }
                Instruction::DivideFusedMultiplyAdd {
                    destination,
                    first,
                    second,
                    third,
                    predicate,
                } => {
                    if self.scalar(function, allocation, *predicate)? != VCC_LO {
                        return Err(Gfx10Error::Unsupported(
                            "divide fmas requires vcc_lo predicate",
                        ));
                    }
                    encoder.vop3a(
                        0x16f,
                        self.vector_destination(function, allocation, *destination)?,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            self.source(function, allocation, *third)?,
                        ],
                        0,
                        0,
                    )?;
                }
                Instruction::DivideFixup {
                    destination,
                    first,
                    second,
                    third,
                } => encoder.vop3a(
                    0x15f,
                    self.vector_destination(function, allocation, *destination)?,
                    [
                        self.source(function, allocation, *first)?,
                        self.source(function, allocation, *second)?,
                        self.source(function, allocation, *third)?,
                    ],
                    0,
                    0,
                )?,
                Instruction::FloatCompare {
                    comparison,
                    destination,
                    first,
                    second,
                } => {
                    let destination =
                        self.predicate_destination(function, allocation, *destination)?;
                    match comparison {
                        FloatComparisonKind::AlwaysTrue => {
                            encoder.sop1(3, destination, Source::Immediate(u32::MAX))?
                        }
                        FloatComparisonKind::AlwaysFalse => {
                            encoder.sop1(3, destination, Source::Immediate(0))?
                        }
                        other => {
                            let opcode = match other {
                                FloatComparisonKind::OrderedLessThan => 1,
                                FloatComparisonKind::OrderedEqual => 2,
                                FloatComparisonKind::OrderedLessThanOrEqual => 3,
                                FloatComparisonKind::OrderedGreaterThan => 4,
                                FloatComparisonKind::OrderedNotEqual => 5,
                                FloatComparisonKind::OrderedGreaterThanOrEqual => 6,
                                FloatComparisonKind::UnorderedLessThan => 9,
                                FloatComparisonKind::UnorderedEqual => 10,
                                FloatComparisonKind::UnorderedLessThanOrEqual => 11,
                                FloatComparisonKind::UnorderedGreaterThan => 12,
                                FloatComparisonKind::UnorderedNotEqual => 13,
                                FloatComparisonKind::UnorderedGreaterThanOrEqual => 14,
                                _ => unreachable!(),
                            };
                            encoder.vop3b(
                                opcode,
                                destination,
                                0,
                                [
                                    self.source(function, allocation, *first)?,
                                    self.source(function, allocation, *second)?,
                                    Source::Scalar(0),
                                ],
                            )?;
                        }
                    }
                }
                Instruction::Select {
                    destination,
                    condition,
                    when_true,
                    when_false,
                } => {
                    let predicate = self.scalar(function, allocation, *condition)?;
                    encoder.vop3a(
                        0x101,
                        self.vector_destination(function, allocation, *destination)?,
                        [
                            self.source(function, allocation, *when_false)?,
                            self.source(function, allocation, *when_true)?,
                            Source::Scalar(predicate),
                        ],
                        0,
                        0,
                    )?;
                }
                Instruction::IntegerMultiply {
                    destination,
                    first,
                    second,
                } => encoder.vop3a(
                    0x169,
                    self.vector_destination(function, allocation, *destination)?,
                    [
                        self.source(function, allocation, *first)?,
                        self.source(function, allocation, *second)?,
                        Source::Scalar(0),
                    ],
                    0,
                    0,
                )?,
                Instruction::IntegerShiftRight32 {
                    destination,
                    first,
                    second,
                } => encoder.vop3a(
                    0x118,
                    self.vector_destination(function, allocation, *destination)?,
                    [
                        self.source(function, allocation, *second)?,
                        self.source(function, allocation, *first)?,
                        Source::Scalar(0),
                    ],
                    0,
                    0,
                )?,
                Instruction::AddressLow {
                    destination,
                    destination_element,
                    carry,
                    first,
                    second,
                } => {
                    let carry = match carry {
                        Some(value) => self.predicate_destination(function, allocation, *value)?,
                        None => NULL,
                    };
                    encoder.vop3b(
                        0x30f,
                        self.destination(
                            function,
                            allocation,
                            *destination,
                            *destination_element,
                            RegisterClass::Vector,
                        )?,
                        carry,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            Source::Scalar(0),
                        ],
                    )?;
                }
                Instruction::AddressHigh {
                    destination,
                    destination_element,
                    first,
                    second,
                    carry,
                } => {
                    let carry = self.scalar(function, allocation, *carry)?;
                    encoder.vop3b(
                        0x128,
                        self.destination(
                            function,
                            allocation,
                            *destination,
                            *destination_element,
                            RegisterClass::Vector,
                        )?,
                        NULL,
                        [
                            self.source(function, allocation, *first)?,
                            self.source(function, allocation, *second)?,
                            Source::Scalar(carry),
                        ],
                    )?;
                }
                Instruction::LoadGlobal {
                    destination,
                    register_count,
                    address,
                    offset,
                } => {
                    let register = self.vector(function, allocation, *address)?;
                    self.validate_address(function, *address)?;
                    self.validate_register_block(function, *destination, *register_count)?;
                    let destination =
                        self.vector_destination(function, allocation, *destination)?;
                    encoder.global_memory(
                        false,
                        *register_count,
                        destination,
                        register,
                        *offset,
                    )?;
                }
                Instruction::StoreGlobal {
                    register_count,
                    address,
                    offset,
                    source,
                } => {
                    let register = self.vector(function, allocation, *address)?;
                    self.validate_address(function, *address)?;
                    let source_register = self.vector(function, allocation, *source)?;
                    if let Operand::Value { value, element } = source
                        && u16::from(*element) + u16::from(*register_count)
                            > u16::from(function.description(*value).register_count)
                    {
                        return Err(Gfx10Error::Unsupported("store source register block"));
                    }
                    encoder.global_memory(
                        true,
                        *register_count,
                        source_register,
                        register,
                        *offset,
                    )?;
                }
                Instruction::Interpolate {
                    second_step,
                    destination,
                    coordinates,
                    previous_value,
                    primitive_mask,
                    attribute,
                    channel,
                } => {
                    let coordinate = self.vector(function, allocation, *coordinates)?;
                    let destination =
                        self.vector_destination(function, allocation, *destination)?;
                    if *second_step {
                        let previous = self.vector(
                            function,
                            allocation,
                            (*previous_value)
                                .ok_or(Gfx10Error::Malformed("missing first interpolation step"))?,
                        )?;
                        if previous != destination {
                            encoder.vop1(1, destination, Source::Vector(previous))?;
                        }
                    } else if previous_value.is_some() {
                        return Err(Gfx10Error::Malformed(
                            "first interpolation step has previous value",
                        ));
                    }
                    let mask_register = self.scalar(function, allocation, *primitive_mask)?;
                    if current_interpolation_mask != Some(mask_register) {
                        encoder.sop1(3, M0, Source::Scalar(mask_register))?;
                        current_interpolation_mask = Some(mask_register);
                    }
                    let attribute =
                        u8::try_from(*attribute).map_err(|_| Gfx10Error::ImmediateRange)?;
                    let channel = u8::try_from(*channel).map_err(|_| Gfx10Error::ImmediateRange)?;
                    encoder.interp(*second_step, destination, coordinate, attribute, channel)?;
                }
                Instruction::Export {
                    target,
                    values,
                    mask,
                    done,
                    valid_mask,
                } => {
                    let target = u8::try_from(*target).map_err(|_| Gfx10Error::ImmediateRange)?;
                    let mut registers = [0u8; 4];
                    for index in 0..4 {
                        if mask & (1 << index) != 0 {
                            registers[index] = self.vector(function, allocation, values[index])?;
                            let register = usize::from(registers[index]);
                            export_registers[register / 64] |= 1u64 << (register % 64);
                        }
                    }
                    encoder.export(target, registers, *mask, *done, *valid_mask)?;
                    pending_exports = true;
                }
                Instruction::ZeroExecutionMask => encoder.sop1(3, EXEC_LO, Source::Immediate(0))?,
                Instruction::AndExecutionMaskWithPredicate { predicate } => {
                    let predicate = self.scalar(function, allocation, *predicate)?;
                    encoder.sop2(
                        0x0e,
                        EXEC_LO,
                        Source::Scalar(EXEC_LO),
                        Source::Scalar(predicate),
                    )?;
                }
                Instruction::WaitForLoads => {
                    encoder.sopp(0x0c, 0);
                    // This encoding also waits for expcnt and lgkmcnt.
                    export_registers.fill(0);
                    pending_exports = false;
                }
                Instruction::Return => {
                    encoder.sopp(1, 0);
                    // ENDPGM completes outstanding exports before releasing VGPRs.
                    export_registers.fill(0);
                    pending_exports = false;
                }
                Instruction::FloatDivide { .. }
                | Instruction::ExponentialBaseTwo { .. }
                | Instruction::Kill { .. } => {
                    return Err(Gfx10Error::Malformed("unexpanded instruction"));
                }
            }
        }
        for (position, label) in branches {
            let target = labels
                .iter()
                .find(|(id, _)| *id == label)
                .map(|(_, position)| *position)
                .ok_or(Gfx10Error::Malformed("branch to unknown label"))?;
            encoder.patch_branch(position, target)?;
        }
        Ok(encoder.code)
    }

    fn format_instruction(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        index: usize,
    ) -> String {
        let instruction = match function.instructions.get(index) {
            Some(instruction) => instruction,
            None => return format!("invalid instruction index {index}"),
        };
        let operand = |source: Operand| match self.source(function, allocation, source) {
            Ok(Source::Vector(register)) => format!("v{register}"),
            Ok(Source::Scalar(register)) => scalar_name(register),
            Ok(Source::Immediate(bits)) => format!("0x{bits:08x}"),
            Err(error) => format!("invalid operand: {error:?}"),
        };
        let destination = |value| operand(value_operand(value));
        match instruction {
            Instruction::Constant {
                destination: value,
                bits,
            } => format!(
                "{}_mov_b32 {}, 0x{bits:08x}",
                if function.description(*value).register_class == RegisterClass::Vector {
                    "v"
                } else {
                    "s"
                },
                destination(*value)
            ),
            Instruction::Copy {
                destination: value,
                source,
            } => format!(
                "{}_mov_b32 {}, {}",
                if function.description(*value).register_class == RegisterClass::Vector {
                    "v"
                } else {
                    "s"
                },
                destination(*value),
                operand(*source)
            ),
            Instruction::FloatBinary {
                operation,
                destination: value,
                first,
                second,
            } => format!(
                "v_{}_f32 {}, {}, {}",
                match operation {
                    FloatBinaryOperation::Add => "add",
                    FloatBinaryOperation::Subtract => "sub",
                    FloatBinaryOperation::Multiply => "mul",
                },
                destination(*value),
                operand(*first),
                operand(*second)
            ),
            Instruction::FloatDivide {
                destination: value,
                first,
                second,
            } => format!(
                "fdiv {}, {}, {}",
                destination(*value),
                operand(*first),
                operand(*second)
            ),
            Instruction::FusedMultiplyAdd {
                destination: value,
                first,
                second,
                third,
                negate_first,
            } => format!(
                "v_fma_f32 {}, {}{}, {}, {}",
                destination(*value),
                if *negate_first { "-" } else { "" },
                operand(*first),
                operand(*second),
                operand(*third)
            ),
            Instruction::FusedMultiplyAccumulate {
                destination: value,
                first,
                second,
            } => format!(
                "v_fmac_f32 {}, {}, {}",
                destination(*value),
                operand(*first),
                operand(*second)
            ),
            Instruction::Reciprocal {
                destination: value,
                source,
            } => format!("v_rcp_f32 {}, {}", destination(*value), operand(*source)),
            Instruction::ExponentialBaseTwo {
                destination: value,
                source,
            } => format!("exp2 {}, {}", destination(*value), operand(*source)),
            Instruction::HardwareExponentialBaseTwo {
                destination: value,
                source,
            } => format!("v_exp_f32 {}, {}", destination(*value), operand(*source)),
            Instruction::LoadExponent {
                destination: value,
                first,
                second,
            } => format!(
                "v_ldexp_f32 {}, {}, {}",
                destination(*value),
                operand(*first),
                operand(*second)
            ),
            Instruction::DivideScale {
                destination: value,
                predicate,
                first,
                second,
                third,
            } => format!(
                "v_div_scale_f32 {}, {}, {}, {}, {}",
                destination(*value),
                predicate.map_or_else(|| scalar_name(NULL), destination),
                operand(*first),
                operand(*second),
                operand(*third)
            ),
            Instruction::DivideFusedMultiplyAdd {
                destination: value,
                first,
                second,
                third,
                predicate,
            } => format!(
                "v_div_fmas_f32 {}, {}, {}, {} (predicate {})",
                destination(*value),
                operand(*first),
                operand(*second),
                operand(*third),
                operand(*predicate)
            ),
            Instruction::DivideFixup {
                destination: value,
                first,
                second,
                third,
            } => format!(
                "v_div_fixup_f32 {}, {}, {}, {}",
                destination(*value),
                operand(*first),
                operand(*second),
                operand(*third)
            ),
            Instruction::FloatCompare {
                comparison,
                destination: value,
                first,
                second,
            } => match comparison {
                FloatComparisonKind::AlwaysTrue => format!("s_mov_b32 {}, -1", destination(*value)),
                FloatComparisonKind::AlwaysFalse => format!("s_mov_b32 {}, 0", destination(*value)),
                _ => format!(
                    "v_cmp_{}_f32 {}, {}, {}",
                    comparison_mnemonic(*comparison),
                    destination(*value),
                    operand(*first),
                    operand(*second)
                ),
            },
            Instruction::Select {
                destination: value,
                condition,
                when_true,
                when_false,
            } => format!(
                "v_cndmask_b32 {}, {}, {}, {}",
                destination(*value),
                operand(*when_false),
                operand(*when_true),
                operand(*condition)
            ),
            Instruction::IntegerMultiply {
                destination: value,
                first,
                second,
            } => format!(
                "v_mul_lo_u32 {}, {}, {}",
                destination(*value),
                operand(*first),
                operand(*second)
            ),
            Instruction::IntegerShiftRight32 {
                destination: value,
                first,
                second,
            } => format!(
                "v_ashrrev_i32 {}, {}, {}",
                destination(*value),
                operand(*second),
                operand(*first)
            ),
            Instruction::AddressLow {
                destination: value,
                destination_element,
                carry,
                first,
                second,
            } => format!(
                "v_add_co_u32 {}, {}, {}, {}",
                operand(Operand::Value {
                    value: *value,
                    element: *destination_element
                }),
                carry.map_or_else(|| scalar_name(NULL), destination),
                operand(*first),
                operand(*second)
            ),
            Instruction::AddressHigh {
                destination: value,
                destination_element,
                first,
                second,
                carry,
            } => format!(
                "v_add_co_ci_u32 {}, {}, {}, {}, {}",
                operand(Operand::Value {
                    value: *value,
                    element: *destination_element
                }),
                scalar_name(NULL),
                operand(*first),
                operand(*second),
                operand(*carry)
            ),
            Instruction::LoadGlobal {
                destination: value,
                register_count,
                address,
                offset,
            } => format!(
                "global_load_dword{} {}, {}, off offset:{offset}",
                if *register_count == 1 { "" } else { "x4" },
                destination(*value),
                operand(*address)
            ),
            Instruction::StoreGlobal {
                register_count,
                address,
                offset,
                source,
            } => format!(
                "global_store_dword{} {}, {}, off offset:{offset}",
                if *register_count == 1 { "" } else { "x4" },
                operand(*address),
                operand(*source)
            ),
            Instruction::Interpolate {
                second_step,
                destination: value,
                coordinates,
                previous_value,
                primitive_mask,
                attribute,
                channel,
            } => format!(
                "v_interp_p{}_f32 {}, {}, attr{}.{} (m0 <- {}){}",
                if *second_step { 2 } else { 1 },
                destination(*value),
                operand(*coordinates),
                attribute,
                ["x", "y", "z", "w"]
                    .get(*channel as usize)
                    .copied()
                    .unwrap_or("invalid"),
                operand(*primitive_mask),
                previous_value.map_or_else(String::new, |value| format!(
                    " (previous {})",
                    operand(value)
                ))
            ),
            Instruction::Export {
                target,
                values,
                mask,
                done,
                valid_mask,
            } => format!(
                "exp {} {}, {}, {}, {} en:{mask:x} done:{} vm:{}",
                target,
                operand(values[0]),
                operand(values[1]),
                operand(values[2]),
                operand(values[3]),
                done,
                valid_mask
            ),
            Instruction::Kill { condition } => format!("kill {}", operand(*condition)),
            Instruction::WaitForLoads => String::from("s_waitcnt vmcnt(0)"),
            Instruction::ZeroExecutionMask => String::from("s_mov_b32 exec_lo, 0"),
            Instruction::AndExecutionMaskWithPredicate { predicate } => {
                format!("s_and_b32 exec_lo, exec_lo, {}", operand(*predicate))
            }
            Instruction::BranchIfNoActiveLanes { label } => format!("s_cbranch_execz label{label}"),
            Instruction::Branch { label } => format!("s_branch label{label}"),
            Instruction::BranchIfPredicate { predicate, label } => format!(
                "s_and_b32 null, exec_lo, {}; s_cbranch_scc1 label{label}",
                operand(*predicate)
            ),
            Instruction::Label { id } => format!("label{id}:"),
            Instruction::Return => String::from("s_endpgm"),
        }
    }

    fn configuration(&self, stage: ShaderStage, summary: &Summary) -> Vec<(u32, u32)> {
        let vgpr_blocks = summary
            .number_of_vector_registers
            .max(if stage == ShaderStage::Pixel { 2 } else { 1 })
            .div_ceil(8)
            - 1;
        let mut config = vec![
            (
                match stage {
                    ShaderStage::Vertex => SPI_SHADER_PGM_RSRC1_VS,
                    ShaderStage::Pixel => SPI_SHADER_PGM_RSRC1_PS,
                },
                vgpr_blocks,
            ),
            (SPI_TMPRING_SIZE, 0),
        ];
        if stage == ShaderStage::Pixel {
            config.push((SPI_SHADER_PGM_RSRC2_PS, 0));
            // The PS entry ABI places perspective-center coordinates in v[0:1].
            config.push((SPI_PS_INPUT_ENA, InterpolationMode::Center as u32));
            config.push((SPI_PS_INPUT_ADDR, InterpolationMode::Center as u32));
        }
        config.extend_from_slice(&[(4, 0), (8, 0)]);
        config
    }
}

impl Gfx10 {
    fn validate_address(
        &self,
        function: &MachineFunction,
        address: Operand,
    ) -> Result<(), Gfx10Error> {
        match address {
            Operand::Value { value, element }
                if function.description(value).register_class == RegisterClass::Vector
                    && u16::from(element) + 2
                        <= u16::from(function.description(value).register_count) =>
            {
                Ok(())
            }
            Operand::Physical {
                register_class: RegisterClass::Vector,
                register,
            } if register < 255 => Ok(()),
            _ => Err(Gfx10Error::Unsupported(
                "global address requires a vector register pair",
            )),
        }
    }
    fn validate_register_block(
        &self,
        function: &MachineFunction,
        destination: ValueId,
        count: u8,
    ) -> Result<(), Gfx10Error> {
        if function.description(destination).register_count < count {
            return Err(Gfx10Error::Unsupported(
                "global load destination register block",
            ));
        }
        Ok(())
    }
}

fn comparison_mnemonic(comparison: FloatComparisonKind) -> &'static str {
    match comparison {
        FloatComparisonKind::OrderedLessThan => "lt",
        FloatComparisonKind::OrderedEqual => "eq",
        FloatComparisonKind::OrderedLessThanOrEqual => "le",
        FloatComparisonKind::OrderedGreaterThan => "gt",
        FloatComparisonKind::OrderedNotEqual => "lg",
        FloatComparisonKind::OrderedGreaterThanOrEqual => "ge",
        FloatComparisonKind::UnorderedLessThan => "nge",
        FloatComparisonKind::UnorderedEqual => "nlg",
        FloatComparisonKind::UnorderedLessThanOrEqual => "ngt",
        FloatComparisonKind::UnorderedGreaterThan => "nle",
        FloatComparisonKind::UnorderedNotEqual => "neq",
        FloatComparisonKind::UnorderedGreaterThanOrEqual => "nlt",
        FloatComparisonKind::AlwaysTrue => "true",
        FloatComparisonKind::AlwaysFalse => "false",
    }
}

fn scalar_name(register: u8) -> String {
    match register {
        VCC_LO => String::from("vcc_lo"),
        M0 => String::from("m0"),
        NULL => String::from("null"),
        EXEC_LO => String::from("exec_lo"),
        _ => format!("s{register}"),
    }
}

#[derive(Clone, Copy)]
enum Source {
    Scalar(u8),
    Vector(u8),
    Immediate(u32),
}

struct Encoder {
    code: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { code: Vec::new() }
    }
    fn pc(&self) -> usize {
        self.code.len()
    }
    fn word(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
    }
    fn patch_branch(&mut self, position: usize, target: usize) -> Result<(), Gfx10Error> {
        if (position | target) & 3 != 0
            || position + 4 > self.code.len()
            || target > self.code.len()
        {
            return Err(Gfx10Error::Malformed("invalid branch offset"));
        }
        let displacement = (target as i64 - position as i64 - 4) / 4;
        let displacement = i16::try_from(displacement).map_err(|_| Gfx10Error::ImmediateRange)?;
        self.code[position..position + 2].copy_from_slice(&displacement.to_le_bytes());
        Ok(())
    }
    fn source(source: Source, literal: &mut Option<u32>) -> Result<u16, Gfx10Error> {
        match source {
            Source::Scalar(register) if register < 128 => Ok(register.into()),
            Source::Scalar(_) => Err(Gfx10Error::Unsupported("scalar register out of range")),
            Source::Vector(register) => Ok(256 + u16::from(register)),
            Source::Immediate(bits) => {
                let signed = bits as i32;
                let inline = match signed {
                    0..=64 => Some((128 + signed) as u16),
                    -16..=-1 => Some((192 - signed) as u16),
                    _ => match bits {
                        0x3f000000 => Some(240),
                        0xbf000000 => Some(241),
                        0x3f800000 => Some(242),
                        0xbf800000 => Some(243),
                        0x40000000 => Some(244),
                        0xc0000000 => Some(245),
                        0x40800000 => Some(246),
                        0xc0800000 => Some(247),
                        0x3e22f983 => Some(248),
                        _ => None,
                    },
                };
                if let Some(inline) = inline {
                    return Ok(inline);
                }
                match literal {
                    Some(previous) if *previous != bits => {
                        return Err(Gfx10Error::MultipleLiterals);
                    }
                    Some(_) => {}
                    None => *literal = Some(bits),
                }
                Ok(255)
            }
        }
    }
    fn emit_literal(&mut self, literal: Option<u32>) {
        if let Some(literal) = literal {
            self.word(literal);
        }
    }
    fn sop1(&mut self, opcode: u8, destination: u8, source: Source) -> Result<(), Gfx10Error> {
        if destination >= 128 {
            return Err(Gfx10Error::Unsupported("scalar destination out of range"));
        }
        let mut literal = None;
        let source = Self::source(source, &mut literal)?;
        if source > 255 {
            return Err(Gfx10Error::Unsupported(
                "vector source in scalar instruction",
            ));
        }
        self.word(
            0xbe80_0000
                | (u32::from(destination) << 16)
                | (u32::from(opcode) << 8)
                | u32::from(source),
        );
        self.emit_literal(literal);
        Ok(())
    }
    fn sop2(
        &mut self,
        opcode: u8,
        destination: u8,
        first: Source,
        second: Source,
    ) -> Result<(), Gfx10Error> {
        if destination >= 128 {
            return Err(Gfx10Error::Unsupported("scalar destination out of range"));
        }
        let mut literal = None;
        let first = Self::source(first, &mut literal)?;
        let second = Self::source(second, &mut literal)?;
        if first > 255 || second > 255 {
            return Err(Gfx10Error::Unsupported(
                "vector source in scalar instruction",
            ));
        }
        self.word(
            0x8000_0000
                | (u32::from(opcode) << 23)
                | (u32::from(destination) << 16)
                | (u32::from(second) << 8)
                | u32::from(first),
        );
        self.emit_literal(literal);
        Ok(())
    }
    fn sopp(&mut self, opcode: u8, immediate: i16) {
        self.word(0xbf80_0000 | (u32::from(opcode) << 16) | u32::from(immediate as u16));
    }
    fn vop1(&mut self, opcode: u8, destination: u8, source: Source) -> Result<(), Gfx10Error> {
        let mut literal = None;
        let source = Self::source(source, &mut literal)?;
        self.word(
            0x7e00_0000
                | (u32::from(destination) << 17)
                | (u32::from(opcode) << 9)
                | u32::from(source),
        );
        self.emit_literal(literal);
        Ok(())
    }
    fn vop3a(
        &mut self,
        opcode: u16,
        destination: u8,
        sources: [Source; 3],
        absolute: u8,
        negative: u8,
    ) -> Result<(), Gfx10Error> {
        self.vop3(opcode, destination, 0, sources, absolute, negative, false)
    }
    fn vop3b(
        &mut self,
        opcode: u16,
        destination: u8,
        scalar_destination: u8,
        sources: [Source; 3],
    ) -> Result<(), Gfx10Error> {
        self.vop3(opcode, destination, scalar_destination, sources, 0, 0, true)
    }
    fn vop3(
        &mut self,
        opcode: u16,
        destination: u8,
        scalar_destination: u8,
        sources: [Source; 3],
        absolute: u8,
        negative: u8,
        has_scalar_destination: bool,
    ) -> Result<(), Gfx10Error> {
        if opcode > 0x3ff
            || absolute > 7
            || negative > 7
            || (has_scalar_destination && scalar_destination >= 128)
        {
            return Err(Gfx10Error::ImmediateRange);
        }
        let mut literal = None;
        let first = u32::from(Self::source(sources[0], &mut literal)?);
        let second = u32::from(Self::source(sources[1], &mut literal)?);
        let third = u32::from(Self::source(sources[2], &mut literal)?);
        // RDNA2 permits at most two distinct uniform/literal sources on the
        // VOP3 constant bus. The third field is ignored by two-input opcodes.
        let used_sources = if matches!(opcode, 0x101 | 0x14b | 0x15f | 0x16d | 0x16f | 0x128) {
            3
        } else {
            2
        };
        let mut scalar_sources = [u32::MAX; 2];
        for &source in [first, second, third].iter().take(used_sources) {
            if source < 128 || source == 255 {
                if scalar_sources.contains(&source) {
                    continue;
                }
                let Some(slot) = scalar_sources.iter_mut().find(|slot| **slot == u32::MAX) else {
                    return Err(Gfx10Error::Unsupported(
                        "VOP3 constant bus has more than two sources",
                    ));
                };
                *slot = source;
            }
        }
        let modifiers = if has_scalar_destination {
            u32::from(scalar_destination)
        } else {
            u32::from(absolute)
        };
        self.word(
            0xd400_0000 | (u32::from(opcode) << 16) | (modifiers << 8) | u32::from(destination),
        );
        self.word((u32::from(negative) << 29) | (third << 18) | (second << 9) | first);
        self.emit_literal(literal);
        Ok(())
    }
    fn interp(
        &mut self,
        second_step: bool,
        destination: u8,
        coordinate: u8,
        attribute: u8,
        channel: u8,
    ) -> Result<(), Gfx10Error> {
        if attribute > 31 || channel > 3 {
            return Err(Gfx10Error::ImmediateRange);
        }
        self.word(
            0xc800_0000
                | (u32::from(destination) << 18)
                | (u32::from(second_step) << 16)
                | ((u32::from(attribute) << 2 | u32::from(channel)) << 8)
                | u32::from(coordinate),
        );
        Ok(())
    }
    fn global_memory(
        &mut self,
        store: bool,
        register_count: u8,
        data: u8,
        address: u8,
        offset: u32,
    ) -> Result<(), Gfx10Error> {
        if !matches!(register_count, 1 | 4)
            || offset > 4095
            || u16::from(data) + u16::from(register_count) > 256
            || address == 255
        {
            return Err(Gfx10Error::Unsupported("global memory operand or offset"));
        }
        let opcode = match (store, register_count) {
            (false, 1) => 0xdc30_8000,
            (false, 4) => 0xdc38_8000,
            (true, 1) => 0xdc70_8000,
            (true, 4) => 0xdc78_8000,
            _ => unreachable!(),
        };
        self.word(opcode | offset);
        self.word(if store {
            (0x7d << 16) | (u32::from(data) << 8) | u32::from(address)
        } else {
            (u32::from(data) << 24) | (0x7d << 16) | u32::from(address)
        });
        Ok(())
    }
    fn export(
        &mut self,
        target: u8,
        values: [u8; 4],
        mask: u8,
        done: bool,
        valid_mask: bool,
    ) -> Result<(), Gfx10Error> {
        if target > 63 || mask > 15 {
            return Err(Gfx10Error::ImmediateRange);
        }
        self.word(
            0xf800_0000
                | (u32::from(valid_mask) << 12)
                | (u32::from(done) << 11)
                | (u32::from(target) << 4)
                | u32::from(mask),
        );
        self.word(
            (u32::from(values[3]) << 24)
                | (u32::from(values[2]) << 16)
                | (u32::from(values[1]) << 8)
                | u32::from(values[0]),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{EXP_POS0, Encoder, Gfx10Error, Source};
    use alloc::vec::Vec;
    use volt_codegen::machine::{
        Instruction, MachineFunction, Operand, RegisterClass, ValuePurpose,
    };
    use volt_codegen::register_allocation::allocate_registers;
    use volt_codegen::targets::Target;

    /// Encodings in these tests are the expected bytes emitted by LLVM's
    /// AMDGPU assembler: `llvm-mc -triple=amdgcn -mcpu=gfx1032 -show-encoding`.
    fn instruction_words(encoder: &Encoder) -> Vec<u32> {
        encoder
            .code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect()
    }

    #[test]
    fn export_source_waits_before_its_vgpr_is_reused() {
        let target = super::for_processor("gfx1036").unwrap();
        let mut function = MachineFunction::default();
        let position = function.add_value(
            RegisterClass::Vector,
            1,
            Some(0),
            ValuePurpose::Parameter(0),
        );
        let zero = function.add_value(RegisterClass::Vector, 1, None, ValuePurpose::Temporary);
        function.instructions.extend([
            Instruction::Export {
                target: EXP_POS0,
                values: [Operand::Value {
                    value: position,
                    element: 0,
                }; 4],
                mask: 15,
                done: true,
                valid_mask: false,
            },
            Instruction::Copy {
                destination: zero,
                source: Operand::Immediate(0),
            },
            Instruction::Export {
                target: 34,
                values: [Operand::Value {
                    value: zero,
                    element: 0,
                }; 4],
                mask: 15,
                done: false,
                valid_mask: false,
            },
            Instruction::Return,
        ]);
        let allocation = allocate_registers(&function, &target).unwrap();
        assert_eq!(
            allocation.bases[position.index()],
            allocation.bases[zero.index()]
        );
        let code = target.encode_function(&function, &allocation).unwrap();
        let words: Vec<u32> = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        // The POS0 operand remains live in hardware after the allocator's last
        // logical use. Wait for its asynchronous read before writing the zero.
        assert_eq!(
            words,
            [
                0xf800_08cf,
                0,
                0xbf8c_ff0f,
                0x7e00_0280,
                0xf800_022f,
                0,
                0xbf81_0000,
            ]
        );
    }

    #[test]
    fn vector_multiply_and_arithmetic_shift_match_gfx1032_assembler() {
        let mut encoder = Encoder::new();
        // v_mul_lo_u32 v0, v0, 56
        encoder
            .vop3a(
                0x169,
                0,
                [Source::Vector(0), Source::Immediate(56), Source::Scalar(0)],
                0,
                0,
            )
            .unwrap();
        // v_ashrrev_i32 v1, 31, v0 — the sign-extend used by 64-bit GEP indices.
        encoder
            .vop3a(
                0x118,
                1,
                [Source::Immediate(31), Source::Vector(0), Source::Scalar(0)],
                0,
                0,
            )
            .unwrap();

        assert_eq!(
            instruction_words(&encoder),
            [0xd569_0000, 0x0001_7100, 0xd518_0001, 0x0002_009f]
        );
    }

    #[test]
    fn global_store_matches_gfx1032_assembler() {
        let mut encoder = Encoder::new();
        encoder.global_memory(true, 1, 4, 2, 8).unwrap();

        assert_eq!(instruction_words(&encoder), [0xdc70_8008, 0x007d_0402]);
    }

    #[test]
    fn distinct_literal_constants_are_rejected() {
        let mut encoder = Encoder::new();
        let result = encoder.vop3a(
            0x14b,
            0,
            [
                Source::Immediate(0x3f12_3456),
                Source::Immediate(0x4012_3456),
                Source::Vector(1),
            ],
            0,
            0,
        );

        assert_eq!(result, Err(Gfx10Error::MultipleLiterals));
    }

    #[test]
    fn floating_point_compare_matches_gfx1032_assembler() {
        let mut encoder = Encoder::new();
        // v_cmp_lt_f32 vcc_lo, 1.0, v2
        encoder
            .vop3b(
                1,
                super::VCC_LO,
                0,
                [
                    Source::Immediate(1.0f32.to_bits()),
                    Source::Vector(2),
                    Source::Scalar(0),
                ],
            )
            .unwrap();

        assert_eq!(instruction_words(&encoder), [0xd401_006a, 0x0002_04f2]);
    }
}

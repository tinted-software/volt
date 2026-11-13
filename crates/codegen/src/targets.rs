//! GPU-specific ABI, register allocation results, and machine-code emission traits.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::machine::{Instruction, MachineFunction, RegisterClass};

/// Calling convention and hardware entry state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderStage {
    Vertex,
    Pixel,
}

/// Pixel barycentric interpolation input. Values are SPI_PS_INPUT_ENA bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpolationMode {
    Sample = 1 << 0,
    Center = 1 << 1,
    Centroid = 1 << 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParameterLocation {
    Fixed(RegisterClass, u8),
    Unavailable,
}

#[derive(Clone, Copy, Debug)]
pub struct ParameterDescription {
    /// Number of 32-bit registers occupied by this parameter.
    pub size: u8,
    /// Whether the parameter has a vector LLVM type (not the register class).
    pub vector: bool,
}

#[derive(Clone, Debug)]
pub struct Signature {
    pub stage: ShaderStage,
    /// Parameters passed in scalar registers at the beginning of the signature.
    pub number_of_scalar_parameters: u32,
    pub parameters: Vec<ParameterDescription>,
}

/// Physical base register of each machine value, indexed by [`crate::machine::ValueId`].
#[derive(Clone, Debug, Default)]
pub struct Allocation {
    pub bases: Vec<u8>,
}

/// Shader resources and hardware operations relevant to loader configuration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub number_of_vector_registers: u32,
    pub number_of_scalar_registers: u32,
    pub uses_discard: bool,
    pub interpolation_modes: u32,
}

impl Summary {
    pub fn from_function(function: &MachineFunction, allocation: &Allocation) -> Self {
        let mut summary = Self::default();
        for (index, value) in function.values.iter().enumerate() {
            let Some(&base) = allocation.bases.get(index) else {
                continue;
            };
            let end = u32::from(base) + u32::from(value.register_count);
            match value.register_class {
                RegisterClass::Vector => {
                    summary.number_of_vector_registers = summary.number_of_vector_registers.max(end)
                }
                RegisterClass::Scalar => {
                    summary.number_of_scalar_registers = summary.number_of_scalar_registers.max(end)
                }
                // Special predicate VCC_LO does not consume the ordinary SGPR file.
                RegisterClass::Predicate => {}
            }
        }
        for instruction in &function.instructions {
            match instruction {
                Instruction::Kill { .. }
                | Instruction::ZeroExecutionMask
                | Instruction::AndExecutionMaskWithPredicate { .. } => summary.uses_discard = true,
                Instruction::Interpolate { .. } => {
                    summary.interpolation_modes |= InterpolationMode::Center as u32
                }
                _ => {}
            }
        }
        summary
    }
}

pub trait Target {
    type Error: fmt::Debug + fmt::Display;

    fn processor_name(&self) -> &'static str;
    fn wave_size(&self) -> u32;
    fn maximum_vector_registers(&self) -> u32;
    fn maximum_scalar_registers(&self) -> u32;
    fn maximum_predicate_registers(&self) -> u32;
    /// Physical register number used for the first predicate register.
    fn predicate_register_base(&self) -> u8;
    fn parameter_location(&self, signature: &Signature, index: u32) -> ParameterLocation;
    fn interpolation_coordinates(&self, mode: InterpolationMode) -> Option<(u8, u8)>;
    fn perspective_mode(&self) -> InterpolationMode;
    fn expand_pseudo_instructions(&self, function: &mut MachineFunction)
    -> Result<(), Self::Error>;
    fn encode_function(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
    ) -> Result<Vec<u8>, Self::Error>;
    fn format_instruction(
        &self,
        function: &MachineFunction,
        allocation: &Allocation,
        index: usize,
    ) -> String;
    fn configuration(&self, stage: ShaderStage, summary: &Summary) -> Vec<(u32, u32)>;
}

//! Deterministic allocation of contiguous machine-value register blocks.
//! Reads precede writes at each instruction, so a destination can reuse the
//! register of a source whose lifetime ends at that instruction.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::machine::{Instruction, MachineFunction, Operand, RegisterClass, ValueId, ValuePurpose};
use crate::targets::{Allocation, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterAllocationError {
    InvalidValue(ValueId),
    EmptyValue(ValueId),
    InvalidElement {
        value: ValueId,
        element: u8,
    },
    InvalidWidth {
        value: ValueId,
        element: u8,
        width: u8,
    },
    ReadBeforeDefinition {
        value: ValueId,
        element: u8,
        instruction: usize,
    },
    InvalidTiedOperand {
        instruction: usize,
    },
    InputWithoutFixedRegister(ValueId),
    WrongRegisterClass {
        value: ValueId,
        expected: RegisterClass,
    },
    InvalidPhysicalRegister {
        register_class: RegisterClass,
        register: u8,
    },
    InvalidRegisterRange(ValueId),
    ConflictingFixedRegisters {
        first: ValueId,
        second: ValueId,
    },
    RegistersExhausted(ValueId),
}

impl fmt::Display for RegisterAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "register allocation failed: {self:?}")
    }
}

#[derive(Clone, Copy)]
struct Lifetime {
    start: usize,
    end: usize,
}

impl Lifetime {
    fn intersects(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    fn touch(&mut self, point: usize) {
        self.start = self.start.min(point);
        self.end = self.end.max(point);
    }
}

#[derive(Clone, Copy)]
struct PhysicalUse {
    register_class: RegisterClass,
    register: u8,
    width: u8,
    point: usize,
}

fn register_bounds(target: &impl Target, register_class: RegisterClass) -> (u32, u32) {
    match register_class {
        RegisterClass::Scalar => (0, target.maximum_scalar_registers()),
        RegisterClass::Vector => (0, target.maximum_vector_registers()),
        RegisterClass::Predicate => {
            let start = u32::from(target.predicate_register_base());
            (
                start,
                start.saturating_add(target.maximum_predicate_registers()),
            )
        }
    }
}

fn physical_range_valid(
    target: &impl Target,
    register_class: RegisterClass,
    register: u8,
    width: u8,
) -> bool {
    let (start, end) = register_bounds(target, register_class);
    width > 0
        && u32::from(register) >= start
        && u32::from(register) + u32::from(width) <= end.min(256)
}

fn overlaps(first_start: u32, first_size: u8, second_start: u32, second_size: u8) -> bool {
    first_start < second_start + u32::from(second_size)
        && second_start < first_start + u32::from(first_size)
}

fn operand_width(instruction: &Instruction, operand: Operand) -> u8 {
    match instruction {
        Instruction::LoadGlobal { address, .. } if *address == operand => 2,
        Instruction::StoreGlobal {
            address,
            source,
            register_count,
            ..
        } => {
            let address_width = if *address == operand { 2 } else { 1 };
            let source_width = if *source == operand {
                *register_count
            } else {
                1
            };
            address_width.max(source_width)
        }
        _ => 1,
    }
}

fn require_predicate(
    function: &MachineFunction,
    value: ValueId,
) -> Result<(), RegisterAllocationError> {
    let description = function
        .values
        .get(value.index())
        .ok_or(RegisterAllocationError::InvalidValue(value))?;
    if description.register_class != RegisterClass::Predicate {
        return Err(RegisterAllocationError::WrongRegisterClass {
            value,
            expected: RegisterClass::Predicate,
        });
    }
    Ok(())
}

/// Allocate every value, reserving fixed ABI registers for their entire live
/// range and raw physical source registers from entry until their last use.
/// The lowest available contiguous block is chosen in instruction/value order.
/// No spilling occurs; an impossible assignment returns an explicit error.
pub fn allocate_registers(
    function: &MachineFunction,
    target: &impl Target,
) -> Result<Allocation, RegisterAllocationError> {
    let value_count = function.values.len();
    let mut lifetimes: Vec<Option<Lifetime>> = vec![None; value_count];
    // Up to 255 elements per value, without a separate heap allocation per value.
    let mut defined_elements = vec![[0u64; 4]; value_count];
    let mut physical_uses = Vec::new();

    for (index, description) in function.values.iter().enumerate() {
        let value = ValueId(index as u32);
        if description.register_count == 0 {
            return Err(RegisterAllocationError::EmptyValue(value));
        }
        if let Some(register) = description.fixed_register {
            if !physical_range_valid(
                target,
                description.register_class,
                register,
                description.register_count,
            ) {
                return Err(RegisterAllocationError::InvalidRegisterRange(value));
            }
        } else if matches!(
            description.purpose,
            ValuePurpose::Parameter(_) | ValuePurpose::InterpolationCoordinates
        ) {
            return Err(RegisterAllocationError::InputWithoutFixedRegister(value));
        }
    }

    for (instruction_index, instruction) in function.instructions.iter().enumerate() {
        let read_point = instruction_index * 2;
        let write_point = read_point + 1;
        let mut error = None;
        MachineFunction::visit_uses(instruction, |operand| {
            if error.is_some() {
                return;
            }
            let width = operand_width(instruction, operand);
            match operand {
                Operand::Value { value, element } => {
                    let Some(description) = function.values.get(value.index()) else {
                        error = Some(RegisterAllocationError::InvalidValue(value));
                        return;
                    };
                    if element >= description.register_count {
                        error = Some(RegisterAllocationError::InvalidElement { value, element });
                        return;
                    }
                    if width == 0
                        || u16::from(element) + u16::from(width)
                            > u16::from(description.register_count)
                    {
                        error = Some(RegisterAllocationError::InvalidWidth {
                            value,
                            element,
                            width,
                        });
                        return;
                    }
                    let input = matches!(
                        description.purpose,
                        ValuePurpose::Parameter(_) | ValuePurpose::InterpolationCoordinates
                    );
                    for current in element..element + width {
                        let mask = 1u64 << (current % 64);
                        if defined_elements[value.index()][usize::from(current / 64)] & mask == 0
                            && !input
                        {
                            error = Some(RegisterAllocationError::ReadBeforeDefinition {
                                value,
                                element: current,
                                instruction: instruction_index,
                            });
                            return;
                        }
                    }
                    let start = if input { 0 } else { read_point };
                    match &mut lifetimes[value.index()] {
                        Some(lifetime) => {
                            lifetime.touch(read_point);
                            lifetime.touch(start);
                        }
                        slot @ None => {
                            *slot = Some(Lifetime {
                                start,
                                end: read_point,
                            })
                        }
                    }
                }
                Operand::Physical {
                    register_class,
                    register,
                } => {
                    if !physical_range_valid(target, register_class, register, width) {
                        error = Some(RegisterAllocationError::InvalidPhysicalRegister {
                            register_class,
                            register,
                        });
                        return;
                    }
                    physical_uses.push(PhysicalUse {
                        register_class,
                        register,
                        width,
                        point: read_point,
                    });
                }
                Operand::Immediate(_) => {}
            }
        });
        if let Some(error) = error {
            return Err(error);
        }

        if let Instruction::Interpolate {
            second_step,
            destination,
            previous_value,
            ..
        } = instruction
        {
            let tied = Some(Operand::Value {
                value: *destination,
                element: 0,
            });
            if (*second_step && *previous_value != tied)
                || (!*second_step && previous_value.is_some())
            {
                return Err(RegisterAllocationError::InvalidTiedOperand {
                    instruction: instruction_index,
                });
            }
        }

        match instruction {
            Instruction::FloatCompare { destination, .. } => {
                require_predicate(function, *destination)?
            }
            Instruction::DivideScale {
                predicate: Some(predicate),
                ..
            }
            | Instruction::AddressLow {
                carry: Some(predicate),
                ..
            } => require_predicate(function, *predicate)?,
            Instruction::AddressHigh {
                carry: Operand::Value { value, .. },
                ..
            }
            | Instruction::DivideFusedMultiplyAdd {
                predicate: Operand::Value { value, .. },
                ..
            } => require_predicate(function, *value)?,
            _ => {}
        }

        MachineFunction::visit_definitions(instruction, |value, element| {
            if error.is_some() {
                return;
            }
            let Some(description) = function.values.get(value.index()) else {
                error = Some(RegisterAllocationError::InvalidValue(value));
                return;
            };
            if element >= description.register_count {
                error = Some(RegisterAllocationError::InvalidElement { value, element });
                return;
            }
            defined_elements[value.index()][usize::from(element / 64)] |= 1u64 << (element % 64);
            match &mut lifetimes[value.index()] {
                Some(lifetime) => lifetime.touch(write_point),
                slot @ None => {
                    *slot = Some(Lifetime {
                        start: write_point,
                        end: write_point,
                    })
                }
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }

    // Backward edges keep live-in values alive through the end of the loop,
    // even if their last textual use is before the backward branch.
    for (instruction_index, instruction) in function.instructions.iter().enumerate() {
        let label = match instruction {
            Instruction::Branch { label }
            | Instruction::BranchIfPredicate { label, .. }
            | Instruction::BranchIfNoActiveLanes { label } => *label,
            _ => continue,
        };
        if let Some(target_index) = function
            .instructions
            .iter()
            .position(|candidate| matches!(candidate, Instruction::Label { id } if *id == label))
            && target_index <= instruction_index
        {
            let start = target_index * 2;
            let end = instruction_index * 2;
            for lifetime in lifetimes.iter_mut().flatten() {
                if lifetime.start <= start && lifetime.end >= start && lifetime.end < end {
                    lifetime.end = end;
                }
            }
            for physical_use in &mut physical_uses {
                if start <= physical_use.point && physical_use.point < end {
                    physical_use.point = end;
                }
            }
        }
    }

    let mut bases = vec![0u8; value_count];
    let mut allocated = vec![false; value_count];
    // Fixed assignments go first. The later allocation pass cannot evict one.
    for (index, description) in function.values.iter().enumerate() {
        let Some(register) = description.fixed_register else {
            continue;
        };
        bases[index] = register;
        allocated[index] = true;
        if let Some(lifetime) = lifetimes[index] {
            for earlier in 0..index {
                let other = &function.values[earlier];
                if other.register_class == description.register_class
                    && other.fixed_register.is_some()
                    && lifetimes[earlier]
                        .is_some_and(|other_lifetime| lifetime.intersects(other_lifetime))
                    && overlaps(
                        u32::from(register),
                        description.register_count,
                        u32::from(bases[earlier]),
                        other.register_count,
                    )
                {
                    return Err(RegisterAllocationError::ConflictingFixedRegisters {
                        first: ValueId(earlier as u32),
                        second: ValueId(index as u32),
                    });
                }
            }
        }
    }

    let mut pending: Vec<usize> = (0..value_count)
        .filter(|index| !allocated[*index])
        .collect();
    pending.sort_by_key(|index| {
        (
            lifetimes[*index].map_or(usize::MAX, |lifetime| lifetime.start),
            *index,
        )
    });
    for index in pending {
        let description = &function.values[index];
        let (lower_bound, upper_bound) = register_bounds(target, description.register_class);
        let upper_bound = upper_bound.min(256);
        let Some(last_base) = upper_bound.checked_sub(u32::from(description.register_count)) else {
            return Err(RegisterAllocationError::RegistersExhausted(ValueId(
                index as u32,
            )));
        };
        let mut chosen = None;
        for base in lower_bound..=last_base {
            if let Some(lifetime) = lifetimes[index] {
                if function
                    .values
                    .iter()
                    .enumerate()
                    .any(|(other_index, other)| {
                        allocated[other_index]
                            && other.register_class == description.register_class
                            && lifetimes[other_index]
                                .is_some_and(|other_lifetime| lifetime.intersects(other_lifetime))
                            && overlaps(
                                base,
                                description.register_count,
                                u32::from(bases[other_index]),
                                other.register_count,
                            )
                    })
                {
                    continue;
                }
                if physical_uses.iter().any(|physical_use| {
                    physical_use.register_class == description.register_class
                        && lifetime.start <= physical_use.point
                        && overlaps(
                            base,
                            description.register_count,
                            u32::from(physical_use.register),
                            physical_use.width,
                        )
                }) {
                    continue;
                }
            }
            chosen = Some(base as u8);
            break;
        }
        bases[index] = chosen.ok_or(RegisterAllocationError::RegistersExhausted(ValueId(
            index as u32,
        )))?;
        allocated[index] = true;
    }

    Ok(Allocation { bases })
}

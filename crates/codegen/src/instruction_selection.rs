//! Selection of the single-block LLVM-dialect shader entry point into machine IR.
//! Constants remain operands until an instruction requires a physical vector
//! register; addresses retain their byte offset so global loads can fold it.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use pliron::builtin::attributes::{FPSingleAttr, IntegerAttr};
use pliron::builtin::op_interfaces::{
    AtMostOneRegionInterface, SingleBlockRegionInterface, SymbolOpInterface,
};
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FP32Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::Typed;
use pliron::utils::apfloat::Float as _;
use pliron::value::Value;
use pliron_llvm::attributes::FCmpPredicateAttr;
use pliron_llvm::ops::{
    CallIntrinsicOp, ConstantOp, ExtractElementOp, FCmpOp, FuncOp, GepIndex, GetElementPtrOp,
};
use pliron_llvm::types::{PointerType, VectorType};

use crate::machine::{
    FloatBinaryOperation, FloatComparisonKind, Instruction, MachineFunction, Operand,
    RegisterClass, ValueId, ValuePurpose,
};
use crate::targets::{ParameterLocation, ShaderStage, Signature, Target};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    MissingEntry,
    InvalidSignature(&'static str),
    UnsupportedOperation(String),
    Unsupported(&'static str),
    UndefinedValue,
}

#[derive(Clone, Copy)]
enum SelectedValue {
    Bits(u32),
    Register(ValueId, u8),
    Address { pair: ValueId, offset: u32 },
}

struct Selector<'a> {
    context: &'a Context,
    function: MachineFunction,
    values: HashMap<Value, SelectedValue>,
    predicate_register: u8,
    outstanding_loads: bool,
}

impl Selector<'_> {
    fn value(&self, value: Value) -> Result<SelectedValue, SelectionError> {
        self.values
            .get(&value)
            .copied()
            .ok_or(SelectionError::UndefinedValue)
    }

    fn operand(&self, value: Value) -> Result<Operand, SelectionError> {
        match self.value(value)? {
            SelectedValue::Bits(bits) => Ok(Operand::Immediate(bits)),
            SelectedValue::Register(register, element) => Ok(Operand::Value {
                value: register,
                element,
            }),
            SelectedValue::Address { .. } => {
                Err(SelectionError::Unsupported("pointer used as scalar"))
            }
        }
    }

    fn constant(&self, value: Value) -> Result<u32, SelectionError> {
        match self.value(value)? {
            SelectedValue::Bits(bits) => Ok(bits),
            _ => Err(SelectionError::Unsupported(
                "expected constant integer argument",
            )),
        }
    }

    fn temporary(&mut self, register_class: RegisterClass, register_count: u8) -> ValueId {
        self.function.add_value(
            register_class,
            register_count,
            None,
            ValuePurpose::Temporary,
        )
    }

    fn vector_operand(&mut self, value: Value) -> Result<Operand, SelectionError> {
        let source = self.operand(value)?;
        match source {
            Operand::Value {
                value: register, ..
            } if self.function.description(register).register_class == RegisterClass::Vector => {
                Ok(source)
            }
            _ => {
                let destination = self.temporary(RegisterClass::Vector, 1);
                self.function.instructions.push(Instruction::Copy {
                    destination,
                    source,
                });
                Ok(Operand::Value {
                    value: destination,
                    element: 0,
                })
            }
        }
    }

    fn address(&mut self, pair: ValueId, index: Operand) -> ValueId {
        let destination = self.temporary(RegisterClass::Vector, 2);
        let carry = self.function.add_value(
            RegisterClass::Predicate,
            1,
            Some(self.predicate_register),
            ValuePurpose::Temporary,
        );
        let high_index = match index {
            Operand::Immediate(bits) => Operand::Immediate(((bits as i32) >> 31) as u32),
            _ => {
                let high = self.temporary(RegisterClass::Vector, 1);
                self.function
                    .instructions
                    .push(Instruction::IntegerShiftRight32 {
                        destination: high,
                        first: index,
                        second: Operand::Immediate(31),
                    });
                Operand::Value {
                    value: high,
                    element: 0,
                }
            }
        };
        self.function.instructions.push(Instruction::AddressLow {
            destination,
            destination_element: 0,
            carry: Some(carry),
            first: Operand::Value {
                value: pair,
                element: 0,
            },
            second: index,
        });
        self.function.instructions.push(Instruction::AddressHigh {
            destination,
            destination_element: 1,
            first: Operand::Value {
                value: pair,
                element: 1,
            },
            second: high_index,
            carry: Operand::Value {
                value: carry,
                element: 0,
            },
        });
        destination
    }

    fn select_operation(
        &mut self,
        operation_pointer: Ptr<Operation>,
    ) -> Result<(), SelectionError> {
        let operation = operation_pointer.deref(self.context);
        let operation_name = format!(
            "{}",
            Operation::get_opid(operation_pointer, self.context).disp(self.context)
        );
        let result = operation.results().next();
        let operand = |index| operation.get_operand(index);
        match operation_name.as_str() {
            "llvm.constant" => {
                let constant = Operation::get_op::<ConstantOp>(operation_pointer, self.context)
                    .ok_or(SelectionError::Unsupported("invalid constant operation"))?;
                let attribute = constant
                    .get_attr_llvm_constant_value(self.context)
                    .ok_or(SelectionError::Unsupported("constant without value"))?;
                let bits = if let Some(float) = attribute.downcast_ref::<FPSingleAttr>() {
                    float.0.to_bits() as u32
                } else if let Some(integer) = attribute.downcast_ref::<IntegerAttr>() {
                    let width = integer.get_type().deref(self.context).width();
                    if width > 32 || width == 0 {
                        return Err(SelectionError::Unsupported(
                            "integer constant wider than 32 bits",
                        ));
                    }
                    integer.value().to_u64() as u32
                } else {
                    return Err(SelectionError::Unsupported("non-scalar constant"));
                };
                self.values.insert(
                    result.ok_or(SelectionError::Unsupported("constant without result"))?,
                    SelectedValue::Bits(bits),
                );
            }
            "llvm.extractelement" => {
                let result = result.ok_or(SelectionError::Unsupported("extract without result"))?;
                // Builders may declare and extract unused barycentric inputs even
                // when no interpolation is emitted. Such inputs are not enabled
                // by the pixel ABI and must not be read.
                if result.num_uses(self.context) == 0 {
                    return Ok(());
                }
                let index = self.constant(operand(1))?;
                let vector = self.value(operand(0))?;
                let SelectedValue::Register(register, 0) = vector else {
                    return Err(SelectionError::Unsupported("extract from non-vector"));
                };
                if index >= self.function.description(register).register_count as u32 {
                    return Err(SelectionError::Unsupported("vector extract out of bounds"));
                }
                self.values
                    .insert(result, SelectedValue::Register(register, index as u8));
            }
            "llvm.mul" | "llvm.fadd" | "llvm.fsub" | "llvm.fmul" | "llvm.fdiv" => {
                let first = self.operand(operand(0))?;
                let second = self.operand(operand(1))?;
                let folded = match (first, second) {
                    (Operand::Immediate(first), Operand::Immediate(second)) => {
                        Some(if operation_name == "llvm.mul" {
                            first.wrapping_mul(second)
                        } else {
                            let (first, second) = (f32::from_bits(first), f32::from_bits(second));
                            match operation_name.as_str() {
                                "llvm.fadd" => (first + second).to_bits(),
                                "llvm.fsub" => (first - second).to_bits(),
                                "llvm.fmul" => (first * second).to_bits(),
                                _ => (first / second).to_bits(),
                            }
                        })
                    }
                    _ => None,
                };
                if let Some(bits) = folded {
                    self.values
                        .insert(result.unwrap(), SelectedValue::Bits(bits));
                } else {
                    let destination = self.temporary(RegisterClass::Vector, 1);
                    let instruction = match operation_name.as_str() {
                        "llvm.mul" => Instruction::IntegerMultiply {
                            destination,
                            first,
                            second,
                        },
                        "llvm.fdiv" => Instruction::FloatDivide {
                            destination,
                            first,
                            second,
                        },
                        name => Instruction::FloatBinary {
                            operation: match name {
                                "llvm.fadd" => FloatBinaryOperation::Add,
                                "llvm.fsub" => FloatBinaryOperation::Subtract,
                                _ => FloatBinaryOperation::Multiply,
                            },
                            destination,
                            first,
                            second,
                        },
                    };
                    self.function.instructions.push(instruction);
                    self.values
                        .insert(result.unwrap(), SelectedValue::Register(destination, 0));
                }
            }
            "llvm.fcmp" => {
                let compare = Operation::get_op::<FCmpOp>(operation_pointer, self.context)
                    .ok_or(SelectionError::Unsupported("invalid comparison"))?;
                let predicate = compare.predicate(self.context);
                let first = self.operand(operand(0))?;
                let second = self.operand(operand(1))?;
                if let (Operand::Immediate(first_bits), Operand::Immediate(second_bits)) =
                    (first, second)
                {
                    let (first, second) = (f32::from_bits(first_bits), f32::from_bits(second_bits));
                    let unordered = first.is_nan() || second.is_nan();
                    let passes = match predicate {
                        FCmpPredicateAttr::False => false,
                        FCmpPredicateAttr::True => true,
                        FCmpPredicateAttr::OEQ => first == second,
                        FCmpPredicateAttr::OGT => first > second,
                        FCmpPredicateAttr::OGE => first >= second,
                        FCmpPredicateAttr::OLT => first < second,
                        FCmpPredicateAttr::OLE => first <= second,
                        FCmpPredicateAttr::ONE => !unordered && first != second,
                        FCmpPredicateAttr::ORD => !unordered,
                        FCmpPredicateAttr::UEQ => unordered || first == second,
                        FCmpPredicateAttr::UGT => unordered || first > second,
                        FCmpPredicateAttr::UGE => unordered || first >= second,
                        FCmpPredicateAttr::ULT => unordered || first < second,
                        FCmpPredicateAttr::ULE => unordered || first <= second,
                        FCmpPredicateAttr::UNE => unordered || first != second,
                        FCmpPredicateAttr::UNO => unordered,
                    };
                    self.values
                        .insert(result.unwrap(), SelectedValue::Bits(u32::from(passes)));
                } else if matches!(
                    predicate,
                    FCmpPredicateAttr::False | FCmpPredicateAttr::True
                ) {
                    self.values.insert(
                        result.unwrap(),
                        SelectedValue::Bits(u32::from(predicate == FCmpPredicateAttr::True)),
                    );
                } else {
                    let comparison = match predicate {
                        FCmpPredicateAttr::False | FCmpPredicateAttr::True => unreachable!(),
                        FCmpPredicateAttr::OEQ => FloatComparisonKind::OrderedEqual,
                        FCmpPredicateAttr::OGT => FloatComparisonKind::OrderedGreaterThan,
                        FCmpPredicateAttr::OGE => FloatComparisonKind::OrderedGreaterThanOrEqual,
                        FCmpPredicateAttr::OLT => FloatComparisonKind::OrderedLessThan,
                        FCmpPredicateAttr::OLE => FloatComparisonKind::OrderedLessThanOrEqual,
                        FCmpPredicateAttr::ONE => FloatComparisonKind::OrderedNotEqual,
                        FCmpPredicateAttr::UEQ => FloatComparisonKind::UnorderedEqual,
                        FCmpPredicateAttr::UGT => FloatComparisonKind::UnorderedGreaterThan,
                        FCmpPredicateAttr::UGE => FloatComparisonKind::UnorderedGreaterThanOrEqual,
                        FCmpPredicateAttr::ULT => FloatComparisonKind::UnorderedLessThan,
                        FCmpPredicateAttr::ULE => FloatComparisonKind::UnorderedLessThanOrEqual,
                        FCmpPredicateAttr::UNE => FloatComparisonKind::UnorderedNotEqual,
                        _ => {
                            return Err(SelectionError::Unsupported(
                                "ordered/unordered NaN-only comparison",
                            ));
                        }
                    };
                    let destination = self.function.add_value(
                        RegisterClass::Predicate,
                        1,
                        Some(self.predicate_register),
                        ValuePurpose::Temporary,
                    );
                    self.function.instructions.push(Instruction::FloatCompare {
                        comparison,
                        destination,
                        first,
                        second,
                    });
                    self.values
                        .insert(result.unwrap(), SelectedValue::Register(destination, 0));
                }
            }
            "llvm.select" => {
                let condition = self.operand(operand(0))?;
                if let Operand::Immediate(bits) = condition {
                    self.values.insert(
                        result.unwrap(),
                        self.value(operand(if bits == 0 { 2 } else { 1 }))?,
                    );
                } else {
                    let when_true = self.operand(operand(1))?;
                    let when_false = self.operand(operand(2))?;
                    let destination = self.temporary(RegisterClass::Vector, 1);
                    self.function.instructions.push(Instruction::Select {
                        destination,
                        condition,
                        when_true,
                        when_false,
                    });
                    self.values
                        .insert(result.unwrap(), SelectedValue::Register(destination, 0));
                }
            }
            "llvm.gep" => {
                let gep = Operation::get_op::<GetElementPtrOp>(operation_pointer, self.context)
                    .ok_or(SelectionError::Unsupported("invalid GEP"))?;
                let source_type = gep.src_elem_type(self.context);
                if !source_type.deref(self.context).is::<IntegerType>()
                    || source_type
                        .deref(self.context)
                        .downcast_ref::<IntegerType>()
                        .unwrap()
                        .width()
                        != 8
                {
                    return Err(SelectionError::Unsupported(
                        "GEP requires byte-sized elements",
                    ));
                }
                let SelectedValue::Address {
                    mut pair,
                    mut offset,
                } = self.value(operand(0))?
                else {
                    return Err(SelectionError::Unsupported("GEP requires global address"));
                };
                for index in gep.indices(self.context) {
                    match index {
                        GepIndex::Constant(addend) => {
                            if let Some(combined) =
                                offset.checked_add(addend).filter(|value| *value <= 4095)
                            {
                                offset = combined;
                            } else {
                                if offset != 0 {
                                    pair = self.address(pair, Operand::Immediate(offset));
                                }
                                pair = self.address(pair, Operand::Immediate(addend));
                                offset = 0;
                            }
                        }
                        GepIndex::Value(index) => {
                            if offset != 0 {
                                pair = self.address(pair, Operand::Immediate(offset));
                                offset = 0;
                            }
                            let index_operand = self.operand(index)?;
                            pair = self.address(pair, index_operand);
                        }
                    }
                }
                self.values
                    .insert(result.unwrap(), SelectedValue::Address { pair, offset });
            }
            "llvm.load" => {
                let SelectedValue::Address { pair, offset } = self.value(operand(0))? else {
                    return Err(SelectionError::Unsupported("load requires global address"));
                };
                let loaded_type = result.unwrap().get_type(self.context);
                if !loaded_type.deref(self.context).is::<FP32Type>() {
                    return Err(SelectionError::Unsupported("global load requires f32"));
                }
                let destination = self.temporary(RegisterClass::Vector, 1);
                self.function.instructions.push(Instruction::LoadGlobal {
                    destination,
                    register_count: 1,
                    address: Operand::Value {
                        value: pair,
                        element: 0,
                    },
                    offset,
                });
                self.outstanding_loads = true;
                self.values
                    .insert(result.unwrap(), SelectedValue::Register(destination, 0));
            }
            "llvm.call_intrinsic" => {
                let intrinsic =
                    Operation::get_op::<CallIntrinsicOp>(operation_pointer, self.context)
                        .ok_or(SelectionError::Unsupported("invalid intrinsic"))?;
                let name_attribute = intrinsic
                    .get_attr_llvm_intrinsic_name(self.context)
                    .ok_or(SelectionError::Unsupported("intrinsic without name"))?;
                match name_attribute.as_str() {
                    "llvm.amdgcn.interp.p1" | "llvm.amdgcn.interp.p2" => {
                        let second_step = name_attribute.as_str().ends_with(".p2");
                        let (coordinate, channel, attribute, previous_value, primitive_mask) =
                            if second_step {
                                if operation.get_num_operands() != 5 {
                                    return Err(SelectionError::Unsupported(
                                        "interp.p2 argument count",
                                    ));
                                }
                                (
                                    operand(1),
                                    self.constant(operand(2))?,
                                    self.constant(operand(3))?,
                                    Some(self.operand(operand(0))?),
                                    self.operand(operand(4))?,
                                )
                            } else {
                                if operation.get_num_operands() != 4 {
                                    return Err(SelectionError::Unsupported(
                                        "interp.p1 argument count",
                                    ));
                                }
                                (
                                    operand(0),
                                    self.constant(operand(1))?,
                                    self.constant(operand(2))?,
                                    None,
                                    self.operand(operand(3))?,
                                )
                            };
                        if attribute >= 32 || channel >= 4 {
                            return Err(SelectionError::Unsupported(
                                "interpolation attribute/channel out of range",
                            ));
                        }
                        let destination = self.temporary(RegisterClass::Vector, 1);
                        let previous_value = if let Some(previous) = previous_value {
                            self.function.instructions.push(Instruction::Copy {
                                destination,
                                source: previous,
                            });
                            Some(Operand::Value {
                                value: destination,
                                element: 0,
                            })
                        } else {
                            None
                        };
                        self.function.instructions.push(Instruction::Interpolate {
                            second_step,
                            destination,
                            coordinates: self.operand(coordinate)?,
                            previous_value,
                            primitive_mask,
                            attribute,
                            channel,
                        });
                        self.values
                            .insert(result.unwrap(), SelectedValue::Register(destination, 0));
                    }
                    "llvm.amdgcn.exp.f32" => {
                        if operation.get_num_operands() != 8 {
                            return Err(SelectionError::Unsupported("export argument count"));
                        }
                        if self.outstanding_loads {
                            self.function.instructions.push(Instruction::WaitForLoads);
                            self.outstanding_loads = false;
                        }
                        let target = self.constant(operand(0))?;
                        let mask = self.constant(operand(1))?;
                        let done = self.constant(operand(6))? != 0;
                        let valid_mask = self.constant(operand(7))? != 0;
                        if target >= 64 || mask > 15 {
                            return Err(SelectionError::Unsupported(
                                "export target/mask out of range",
                            ));
                        }
                        let mut values = [Operand::Immediate(0); 4];
                        let mut materialized_constants: [Option<(u32, Operand)>; 4] = [None; 4];
                        for channel in 0..4 {
                            if mask & (1 << channel) == 0 {
                                continue;
                            }
                            let argument = operand(channel + 2);
                            values[channel] = if let SelectedValue::Bits(bits) =
                                self.value(argument)?
                            {
                                if let Some((_, previous)) = materialized_constants
                                    .iter()
                                    .flatten()
                                    .find(|(previous_bits, _)| *previous_bits == bits)
                                {
                                    *previous
                                } else {
                                    let materialized = self.vector_operand(argument)?;
                                    materialized_constants[channel] = Some((bits, materialized));
                                    materialized
                                }
                            } else {
                                self.vector_operand(argument)?
                            };
                        }
                        let inactive_source = if mask == 0 {
                            let destination = self.temporary(RegisterClass::Vector, 1);
                            self.function.instructions.push(Instruction::Copy {
                                destination,
                                source: Operand::Immediate(0),
                            });
                            Operand::Value {
                                value: destination,
                                element: 0,
                            }
                        } else {
                            values[mask.trailing_zeros() as usize]
                        };
                        for (channel, value) in values.iter_mut().enumerate() {
                            if mask & (1 << channel) == 0 {
                                *value = inactive_source;
                            }
                        }
                        self.function.instructions.push(Instruction::Export {
                            target,
                            values,
                            mask: mask as u8,
                            done,
                            valid_mask,
                        });
                    }
                    "llvm.amdgcn.kill" => {
                        if operation.get_num_operands() != 1 {
                            return Err(SelectionError::Unsupported("kill argument count"));
                        }
                        let condition = self.operand(operand(0))?;
                        match condition {
                            Operand::Immediate(0) => self
                                .function
                                .instructions
                                .push(Instruction::ZeroExecutionMask),
                            Operand::Immediate(_) => {}
                            _ => self
                                .function
                                .instructions
                                .push(Instruction::Kill { condition }),
                        }
                    }
                    "llvm.exp2.f32" => {
                        if operation.get_num_operands() != 1 {
                            return Err(SelectionError::Unsupported("exp2 argument count"));
                        }
                        let destination = self.temporary(RegisterClass::Vector, 1);
                        self.function
                            .instructions
                            .push(Instruction::ExponentialBaseTwo {
                                destination,
                                source: self.operand(operand(0))?,
                            });
                        self.values
                            .insert(result.unwrap(), SelectedValue::Register(destination, 0));
                    }
                    name => {
                        return Err(SelectionError::UnsupportedOperation(format!(
                            "llvm.call_intrinsic @{name}"
                        )));
                    }
                }
            }
            "llvm.return" => {
                if operation.get_num_operands() != 0 {
                    return Err(SelectionError::Unsupported("non-void return"));
                }
                self.function.instructions.push(Instruction::Return);
            }
            _ => return Err(SelectionError::UnsupportedOperation(operation_name)),
        }
        Ok(())
    }
}

/// Select the shader's `@main` entry point without using LLVM's C library.
pub fn select_instructions(
    context: &Context,
    module: Ptr<Operation>,
    signature: &Signature,
    target: &impl Target,
) -> Result<MachineFunction, SelectionError> {
    let module =
        Operation::get_op::<ModuleOp>(module, context).ok_or(SelectionError::MissingEntry)?;
    let mut entry = None;
    for pointer in module.get_body(context, 0).deref(context).iter(context) {
        if let Some(function) = Operation::get_op::<FuncOp>(pointer, context)
            && function.get_symbol_name(context).as_ref() == "main"
            && entry.replace(function).is_some()
        {
            return Err(SelectionError::InvalidSignature(
                "multiple @main definitions",
            ));
        }
    }
    let entry = entry.ok_or(SelectionError::MissingEntry)?;
    let block = entry
        .get_entry_block(context)
        .ok_or(SelectionError::MissingEntry)?;
    if entry
        .get_region(context)
        .ok_or(SelectionError::MissingEntry)?
        .deref(context)
        .iter(context)
        .count()
        != 1
    {
        return Err(SelectionError::Unsupported("multiple basic blocks"));
    }
    let arguments: Vec<Value> = block.deref(context).arguments().collect();
    if arguments.len() != signature.parameters.len()
        || signature.number_of_scalar_parameters as usize > arguments.len()
    {
        return Err(SelectionError::InvalidSignature("parameter count"));
    }
    let mut selector = Selector {
        context,
        function: MachineFunction::default(),
        values: HashMap::new(),
        predicate_register: target.predicate_register_base(),
        outstanding_loads: false,
    };
    for (index, &argument) in arguments.iter().enumerate() {
        let parameter_type = argument.get_type(context);
        let (register_count, vector) = if let Some(pointer) =
            parameter_type.deref(context).downcast_ref::<PointerType>()
        {
            if pointer.address_space() != 1 {
                return Err(SelectionError::Unsupported("non-global pointer ABI"));
            }
            (2, false)
        } else if let Some(vector_type) = parameter_type.deref(context).downcast_ref::<VectorType>()
        {
            if vector_type.num_elements() > u8::MAX as u32
                || vector_type.is_scalable()
                || !vector_type.elem_type().deref(context).is::<FP32Type>()
            {
                return Err(SelectionError::Unsupported("vector ABI type"));
            }
            (vector_type.num_elements() as u8, true)
        } else if parameter_type.deref(context).is::<FP32Type>() {
            (1, false)
        } else if let Some(integer) = parameter_type.deref(context).downcast_ref::<IntegerType>() {
            if integer.width() != 32 {
                return Err(SelectionError::Unsupported("integer ABI type"));
            }
            (1, false)
        } else {
            return Err(SelectionError::Unsupported("parameter ABI type"));
        };
        if signature.parameters[index].size != register_count
            || signature.parameters[index].vector != vector
        {
            return Err(SelectionError::InvalidSignature("parameter type"));
        }
        if vector && argument.num_uses(context) == 0 {
            continue;
        }
        // The builder emits unused barycentric extracts even for pixel
        // variants that never interpolate. These inputs are unavailable unless
        // the shader actually consumes them.
        let unused_coordinates = signature.stage == ShaderStage::Pixel
            && vector
            && argument.uses(context).iter().all(|usage| {
                let pointer = usage.user_op();
                Operation::get_op::<ExtractElementOp>(pointer, context).is_some()
                    && pointer
                        .deref(context)
                        .results()
                        .all(|result| result.num_uses(context) == 0)
            });
        if unused_coordinates {
            continue;
        }
        let (register_class, fixed_register) =
            match target.parameter_location(signature, index as u32) {
                ParameterLocation::Fixed(register_class, register) => (register_class, register),
                ParameterLocation::Unavailable => {
                    return Err(SelectionError::Unsupported(
                        "used unavailable ABI parameter",
                    ));
                }
            };
        let parameter_value = selector.function.add_value(
            register_class,
            register_count,
            Some(fixed_register),
            ValuePurpose::Parameter(index as u32),
        );
        selector.values.insert(
            argument,
            if parameter_type.deref(context).is::<PointerType>() {
                SelectedValue::Address {
                    pair: parameter_value,
                    offset: 0,
                }
            } else {
                SelectedValue::Register(parameter_value, 0)
            },
        );
    }
    for operation in block.deref(context).iter(context) {
        selector.select_operation(operation)?;
    }
    if !matches!(
        selector.function.instructions.last(),
        Some(Instruction::Return)
    ) {
        return Err(SelectionError::Unsupported("entry point without return"));
    }
    Ok(selector.function)
}

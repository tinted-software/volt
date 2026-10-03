#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod addrfold;
pub mod ir;
pub mod wimmer;

pub use addrfold::{Addr, Analysis, analyze};
pub use ir::Func;
pub use wimmer::{
    Action, ActionKind, AllocError, Allocation, CallSite, ClassRegs, EdgeMoves, FixedAssign,
    Interval, Location, Move, Range, RegClass, RegDescription, RegWidth, Segment, UseKind, UsePos,
    UsedSaved, Violation, ViolationKind, allocate, build_intervals, order_moves, verify_intervals,
};

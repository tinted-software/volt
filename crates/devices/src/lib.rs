//! Memory-mapped devices behind the guest bus: address routing plus models.
extern crate alloc;

pub mod bus;
pub mod pl011;

pub use bus::{Bus, BusDevice, Device};
pub use pl011::{LEN as PL011_LEN, Pl011};

extern crate alloc;

use alloc::vec::Vec;

#[derive(Debug, Default)]
pub struct Pool<T> {
    items: Vec<T>,
}

impl<T> Pool<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn append(&mut self, item: T) -> u32 {
        let index = self.items.len() as u32;
        self.items.push(item);
        index
    }

    pub fn get(&self, index: u32) -> &T {
        &self.items[index as usize]
    }

    pub fn get_mut(&mut self, index: u32) -> &mut T {
        &mut self.items[index as usize]
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_round_trip() {
        let mut pool = Pool::new();
        let a = pool.append(10u32);
        let b = pool.append(20u32);
        assert_ne!(a, b);
        assert_eq!(10, *pool.get(a));
        assert_eq!(20, *pool.get(b));
    }
}

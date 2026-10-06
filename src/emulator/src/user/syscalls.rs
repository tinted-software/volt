// Copyright 2026 LilithSemi
// SPDX-License-Identifier: Apache-2.0
// Modified: Rust port of Mirage lib/mirage-jit/user/Syscalls.zig.
//! Linux AArch64 syscall ABI with owned host descriptors and translated memory.
//! Signals, clone/futex/threads, file-backed/fixed mappings remain unsupported.
use super::{Error, loader::round_page, memory::Space};
use crate::aarch64::Cpu;
use crate::memory::GuestMemory;
use core::mem::MaybeUninit;

const PAGE: u64 = 4096;
const USER_LIMIT: u64 = 0x0001_0000_0000_0000;
const MAX_RW_COUNT: u64 = 0x7fff_f000;
const IOV_MAX: usize = 1024;

pub struct State {
    fds: Vec<Option<libc::c_int>>,
    brk: u64,
    brk_min: u64,
    mapped_brk: u64,
    mmap_next: u64,
    clear_child_tid: u64,
}
impl Drop for State {
    fn drop(&mut self) {
        for fd in self.fds.iter().flatten() {
            unsafe {
                libc::close(*fd);
            }
        }
    }
}
impl State {
    pub fn new(brk: u64) -> Result<Self, Error> {
        if !cfg!(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )) {
            return Err(Error::UnsupportedHost);
        }
        let mut state = Self {
            fds: vec![None; 3],
            brk,
            brk_min: brk,
            mapped_brk: round_page(brk)?,
            mmap_next: 0x2000_0000_0000,
            clear_child_tid: 0,
        };
        for fd in 0..3 {
            let owned = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
            if owned >= 0 {
                state.fds[fd as usize] = Some(owned);
            } else {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EBADF) {
                    return Err(error.into());
                }
            }
        }
        Ok(state)
    }
    pub fn dispatch(&mut self, space: &mut Space, cpu: &mut Cpu) -> Result<Option<u8>, Error> {
        let a: [u64; 6] = cpu.x[..6].try_into().unwrap();
        if cpu.x[8] == 93 || cpu.x[8] == 94 {
            if self.clear_child_tid != 0 {
                let _ = space.write(self.clear_child_tid, &[0; 4]);
            }
            cpu.x[0] = 0;
            return Ok(Some(a[0] as u8));
        }
        cpu.x[0] = match cpu.x[8] {
            56 => self.openat(space, a),
            57 => self.close(a[0]),
            62 => match self.host_fd(a[0]) {
                Some(fd) => host_result(unsafe { libc::lseek(fd, a[1] as i64, a[2] as i32) }),
                None => failure(libc::EBADF),
            },
            63 => self.read_write(space, a, false),
            64 => self.read_write(space, a, true),
            66 => self.writev(space, a),
            96 => {
                self.clear_child_tid = a[0];
                host_result(unsafe { libc::syscall(libc::SYS_gettid) } as i64)
            }
            113 => self.clock_gettime(space, a),
            160 => self.uname(space, a[0]),
            172 => (unsafe { libc::getpid() }) as u64,
            174 => (unsafe { libc::getuid() }) as u64,
            175 => (unsafe { libc::geteuid() }) as u64,
            176 => (unsafe { libc::getgid() }) as u64,
            177 => (unsafe { libc::getegid() }) as u64,
            178 => host_result(unsafe { libc::syscall(libc::SYS_gettid) } as i64),
            214 => self.change_brk(space, a[0]),
            215 => self.unmap(space, a),
            222 => self.mmap(space, a),
            226 => self.mprotect(space, a),
            278 => self.getrandom(space, a),
            _ => failure(libc::ENOSYS),
        };
        Ok(None)
    }
    fn host_fd(&self, guest: u64) -> Option<libc::c_int> {
        let index = guest as u32 as i32;
        if index < 0 {
            return None;
        }
        self.fds.get(index as usize).copied().flatten()
    }
    fn read_write(&self, space: &mut Space, a: [u64; 6], writing: bool) -> u64 {
        let Some(fd) = self.host_fd(a[0]) else {
            return failure(libc::EBADF);
        };
        let count = a[2].min(MAX_RW_COUNT) as usize;
        if space
            .check(a[1], count, if writing { 1 } else { 2 })
            .is_err()
        {
            return failure(libc::EFAULT);
        }
        if let Ok(bytes) = space.slice_mut(a[1], count) {
            return host_result(unsafe {
                if writing {
                    libc::write(fd, bytes.as_ptr().cast(), bytes.len())
                } else {
                    libc::read(fd, bytes.as_mut_ptr().cast(), bytes.len())
                }
            } as i64);
        }
        let mut vectors = [libc::iovec {
            iov_base: core::ptr::null_mut(),
            iov_len: 0,
        }; IOV_MAX];
        let Ok(n) = vectors_for(space, a[1], count, &mut vectors) else {
            return failure(libc::EFAULT);
        };
        host_result(unsafe {
            if writing {
                libc::writev(fd, vectors.as_ptr(), n as i32)
            } else {
                libc::readv(fd, vectors.as_ptr(), n as i32)
            }
        } as i64)
    }
    fn writev(&self, space: &mut Space, a: [u64; 6]) -> u64 {
        let Some(fd) = self.host_fd(a[0]) else {
            return failure(libc::EBADF);
        };
        let count = a[2] as u32 as i32;
        if !(0..=IOV_MAX as i32).contains(&count) {
            return failure(libc::EINVAL);
        }
        let count = count as usize;
        let mut guest = [0u8; IOV_MAX * 16];
        if space.read(a[1], &mut guest[..count * 16]).is_err() {
            return failure(libc::EFAULT);
        }
        let mut vectors = [libc::iovec {
            iov_base: core::ptr::null_mut(),
            iov_len: 0,
        }; IOV_MAX];
        let mut used = 0;
        let mut total = 0u64;
        let mut remaining = MAX_RW_COUNT as usize;
        for pair in guest[..count * 16].chunks_exact(16) {
            let address = u64::from_le_bytes(pair[..8].try_into().unwrap());
            let len = u64::from_le_bytes(pair[8..].try_into().unwrap());
            let Some(sum) = total.checked_add(len) else {
                return failure(libc::EINVAL);
            };
            total = sum;
            if total > i64::MAX as u64 {
                return failure(libc::EINVAL);
            }
            let size = len.min(remaining as u64) as usize;
            if size == 0 || used == vectors.len() {
                continue;
            }
            if space.check(address, size, 1).is_err() {
                return failure(libc::EFAULT);
            }
            let Ok(n) = vectors_for(space, address, size, &mut vectors[used..]) else {
                return failure(libc::EFAULT);
            };
            used += n;
            remaining -= size;
        }
        host_result(unsafe { libc::writev(fd, vectors.as_ptr(), used as i32) } as i64)
    }
    fn openat(&mut self, space: &Space, a: [u64; 6]) -> u64 {
        let mut path = [0u8; 4097];
        let mut len = 0;
        while len < 4096 {
            let Some(address) = a[1].checked_add(len as u64) else {
                return failure(libc::EFAULT);
            };
            if space.read(address, &mut path[len..len + 1]).is_err() {
                return failure(libc::EFAULT);
            }
            if path[len] == 0 {
                break;
            }
            len += 1;
        }
        if len == 4096 {
            return failure(libc::ENAMETOOLONG);
        }
        if len == 0 {
            return failure(libc::ENOENT);
        }
        let guest_dir = a[0] as u32 as i32;
        let dir = if path[0] == b'/' || guest_dir == libc::AT_FDCWD {
            libc::AT_FDCWD
        } else {
            match self.host_fd(a[0]) {
                Some(fd) => fd,
                None => return failure(libc::EBADF),
            }
        };
        let guest_flags = a[2] as u32;
        let known = 3
            | 0x40
            | 0x80
            | 0x100
            | 0x200
            | 0x400
            | 0x800
            | 0x1000
            | 0x2000
            | 0x4000
            | 0x8000
            | 0x10000
            | 0x20000
            | 0x40000
            | 0x80000
            | 0x100000
            | 0x200000
            | 0x400000;
        if guest_flags & !known != 0 {
            return failure(libc::EINVAL);
        }
        let mut flags = (guest_flags & !0x3c000) as i32;
        if guest_flags & 0x4000 != 0 {
            flags |= libc::O_DIRECTORY;
        }
        if guest_flags & 0x8000 != 0 {
            flags |= libc::O_NOFOLLOW;
        }
        if guest_flags & 0x10000 != 0 {
            flags |= libc::O_DIRECT;
        }
        flags |= libc::O_CLOEXEC;
        let fd = unsafe { libc::openat(dir, path.as_ptr().cast(), flags, a[3] as libc::mode_t) };
        if fd < 0 {
            return host_result(i64::from(fd));
        }
        if let Some(index) = self.fds.iter().position(Option::is_none) {
            self.fds[index] = Some(fd);
            return index as u64;
        }
        if self.fds.len() > i32::MAX as usize {
            unsafe {
                libc::close(fd);
            }
            return failure(libc::EMFILE);
        }
        if self.fds.try_reserve(1).is_err() {
            unsafe {
                libc::close(fd);
            }
            return failure(libc::ENOMEM);
        }
        self.fds.push(Some(fd));
        (self.fds.len() - 1) as u64
    }
    fn close(&mut self, guest: u64) -> u64 {
        let Some(fd) = self.host_fd(guest) else {
            return failure(libc::EBADF);
        };
        self.fds[guest as u32 as usize] = None;
        host_result(i64::from(unsafe { libc::close(fd) }))
    }
    fn clock_gettime(&self, space: &mut Space, a: [u64; 6]) -> u64 {
        if space.check(a[1], 16, 2).is_err() {
            return failure(libc::EFAULT);
        }
        let mut id = a[0] as u32;
        if (id as i32) < 0 && id & 7 == 3 {
            let Some(fd) = self.host_fd(u64::from(!id >> 3)) else {
                return failure(libc::EBADF);
            };
            id = (!(fd as u32) << 3) | 3;
        }
        let mut time = MaybeUninit::<libc::timespec>::uninit();
        let result = unsafe { libc::clock_gettime(id as libc::clockid_t, time.as_mut_ptr()) };
        if result < 0 {
            return host_result(i64::from(result));
        }
        let time = unsafe { time.assume_init() };
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&(time.tv_sec as i64).to_le_bytes());
        bytes[8..].copy_from_slice(&(time.tv_nsec as i64).to_le_bytes());
        if space.write(a[1], &bytes).is_err() {
            return failure(libc::EFAULT);
        }
        0
    }
    fn uname(&self, space: &mut Space, address: u64) -> u64 {
        if space.check(address, 6 * 65, 2).is_err() {
            return failure(libc::EFAULT);
        }
        let mut host = MaybeUninit::<libc::utsname>::uninit();
        let result = unsafe { libc::uname(host.as_mut_ptr()) };
        if result < 0 {
            return host_result(i64::from(result));
        }
        let host = unsafe { host.assume_init() };
        let mut bytes = [0; 6 * 65];
        for (i, field) in [
            &host.sysname,
            &host.nodename,
            &host.release,
            &host.version,
            &host.machine,
            &host.domainname,
        ]
        .into_iter()
        .enumerate()
        {
            for (out, value) in bytes[i * 65..(i + 1) * 65].iter_mut().zip(field) {
                *out = *value as u8;
            }
        }
        bytes[260..325].fill(0);
        bytes[260..267].copy_from_slice(b"aarch64");
        if space.write(address, &bytes).is_err() {
            return failure(libc::EFAULT);
        }
        0
    }
    fn getrandom(&self, space: &mut Space, a: [u64; 6]) -> u64 {
        let count = a[1].min(MAX_RW_COUNT) as usize;
        if space.check(a[0], count, 2).is_err() {
            return failure(libc::EFAULT);
        }
        let len = if space.slice(a[0], count).is_ok() {
            count
        } else {
            match space.chunk_len(a[0], count) {
                Ok(n) => n,
                Err(_) => return failure(libc::EFAULT),
            }
        };
        let Ok(bytes) = space.slice_mut(a[0], len) else {
            return failure(libc::EFAULT);
        };
        host_result(
            unsafe { libc::getrandom(bytes.as_mut_ptr().cast(), bytes.len(), a[2] as u32) } as i64,
        )
    }
    fn change_brk(&mut self, space: &mut Space, requested: u64) -> u64 {
        if requested == 0 || requested < self.brk_min || requested >= USER_LIMIT {
            return self.brk;
        }
        let Ok(end) = round_page(requested) else {
            return self.brk;
        };
        if end > self.mapped_brk {
            if space
                .map(self.mapped_brk, (end - self.mapped_brk) as usize)
                .is_err()
            {
                return self.brk;
            }
        } else if end < self.mapped_brk
            && space.unmap(end, (self.mapped_brk - end) as usize).is_err()
        {
            return self.brk;
        }
        self.brk = requested;
        self.mapped_brk = end;
        requested
    }
    fn mmap(&mut self, space: &mut Space, a: [u64; 6]) -> u64 {
        if a[1] == 0 {
            return failure(libc::EINVAL);
        }
        let prot = a[2] as u32;
        if prot & !7 != 0 {
            return failure(libc::EINVAL);
        }
        let flags = a[3] as u32;
        if flags & 0x20 == 0 {
            return failure(libc::ENOSYS);
        }
        if flags & 3 != 1 && flags & 3 != 2 {
            return failure(libc::EINVAL);
        }
        if flags & !0x23 != 0 {
            return failure(libc::ENOSYS);
        }
        if a[5] % PAGE != 0 {
            return failure(libc::EINVAL);
        }
        let Ok(len) = round_page(a[1]) else {
            return failure(libc::ENOMEM);
        };
        if len >= USER_LIMIT {
            return failure(libc::ENOMEM);
        }
        let mut address = self.mmap_next;
        loop {
            let Some(end) = address.checked_add(len) else {
                return failure(libc::ENOMEM);
            };
            if end > USER_LIMIT {
                return failure(libc::ENOMEM);
            }
            if let Some(region) = space
                .regions
                .iter()
                .find(|r| address < r.end() && end > r.address)
            {
                let Ok(next) = round_page(region.end()) else {
                    return failure(libc::ENOMEM);
                };
                address = next;
            } else {
                break;
            }
        }
        if space.map(address, len as usize).is_err() {
            return failure(libc::ENOMEM);
        }
        if space.protect(address, len as usize, prot as u8).is_err() {
            let _ = space.unmap(address, len as usize);
            return failure(libc::ENOMEM);
        }
        self.mmap_next = address + len;
        address
    }
    fn unmap(&self, space: &mut Space, a: [u64; 6]) -> u64 {
        if a[0] % PAGE != 0 || a[1] == 0 {
            return failure(libc::EINVAL);
        }
        let Ok(len) = round_page(a[1]) else {
            return failure(libc::EINVAL);
        };
        if a[0].checked_add(len).is_none() {
            return failure(libc::EINVAL);
        }
        if space.unmap(a[0], len as usize).is_err() {
            return failure(libc::ENOMEM);
        }
        0
    }
    fn mprotect(&self, space: &mut Space, a: [u64; 6]) -> u64 {
        if a[0] % PAGE != 0 {
            return failure(libc::EINVAL);
        }
        let prot = a[2] as u32;
        if prot & !7 != 0 {
            return failure(libc::EINVAL);
        }
        let Ok(len) = round_page(a[1]) else {
            return failure(libc::EINVAL);
        };
        if a[0].checked_add(len).is_none() {
            return failure(libc::EINVAL);
        }
        if len == 0 {
            return 0;
        }
        if space.protect(a[0], len as usize, prot as u8).is_err() {
            return failure(libc::ENOMEM);
        }
        0
    }
}
fn vectors_for(
    space: &mut Space,
    address: u64,
    len: usize,
    vectors: &mut [libc::iovec],
) -> Result<usize, crate::memory::MemoryError> {
    let mut offset = 0;
    let mut count = 0;
    while offset < len && count < vectors.len() {
        let at = address + offset as u64;
        let n = space.chunk_len(at, len - offset)?;
        // Backing allocations remain stable throughout the host syscall. These
        // raw pointers never escape dispatch and are not guest addresses.
        let pointer = space.host_pointer(at, n)?;
        vectors[count] = libc::iovec {
            iov_base: pointer.cast(),
            iov_len: n,
        };
        count += 1;
        offset += n;
    }
    Ok(count)
}
fn failure(errno: i32) -> u64 {
    (-i64::from(errno)) as u64
}
fn host_result(result: i64) -> u64 {
    if result == -1 {
        failure(
            std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        )
    } else {
        result as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn call(state: &mut State, space: &mut Space, number: u64, a: [u64; 6]) -> u64 {
        let mut cpu = Cpu::default();
        cpu.x[..6].copy_from_slice(&a);
        cpu.x[8] = number;
        assert_eq!(state.dispatch(space, &mut cpu).unwrap(), None);
        cpu.x[0]
    }
    #[test]
    fn syscall_errors_permissions_and_brk() {
        let mut space = Space::new();
        let mut state = State::new(0x10000).unwrap();
        assert_eq!(
            call(&mut state, &mut space, 65535, [0; 6]),
            failure(libc::ENOSYS)
        );
        assert_eq!(
            call(&mut state, &mut space, 63, [99999, 0, 1, 0, 0, 0]),
            failure(libc::EBADF)
        );
        assert_eq!(
            call(&mut state, &mut space, 214, [0x11000, 0, 0, 0, 0, 0]),
            0x11000
        );
        assert_eq!(
            call(&mut state, &mut space, 214, [0x12000, 0, 0, 0, 0, 0]),
            0x12000
        );
        space
            .write(0x10ffd, &0x123456789abcdef0u64.to_le_bytes())
            .unwrap();
        let mut bytes = [0; 8];
        space.read(0x10ffd, &mut bytes).unwrap();
        assert_eq!(u64::from_le_bytes(bytes), 0x123456789abcdef0);
        assert_eq!(
            call(&mut state, &mut space, 226, [0x10000, 4096, 1, 0, 0, 0]),
            0
        );
        assert!(space.write(0x10000, &[1]).is_err());
        assert_eq!(
            call(&mut state, &mut space, 215, [0x10000, 4096, 0, 0, 0, 0]),
            0
        );
        assert!(space.read(0x10000, &mut [0]).is_err());
        assert_eq!(
            call(&mut state, &mut space, 222, [0, 4096, 3, 0x32, 0, 0]),
            failure(libc::ENOSYS)
        );
    }
}

#[cfg(test)]
mod host_io_tests {
    use super::*;
    #[test]
    fn real_pipe_io_crosses_mappings_and_preserves_guest_descriptors() {
        let mut space = Space::new();
        space.map(0x1000, 4).unwrap();
        space.map(0x1004, 4).unwrap();
        space.map(0x2000, 32).unwrap();
        space.write(0x1000, b"abcdefgh").unwrap();
        let mut state = State::new(0x10000).unwrap();
        let mut pipe = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        state.fds.push(Some(pipe[0]));
        state.fds.push(Some(pipe[1]));
        assert_eq!(
            state.read_write(&mut space, [4, 0x1000, 8, 0, 0, 0], true),
            8
        );
        space.write(0x1000, &[0; 8]).unwrap();
        assert_eq!(
            state.read_write(&mut space, [3, 0x1000, 8, 0, 0, 0], false),
            8
        );
        let mut bytes = [0; 8];
        space.read(0x1000, &mut bytes).unwrap();
        assert_eq!(&bytes, b"abcdefgh");
        let mut vectors = [0; 32];
        vectors[..8].copy_from_slice(&0x1000u64.to_le_bytes());
        vectors[8..16].copy_from_slice(&4u64.to_le_bytes());
        vectors[16..24].copy_from_slice(&0x1002u64.to_le_bytes());
        vectors[24..].copy_from_slice(&4u64.to_le_bytes());
        space.write(0x2000, &vectors).unwrap();
        assert_eq!(state.writev(&mut space, [4, 0x2000, 2, 0, 0, 0]), 8);
        assert_eq!(
            unsafe { libc::read(pipe[0], bytes.as_mut_ptr().cast(), 8) },
            8
        );
        assert_eq!(&bytes, b"abcdcdef");
        assert_eq!(state.close(4), 0);
        assert_eq!(state.close(4), failure(libc::EBADF));
        assert_eq!(
            state.read_write(&mut space, [4, 0x1000, 8, 0, 0, 0], true),
            failure(libc::EBADF)
        );
    }
}

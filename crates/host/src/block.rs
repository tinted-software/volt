use std::{
    fs::{File, OpenOptions},
    path::Path,
};
use volt_platform::BlockBackend;

/// Host file-backed block storage for virtio-blk, streaming sectors on demand.
pub struct FileBackend {
    file: File,
    len: u64,
}

impl FileBackend {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .or_else(|_| File::open(path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        Ok(Self { file, len })
    }
}

impl BlockBackend for FileBackend {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset).map_err(|_| ())
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset)).map_err(|_| ())?;
            self.file.read_exact(buf).map_err(|_| ())
        }
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(buf, offset).map_err(|_| ())
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom, Write};
            self.file.seek(SeekFrom::Start(offset)).map_err(|_| ())?;
            self.file.write_all(buf).map_err(|_| ())
        }
    }

    fn flush(&mut self) -> Result<(), ()> {
        self.file.sync_all().map_err(|_| ())
    }
}

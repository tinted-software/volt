use std::io::{BufWriter, Stdout, Write};
use volt_platform::Console;

/// Buffered stdout sink implementing [`Console`].
pub struct StdoutConsole {
    output: BufWriter<Stdout>,
}

impl Default for StdoutConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl StdoutConsole {
    pub fn new() -> Self {
        Self {
            output: BufWriter::new(std::io::stdout()),
        }
    }
}

impl Console for StdoutConsole {
    fn write_byte(&mut self, byte: u8) {
        let _ = self.output.write_all(&[byte]);
        if byte == b'\n' {
            let _ = self.output.flush();
        }
    }

    fn flush(&mut self) {
        let _ = self.output.flush();
    }
}

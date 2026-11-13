ENTRY(_start)

SECTIONS
{
    . = 0x40100000;

    .text : ALIGN(4096) {
        *(.text.boot)
        *(.text .text.*)
    }

    .rodata : ALIGN(4096) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(4096) {
        *(.data .data.*)
    }

    .bss (NOLOAD) : ALIGN(4096) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }

    . = ALIGN(4096);
    __stack_bottom = .;
    . += 0x40000; /* 256 KiB stack */
    __stack_top = .;
}

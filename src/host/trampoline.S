# Shared tail for unimplemented-import trampolines.
#
# Each trampoline loads its ImportInfo* into %rax and jumps here. We save the six integer
# argument registers, call lpn_on_unimplemented(info, regs) and return its result to the
# guest as if the import had run.

    .text
    .globl lpn_unimplemented_stub
    .type lpn_unimplemented_stub, @function
lpn_unimplemented_stub:
    push %r9
    push %r8
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    mov %rax, %rdi          # ImportInfo*
    mov %rsp, %rsi          # uint64_t regs[6] = rdi, rsi, rdx, rcx, r8, r9
    sub $8, %rsp            # 8 (return address) + 48 + 8 keeps %rsp 16-byte aligned
    call lpn_on_unimplemented
    add $8, %rsp
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %r8
    pop %r9
    ret
    .size lpn_unimplemented_stub, . - lpn_unimplemented_stub

    .section .note.GNU-stack, "", @progbits

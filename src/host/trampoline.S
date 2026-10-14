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

# Shared tail for traced HLE imports (--trace-hle). The trampoline puts its ImportInfo* in
# %r10. We save every argument register (integer and vector, plus %rax for varargs), call
# lpn_on_traced(info), which returns the real implementation, and tail-jump to it.

    .globl lpn_trace_stub
    .type lpn_trace_stub, @function
lpn_trace_stub:
    push %rax
    push %r9
    push %r8
    push %rcx
    push %rdx
    push %rsi
    push %rdi               # 8 (return address) + 56 keeps %rsp 16-byte aligned
    sub $128, %rsp
    movdqu %xmm0, 0(%rsp)
    movdqu %xmm1, 16(%rsp)
    movdqu %xmm2, 32(%rsp)
    movdqu %xmm3, 48(%rsp)
    movdqu %xmm4, 64(%rsp)
    movdqu %xmm5, 80(%rsp)
    movdqu %xmm6, 96(%rsp)
    movdqu %xmm7, 112(%rsp)
    mov %r10, %rdi
    call lpn_on_traced
    mov %rax, %r11
    movdqu 0(%rsp), %xmm0
    movdqu 16(%rsp), %xmm1
    movdqu 32(%rsp), %xmm2
    movdqu 48(%rsp), %xmm3
    movdqu 64(%rsp), %xmm4
    movdqu 80(%rsp), %xmm5
    movdqu 96(%rsp), %xmm6
    movdqu 112(%rsp), %xmm7
    add $128, %rsp
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %r8
    pop %r9
    pop %rax
    jmp *%r11
    .size lpn_trace_stub, . - lpn_trace_stub

    .section .note.GNU-stack, "", @progbits

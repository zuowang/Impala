	.file	"mul.c"
	.section	.text.unlikely._Z19int128_mul_overflownnPn,"axG",@progbits,_Z19int128_mul_overflownnPn,comdat
.LCOLDB0:
	.section	.text._Z19int128_mul_overflownnPn,"axG",@progbits,_Z19int128_mul_overflownnPn,comdat
.LHOTB0:
	.p2align 4,,15
	.weak	_Z19int128_mul_overflownnPn
	.type	_Z19int128_mul_overflownnPn, @function
_Z19int128_mul_overflownnPn:
.LFB0:
	.cfi_startproc
	movq	%rdi, %r10
	xorl	%r9d, %r9d
	movq	%rdx, %rdi
	movq	%r10, %rax
	sarq	$63, %rdx
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	sarq	$63, %rax
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	cmpq	%rsi, %rax
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	jne	.L4
	cmpq	%rcx, %rdx
	jne	.L5
	movq	%r10, %rax
	imulq	%rdi
	movq	%rax, %rdi
	movq	%rdx, %rax
.L2:
	movq	%rax, 8(%r8)
	movl	%r9d, %eax
	movq	%rdi, (%r8)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	andl	$1, %eax
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L4:
	.cfi_restore_state
	cmpq	%rcx, %rdx
	jne	.L7
	movq	%r10, -16(%rsp)
	movq	%rsi, -8(%rsp)
	movq	%rsi, %rbp
	movq	%rdi, %rbx
.L6:
	movq	%r10, %rax
	mulq	%rdi
	movq	%rax, %r11
	movq	%rbx, %rax
	movq	%rdx, %r12
	mulq	%rbp
	testq	%rbp, %rbp
	movq	%rax, %r13
	movq	%rdx, %r14
	jns	.L8
	xorl	%eax, %eax
	subq	%rax, %r13
	sbbq	%rbx, %r14
.L8:
	testq	%rbx, %rbx
	jns	.L9
	subq	-16(%rsp), %r13
	sbbq	-8(%rsp), %r14
.L9:
	movq	%r12, %rax
	xorl	%edx, %edx
	addq	%r13, %rax
	movq	%rax, %rbp
	adcq	%r14, %rdx
	sarq	$63, %rbp
	cmpq	%rdx, %rbp
	jne	.L10
	movq	%r11, %rdi
	jmp	.L2
	.p2align 4,,10
	.p2align 3
.L5:
	movq	%rdi, -16(%rsp)
	movq	%rcx, -8(%rsp)
	movq	%rcx, %rbp
	movq	%r10, %rbx
	jmp	.L6
	.p2align 4,,10
	.p2align 3
.L7:
	movq	%rsi, %rbx
	movq	%rcx, %rax
	imulq	%rdi, %rbx
	imulq	%r10, %rax
	addq	%rax, %rbx
	movq	%r10, %rax
	mulq	%rdi
	leaq	(%rbx,%rdx), %r13
	leaq	1(%rsi), %rdx
	movq	%rax, %rdi
	cmpq	$1, %rdx
	movq	%r13, %rax
	ja	.L3
	leaq	1(%rcx), %rdx
	cmpq	$1, %rdx
	ja	.L3
	cmpq	%rcx, %rsi
	jne	.L11
	testq	%r13, %r13
	jns	.L2
.L3:
	movl	$1, %r9d
	jmp	.L2
	.p2align 4,,10
	.p2align 3
.L11:
	testq	%r13, %r13
	js	.L2
	jmp	.L3
.L10:
	imulq	%rdi, %rsi
	movq	%r10, %rax
	imulq	%r10, %rcx
	mulq	%rdi
	addq	%rsi, %rcx
	addq	%rdx, %rcx
	movq	%rax, %rdi
	movq	%rcx, %rax
	jmp	.L3
	.cfi_endproc
.LFE0:
	.size	_Z19int128_mul_overflownnPn, .-_Z19int128_mul_overflownnPn
	.section	.text.unlikely._Z19int128_mul_overflownnPn,"axG",@progbits,_Z19int128_mul_overflownnPn,comdat
.LCOLDE0:
	.section	.text._Z19int128_mul_overflownnPn,"axG",@progbits,_Z19int128_mul_overflownnPn,comdat
.LHOTE0:
	.section	.text.unlikely._Z19int128_add_overflownnPn,"axG",@progbits,_Z19int128_add_overflownnPn,comdat
.LCOLDB1:
	.section	.text._Z19int128_add_overflownnPn,"axG",@progbits,_Z19int128_add_overflownnPn,comdat
.LHOTB1:
	.p2align 4,,15
	.weak	_Z19int128_add_overflownnPn
	.type	_Z19int128_add_overflownnPn, @function
_Z19int128_add_overflownnPn:
.LFB1:
	.cfi_startproc
	movq	%rsi, %r10
	movq	%rdi, %rsi
	movq	%rdi, %r9
	xorl	%eax, %eax
	movq	%r10, %rdi
	addq	%rdx, %rsi
	adcq	%rcx, %rdi
	testq	%rcx, %rcx
	js	.L17
	cmpq	%r10, %rdi
	jle	.L20
.L15:
	movq	%rsi, (%r8)
	movq	%rdi, 8(%r8)
	andl	$1, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L17:
	cmpq	%r10, %rdi
	jl	.L15
	jle	.L21
.L16:
	movl	$1, %eax
	movq	%rsi, (%r8)
	movq	%rdi, 8(%r8)
	andl	$1, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L20:
	jl	.L16
	cmpq	%r9, %rsi
	jnb	.L15
	jmp	.L16
	.p2align 4,,10
	.p2align 3
.L21:
	cmpq	%r9, %rsi
	jbe	.L15
	jmp	.L16
	.cfi_endproc
.LFE1:
	.size	_Z19int128_add_overflownnPn, .-_Z19int128_add_overflownnPn
	.section	.text.unlikely._Z19int128_add_overflownnPn,"axG",@progbits,_Z19int128_add_overflownnPn,comdat
.LCOLDE1:
	.section	.text._Z19int128_add_overflownnPn,"axG",@progbits,_Z19int128_add_overflownnPn,comdat
.LHOTE1:
	.ident	"GCC: (Ubuntu 5.3.0-3ubuntu1~14.04) 5.3.0 20151204"
	.section	.note.GNU-stack,"",@progbits

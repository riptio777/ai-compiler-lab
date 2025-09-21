	.build_version macos, 15, 0	sdk_version 15, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_mat4x4_mul_16acc               ; -- Begin function mat4x4_mul_16acc
	.p2align	2
_mat4x4_mul_16acc:                      ; @mat4x4_mul_16acc
	.cfi_startproc
; %bb.0:                                ; %entry
	add	x8, x1, #8
	add	x9, x0, #32
	movi.2d	v1, #0000000000000000
	mov	w10, #4                         ; =0x4
	movi.2d	v3, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v0, #0000000000000000
	subs	x10, x10, #1
	b.lo	LBB0_2
LBB0_1:                                 ; %for.inc
                                        ; =>This Inner Loop Header: Depth=1
	add	x11, x9, #16
	ldp	s5, s4, [x8]
	sub	x12, x9, #16
	ldur	s6, [x9, #-32]
	ld1.s	{ v6 }[1], [x12]
	ldp	s16, s7, [x8, #-8]
	ld1.s	{ v6 }[2], [x9], #4
	ld1.s	{ v6 }[3], [x11]
	fmla.4s	v1, v6, v16[0]
	fmla.4s	v3, v6, v7[0]
	fmla.4s	v2, v6, v5[0]
	fmla.4s	v0, v6, v4[0]
	add	x8, x8, #16
	subs	x10, x10, #1
	b.hs	LBB0_1
LBB0_2:                                 ; %for.end
	mov	s4, v1[1]
	mov	s5, v1[2]
	mov	s6, v1[3]
	stp	s1, s3, [x2]
	mov	s1, v3[1]
	mov	s7, v3[2]
	mov	s3, v3[3]
	stp	s2, s0, [x2, #8]
	mov	s16, v2[1]
	mov	s17, v2[2]
	mov	s2, v2[3]
	stp	s4, s1, [x2, #16]
	mov	s1, v0[1]
	mov	s4, v0[2]
	mov	s0, v0[3]
	stp	s16, s1, [x2, #24]
	stp	s5, s7, [x2, #32]
	stp	s17, s4, [x2, #40]
	stp	s6, s3, [x2, #48]
	stp	s2, s0, [x2, #56]
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

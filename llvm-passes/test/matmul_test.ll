; RUN: opt -load-pass-plugin=%shlibdir/libMatMulToNeon%shlibext -passes=matmul2neon -S < %s | FileCheck %s

define void @matmul_naive(float* %A, float* %B, float* %C, i32 %N) {
entry:
  br label %outer

outer:                                            ; preds = %outer.inc, %entry
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.inc ]
  %i.end = icmp slt i32 %i, %N
  br i1 %i.end, label %inner, label %exit

inner:                                            ; preds = %inner.inc, %outer
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner.inc ]
  %j.end = icmp slt i32 %j, %N
  br i1 %j.end, label %compute, label %outer.inc

compute:                                          ; preds = %inner
  %sum = alloca float
  store float 0.0, float* %sum
  br label %kloop

kloop:                                            ; preds = %kloop, %compute
  %k = phi i32 [ 0, %compute ], [ %k.next, %kloop ]
  %k.end = icmp slt i32 %k, %N
  br i1 %k.end, label %body, label %inner.inc

body:                                             ; preds = %kloop
  %a.ptr = getelementptr float, float* %A, i32 %k
  %a = load float, float* %a.ptr
  %b.ptr = getelementptr float, float* %B, i32 %k
  %b = load float, float* %b.ptr
  %prod = fmul float %a, %b
  %old = load float, float* %sum
  %new = fadd float %old, %prod
  store float %new, float* %sum
  %k.next = add i32 %k, 1
  br label %kloop

inner.inc:                                        ; preds = %kloop
  %sum.val = load float, float* %sum
  %c.ptr = getelementptr float, float* %C, i32 %j
  store float %sum.val, float* %c.ptr
  %j.next = add i32 %j, 1
  br label %inner

outer.inc:                                        ; preds = %inner
  %i.next = add i32 %i, 1
  br label %outer

exit:                                             ; preds = %outer
  ret void
}
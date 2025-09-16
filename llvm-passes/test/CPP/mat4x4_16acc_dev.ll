; ModuleID = 'mat4x4_16acc.ll'
source_filename = "mat4x4_16acc.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define void @mat4x4_mul_16acc(ptr noundef %A, ptr noundef %B, ptr noundef %C) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %c31.0 = phi float [ 0.000000e+00, %entry ], [ %add52, %for.inc ]
  %c30.0 = phi float [ 0.000000e+00, %entry ], [ %add50, %for.inc ]
  %c23.0 = phi float [ 0.000000e+00, %entry ], [ %add48, %for.inc ]
  %c22.0 = phi float [ 0.000000e+00, %entry ], [ %add46, %for.inc ]
  %c21.0 = phi float [ 0.000000e+00, %entry ], [ %add44, %for.inc ]
  %c20.0 = phi float [ 0.000000e+00, %entry ], [ %add42, %for.inc ]
  %c13.0 = phi float [ 0.000000e+00, %entry ], [ %add40, %for.inc ]
  %c12.0 = phi float [ 0.000000e+00, %entry ], [ %add38, %for.inc ]
  %c11.0 = phi float [ 0.000000e+00, %entry ], [ %add36, %for.inc ]
  %c10.0 = phi float [ 0.000000e+00, %entry ], [ %add34, %for.inc ]
  %c03.0 = phi float [ 0.000000e+00, %entry ], [ %add32, %for.inc ]
  %c02.0 = phi float [ 0.000000e+00, %entry ], [ %add30, %for.inc ]
  %c01.0 = phi float [ 0.000000e+00, %entry ], [ %add28, %for.inc ]
  %c00.0 = phi float [ 0.000000e+00, %entry ], [ %add26, %for.inc ]
  %c32.0 = phi float [ 0.000000e+00, %entry ], [ %add54, %for.inc ]
  %c33.0 = phi float [ 0.000000e+00, %entry ], [ %add56, %for.inc ]
  %exitcond = icmp ne i64 %indvars.iv, 4
  br i1 %exitcond, label %for.inc, label %for.end

for.inc:                                          ; preds = %for.cond
  %0 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx9 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %1 = load float, ptr %arrayidx9, align 4
  %2 = shl nuw nsw i64 %indvars.iv, 2
  %3 = getelementptr inbounds nuw float, ptr %B, i64 %2
  %arrayidx24 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %4 = load float, ptr %arrayidx24, align 4
  %mul55 = fmul float %1, %4
  %add56 = fadd float %c33.0, %mul55
  %5 = shl nuw nsw i64 %indvars.iv, 2
  %6 = getelementptr inbounds nuw float, ptr %B, i64 %5
  %arrayidx20 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %7 = load float, ptr %arrayidx20, align 4
  %mul53 = fmul float %1, %7
  %add54 = fadd float %c32.0, %mul53
  %8 = shl nuw nsw i64 %indvars.iv, 2
  %9 = getelementptr inbounds nuw float, ptr %B, i64 %8
  %arrayidx16 = getelementptr inbounds nuw i8, ptr %9, i64 4
  %10 = load float, ptr %arrayidx16, align 4
  %mul51 = fmul float %1, %10
  %add52 = fadd float %c31.0, %mul51
  %11 = shl nuw nsw i64 %indvars.iv, 2
  %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %11
  %12 = load float, ptr %arrayidx12, align 4
  %mul49 = fmul float %1, %12
  %add50 = fadd float %c30.0, %mul49
  %13 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx6 = getelementptr inbounds nuw i8, ptr %13, i64 32
  %14 = load float, ptr %arrayidx6, align 4
  %mul47 = fmul float %14, %4
  %add48 = fadd float %c23.0, %mul47
  %mul45 = fmul float %14, %7
  %add46 = fadd float %c22.0, %mul45
  %mul43 = fmul float %14, %10
  %add44 = fadd float %c21.0, %mul43
  %mul41 = fmul float %14, %12
  %add42 = fadd float %c20.0, %mul41
  %15 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx3 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %16 = load float, ptr %arrayidx3, align 4
  %mul39 = fmul float %16, %4
  %add40 = fadd float %c13.0, %mul39
  %mul37 = fmul float %16, %7
  %add38 = fadd float %c12.0, %mul37
  %mul35 = fmul float %16, %10
  %add36 = fadd float %c11.0, %mul35
  %mul33 = fmul float %16, %12
  %add34 = fadd float %c10.0, %mul33
  %arrayidx = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %17 = load float, ptr %arrayidx, align 4
  %mul31 = fmul float %17, %4
  %add32 = fadd float %c03.0, %mul31
  %mul29 = fmul float %17, %7
  %add30 = fadd float %c02.0, %mul29
  %mul27 = fmul float %17, %10
  %add28 = fadd float %c01.0, %mul27
  %mul25 = fmul float %17, %12
  %add26 = fadd float %c00.0, %mul25
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %c31.0.lcssa = phi float [ %c31.0, %for.cond ]
  %c30.0.lcssa = phi float [ %c30.0, %for.cond ]
  %c23.0.lcssa = phi float [ %c23.0, %for.cond ]
  %c22.0.lcssa = phi float [ %c22.0, %for.cond ]
  %c21.0.lcssa = phi float [ %c21.0, %for.cond ]
  %c20.0.lcssa = phi float [ %c20.0, %for.cond ]
  %c13.0.lcssa = phi float [ %c13.0, %for.cond ]
  %c12.0.lcssa = phi float [ %c12.0, %for.cond ]
  %c11.0.lcssa = phi float [ %c11.0, %for.cond ]
  %c10.0.lcssa = phi float [ %c10.0, %for.cond ]
  %c03.0.lcssa = phi float [ %c03.0, %for.cond ]
  %c02.0.lcssa = phi float [ %c02.0, %for.cond ]
  %c01.0.lcssa = phi float [ %c01.0, %for.cond ]
  %c00.0.lcssa = phi float [ %c00.0, %for.cond ]
  %c32.0.lcssa = phi float [ %c32.0, %for.cond ]
  %c33.0.lcssa = phi float [ %c33.0, %for.cond ]
  store float %c00.0.lcssa, ptr %C, align 4
  %arrayidx58 = getelementptr inbounds nuw i8, ptr %C, i64 4
  store float %c01.0.lcssa, ptr %arrayidx58, align 4
  %arrayidx59 = getelementptr inbounds nuw i8, ptr %C, i64 8
  store float %c02.0.lcssa, ptr %arrayidx59, align 4
  %arrayidx60 = getelementptr inbounds nuw i8, ptr %C, i64 12
  store float %c03.0.lcssa, ptr %arrayidx60, align 4
  %arrayidx61 = getelementptr inbounds nuw i8, ptr %C, i64 16
  store float %c10.0.lcssa, ptr %arrayidx61, align 4
  %arrayidx62 = getelementptr inbounds nuw i8, ptr %C, i64 20
  store float %c11.0.lcssa, ptr %arrayidx62, align 4
  %arrayidx63 = getelementptr inbounds nuw i8, ptr %C, i64 24
  store float %c12.0.lcssa, ptr %arrayidx63, align 4
  %arrayidx64 = getelementptr inbounds nuw i8, ptr %C, i64 28
  store float %c13.0.lcssa, ptr %arrayidx64, align 4
  %arrayidx65 = getelementptr inbounds nuw i8, ptr %C, i64 32
  store float %c20.0.lcssa, ptr %arrayidx65, align 4
  %arrayidx66 = getelementptr inbounds nuw i8, ptr %C, i64 36
  store float %c21.0.lcssa, ptr %arrayidx66, align 4
  %arrayidx67 = getelementptr inbounds nuw i8, ptr %C, i64 40
  store float %c22.0.lcssa, ptr %arrayidx67, align 4
  %arrayidx68 = getelementptr inbounds nuw i8, ptr %C, i64 44
  store float %c23.0.lcssa, ptr %arrayidx68, align 4
  %arrayidx69 = getelementptr inbounds nuw i8, ptr %C, i64 48
  store float %c30.0.lcssa, ptr %arrayidx69, align 4
  %arrayidx70 = getelementptr inbounds nuw i8, ptr %C, i64 52
  store float %c31.0.lcssa, ptr %arrayidx70, align 4
  %arrayidx71 = getelementptr inbounds nuw i8, ptr %C, i64 56
  store float %c32.0.lcssa, ptr %arrayidx71, align 4
  %arrayidx72 = getelementptr inbounds nuw i8, ptr %C, i64 60
  store float %c33.0.lcssa, ptr %arrayidx72, align 4
  ret void
}

attributes #0 = { mustprogress noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 21.0.0git (https://github.com/riptio777/llvm-project.git 0053de4c88f93e41709dccc38d1df1946b1197cc)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}

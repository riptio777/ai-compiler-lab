; ModuleID = 'mat4x4_16acc_dev.ll'
source_filename = "mat4x4_16acc.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define void @mat4x4_mul_16acc(ptr noundef %A, ptr noundef %B, ptr noundef %C) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cvec0 = phi <4 x float> [ zeroinitializer, %entry ], [ %38, %for.inc ]
  %cvec1 = phi <4 x float> [ zeroinitializer, %entry ], [ %43, %for.inc ]
  %cvec2 = phi <4 x float> [ zeroinitializer, %entry ], [ %48, %for.inc ]
  %cvec3 = phi <4 x float> [ zeroinitializer, %entry ], [ %53, %for.inc ]
  %0 = extractelement <4 x float> %cvec0, i64 0
  %1 = extractelement <4 x float> %cvec0, i64 1
  %2 = extractelement <4 x float> %cvec0, i64 2
  %3 = extractelement <4 x float> %cvec0, i64 3
  %4 = extractelement <4 x float> %cvec1, i64 0
  %5 = extractelement <4 x float> %cvec1, i64 1
  %6 = extractelement <4 x float> %cvec1, i64 2
  %7 = extractelement <4 x float> %cvec1, i64 3
  %8 = extractelement <4 x float> %cvec2, i64 0
  %9 = extractelement <4 x float> %cvec2, i64 1
  %10 = extractelement <4 x float> %cvec2, i64 2
  %11 = extractelement <4 x float> %cvec2, i64 3
  %12 = extractelement <4 x float> %cvec3, i64 0
  %13 = extractelement <4 x float> %cvec3, i64 1
  %14 = extractelement <4 x float> %cvec3, i64 2
  %15 = extractelement <4 x float> %cvec3, i64 3
  %exitcond = icmp ne i64 %indvars.iv, 4
  br i1 %exitcond, label %for.inc, label %for.end

for.inc:                                          ; preds = %for.cond
  %16 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx9 = getelementptr inbounds nuw i8, ptr %16, i64 48
  %17 = load float, ptr %arrayidx9, align 4
  %18 = shl nuw nsw i64 %indvars.iv, 2
  %19 = getelementptr inbounds nuw float, ptr %B, i64 %18
  %arrayidx24 = getelementptr inbounds nuw i8, ptr %19, i64 12
  %20 = load float, ptr %arrayidx24, align 4
  %21 = shl nuw nsw i64 %indvars.iv, 2
  %22 = getelementptr inbounds nuw float, ptr %B, i64 %21
  %arrayidx20 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %23 = load float, ptr %arrayidx20, align 4
  %24 = shl nuw nsw i64 %indvars.iv, 2
  %25 = getelementptr inbounds nuw float, ptr %B, i64 %24
  %arrayidx16 = getelementptr inbounds nuw i8, ptr %25, i64 4
  %26 = load float, ptr %arrayidx16, align 4
  %27 = shl nuw nsw i64 %indvars.iv, 2
  %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %27
  %28 = load float, ptr %arrayidx12, align 4
  %29 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx6 = getelementptr inbounds nuw i8, ptr %29, i64 32
  %30 = load float, ptr %arrayidx6, align 4
  %31 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %arrayidx3 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %32 = load float, ptr %arrayidx3, align 4
  %arrayidx = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
  %33 = load float, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %.splatinsert = insertelement <4 x float> poison, float %28, i64 0
  %.splat = shufflevector <4 x float> %.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %34 = insertelement <4 x float> poison, float %33, i64 0
  %35 = insertelement <4 x float> %34, float %32, i64 1
  %36 = insertelement <4 x float> %35, float %30, i64 2
  %37 = insertelement <4 x float> %36, float %17, i64 3
  %38 = call <4 x float> @llvm.fma.v4f32(<4 x float> %37, <4 x float> %.splat, <4 x float> %cvec0)
  %.splatinsert1 = insertelement <4 x float> poison, float %26, i64 0
  %.splat2 = shufflevector <4 x float> %.splatinsert1, <4 x float> poison, <4 x i32> zeroinitializer
  %39 = insertelement <4 x float> poison, float %33, i64 0
  %40 = insertelement <4 x float> %39, float %32, i64 1
  %41 = insertelement <4 x float> %40, float %30, i64 2
  %42 = insertelement <4 x float> %41, float %17, i64 3
  %43 = call <4 x float> @llvm.fma.v4f32(<4 x float> %42, <4 x float> %.splat2, <4 x float> %cvec1)
  %.splatinsert3 = insertelement <4 x float> poison, float %23, i64 0
  %.splat4 = shufflevector <4 x float> %.splatinsert3, <4 x float> poison, <4 x i32> zeroinitializer
  %44 = insertelement <4 x float> poison, float %33, i64 0
  %45 = insertelement <4 x float> %44, float %32, i64 1
  %46 = insertelement <4 x float> %45, float %30, i64 2
  %47 = insertelement <4 x float> %46, float %17, i64 3
  %48 = call <4 x float> @llvm.fma.v4f32(<4 x float> %47, <4 x float> %.splat4, <4 x float> %cvec2)
  %.splatinsert5 = insertelement <4 x float> poison, float %20, i64 0
  %.splat6 = shufflevector <4 x float> %.splatinsert5, <4 x float> poison, <4 x i32> zeroinitializer
  %49 = insertelement <4 x float> poison, float %33, i64 0
  %50 = insertelement <4 x float> %49, float %32, i64 1
  %51 = insertelement <4 x float> %50, float %30, i64 2
  %52 = insertelement <4 x float> %51, float %17, i64 3
  %53 = call <4 x float> @llvm.fma.v4f32(<4 x float> %52, <4 x float> %.splat6, <4 x float> %cvec3)
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %c31.0.lcssa = phi float [ %7, %for.cond ]
  %c30.0.lcssa = phi float [ %3, %for.cond ]
  %c23.0.lcssa = phi float [ %14, %for.cond ]
  %c22.0.lcssa = phi float [ %10, %for.cond ]
  %c21.0.lcssa = phi float [ %6, %for.cond ]
  %c20.0.lcssa = phi float [ %2, %for.cond ]
  %c13.0.lcssa = phi float [ %13, %for.cond ]
  %c12.0.lcssa = phi float [ %9, %for.cond ]
  %c11.0.lcssa = phi float [ %5, %for.cond ]
  %c10.0.lcssa = phi float [ %1, %for.cond ]
  %c03.0.lcssa = phi float [ %12, %for.cond ]
  %c02.0.lcssa = phi float [ %8, %for.cond ]
  %c01.0.lcssa = phi float [ %4, %for.cond ]
  %c00.0.lcssa = phi float [ %0, %for.cond ]
  %c32.0.lcssa = phi float [ %11, %for.cond ]
  %c33.0.lcssa = phi float [ %15, %for.cond ]
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

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
declare <4 x float> @llvm.fma.v4f32(<4 x float> noundef, <4 x float> noundef, <4 x float> noundef) #0

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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

// Deal with 4x4 matrices now
#define MAT_COL 4
#define MAT_ROW 4
#define ELEM_SIZE 4 // float size

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// Row/col info of each target Phi Node update
struct RCInfo {
    PHINode *PN; // %c31.0 = phi float [ 0.000000e+00, %entry ], [ %add52, %for.inc ]
    Instruction *Add; // %add52 = fadd float %c31.0, %mul51
    Instruction *Mul; // %mul51 = fmul float %1, %10
    Value *A; // %1 (%1 = load float, ptr %arrayidx9, align 4)
    Value *B; // %10 (%10 = load float, ptr %arrayidx16, align 4)
    /*
    %0 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
    arrayidx9 = getelementptr inbounds nuw i8, ptr %0, i64 48
    => A[k+48] => A[3][k] (in matrix representation)
    */
    int Row = -1;

    /*
      %8 = shl nuw nsw i64 %indvars.iv, 2
      %9 = getelementptr inbounds nuw float, ptr %B, i64 %8
      %arrayidx16 = getelementptr inbounds nuw i8, ptr %9, i64 4
      %10 = load float, ptr %arrayidx16, align 4
    => B[4*k + 4] => B[k][1] (in matrix representation)
    */

    /*
    Example when col is 0
    %11 = shl nuw nsw i64 %indvars.iv, 2
    %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %11
    %12 = load float, ptr %arrayidx12, align 4
    */
    int Col = -1;
};

struct LPInfo {
    Value *ABase; // Base pointer to matrix %A
    Value *BBase; // Base pointer to matrix %B

    PHINode *IVPhi; // Induction Phi, for example k as in A[i][k] * B[k][j]
};

static int tryInferRowColIndex(Value *Ptr, LPInfo &LP) {
/*  For A: %1 = load float, ptr %arrayidx9, align 4
    A.getPointerOperand => %arrayidx9
    current parameter: %arrayidx9-> as GEP => get last constant => index
    index/12 == row index from A
 */
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        Value *Index = GEP->getOperand(GEP->getNumOperands()-1);
        if (auto *ConstIntIdx = dyn_cast<ConstantInt>(Index)) {
            return static_cast<int>(ConstIntIdx->getZExtValue());
        } else if (match(GEP, m_GEP(m_Specific(LP.ABase), m_Specific(LP.IVPhi)))){
            return 0;
        /*
            Match for row 0 of Matrix A
            %arrayidx = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
        */
          /*
            %11 = shl nuw nsw i64 %indvars.iv, 2
            %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %11
            column 0
          */ 
        } else if (Value *IVInst; match(GEP, m_GEP(m_Specific(LP.BBase), m_Value(IVInst)))) {
            // %11 = shl nuw nsw i64 %indvars.iv, 2 
            if (auto *BO = dyn_cast<BinaryOperator>(IVInst)) 
                if (BO->getOpcode() == Instruction::Shl && BO->getOperand(0) == LP.IVPhi) {
                    return 0;
                }
        }
    }

    return -1;
}

static void assignRolColFor4x4(RCInfo &RC, LPInfo &LP) {
    // Grab the load instructions
    auto *AL = dyn_cast<LoadInst>(RC.A);
    auto *BL = dyn_cast<LoadInst>(RC.B);

    // TODO: handle cases where GEP is replaced with ExtractElement by
    // mem2ref/instcombine for AL
    
    // Row from %A
    // e.g., AL: %1 = load float, ptr %arrayidx9, align 4
    // pass %arrayidx9 to tryInferRowColIndex
    if (AL) {
        int idx = tryInferRowColIndex(AL->getPointerOperand(), LP);
        if (idx > -1) {
            RC.Row = idx / (MAT_COL * ELEM_SIZE);
            //errs() << "Row: " << RC.Row << "\n";
        }
    }

    if (BL) {
        int idx = tryInferRowColIndex(BL->getPointerOperand(), LP);
        if (idx > -1) {
            RC.Col = idx / ELEM_SIZE;
            //errs() << "Col: " << RC.Col << "\n";

        }
    }

}
// Check if this phi node is zero-initiated and of float value type
// E.g., %c00 = phi float [ 0.0, %entry ], [ %c00.n, %kloop ]
static bool isFloatAccPhi(PHINode *Phi) {
    if (!Phi || !Phi->getType()->isFloatTy()) {
        return false;
    }

    if (Phi->getNumIncomingValues() != 2) {
        return false;
    }

    // Make sure it's zero initiated
    if (auto *CF = dyn_cast<ConstantFP>(Phi->getIncomingValue(0))) {
        if (CF->isZero()) {
            errs() << "Is float acc phi\n";
            return true;
        }
    }
    return false;
}

static bool matchFaddOfFmulReduction(PHINode *PN, BasicBlock *Latch,
                                     RCInfo &RCResult) {
    // TODO: maybe use isFloatTy?
    if (!PN || !PN->getType()->isFloatingPointTy()) return false;

    // Find the latch update in the phi node
    // TODO: maybe need to check if the PN has exactly 2 incoming values
    int LatchIdx = PN->getBasicBlockIndex(Latch);

    if (LatchIdx < 0) return false;
    
    Value *LatchVal = PN->getIncomingValue(LatchIdx); 
    if (!match(LatchVal, m_FAdd(m_Value(), m_Value()))) return false;
    
    Instruction *AddI;
    AddI = dyn_cast<Instruction>(LatchVal);
    if (!AddI || AddI->getParent() != Latch) return false;
    errs() << "Add instruction: ";
    AddI->dump();

    Value *Prod = nullptr;
    if (!(match(LatchVal, m_FAdd(m_Specific(PN), m_Value(Prod))) ||
          match(LatchVal, m_FAdd(m_Value(Prod), m_Specific(PN))))) {
        return false;
    }

    // e.g., Prod ==   %mul51 = fmul float %1, %10
    Value *LHSop = nullptr, *RHSop = nullptr;
    if (!match(Prod, m_FMul(m_Value(LHSop), m_Value(RHSop)))) {
        return false;
    }
    Value *A, *B;
    Instruction *MulI;
    A = LHSop;
    B = RHSop;
    MulI = dyn_cast<Instruction>(Prod);
    RCResult = {PN, cast<Instruction>(LatchVal), cast<Instruction>(MulI), A, B, -1, -1};
    return true;
}

// TODO: Remove, printing for debugging purpose
void printRC(const RCInfo &RC) {
    RC.A->dump();
    RC.B->dump();
    RC.Add->dump();
    RC.Mul->dump();
}

class MatMulToNeonPass : public PassInfoMixin<MatMulToNeonPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MatMulToNeanPass" << "\n";
        bool Changed = false;
        Module *M = F.getParent();
        SmallVector<Instruction*, 8> ToErase;

        auto &LI = AM.getResult<LoopAnalysis>(F);
        auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
       // auto &AC = FAM.getResult<llvm::AssumptionAnalysis>(F);     // returns AssumptionCache
        //auto &TLI = FAM.getResult<llvm::TargetLibraryAnalysis>(F); // returns TargetLibraryInfo
        auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
        LPInfo LP;
        // Add function arguments, right now we assume 
        // that Matrix A is passed in as arg0 and B as arg1
        assert(F.arg_size() >= 2 && "This function should have at least 2 arguments");
        // TODO: Not robust handling
        LP.ABase = F.getArg(0);
        LP.BBase = F.getArg(1);

        for (Loop *L : LI) {
            SmallVector<Loop*, 8> LoopList;
            //L->getInnerLoopsInPreorder(L, LoopList);
            LoopList = L->getLoopsInPreorder();
            for (Loop *subLoop : LoopList) {
                if (subLoop->getSubLoops().empty()) {
                    // found innermost loop
                    tryRewrite4x4Kernel(*subLoop, SE, LP);
                }
            }
        }


     /*    IRBuilder<> Builder(F.getContext());
        Builder.SetInsertPoint(F.getEntryBlock().getTerminator());

        Function *Ctpop = Intrinsic::getOrInsertDeclaration(M, Intrinsic::ctpop, {Builder.getInt32Ty()});
        AttributeList AL = F.getAttributes();
        Ctpop->setAttributes(AL);
        
        //auto ArgIt = F.arg_begin();
        LLVMContext &Ctx = F.getContext();
        Value *X = Builder.getInt32(42);
        Value *Pop = Builder.CreateCall(Ctpop, {X}, "popcnt");
 */        
     //   Constant *C1 = ConstantFP::get(Type::getFloatTy(Ctx), 1.5f); 
     //   APFloat APF(1.23);
     //   Constant *C = ConstantFP::get(Ctx, APF);
        //CallInst *CI = Builder.CreateCall(Sqrt, {C}, "sqrt");
                            errs() << "**************** START ******************" << "\n";
     //   errs() << APF << "\n";
     //   errs() << *C << "\n";
        //CI->dump();
                            errs() << "**************** END ******************" << "\n";

        //CI->print(errs()); errs() << "\n";


        /*
        for (Instruction *I : ToErase) {
            I->eraseFromParent();
        }
        */
        return PreservedAnalyses::none();
    }


    bool tryRewrite4x4Kernel(Loop &L, ScalarEvolutionAnalysis::Result &SE, LPInfo &LP) {
        errs() << "Try Rewrite 4x4 Kernel\n";
        BasicBlock *Header = L.getHeader();
        BasicBlock *Body = Header; // deal with single-block loops for now
        BasicBlock *Latch = L.getLoopLatch();
        if (!Header) return false;

        // Collect candidate accumulator PHIs
        SmallVector<PHINode*, 16> Accs;

        RecurrenceDescriptor RD;
        InductionDescriptor ID;
        Value *A = nullptr, *B = nullptr;
        Instruction *AddI = nullptr, *MulI = nullptr;
        PHINode *IVPhi = nullptr;

        // Group target PHIs by cols
        SmallVector<RCInfo, 4> Col0Vec;
        SmallVector<RCInfo, 4> Col1Vec;
        SmallVector<RCInfo, 4> Col2Vec;
        SmallVector<RCInfo, 4> Col3Vec;

        SmallVector<SmallVector<RCInfo, 4>, 4> ColVecs(4);

        for (auto &I : *Header) {
            if (auto *Phi = dyn_cast<PHINode>(&I)) {
                errs() << "Checking phi nodes\n";
                RCInfo RC;
                if (InductionDescriptor::isInductionPHI(Phi, &L, &SE, ID)) {
                    LP.IVPhi = Phi;
                }
                if (matchFaddOfFmulReduction(Phi, Latch, RC)) {
                    errs() << "Matched phi node\n";
                    //printRC(RC);
                    /*
                    At this point LP should contain info for ABase, BBase and Induction Phi
                    */
                    assignRolColFor4x4(RC, LP); 
                    assert(RC.Col >= 0 && RC.Col < 4 && "RC column value is out of range.");
                    //errs() << "RC.Col: " << RC.Col;
                    ColVecs[RC.Col].push_back(RC);

                    Accs.push_back(Phi);
                }
            }
        }

        // TODO: move this check earlier
        if (Accs.size() < 16) { 
            return false; 
        }

        for (int i = 0; i < ColVecs.size(); i++) {
            sort(ColVecs[i], [](const RCInfo &rhs, const RCInfo &lhs) {
                return rhs.Row < lhs.Row;
            });
            errs() << "*********** Col: " << i << "**********\n";
            for (int j = 0; j < ColVecs[i].size(); j++) {
                //printRC(ColVecs[i][j]);
                errs() << "Row: " << ColVecs[i][j].Row << "\n";
                errs() << "Col: " << ColVecs[i][j].Col << "\n";
            }
        }

        IRBuilder<> Builder(Latch->getTerminator());
    
        

        return true;
    }
};

} // end of anonymous namespace

llvm::PassPluginLibraryInfo getMatMulToNeonPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "MatMulToNeonPass",
            LLVM_VERSION_STRING,
            [] (PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                        ArrayRef<PassBuilder::PipelineElement>) {
                            if (Name == "matmul2neon") {
                                FPM.addPass(MatMulToNeonPass());
                                return true;
                            }
                            return false;
                    });
                // Ensure standard function analyses (Loop/DomTree/etc.) are registered
                PB.registerAnalysisRegistrationCallback(
                    [](FunctionAnalysisManager &FAM) {
                    //registerFunctionAnalyses(FAM);  // <-- critical line

                    FAM.registerPass([&] { return LoopAnalysis(); });
                    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
                    FAM.registerPass([&] {return ScalarEvolutionAnalysis(); });
                    FAM.registerPass([&]{ return AssumptionAnalysis(); });
                    FAM.registerPass([&]{ return TargetIRAnalysis(); });
                    FAM.registerPass([&]{ return TargetLibraryAnalysis(); });
                });
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getMatMulToNeonPassPluginInfo();
}
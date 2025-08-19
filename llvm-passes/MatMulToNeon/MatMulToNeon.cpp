#include "llvm/IR/Analysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class MatMulToNeanPass : public PassInfoMixin<MatMulToNeanPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MatMulToNeanPass" << "\n";

        return PreservedAnalyses::all();
    }
};

} // end of anonymous namespace
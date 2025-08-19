#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class MatMulToNeonPass : public PassInfoMixin<MatMulToNeonPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MatMulToNeanPass" << "\n";

        return PreservedAnalyses::all();
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
                
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getMatMulToNeonPassPluginInfo();
}
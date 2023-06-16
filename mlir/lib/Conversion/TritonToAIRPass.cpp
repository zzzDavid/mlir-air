//===- TritonToAIRPass.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Conversion/TritonToAIRPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "triton/Dialect/Triton/IR/Types.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Ops.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "air/Util/Util.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

#define DEBUG_TYPE "triton-to-air"

using namespace mlir;
using namespace mlir::arith;
using namespace xilinx;
using namespace xilinx::air;

namespace {


void analyzeLoadOp(func::FuncOp func) {
  // get all tt.get_program_id operations
  SmallVector<mlir::triton::GetProgramIdOp, 4> getProgramIdOps;
  func.walk([&](mlir::triton::GetProgramIdOp op) {
    getProgramIdOps.push_back(op);
  });
  // find use-def chains for each get_program_id that ends with tt.splat
  for (auto getProgramIdOp : getProgramIdOps) {
    // create an affine expression for the index
    auto idExpr = getAffineDimExpr(0, func.getContext());
    for (auto user : getProgramIdOp.getResult().getUsers()) {
      // if the user is a multiply op
      if (isa<arith::MulIOp>(user)) {
        if (auto constOp = user->getOperands()[1].getDefiningOp<arith::ConstantOp>()) {
          auto constVal = constOp.getValue().cast<IntegerAttr>().getInt();
          auto constExpr = getAffineConstantExpr(constVal, func.getContext());
          auto affineMap = AffineMap::get(1, 0, idExpr * constExpr);
          llvm::outs() << "affine map: " << affineMap << "\n";
        }
      }
    }
  }
}



class TritonToAIRPass : public TritonToAIRBase<TritonToAIRPass> {

public:
  TritonToAIRPass() = default;
  TritonToAIRPass(const TritonToAIRPass &pass) {}

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    // get all functions
    for (auto func : module.getOps<func::FuncOp>()) {
      analyzeLoadOp(func);
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createTritonToAIRPass() {
  return std::make_unique<TritonToAIRPass>();
}

} // namespace air
} // namespace xilinx

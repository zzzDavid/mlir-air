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
#include "triton/Dialect/Triton/IR/Ops.h.inc"

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



class TritonToAIRPass : public TritonToAIRBase<TritonToAIRPass> {

public:
  TritonToAIRPass() = default;
  TritonToAIRPass(const TritonToAIRPass &pass) {}

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();
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

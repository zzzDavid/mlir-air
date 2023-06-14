//===- AIRToAsyncPass.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_AIR_PASS_H
#define TRITON_TO_AIR_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createTritonToAIRPass();

} // namespace air
} // namespace xilinx

#endif // TRITON_TO_AIR_PASS_H

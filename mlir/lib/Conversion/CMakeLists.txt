# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_air_library(AIRConversionPasses
TritonToAIRPass.cpp
ConvertToAIRPass.cpp
AIRLoweringPass.cpp
AIRRtToLLVMPass.cpp
AIRToAIEPass.cpp
AIRToAsyncPass.cpp
AIRPipeline.cpp
Passes.cpp

DEPENDS
AIRConversionPassIncGen
AIRDialect
AIRRtDialect

LINK_COMPONENTS
Core

LINK_LIBS
AIRDialect
AIRRtDialect
AIRUtil
AIE
MLIRIR
MLIRLinalgUtils
MLIRLinalgTransforms
MLIRSupport
MLIRTransforms
)

target_include_directories(AIRConversionPasses PRIVATE 
    "/home/nz264/scratch/triton/python/build/include"
    "/home/nz264/scratch/triton/include"
    )
target_link_libraries(AIRConversionPasses PRIVATE 
    "/home/nz264/scratch/triton/python/build/lib/Dialect/Triton/IR/libTritonIR.a"
    "/home/nz264/scratch/triton/python/build/lib/Dialect/Triton/Transforms/libTritonTransforms.a"
    )
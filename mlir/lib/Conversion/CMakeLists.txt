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

target_include_directories(AIRConversionPasses PRIVATE "/home/niansong/triton/python/build/cmake.linux-x86_64-cpython-3.11/include")

SYSROOT_DIR=/group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
LLVM_DIR=/home/niansong/mlir-air/utils/llvm
CMAKEMODULES_DIR=/home/niansong/mlir-air/utils/mlir-aie/cmake/modulesXilinx
MLIR_AIE_DIR=/home/niansong/mlir-air/utils/mlir-aie

BUILD_DIR=build
INSTALL_DIR=install

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`


# This also works

cmake .. \
    -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86.cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
    -DLibXAIE_ROOT=`pwd`/../aienginev2/install \
    -DAIR_RUNTIME_TARGETS:STRING="x86" \
    -Dx86_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86.cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 


# cmake .. \
#     -GNinja \
#     -DCMAKE_C_COMPILER=clang \
#     -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
#     -DCMAKE_TOOLCHAIN_FILE_OPT=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm_petalinux.cmake \
#     -DArch=arm64 \
#     -DgccVer=10.2.0 \
#     -DCMAKE_USE_TOOLCHAIN=FALSE \
#     -DCMAKE_USE_TOOLCHAIN_AIRHOST=TRUE \
#     -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
#     -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
#     -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
#     -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
#     -DBUILD_SHARED_LIBS=OFF \
#     -DLLVM_USE_LINKER=lld \
#     -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
#     -DCMAKE_TOOLCHAIN_FILE=/home/niansong/mlir-air/cmake/modules/toolchain_x86.cmake \
#     -DAIR_RUNTIME_TARGETS:STRING="x86" \
#     -DXILINX_XAIE_INCLUDE_DIR=${XILINX_XAIE_INCLUDE_DIR} \
#     -DXILINX_XAIE_LIBS=${XILINX_XAIE_LIBS} \ 

# cmake .. \
#     -GNinja \
#     -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86.cmake \
#     -DCMAKE_BUILD_TYPE=Debug \
#     -DLLVM_ENABLE_ASSERTIONS=ON \ 
#     -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
#     -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
#     -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
#     -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
#     -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
#     -DBUILD_SHARED_LIBS=OFF \
#     -DLLVM_USE_LINKER=lld \
#     -DAIR_RUNTIME_TARGETS:STRING="x86" \
#     -Dx86_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86.cmake \
#     -DXILINX_XAIE_INCLUDE_DIR=${XILINX_XAIE_INCLUDE_DIR} \
#     -DXILINX_XAIE_LIBS=${XILINX_XAIE_LIBS} \
#     -DCMAKE_INSTALL_PREFIX=install

ninja
ninja install

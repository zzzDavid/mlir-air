// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt -canonicalize  %s | FileCheck %s

// CHECK-LABEL: func @launch
// CHECK: air.launch_herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
// CHECK:   air.herd_terminator
// CHECK: }
func @launch(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.launch_herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}

// CHECK-LABEL: func @launch_async
// CHECK: air.launch_herd async [{{.*}}] tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
// CHECK:   air.herd_terminator
// CHECK: }
func @launch_async(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.launch_herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  air.wait_all [%e1]
  return
}
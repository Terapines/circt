//===--- Passes.td - Moore pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_TD
#define CIRCT_DIALECT_MOORE_MOOREPASSES_TD

include "mlir/Pass/PassBase.td"

def CombineNonBlockings : Pass<"moore-combine-nonblockings", "moore::SVModuleOp"> {
    let summary = "Combine moore.nonblocking_assign ops";
    let description = [{
    }];
    let constructor = "circt::moore::createCombineNonBlockingsPass()";
}

def DetectRegisters : Pass<"moore-detect-registers", "moore::SVModuleOp"> {
    let summary = "";
    let description = [{
    }];
    let constructor = "circt::moore::createDetectRegistersPass()";
}

def InfoCollection : Pass<"info-collection", "mlir::ModuleOp"> {
    let summary = "Collect all nets and variables declarations";
    let description = [{
      This pass will perform a pre-pass over the IR to find all concurrently-
      assigned nets and variables. Basically go through all net/variable 
      declarations and collect the connected value in a hash map (either the
      initial value in net declaration or the assigned value).
    }];
    let constructor = "circt::moore::createInfoCollectionPass()";
}

def PullNonBlockingUp : Pass<"moore-pull-nonblocking-up", "moore::SVModuleOp"> {
    let summary = "Pull non-blocking assignments upwards";
    let description = [{
      Improve the level of non-blocking assignments from the inner region to
      its parent region, like from "{if(cond) x <= y;}" to "{x <= y if %cond;}"
    }];
    let constructor = "circt::moore::createPullNonBlockingUpPass()";
}

def SimplifyProcedures : Pass<"moore-simplify-procedures", "moore::SVModuleOp"> {
    let summary = "Simplify procedures";
    let description = [{
      Because we want to introduce mem2reg in the moore dialect to eliminate the
      local temporary variables, if the local variabels exist in the procedure
      body, it can be promoted by mem2reg. But global/module-level variables
      don't be promoted. So this pass is aimed at inserting a local "shadow"
      variable in always blocks for every module-level variable that the
      process modifies. Then perform mem2reg to eliminate the local "shadow"
      variable.
    }];
    let constructor = "circt::moore::createSimplifyProceduresPass()";
}

def LowerConcatRef : Pass<"moore-lower-concatref", "moore::SVModuleOp"> {
    let summary = "Lower moore.concat_ref ops";
    let description = [{
      It's used to disassemble the LHS of assignments that have a form like
      "{a, b} = c" onto "a = c[9001:42];" and "b = c[41:0]". Aimed at
      conveniently lowering this kind of assignment.
    }];
    let constructor = "circt::moore::createLowerConcatRefPass()";
}

def MergeExtractRef : Pass<"moore-merge-extractref", "moore::SVModuleOp"> {
    let summary = "Merge moore.extract_ref ops";
    let description = [{
    }];
    let constructor = "circt::moore::createMergeExtractRefPass()";
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD

//===- InfoCollection.cpp - Collect net/variable declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InfoCollection pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_INFOCOLLECTION
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
struct InfoCollectionPass
    : public circt::moore::impl::InfoCollectionBase<InfoCollectionPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createInfoCollectionPass() {
  return std::make_unique<InfoCollectionPass>();
}

void InfoCollection::addValue(Operation *op) {
  TypeSwitch<Operation *, void>(op)
      // Collect all variables and their initial values.
      .Case<VariableOp>([&](auto op) {
        auto value = op.getInitial();
        assignmentChains[op] = value;
      })
      // Collect all nets and their initial values.
      .Case<NetOp>([&](auto op) {
        auto value = op.getAssignment();
        assignmentChains[op] = value;
      })
      // Update the values of the nets/variables. Just reserve the last
      // assignment.
      .Case<ContinuousAssignOp, BlockingAssignOp>([&](auto op) {
        auto destOp = op.getDst().getDefiningOp();
        auto srcValue = op.getSrc();
        assignmentChains[destOp] = srcValue;
      });
}

extern InfoCollection moore::decl;
void InfoCollectionPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp moduleOp) {
    for (auto &op : moduleOp.getOps())
      decl.addValue(&op);

    return WalkResult::advance();
  });
}

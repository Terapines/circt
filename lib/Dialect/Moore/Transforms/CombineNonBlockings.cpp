//===- CombineNonBlockings.cpp - Combine non-blocking ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_COMBINENONBLOCKINGS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {
struct CombineNonBlockingsPass
    : public circt::moore::impl::CombineNonBlockingsBase<
          CombineNonBlockingsPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createCombineNonBlockingsPass() {
  return std::make_unique<CombineNonBlockingsPass>();
}

void CombineNonBlockingsPass::runOnOperation() {
  OpBuilder build(&getContext());
  llvm::MapVector<Value, NonBlockingAssignOp> assigns;
  getOperation()->walk([&](NonBlockingAssignOp op) {
    if (auto assign = assigns.lookup(op.getDst())) {
      build.setInsertionPointAfter(op);
      auto src = build.create<moore::MuxOp>(
          assign->getLoc(), assign.getEnable(), assign.getSrc(), op.getSrc());
      auto en = build.create<OrOp>(assign->getLoc(), assign.getEnable(),
                                   op.getEnable());
      auto newAssign = build.create<NonBlockingAssignOp>(
          assign->getLoc(), assign.getDst(), src, en);
      assigns[newAssign.getDst()] = newAssign;
      op->erase();
      assign->erase();
    } else
      assigns.insert_or_assign(op.getDst(), op);
  });
}

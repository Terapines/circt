//===- DetectRegisters.cpp -  -----------------===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_DETECTREGISTERS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {
/// Lower `scf.if` into `moore.mux`.
/// Example : if (cond1) a = 1; else if (cond2) a = 2; else a=3;
/// post-conv: %0 = moore.mux %cond1, a=1, %1; %1 = moore.mux %cond2, a=2, a=3;
static LogicalResult lowerSCFIfOp(scf::IfOp ifOp,
                                  ConversionPatternRewriter &rewriter,
                                  DenseMap<Operation *, Value> &muxMap,
                                  SmallVector<Value> &allConds) {

  Value cond, trueValue, falseValue;

  // Traverse the 'else' region.
  for (auto &elseOp : ifOp.getElseRegion().getOps()) {
    // First to find the innermost 'else if' statement.
    if (isa<scf::IfOp>(elseOp))
      if (failed(lowerSCFIfOp(cast<scf::IfOp>(elseOp), rewriter, muxMap,
                              allConds)))
        return failure();

    if (isa<NonBlockingAssignOp>(elseOp)) {
      Operation *dstOp;
      if (auto varOp = elseOp.getOperand(0).getDefiningOp<VariableOp>())
        dstOp = varOp;
      else
        continue;
      auto srcValue = elseOp.getOperand(1);
      muxMap[dstOp] = srcValue;
    }
  }

  if (!ifOp.getElseRegion().empty())
    rewriter.inlineBlockBefore(ifOp.elseBlock(), ifOp);

  // Traverse the 'then' region.
  for (auto &thenOp : ifOp.getThenRegion().getOps()) {

    // First to find the innermost 'if' statement.
    if (isa<scf::IfOp>(thenOp)) {
      if (allConds.empty())
        allConds.push_back(
            ifOp.getCondition().getDefiningOp<ConversionOp>().getInput());
      allConds.push_back(
          thenOp.getOperand(0).getDefiningOp<ConversionOp>().getInput());
      if (failed(lowerSCFIfOp(cast<scf::IfOp>(thenOp), rewriter, muxMap,
                              allConds)))
        return failure();
    }

    cond = ifOp.getCondition().getDefiningOp<ConversionOp>().getInput();
    if (isa<NonBlockingAssignOp>(thenOp)) {
      Operation *dstOp;
      if (auto varOp = thenOp.getOperand(0).getDefiningOp<VariableOp>())
        dstOp = varOp;
      else
        continue;
      trueValue = thenOp.getOperand(1);

      // Maybe just the 'then' region exists. Like if(); no 'else'.
      if (auto value = muxMap.lookup(dstOp))
        falseValue = value;
      else
        falseValue = rewriter.create<ConstantOp>(
            thenOp.getLoc(), cast<IntType>(trueValue.getType()), uint64_t(0));

      auto thenBuilder = ifOp.getThenBodyBuilder();
      thenBuilder.setInsertionPoint(ifOp.thenYield());

      if (allConds.size() > 1) {
        cond = allConds.back();
        allConds.pop_back();
        while (!allConds.empty()) {
          cond =
              thenBuilder.create<AndOp>(ifOp->getLoc(), allConds.back(), cond);
          allConds.pop_back();
        }
      }

      auto muxOp =
          thenBuilder.create<MuxOp>(ifOp.getLoc(), cond, trueValue, falseValue);
      muxMap[dstOp] = muxOp;
    }
  }
  rewriter.inlineBlockBefore(ifOp.thenBlock(), ifOp);
  rewriter.eraseOp(ifOp);
  return success();
}

struct SCFIfOpConversion : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    DenseMap<Operation *, Value> muxMap;
    SmallVector<Value> allConds;
    if (failed(lowerSCFIfOp(op, rewriter, muxMap, allConds)))
      return failure();

    return success();
  }
};

struct SCFYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

struct DetectRegistersPass
    : public circt::moore::impl::DetectRegistersBase<DetectRegistersPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createDetectRegistersPass() {
  return std::make_unique<DetectRegistersPass>();
}

void DetectRegistersPass::runOnOperation() {
  MLIRContext &context = getContext();
  ConversionTarget target(context);

  // target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp ifOp) {
  //   bool thenRegion = true, elseRegion = true;
  //   if (llvm::any_of(ifOp.getThenRegion().getOps(), [&](auto &thenOp) {
  //         return isa<NonBlockingAssignOp>(thenOp) &&
  //                thenOp.getOperand(0).template getDefiningOp<VariableOp>();
  //       }))
  //     thenRegion = false;
  //   if (llvm::any_of(ifOp.getElseRegion().getOps(), [&](auto &elseOp) {
  //         return isa<NonBlockingAssignOp>(elseOp) &&
  //                elseOp.getOperand(0).template getDefiningOp<VariableOp>();
  //       }))
  //     elseRegion = false;
  //   return thenRegion || elseRegion;

  //   return true;
  // });

  // target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp op) {
  //   if (auto ifOp = llvm::dyn_cast_or_null<scf::IfOp>(op->getParentOp()))
  //     return (ifOp.thenYield() == op) || (ifOp.elseYield() == op);
  //   return true;
  // });

  target.addLegalDialect<MooreDialect>();
  RewritePatternSet patterns(&context);
  patterns.add<SCFIfOpConversion, SCFYieldOpConversion>(&context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

//===- MooreToCore.cpp - Moore To Core Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTMOORETOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

using comb::ICmpPredicate;
InfoCollection moore::decl;

namespace {

/// Returns the passed value if the integer width is already correct.
/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if
/// it is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return builder.create<comb::ConcatOp>(loc, ValueRange{zeroExt, value});
  }

  Value hi = builder.create<comb::ExtractOp>(loc, value, targetWidth,
                                             intWidth - targetWidth);
  Value zero = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, hi,
                                              zero, false);
  Value lo = builder.create<comb::ExtractOp>(loc, value, 0, targetWidth);
  Value max = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(targetWidth), -1);
  return builder.create<comb::MuxOp>(loc, isZero, lo, max, false);
}

/// Get the ModulePortInfo from a SVModuleOp.
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> inputs, outputs;
  inputs.reserve(moduleTy.getNumInputs());
  outputs.reserve(moduleTy.getNumOutputs());

  for (auto port : moduleTy.getPorts())
    if (port.dir == hw::ModulePort::Direction::Output) {
      outputs.push_back(
          hw::PortInfo({{port.name, port.type, port.dir}, resultNum++, {}}));
    } else {
      // FIXME: Once we support net<...>, ref<...> type to represent type of
      // special port like inout or ref port which is not a input or output
      // port. It can change to generate corresponding types for direction of
      // port or do specified operation to it. Now inout and ref port is treated
      // as input port.
      inputs.push_back(
          hw::PortInfo({{port.name, port.type, port.dir}, inputNum++, {}}));
    }

  return hw::ModulePortInfo(inputs, outputs);
}

/// Lower `scf.if` into `comb.mux`.
/// Example : if (cond1) a = 1; else if (cond2) a = 2; else a=3;
/// post-conv: %0 = comb.mux %cond1, a=1, %1; %1 = comb.mux %cond2, a=2, a=3;
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

    if (isa<BlockingAssignOp, NonBlockingAssignOp>(elseOp)) {
      auto *dstOp = elseOp.getOperand(0).getDefiningOp();
      auto srcValue = rewriter.getRemappedValue(elseOp.getOperand(1));
      muxMap[dstOp] = srcValue;
    }
  }

  if (ifOp.elseBlock())
    rewriter.inlineBlockBefore(ifOp.elseBlock(), ifOp);

  // Traverse the 'then' region.
  for (auto &thenOp : ifOp.getThenRegion().getOps()) {

    // First to find the innermost 'if' statement.
    if (isa<scf::IfOp>(thenOp)) {
      if (allConds.empty())
        allConds.push_back(ifOp.getCondition());
      allConds.push_back(thenOp.getOperand(0));
      if (failed(lowerSCFIfOp(cast<scf::IfOp>(thenOp), rewriter, muxMap,
                              allConds)))
        return failure();
    }

    cond = ifOp.getCondition();
    if (isa<BlockingAssignOp, NonBlockingAssignOp>(thenOp)) {
      auto *dstOp = thenOp.getOperand(0).getDefiningOp();
      trueValue = rewriter.getRemappedValue(thenOp.getOperand(1));

      // Maybe just the 'then' region exists. Like if(); no 'else'.
      if (auto value = muxMap.lookup(dstOp))
        falseValue = rewriter.getRemappedValue(value);
      else
        falseValue = rewriter.create<hw::ConstantOp>(thenOp.getLoc(),
                                                     trueValue.getType(), 0);

      auto b = ifOp.getThenBodyBuilder();

      if (allConds.size() > 1) {
        cond = b.create<comb::AndOp>(ifOp.getLoc(), allConds, false);
        allConds.pop_back();
      }

      auto muxOp =
          b.create<comb::MuxOp>(ifOp.getLoc(), cond, trueValue, falseValue);
      muxMap[dstOp] = muxOp;
    }
  }
  if (!ifOp.getThenRegion().empty()) {
    rewriter.inlineBlockBefore(ifOp.thenBlock(), ifOp);
    rewriter.eraseOp(ifOp);
  }
  return success();
}

/// Handle "always_comb".
static LogicalResult lowerAlwaysComb(ProcedureOp procedureOp,
                                     ConversionPatternRewriter &rewriter) {
  DenseMap<Operation *, Value> muxMap;
  for (auto &nestOp : procedureOp.getBodyRegion().getOps()) {
    SmallVector<Value> allConds;
    if (isa<scf::IfOp>(nestOp))
      if (failed(lowerSCFIfOp(dyn_cast<scf::IfOp>(nestOp), rewriter, muxMap,
                              allConds)))
        return failure();
  }

  // Update the values of the variables.
  for (auto muxOp : muxMap) {
    auto from = rewriter.getRemappedValue(muxOp.first->getResult(0));
    if (!from)
      return failure();

    StringAttr name;
    if (auto varOp = llvm::dyn_cast_or_null<VariableOp>(muxOp.first))
      name = varOp.getNameAttr();
    else if (auto varOp =
                 muxOp.getFirst()->getOperand(0).getDefiningOp<VariableOp>()) {
      name = varOp.getNameAttr();
    } else if (auto varOp = muxOp.getFirst()
                                ->getOperand(0)
                                .getDefiningOp()
                                ->getOperand(0)
                                .getDefiningOp<VariableOp>()) {
      name = varOp.getNameAttr();
    }

    rewriter.setInsertionPoint(muxOp.first);
    rewriter.create<hw::WireOp>(muxOp.first->getLoc(), muxOp.second, name);
  }

  rewriter.inlineBlockBefore(procedureOp.getBody(), procedureOp);
  return success();
}

/// Lower `always` with an explicit sensitivity list and clock edges to `seq`,
/// otherwise to `comb`.
static LogicalResult lowerAlways(ProcedureOp procedureOp,
                                 ConversionPatternRewriter &rewriter) {
  // The default is a synchronization.
  bool isAsync = false;

  // The default is the sequential logic unit.
  bool isCombination = false;

  // Assume the reset value is at the then region.
  bool atThenRegion = true;

  // Assume we can correctly lower `scf.if`.
  bool falseLowerIf = false;

  Value clk, rst, input, rstValue;

  // Collect all signals.
  DenseMap<Value, moore::Edge> senLists;

  // Collect `comb.mux`.
  DenseMap<Operation *, Value> muxMap;

  if (procedureOp.getBodyRegion().getOps<EventOp>().empty())
    isCombination = true;

  for (auto &nestOp : procedureOp.getBodyRegion().getOps()) {
    TypeSwitch<Operation *>(&nestOp)
        .Case<EventOp>([&](auto op) {
          if (op.getEdge() == moore::Edge::None) {
            isCombination = true;
            return;
          }

          senLists.insert({op.getInput(), op.getEdge()});
          if (auto readOp = op.getInput().template getDefiningOp<ReadOp>())
            for (auto user : readOp.getInput().getUsers())
              if (isa<NotOp, ConversionOp>(user)) {
                isAsync = true;
                auto edge = senLists.lookup(op.getInput());
                // Represent reset value doesn't at `then` region.
                if ((edge == Edge::PosEdge && isa<NotOp>(user)) ||
                    (edge == Edge::NegEdge && isa<ConversionOp>(user)) ||
                    (!isAsync && isa<NotOp>(user)))
                  atThenRegion = false;

                rst = rewriter.getRemappedValue(op.getInput());
                senLists.erase(op.getInput());
              }
        })
        .Case<NonBlockingAssignOp>([&](auto op) {
          StringAttr name;
          if (auto varOp = llvm::dyn_cast_or_null<VariableOp>(
                  op.getDst().getDefiningOp()))
            name = varOp.getNameAttr();
          if (!name)
            if (auto varOp =
                    llvm::dyn_cast_or_null<VariableOp>(op.getDst()
                                                           .getDefiningOp()
                                                           ->getOperand(0)
                                                           .getDefiningOp()
                                                           ->getOperand(0)
                                                           .getDefiningOp()))
              name = varOp.getNameAttr();

          input = rewriter.getRemappedValue(op.getSrc());

          if (senLists.empty()) {
            isCombination = true;
            return;
          }

          // Get the clock signal.
          if (name) {
            if (!clk) {
              clk = rewriter.getRemappedValue(senLists.begin()->first);
              clk = rewriter.create<seq::ToClockOp>(procedureOp->getLoc(), clk);
            }

            auto regOp = rewriter.create<seq::FirRegOp>(procedureOp->getLoc(),
                                                        input, clk, name);

            // Update the variables.
            auto from = rewriter.getRemappedValue(op.getDst());
            rewriter.replaceOp(from.getDefiningOp(), regOp);
          } else
            emitError(procedureOp->getLoc()) << "empty name";
        })
        .Case<scf::IfOp>([&](auto op) {
          SmallVector<Value> allConds;
          if (failed(lowerSCFIfOp(op, rewriter, muxMap, allConds))) {
            falseLowerIf = true;
            return;
          }

          // Get the clock signal.
          clk = rewriter.getRemappedValue(senLists.begin()->first);
          clk = rewriter.create<seq::ToClockOp>(procedureOp->getLoc(), clk);
          if (!rst)
            rst = op.getCondition();

          for (auto muxOp : muxMap) {
            StringAttr name;
            if (auto varOp = llvm::dyn_cast_or_null<VariableOp>(muxOp.first))
              name = varOp.getNameAttr();
            else if (auto varOp = muxOp.getFirst()
                                      ->getOperand(0)
                                      .getDefiningOp<VariableOp>()) {
              name = varOp.getNameAttr();
            } else if (auto varOp = muxOp.getFirst()
                                        ->getOperand(0)
                                        .getDefiningOp()
                                        ->getOperand(0)
                                        .getDefiningOp<VariableOp>()) {
              name = varOp.getNameAttr();
            }

            if (!isa<comb::MuxOp>(muxOp.getSecond().getDefiningOp())) {
              continue;
            }
            auto mux = muxOp.getSecond().getDefiningOp<comb::MuxOp>();

            input = atThenRegion ? mux.getFalseValue() : mux.getTrueValue();
            rstValue = atThenRegion ? mux.getTrueValue() : mux.getFalseValue();

            auto regOp = rst ? rewriter.create<seq::FirRegOp>(
                                   procedureOp->getLoc(), input, clk, name, rst,
                                   rstValue, hw::InnerSymAttr{}, isAsync)
                             : rewriter.create<seq::FirRegOp>(
                                   procedureOp->getLoc(), input, clk, name);

            // Update the variables.
            auto from = rewriter.getRemappedValue(muxOp.first->getResult(0));
            rewriter.replaceOp(from.getDefiningOp(), regOp);
          }
        });

    if (falseLowerIf)
      return failure();

    // Always represents a combinational logic unit.
    // Like `always @(a, b, ...)` and `always @(*)`.
    if (isCombination) {
      if (failed(lowerAlwaysComb(procedureOp, rewriter)))
        return failure();
      return success();
    }
  }

  rewriter.inlineBlockBefore(procedureOp.getBody(), procedureOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Structural Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConversion : public OpConversionPattern<SVModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputOp = op.getOutputOp();
    rewriter.setInsertionPoint(op);

    // Create the hw.module to replace svmoduleOp
    auto hwModuleOp =
        rewriter.create<hw::HWModuleOp>(op.getLoc(), op.getSymNameAttr(),
                                        getModulePortInfo(*typeConverter, op));
    // Make hw.module have the same visibility as the moore.module.
    // The entry/top level module is public, otherwise is private.
    SymbolTable::setSymbolVisibility(hwModuleOp,
                                     SymbolTable::getSymbolVisibility(op));
    rewriter.eraseBlock(hwModuleOp.getBodyBlock());
    rewriter.inlineRegionBefore(op.getBodyRegion(), hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());

    // Rewrite the hw.output op
    rewriter.setInsertionPointToEnd(hwModuleOp.getBodyBlock());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(outputOp, outputOp.getOperands());

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
};

struct InstanceOpConversion : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = op.getInstanceNameAttr();
    auto moduleName = op.getModuleNameAttr();

    // Create the new hw instanceOp to replace the original one.
    rewriter.setInsertionPoint(op);
    auto instOp = rewriter.create<hw::InstanceOp>(
        op.getLoc(), op.getResultTypes(), instName, moduleName, op.getInputs(),
        op.getInputNamesAttr(), op.getOutputNamesAttr(),
        /*Parameter*/ rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    // Replace uses chain and erase the original op.
    op.replaceAllUsesWith(instOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ProcedureOpConversion : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getKind()) {
    case ProcedureKind::AlwaysComb:
      if (failed(lowerAlwaysComb(op, rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::Always:
      if (failed(lowerAlways(op, rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::AlwaysFF:
    case ProcedureKind::AlwaysLatch:
    case ProcedureKind::Initial:
    case ProcedureKind::Final:
      return emitError(op->getLoc(), "Unsupported procedure operation");
    };
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

struct NetOpConversion : public OpConversionPattern<NetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value value = adaptor.getAssignment();
    if (value) {
      if (auto readOp = value.template getDefiningOp<ReadOp>())
        value = rewriter.getRemappedValue(readOp.getInput());
    } else
      value = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<hw::WireOp>(op, value, op.getNameAttr());
    return success();
  }
};

struct VariableOpConversion : public OpConversionPattern<VariableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value value = adaptor.getInitial();
    if (value) {
      if (auto readOp = value.template getDefiningOp<ReadOp>())
        value = rewriter.getRemappedValue(readOp.getInput());
    } else
      value = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<hw::WireOp>(op, value, op.getNameAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct NamedConstantOpConv : public OpConversionPattern<NamedConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NamedConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = typeConverter->convertType(op.getResult().getType());
    SmallString<32> symStr;
    switch (op.getKind()) {
    case NamedConst::Parameter:
      symStr = "parameter";
      break;
    case NamedConst::LocalParameter:
      symStr = "localparameter";
      break;
    case NamedConst::SpecParameter:
      symStr = "specparameter";
      break;
    }
    auto symAttr =
        rewriter.getStringAttr(symStr + Twine("_") + adaptor.getName());
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, resultType, adaptor.getValue(),
                                            op.getNameAttr(),
                                            hw::InnerSymAttr::get(symAttr));
    return success();
  }
};

struct ConcatOpConversion : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

struct ConcatRefOpConversion : public OpConversionPattern<ConcatRefOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType,
                                                   adaptor.getValue());
    return success();
  }
};

struct ExtractOpConversion : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(
        op, resultType, adaptor.getInput(), adaptor.getLowBit());
    return success();
  }
};

struct DynExtractOpConversion : public OpConversionPattern<DynExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto width = typeConverter->convertType(op.getInput().getType())
                     .getIntOrFloatBitWidth();
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getLowBit(), width, op->getLoc());
    Value value =
        rewriter.create<comb::ShrUOp>(op->getLoc(), adaptor.getInput(), amount);

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
    return success();
  }
};

struct ExtractRefOpConversion : public OpConversionPattern<ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(
        op, resultType, adaptor.getInput(), adaptor.getLowBit());
    return success();
  }
};

struct DynExtractRefOpConversion : public OpConversionPattern<DynExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto width = typeConverter->convertType(op.getInput().getType())
                     .getIntOrFloatBitWidth();
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getLowBit(), width, op->getLoc());
    Value value =
        rewriter.create<comb::ShrUOp>(op->getLoc(), adaptor.getInput(), amount);

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
    return success();
  }
};

struct ReduceAndOpConversion : public OpConversionPattern<ReduceAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value max = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              adaptor.getInput(), max);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              adaptor.getInput(), zero);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    if (isa_and_nonnull<IntegerType>(resultType)) {
      Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                adaptor.getInput(), zero);
      return success();
    }
    return failure();
  }
};

struct NotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value zero = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, adaptor.getInput());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct ICmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adapter.getLhs(), adapter.getRhs());
    return success();
  }
};

struct ConversionOpConversion : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getInput(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());

    rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, resultType, amount);
    return success();
  }
};

struct ConditionalOpConversion : public OpConversionPattern<ConditionalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto trueValue =
        adaptor.getTrueRegion().front().getTerminator()->getOperand(0);
    trueValue = rewriter.getRemappedValue(trueValue);

    auto falseValue =
        adaptor.getFalseRegion().front().getTerminator()->getOperand(0);
    falseValue = rewriter.getRemappedValue(falseValue);

    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);

    rewriter.replaceOpWithNewOp<comb::MuxOp>(op, adaptor.getCondition(),
                                             trueValue, falseValue);

    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

struct HWOutputOpConversion : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct HWInstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, convResTypes, op.getInstanceName(), op.getModuleName(),
        adaptor.getOperands(), op.getArgNames(),
        op.getResultNames(), /*Parameter*/
        rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

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

struct CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());
    return success();
  }
};

struct UnrealizedConversionCastConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

struct ShlOpConversion : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.getResult().replaceAllUsesWith(adaptor.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

struct EventOpConversion : public OpConversionPattern<EventOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EventOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp>
struct AssignOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

struct CAssignOpConversion : public OpConversionPattern<ContinuousAssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ContinuousAssignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.getDst().getType());
    StringAttr nameAttr;
    if (auto varOp = op.getDst().getDefiningOp<VariableOp>()) {
      nameAttr = varOp.getNameAttr();
    }
    if (auto netOp = op.getDst().getDefiningOp<NetOp>()) {
      nameAttr = netOp.getNameAttr();
      auto from = rewriter.getRemappedValue(op.getDst());
      rewriter.replaceOpWithNewOp<hw::WireOp>(from.getDefiningOp(), resultType,
                                              adaptor.getSrc(), nameAttr,
                                              hw::InnerSymAttr{});
    }
    rewriter.setInsertionPointAfterValue(op.getDst());
    rewriter.create<hw::WireOp>(op->getLoc(), resultType, adaptor.getSrc(),
                                nameAttr, hw::InnerSymAttr{});
    // } else {
    //   auto reValue = rewriter.getRemappedValue(op.getDst());
    //   if (auto wire = reValue.getDefiningOp<hw::WireOp>())
    //     rewriter.replaceAllUsesWith(wire.getInput(), adaptor.getSrc());
    // }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static bool isMooreType(Type type) { return isa<UnpackedType>(type); }

static bool hasMooreType(TypeRange types) {
  return llvm::any_of(types, isMooreType);
}

static bool hasMooreType(ValueRange values) {
  return hasMooreType(values.getTypes());
}

template <typename Op>
static void addGenericLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    return !hasMooreType(op->getOperands()) && !hasMooreType(op->getResults());
  });
}

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<MooreDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  // target.addLegalDialect<mlir::scf::SCFDialect>();

  addGenericLegality<cf::CondBranchOp>(target);
  addGenericLegality<cf::BranchOp>(target);
  addGenericLegality<func::CallOp>(target);
  addGenericLegality<func::ReturnOp>(target);
  addGenericLegality<UnrealizedConversionCastOp>(target);
  addGenericLegality<scf::WhileOp>(target);
  addGenericLegality<scf::ConditionOp>(target);

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasMooreType(block.getArguments());
    });
    auto resultsConverted = !hasMooreType(op.getResultTypes());
    return argsConverted && resultsConverted;
  });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([](hw::HWModuleOp op) {
    return !hasMooreType(op.getInputTypes()) &&
           !hasMooreType(op.getOutputTypes()) &&
           !hasMooreType(op.getBody().getArgumentTypes());
  });

  target.addDynamicallyLegalOp<hw::InstanceOp>([](hw::InstanceOp op) {
    return !hasMooreType(op.getInputs()) && !hasMooreType(op.getResults());
  });

  target.addDynamicallyLegalOp<hw::OutputOp>(
      [](hw::OutputOp op) { return !hasMooreType(op.getOutputs()); });

  target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp op) {
    if (auto whileOp = llvm::dyn_cast_or_null<scf::WhileOp>(op->getParentOp()))
      return whileOp.getYieldOp() == op;
    // if (auto ifOp = llvm::dyn_cast_or_null<scf::IfOp>(op->getParentOp()))
    //   return (ifOp.thenYield() == op) || (ifOp.elseYield() == op);
    return false;
  });

  // target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) {
  //   llvm::function_ref<bool(Region &)> checkRegion;
  //   checkRegion = [&](auto &region) {
  //     for (auto &op : region.getOps()) {
  //       if (isa<NonBlockingAssignOp>(op))
  //         return false;
  //       if (auto ifOp = dyn_cast<scf::IfOp>(op))
  //         if (!checkRegion(ifOp.getThenRegion()) ||
  //             !checkRegion(ifOp.getElseRegion()))
  //           return false;
  //     }
  //     return true;
  //   };
  //   return checkRegion(op.getThenRegion()) &&
  //   checkRegion(op.getElseRegion());
  // });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](IntType type) {
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  typeConverter.addConversion([&](UnpackedType type) {
    return mlir::IntegerType::get(type.getContext(), *type.getBitSize());
  });

  typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
    if (auto nestedType = llvm::dyn_cast_or_null<IntType>(type.getNestedType()))
      return mlir::IntegerType::get(nestedType.getContext(),
                                    nestedType.getWidth());
    if (auto nestedType =
            llvm::dyn_cast_or_null<UnpackedType>(type.getNestedType()))
      return mlir::IntegerType::get(nestedType.getContext(),
                                    *nestedType.getBitSize());
    return std::nullopt;
  });

  typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
    if (isa<IntType, ArrayType, UnpackedArrayType>(type.getNestedType()))
      return mlir::IntegerType::get(type.getContext(),
                                    type.getBitSize().value());
    return std::nullopt;
  });

  // Valid target types.
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    // Patterns of declaration operations.
    VariableOpConversion, NetOpConversion,
    
    // Patterns of miscellaneous operations.
    ConstantOpConv, ConcatOpConversion, ReplicateOpConversion,
    ExtractOpConversion, DynExtractOpConversion, ExtractRefOpConversion,
    DynExtractRefOpConversion, ConversionOpConversion, NamedConstantOpConv,
    ReadOpConversion, ProcedureOpConversion, EventOpConversion,
    ConcatRefOpConversion,

    // Patterns of unary operations.
    ReduceAndOpConversion, ReduceOrOpConversion, ReduceXorOpConversion,
    BoolCastOpConversion, NotOpConversion, NegOpConversion,

    // Patterns of binary operations.
    BinaryOpConversion<AddOp, comb::AddOp>,
    BinaryOpConversion<SubOp, comb::SubOp>,
    BinaryOpConversion<MulOp, comb::MulOp>,
    BinaryOpConversion<DivUOp, comb::DivUOp>,
    BinaryOpConversion<DivSOp, comb::DivSOp>,
    BinaryOpConversion<ModUOp, comb::ModUOp>,
    BinaryOpConversion<ModSOp, comb::ModSOp>,
    BinaryOpConversion<AndOp, comb::AndOp>,
    BinaryOpConversion<OrOp, comb::OrOp>,
    BinaryOpConversion<XorOp, comb::XorOp>,

    // Patterns of relational operations.
    ICmpOpConversion<UltOp, ICmpPredicate::ult>,
    ICmpOpConversion<SltOp, ICmpPredicate::slt>,
    ICmpOpConversion<UleOp, ICmpPredicate::ule>,
    ICmpOpConversion<SleOp, ICmpPredicate::sle>,
    ICmpOpConversion<UgtOp, ICmpPredicate::ugt>,
    ICmpOpConversion<SgtOp, ICmpPredicate::sgt>,
    ICmpOpConversion<UgeOp, ICmpPredicate::uge>,
    ICmpOpConversion<SgeOp, ICmpPredicate::sge>,
    ICmpOpConversion<EqOp, ICmpPredicate::eq>,
    ICmpOpConversion<NeOp, ICmpPredicate::ne>,
    ICmpOpConversion<CaseEqOp, ICmpPredicate::ceq>,
    ICmpOpConversion<CaseNeOp, ICmpPredicate::cne>,
    ICmpOpConversion<WildcardEqOp, ICmpPredicate::weq>,
    ICmpOpConversion<WildcardNeOp, ICmpPredicate::wne>,
    
    // Patterns of structural operations.
    SVModuleOpConversion, InstanceOpConversion,

    // Patterns of shifting operations.
    ShrOpConversion, ShlOpConversion, AShrOpConversion,

    // Patterns of assignment operations.
    // AssignOpConversion<ContinuousAssignOp>,
    CAssignOpConversion,
    AssignOpConversion<BlockingAssignOp>,
    AssignOpConversion<NonBlockingAssignOp>,

    // Patterns of branch operations.
    CondBranchOpConversion, BranchOpConversion, ConditionalOpConversion,
    SCFIfOpConversion, 

    // Patterns of terminator operations.
    YieldOpConversion, SCFYieldOpConversion,

    // Patterns of other operations outside Moore dialect.
    HWOutputOpConversion, HWInstanceOpConversion, ReturnOpConversion,
    CallOpConversion, UnrealizedConversionCastConversion
  >(typeConverter, context);
  // clang-format on
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);

  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass
    : public circt::impl::ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // auto pm = PassManager::on<ModuleOp>(&context);
  // pm.addPass(moore::createInfoCollectionPass());
  // if (failed(pm.run(module)))
  //   return signalPassFailure();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

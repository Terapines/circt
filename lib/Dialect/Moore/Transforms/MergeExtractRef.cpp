//===- MergeExtractRef.cpp - Merge moore.extract_ref ----------------------===//
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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_MERGEEXTRACTREF
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {

struct IndexedSrc {
  IndexedSrc(uint32_t i, Value s) : index(i), src(s) {}

  uint32_t index;
  Value src;
};

static void
concatExtractRef(DenseMap<Value, SmallVector<IndexedSrc>> &extractRefs,
                 MLIRContext &ctx) {
  mlir::OpBuilder builder(&ctx);
  for (auto &[extrRef, indexedSrcs] : extractRefs) {
    uint32_t lo, hi;
    auto backIndex = indexedSrcs.back().index;
    auto frontIndex = indexedSrcs.front().index;

    if (backIndex < frontIndex) {
      llvm::sort(indexedSrcs, [](const IndexedSrc &a, const IndexedSrc &b) {
        return a.index > b.index;
      });
      lo = backIndex;
      hi = frontIndex + cast<UnpackedType>(indexedSrcs.front().src.getType())
                            .getBitSize()
                            .value();
    } else {
      llvm::sort(indexedSrcs, [](const IndexedSrc &a, const IndexedSrc &b) {
        return a.index < b.index;
      });
      lo = frontIndex;
      hi = backIndex + cast<UnpackedType>(indexedSrcs.back().src.getType())
                           .getBitSize()
                           .value();
    }

    if (indexedSrcs.empty())
      continue;

    auto type = extrRef.getType();
    auto width = cast<RefType>(type).getBitSize().value();
    auto domain = cast<RefType>(type).getDomain();
    auto loc = extrRef.getLoc();

    SmallVector<Value> values;
    for (const auto &indexedSrc : indexedSrcs)
      values.push_back(indexedSrc.src);

    builder.setInsertionPointAfterValue(indexedSrcs.back().src);
    Value concat = builder.create<ConcatOp>(loc, values);

    if (cast<IntType>(concat.getType()).getWidth() == width) {
      ;
    } else if ((lo != uint32_t(0)) || (hi != uint32_t(width))) {
      Value loExtract, hiExtract;
      if (lo != uint32_t(0)) {
        auto resultType = IntType::get(&ctx, lo, domain);
        auto read = builder.create<ReadOp>(
            loc, cast<RefType>(type).getNestedType(), extrRef);
        loExtract = builder.create<ExtractOp>(loc, resultType, read, 0);
      }

      if (hi != uint32_t(width)) {
        auto resultType = IntType::get(&ctx, width - hi, domain);
        auto read = builder.create<ReadOp>(
            loc, cast<RefType>(type).getNestedType(), extrRef);
        hiExtract = builder.create<ExtractOp>(loc, resultType, read, hi);
      }

      if (loExtract && hiExtract) {
        concat = builder.create<ConcatOp>(
            loc, ValueRange{loExtract, concat, hiExtract});
      } else
        concat =
            loExtract
                ? builder.create<ConcatOp>(loc, ValueRange{loExtract, concat})
                : builder.create<ConcatOp>(loc, ValueRange{concat, hiExtract});

    } else {
      emitError(loc)
          << "Unsupported the complex situations of missing some middle bits ";
      return;
    }
    if (type != concat.getType())
      concat = builder.create<ConversionOp>(
          loc, cast<RefType>(type).getNestedType(), concat);
    if (isa<NonBlockingAssignOp>(*extrRef.getUsers().begin()))
      builder.create<NonBlockingAssignOp>(loc, extrRef, concat, Value{});
    else
      builder.create<BlockingAssignOp>(loc, extrRef, concat, Value{});
  }
}

static void
collectExtractRef(Operation *op, Value src,
                  DenseMap<Value, SmallVector<IndexedSrc>> &extractRefs) {

  uint32_t index;
  if (auto extrRef = llvm::dyn_cast_or_null<ExtractRefOp>(op))
    index = extrRef.getLowBit();
  else
    index = 0;

  if (op->getOperand(0).getDefiningOp<VariableOp>()) {
    extractRefs[op->getOperand(0)].emplace_back(index, src);
  } else
    emitError(op->getLoc()) << "Unsupported nested extract_ref op";
}

static void run(Operation *op,
                DenseMap<Value, SmallVector<IndexedSrc>> &extractRefs) {
  TypeSwitch<Operation *, void>(op)
      .Case<BlockingAssignOp, NonBlockingAssignOp>([&](auto op) {
        if (auto extrRef = op.getDst().template getDefiningOp<ExtractRefOp>()) {
          collectExtractRef(extrRef, op.getSrc(), extractRefs);
        }
      })
      .Case<ProcedureOp>([&](auto op) {
        DenseMap<Value, SmallVector<IndexedSrc>> extrRefs(extractRefs);
        for (auto &nestedOp : op.getRegion().getOps())
          run(&nestedOp, extrRefs);
        concatExtractRef(extrRefs, *op.getContext());
      })
      .Case<scf::IfOp>([&](auto op) {
        DenseMap<Value, SmallVector<IndexedSrc>> thenExtrRefs(extractRefs);
        for (auto &thenOp : op.getThenRegion().getOps())
          run(&thenOp, thenExtrRefs);
        concatExtractRef(thenExtrRefs, *op.getContext());

        DenseMap<Value, SmallVector<IndexedSrc>> elseExtrRefs(extractRefs);
        for (auto &elseOp : op.getElseRegion().getOps())
          run(&elseOp, elseExtrRefs);
        concatExtractRef(elseExtrRefs, *op.getContext());
      });
}

struct MergeExtractRefPass
    : public circt::moore::impl::MergeExtractRefBase<MergeExtractRefPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createMergeExtractRefPass() {
  return std::make_unique<MergeExtractRefPass>();
}

void MergeExtractRefPass::runOnOperation() {
  DenseMap<Value, SmallVector<IndexedSrc>> extractRefs;
  for (auto &op : getOperation().getRegion().getOps()) {
    run(&op, extractRefs);
  }

  // concatExtractRef(extractRefs, getContext());

  // for (auto &[extrRef, indexedSrcs] : extractRefs) {
  //   auto name = extrRef.getDefiningOp()->getAttrOfType<StringAttr>("name");
  //   auto type = extrRef.getType();

  //   llvm::sort(indexedSrcs, [](const IndexedSrc &a, const IndexedSrc &b) {
  //     return a.index < b.index;
  //   });

  //   if (indexedSrcs.empty())
  //     continue;

  //   auto lo = indexedSrcs.front().index;
  //   auto hi = indexedSrcs.back().index +
  //             cast<UnpackedType>(indexedSrcs.back().src.getType())
  //                 .getBitSize()
  //                 .value();

  //   auto width = cast<RefType>(type).getBitSize().value();
  //   auto domain = cast<RefType>(type).getDomain();
  //   auto loc = extrRef.getLoc();

  //   SmallVector<Value> values;
  //   for (const auto &indexedSrc : indexedSrcs)
  //     values.push_back(indexedSrc.src);

  //   builder.setInsertionPointAfterValue(indexedSrcs.back().src);
  //   Value concat = builder.create<ConcatOp>(loc, values);

  //   if (cast<IntType>(concat.getType()).getWidth() == width) {
  //     ;
  //   } else if ((lo != uint32_t(0)) || (hi != uint32_t(width))) {
  //     Value loExtract, hiExtract;
  //     if (lo != uint32_t(0)) {
  //       auto resultType = IntType::get(&getContext(), lo, domain);
  //       Value lowBit = builder.create<ConstantOp>(loc, resultType, 0);
  //       auto read = builder.create<ReadOp>(
  //           loc, cast<RefType>(type).getNestedType(), extrRef);
  //       loExtract = builder.create<ExtractOp>(loc, resultType, read, lowBit);
  //     }

  //     if (hi != uint32_t(width)) {
  //       auto resultType = IntType::get(&getContext(), width - hi, domain);
  //       Value lowBit = builder.create<ConstantOp>(loc, resultType, hi);
  //       auto read = builder.create<ReadOp>(
  //           loc, cast<RefType>(type).getNestedType(), extrRef);
  //       hiExtract = builder.create<ExtractOp>(loc, resultType, read, lowBit);
  //     }

  //     if (loExtract && hiExtract) {
  //       concat = builder.create<ConcatOp>(
  //           loc, ValueRange{loExtract, concat, hiExtract});
  //     } else
  //       concat =
  //           loExtract
  //               ? builder.create<ConcatOp>(loc, ValueRange{loExtract,
  //               concat}) : builder.create<ConcatOp>(loc, ValueRange{concat,
  //               hiExtract});

  //   } else {
  //     emitError(loc)
  //         << "Unsupported the complex situations of missing some middle bits
  //         ";
  //     return;
  //   }
  //   builder.create<VariableOp>(loc, type, name, concat);
  //   // builder.create<NonBlockingAssignOp>(loc, extrRef, concat);
  // }
}

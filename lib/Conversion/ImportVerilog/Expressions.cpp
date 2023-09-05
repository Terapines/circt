
#include "ImportVerilogInternals.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/StringRef.h"
#include <slang/ast/Expression.h>
#include <slang/ast/expressions/AssignmentExpressions.h>
#include <slang/ast/expressions/LiteralExpressions.h>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/expressions/OperatorExpressions.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/numeric/ConstantValue.h>

using namespace circt;
using namespace ImportVerilog;


namespace {
struct ExprVisitor {
  Context &context;
  Location loc;
  ExprVisitor(Context &context, Location loc) : context(context), loc(loc), builder(context.rootBuilder) {}
  mlir::OpBuilder builder;

  //----------------------------------------------------------------------------------
  //Literal Expressions visit
  //----------------------------------------------------------------------------------

  Value visit(const slang::ast::IntegerLiteral &expr,const slang::ast::VariableSymbol &var){
    Type tmpType = context.convertType(*var.getDeclaredType());
    uint32_t value = expr.getValue().as<uint32_t>().value();
    return builder.create<moore::VariableDeclOp>(loc,
                    moore::LValueType::get(tmpType),
                    builder.getStringAttr(var.name),
                    value);
  }


  //TODO: 
  Value visit(const slang::ast::RealLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::TimeLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::UnbasedUnsizedIntegerLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::NullLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::UnboundedLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::StringLiteral &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //----------------------------------------------------------------------------------
  //Misc Expressions visit
  //----------------------------------------------------------------------------------

  //TODO:
  Value visit(const slang::ast::NamedValueExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::HierarchicalValueExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::DataTypeExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::TypeReferenceExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ArbitrarySymbolExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::LValueReferenceExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::EmptyArgumentExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::DistExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::CopyClassExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::MinTypMaxExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ClockingEventExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::AssertionInstanceExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::TaggedUnionExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //----------------------------------------------------------------------------------
  //Operator Expressions
  //----------------------------------------------------------------------------------

  //TODO: 
  Value visit(const slang::ast::UnaryExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::BinaryExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ConditionalExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::InsideExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ConcatenationExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ReplicationExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::StreamingConcatenationExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::OpenRangeExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //----------------------------------------------------------------------------------
  //Assignment Expressions
  //----------------------------------------------------------------------------------

  //TODO: 
  Value visit(const slang::ast::AssignmentExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ConversionExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::SimpleAssignmentPatternExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::StructuredAssignmentPatternExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::ReplicatedAssignmentPatternExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::NewArrayExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::NewClassExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::NewCovergroupExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //----------------------------------------------------------------------------------
  //Select Expressions
  //----------------------------------------------------------------------------------

  //TODO: 
  Value visit(const slang::ast::ElementSelectExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::RangeSelectExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //TODO: 
  Value visit(const slang::ast::MemberAccessExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }

  //----------------------------------------------------------------------------------
  //Call Expressions
  //----------------------------------------------------------------------------------

  //TODO: 
  Value visit(const slang::ast::CallExpression &expr,const slang::ast::VariableSymbol &var){
    assert(0 && "TODO");
  }
  

 /// Emit an error for all other types.
  template<typename T, typename... Args>
  Value visit(T&& node, Args&&... args) {
    mlir::emitError(loc, "unsupported expr: ");
    return nullptr;
  }
  /// Emit an error for all other types.
  template<typename T, typename... Args>
  Value visitInvalid(T&& node, Args&&... args) {
    mlir::emitError(loc, "unsupported expr: ");
    return nullptr;
  }
};
} // namespace

void Context::convertExpr(const slang::ast::VariableSymbol &varAst, LocationAttr loc) {
  auto *expr = varAst.getInitializer();
  if (!loc)
    loc = convertLocation(expr->sourceRange.start());
  auto value = expr->visit(ExprVisitor(*this, loc),varAst);
  if(!value)
    return;
  varSymbolTable.insert(varAst.name, value);

}

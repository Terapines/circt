add_circt_dialect_library(CIRCTMooreTransforms
CombineNonBlockings.cpp
DetectRegisters.cpp
InfoCollection.cpp
LowerConcatRef.cpp
MergeExtractRef.cpp
PullNonBlockingUp.cpp
SimplifyProcedures.cpp


  DEPENDS
  CIRCTMooreTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

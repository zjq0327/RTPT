import LoadShadingData;
import PathTracer;
import RenderPasses.ReSTIRPTPass.SharcCache;

// Share the sparse training kernel while binding this pass's PathTracer layout.
#define SHARC_FIXED_X1_TRAINER 1
#include "../ReSTIRPTPass/SharcUpdate.cs.slang"

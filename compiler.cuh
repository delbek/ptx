#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <fstream>
#include "gpuHelpers.cuh"
#include <string>

extern CUmodule module;
extern CUcontext ctx;

std::string loadPTX(std::string filename);
CUfunction compilePTX(std::string filename, std::string kernelName);
void destroyContext();

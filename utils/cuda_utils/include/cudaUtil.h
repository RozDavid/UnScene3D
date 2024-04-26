#pragma once

#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#undef max
#undef min

#include "cutil_inline_runtime.h"

// Enable run time assertion checking in kernel code
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }
//#define cudaAssert(condition)

#if defined(__CUDA_ARCH__)
#define __CONDITIONAL_UNROLL__ #pragma unroll
#else
#define __CONDITIONAL_UNROLL__ 
#endif

// math helpers
#include "cutil_math.h"

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif 

#ifndef slong 
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

#endif

// Util functions
int optimal_thread_num(int view_size){

    if (view_size <= 2){return 2;}
    else if (view_size <= 4){return 4;}
    else if (view_size <= 8){return 8;}
    else {return 16;}

}


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace {
    struct RayCastParams
    {
        int width;
        int height;
        float depthMin;
        float depthMax;
        float* intrinsicsParams;
        float threshSampleDist;
        float rayIncrement;
        int batch_size;
        int view_num;

        __device__ inline
        float getFx(int batch) const { return intrinsicsParams[batch*4 + 0]; }
        __device__ inline
        float getFy(int batch) const { return intrinsicsParams[batch*4 + 1]; }
        __device__ inline
        float getMx(int batch) const { return intrinsicsParams[batch*4 + 2]; }
        __device__ inline
        float getMy(int batch) const { return intrinsicsParams[batch*4 + 3]; }
    };
    struct RayCastSample
    {
        float sdf;
        float alpha;
        uint weight;
    };
}

__device__
inline float3 kinectDepthToSkeleton(float mx, float my, float fx, float fy, uint ux, uint uy, float depth)	{
    const float x = ((float)ux-mx) / fx;
    const float y = ((float)uy-my) / fy;
    return make_float3(depth*x, depth*y, depth);
}
__device__
inline float kinectProjToCameraZ(float depthMin, float depthMax, float z) {
    return z * (depthMax - depthMin) + depthMin;
}
__device__
inline float3 kinectProjToCamera(float depthMin, float depthMax, float mx, float my, float fx, float fy, uint ux, uint uy, float z)	{
    float fSkeletonZ = kinectProjToCameraZ(depthMin, depthMax, z);
    return kinectDepthToSkeleton(mx, my, fx, fy, ux, uy, fSkeletonZ);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

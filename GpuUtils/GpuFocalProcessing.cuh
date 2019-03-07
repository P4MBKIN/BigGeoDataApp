#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned short int pixel;

namespace winGpu
{
	enum FocalOpType
	{
		BoxBlur3 = 0,
		BoxBlur5 = 1,
		BoxBlur7 = 2,
		GaussianBlur3 = 3,
		GaussianBlur5 = 4,
		EdgeDetection3_1 = 5,
		EdgeDetection3_2 = 6,
		EdgeDetection3_3 = 7,
		Sharpen3 = 8,
		UnsharpMasking5 = 9
	};

	struct FocalKernelGpu
	{
		int sideSize;
		int midSize;
		double* ker;
		__host__ __device__ const double* operator[] (int i) const { return ker + i * sideSize; }
		__host__ __device__ double* operator[] (int i) { return ker + i * sideSize; }
		__host__ __device__ size_t size() const { return sideSize * sideSize * sizeof(double); }
	};

	struct FocalRasterGpu
	{
		int height, width;
		const pixel defaultValueConst = 55537;
		pixel defaultValue = 55537;
		pixel* data;
		__host__ __device__ pixel& operator() (int h, int w)
		{
			if (0 <= h && h < height &&
				0 <= w && w < width)
			{
				return *(data + h * width + w);
			}
			else
			{
				return defaultValue;
			}
		}
		__host__ __device__ const pixel& operator() (int h, int w) const
		{
			if (0 <= h && h < height &&
				0 <= w && w < width)
			{
				return *(data + h * width + w);
			}
			else
			{
				return defaultValueConst;
			}
		}
		__host__ __device__ size_t size() const { return height * width * sizeof(pixel); }
	};

	void doFocalOpGpu(pixel* input, int height, int width, pixel* output, int type);
}

#define GPU_BOX_BLUR_3 \
{ 1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9 }

#define GPU_BOX_BLUR_5 \
{ 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25 }

#define GPU_BOX_BLUR_7 \
{ 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49 }

#define GPU_GAUSSIAN_BLUR_3 \
{ 1.0,  2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0 }

#define GPU_GAUSSIAN_BLUR_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, 36.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

#define GPU_EDGE_DETECTION_3_1 \
{ 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0 }

#define GPU_EDGE_DETECTION_3_2 \
{ 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0 }

#define GPU_EDGE_DETECTION_3_3 \
{ -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0 }

#define GPU_SHARPEN_3 \
{ 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0 }

#define GPU_UNSHARP_MASKING_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, -476.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

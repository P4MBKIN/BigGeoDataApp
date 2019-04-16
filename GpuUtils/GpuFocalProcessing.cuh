#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>

typedef short int pixel;

namespace winGpu
{
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
		const pixel defaultValueConst = -9999;
		pixel defaultValue = -9999;
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

	double doFocalOpGpu(pixel* input, int height, int width, pixel* output, std::vector<double> matrix);
}

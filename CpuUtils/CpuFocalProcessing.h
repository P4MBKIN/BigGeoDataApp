#pragma once
#include <stdio.h>

typedef unsigned short int pixel;

namespace winCpu
{
	enum FocalOpTypeCpu
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

	struct FocalKernelCpu
	{
		int sideSize;
		int midSize;
		double* ker;
		const double* operator[] (int i) const { return ker + i * sideSize; }
		double* operator[] (int i) { return ker + i * sideSize; }
		size_t size() const { return sideSize * sideSize * sizeof(double); }
	};

	struct FocalRasterCpu
	{
		int height, width;
		const pixel defaultValueConst = 55537;
		pixel defaultValue = 55537;
		pixel* data;
		pixel& operator() (int h, int w)
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
		const pixel& operator() (int h, int w) const
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
		size_t size() const { return height * width * sizeof(pixel); }
	};

	double doFocalOpCpu(pixel* input, int height, int width, int type);
}

#define CPU_BOX_BLUR_3 \
{ 1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9 }

#define CPU_BOX_BLUR_5 \
{ 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25 }

#define CPU_BOX_BLUR_7 \
{ 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49 }

#define CPU_GAUSSIAN_BLUR_3 \
{ 1.0,  2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0 }

#define CPU_GAUSSIAN_BLUR_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, 36.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

#define CPU_EDGE_DETECTION_3_1 \
{ 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0 }

#define CPU_EDGE_DETECTION_3_2 \
{ 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0 }

#define CPU_EDGE_DETECTION_3_3 \
{ -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0 }

#define CPU_SHARPEN_3 \
{ 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0 }

#define CPU_UNSHARP_MASKING_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, -476.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

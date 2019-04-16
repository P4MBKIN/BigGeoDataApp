#pragma once
#include <stdio.h>
#include <vector>

typedef short int pixel;

namespace winCpu
{
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
		const pixel defaultValueConst = -9999;
		pixel defaultValue = -9999;
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

	double doFocalOpCpu(pixel* input, int height, int width, std::vector<double> matrix);
}

#include <CpuFocalProcessing.h>
#include <CpuTimer.h>
#include <stdio.h>
#include <ctime>
#include <chrono>

using namespace winCpu;

void applyFocalOpCpu(FocalRasterCpu rasterInput, FocalKernelCpu kernel)
{
	for (int h = 0; h < rasterInput.height; h++)
	{
		for (int w = 0; w < rasterInput.width; w++)
		{
			double sum = 0.0;
			pixel resultPixel;
			for (int i = 0; i < kernel.sideSize; ++i)
			{
				for (int j = 0; j < kernel.sideSize; ++j)
				{
					pixel value = rasterInput(h + (i - kernel.midSize), w + (j - kernel.midSize));
					if (value == rasterInput.defaultValue)
					{
						resultPixel = rasterInput(h, w);
						continue;
					}
					sum += kernel[i][j] * value;
				}
			}
			if (sum <= 0)
			{
				sum = 0.0;
			}
			resultPixel = (pixel)sum;
		}
	}
}

double winCpu::doFocalOpCpu(pixel* input, int height, int width, int type)
{
	// Создаем Rater для входных данных
	FocalRasterCpu rasterInput;
	rasterInput.height = height;
	rasterInput.width = width;
	rasterInput.data = input;

	// Создаем Kernel для применения матрицы свертки
	FocalKernelCpu kernel;
	switch (type)
	{
	case FocalOpTypeCpu::BoxBlur3:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_BOX_BLUR_3;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::BoxBlur5:
	{
		kernel.sideSize = 5;
		double mas[] = CPU_BOX_BLUR_5;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::BoxBlur7:
	{
		kernel.sideSize = 7;
		double mas[] = CPU_BOX_BLUR_7;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::GaussianBlur3:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_GAUSSIAN_BLUR_3;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::GaussianBlur5:
	{
		kernel.sideSize = 5;
		double mas[] = CPU_GAUSSIAN_BLUR_5;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::EdgeDetection3_1:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_EDGE_DETECTION_3_1;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::EdgeDetection3_2:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_EDGE_DETECTION_3_2;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::EdgeDetection3_3:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_EDGE_DETECTION_3_3;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::Sharpen3:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_SHARPEN_3;
		kernel.ker = mas;
		break;
	}
	case FocalOpTypeCpu::UnsharpMasking5:
	{
		kernel.sideSize = 3;
		double mas[] = CPU_UNSHARP_MASKING_5;
		kernel.ker = mas;
		break;
	}
	default:
		break;
	}
	kernel.midSize = kernel.sideSize / 2;

	double time;
	CPU_TIMER_START;
	applyFocalOpCpu(rasterInput, kernel);
	CPU_TIMER_STOP(time);

	return time;
}

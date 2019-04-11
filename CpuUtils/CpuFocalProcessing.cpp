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

double winCpu::doFocalOpCpu(pixel* input, int height, int width, std::vector<double> matrix)
{
	// Создаем Rater для входных данных
	FocalRasterCpu rasterInput;
	rasterInput.height = height;
	rasterInput.width = width;
	rasterInput.data = input;

	// Создаем Kernel для применения матрицы свертки
	FocalKernelCpu kernel;
	kernel.sideSize = (int)std::sqrt(matrix.size());
	kernel.ker = matrix.data();
	kernel.midSize = kernel.sideSize / 2;

	double time;
	CPU_TIMER_START;
	applyFocalOpCpu(rasterInput, kernel);
	CPU_TIMER_STOP(time);

	return time;
}

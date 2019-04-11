#include "GpuFocalProcessing.cuh"
#include "GpuTimer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

using namespace winGpu;

__global__ void applyFocalOpGpu(FocalRasterGpu rasterInput, FocalRasterGpu rasterOutput, FocalKernelGpu kernel, int rowIter)
{
	int h = blockDim.x * blockIdx.x + threadIdx.x + rowIter;
	int w = blockDim.y * blockIdx.y + threadIdx.y;
	if (rasterInput.height <= h || rasterInput.width <= w)
	{
		return;
	}

	if (rasterInput(h, w) == rasterInput.defaultValue)
	{
		rasterOutput(h, w) = rasterInput(h, w);
		return;
	}
	double sum = 0.0;
	for (int i = 0; i < kernel.sideSize; ++i)
	{
		for (int j = 0; j < kernel.sideSize; ++j)
		{
			pixel value = rasterInput(h + (i - kernel.midSize), w + (j - kernel.midSize));
			if (value == rasterInput.defaultValue)
			{
				rasterOutput(h, w) = rasterInput(h, w);
				return;
			}
			sum += kernel[i][j] * value;
		}
	}
	if (sum <= 0)
	{
		sum = 0.0;
	}
	rasterOutput(h, w) = (pixel)sum;
}

double winGpu::doFocalOpGpu(pixel* input, int height, int width, pixel* output, std::vector<double> matrix)
{
	// Создаем Rater для входных данных
	FocalRasterGpu rasterInput;
	rasterInput.height = height;
	rasterInput.width = width;
	rasterInput.data = 0;

	// Создаем Rater для выходных данных
	FocalRasterGpu rasterOutput;
	rasterOutput.height = height;
	rasterOutput.width = width;
	rasterOutput.data = 0;

	// Создаем Kernel для применения матрицы свертки
	FocalKernelGpu kernelTemp;
	kernelTemp.sideSize = (int)std::sqrt(matrix.size());
	kernelTemp.ker = matrix.data();
	kernelTemp.midSize = kernelTemp.sideSize / 2;

	FocalKernelGpu kernel;
	kernel.sideSize = kernelTemp.sideSize;
	kernel.midSize = kernelTemp.midSize;
	kernel.ker = 0;

	cudaSetDevice(0);
	cudaMalloc((void**)&rasterInput.data, rasterInput.size());
	cudaMalloc((void**)&rasterOutput.data, rasterOutput.size());
	cudaMalloc((void**)&kernel.ker, kernel.size());
	cudaMemcpy(rasterInput.data, input, rasterInput.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel.ker, kernelTemp.ker, kernel.size(), cudaMemcpyHostToDevice);

	const size_t maxAvaliableCoords = 8000000;
	int countRowsPerIter = maxAvaliableCoords / width;
	int countIter = height / countRowsPerIter + 1;
	const size_t size = width * countRowsPerIter;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(countRowsPerIter / threadsPerBlock.x + 1, width / threadsPerBlock.y + 1);

	float time;
	GPU_TIMER_START;
	for (int i = 0; i < countIter; i++)
	{
		int rowIter = i * countRowsPerIter;
		applyFocalOpGpu << <numBlocks, threadsPerBlock >> > (rasterInput, rasterOutput, kernel, rowIter);
		cudaDeviceSynchronize();
		int k = 5;
	}
	GPU_TIMER_STOP(time);

	cudaMemcpy(output, rasterOutput.data, rasterOutput.size(), cudaMemcpyDeviceToHost);

	cudaFree(rasterInput.data);
	cudaFree(rasterOutput.data);
	cudaFree(kernel.ker);

	return (double)time;
}

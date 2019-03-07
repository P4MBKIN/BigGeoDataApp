#include "GpuFocalProcessing.cuh"
#include "GpuTimer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace winGpu;

__global__ void applyFocalOpGpu(FocalRasterGpu rasterInput, FocalRasterGpu rasterOutput, FocalKernelGpu kernel)
{
	int h = blockDim.x * blockIdx.x + threadIdx.x;
	int w = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;
	if (rasterInput.height <= h || rasterInput.width <= w || 1 < c)
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

double winGpu::doFocalOpGpu(pixel* input, int height, int width, pixel* output, int type)
{
	// ������� Rater ��� ������� ������
	FocalRasterGpu rasterInput;
	rasterInput.height = height;
	rasterInput.width = width;
	rasterInput.data = 0;

	// ������� Rater ��� �������� ������
	FocalRasterGpu rasterOutput;
	rasterOutput.height = height;
	rasterOutput.width = width;
	rasterOutput.data = 0;

	// ������� Kernel ��� ���������� ������� �������
	FocalKernelGpu kernelTemp;
	switch (type)
	{
	case FocalOpTypeGpu::BoxBlur3:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_BOX_BLUR_3;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::BoxBlur5:
	{
		kernelTemp.sideSize = 5;
		double mas[] = GPU_BOX_BLUR_5;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::BoxBlur7:
	{
		kernelTemp.sideSize = 7;
		double mas[] = GPU_BOX_BLUR_7;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::GaussianBlur3:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_GAUSSIAN_BLUR_3;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::GaussianBlur5:
	{
		kernelTemp.sideSize = 5;
		double mas[] = GPU_GAUSSIAN_BLUR_5;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::EdgeDetection3_1:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_EDGE_DETECTION_3_1;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::EdgeDetection3_2:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_EDGE_DETECTION_3_2;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::EdgeDetection3_3:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_EDGE_DETECTION_3_3;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::Sharpen3:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_SHARPEN_3;
		kernelTemp.ker = mas;
		break;
	}
	case FocalOpTypeGpu::UnsharpMasking5:
	{
		kernelTemp.sideSize = 3;
		double mas[] = GPU_UNSHARP_MASKING_5;
		kernelTemp.ker = mas;
		break;
	}
	default:
		break;
	}
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

	dim3 grid = dim3(height / 32 + 1, width / 32 + 1, 1);
	dim3 blocks = dim3(32, 32, 1);

	float time;
	GPU_TIMER_START;
	applyFocalOpGpu << <grid, blocks >> > (rasterInput, rasterOutput, kernel);
	cudaDeviceSynchronize();
	GPU_TIMER_STOP(time);

	cudaMemcpy(output, rasterOutput.data, rasterOutput.size(), cudaMemcpyDeviceToHost);

	cudaFree(rasterInput.data);
	cudaFree(rasterOutput.data);
	cudaFree(kernel.ker);

	return (double)time;
}

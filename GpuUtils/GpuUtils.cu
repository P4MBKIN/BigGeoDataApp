#include "GpuUtils.cuh"
#include "GpuFocalProcessing.cuh"
#include "GpuProjectionProcessing.cuh"
#include "GpuTimer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void addKernelGpu(double* res, const double* a, const double* b)
{
	int i = threadIdx.x;
	res[i] = a[i] + b[i];
}

double winGpu::testPlusGpu(const double* a, const double* b, double* res, size_t size)
{
	double* devA = 0;
	double* devB = 0;
	double* devRes = 0;
	cudaSetDevice(0);
	cudaMalloc((void**)&devRes, size * sizeof(double));
	cudaMalloc((void**)&devA, size * sizeof(double));
	cudaMalloc((void**)&devB, size * sizeof(double));
	cudaMemcpy(devA, a, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, size * sizeof(double), cudaMemcpyHostToDevice);

	float time;
	GPU_TIMER_START;
	addKernelGpu << <1, (int)size >> > (devRes, devA, devB);
	cudaDeviceSynchronize();
	GPU_TIMER_STOP(time);

	cudaMemcpy(res, devRes, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devRes);

	return (double)time;
}

double winGpu::performFocalOpGpu(pixel* input, int height, int width, pixel* output, std::vector<double> matrix)
{
	return winGpu::doFocalOpGpu(input, height, width, output, matrix);
}

double winGpu::performTransformUtmToWgsCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, bool southhemi, double* lon, double* lat)
{
	return winGpu::doTransformUtmToWgsCoordsGpu(xOrigin, yOrigin, xPixelSize, yPixelSize,
		height, width, zone, southhemi, lon, lat);
}

double winGpu::performTransformWgsToUtmCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, double* x, double* y)
{
	return winGpu::doTransformWgsToUtmCoordsGpu(xOrigin, yOrigin, xPixelSize, yPixelSize,
		height, width, zone, x, y);
}

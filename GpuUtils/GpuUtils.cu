#include "GpuUtils.cuh"
#include "GpuFocalProcessing.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void addKernelGpu(double *res, const double *a, const double *b)
{
	int i = threadIdx.x;
	res[i] = a[i] + b[i];
}

void winGpu::testPlusGpu(const double* a, const double* b, double* res, size_t size)
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
	addKernelGpu << <1, (int)size >> > (devRes, devA, devB);
	cudaDeviceSynchronize();
	cudaMemcpy(res, devRes, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devRes);
}

void winGpu::performFocalOpGpu(pixel* input, int height, int width, pixel* output, int type)
{
	winGpu::doFocalOpGpu(input, height, width, output, type);
}
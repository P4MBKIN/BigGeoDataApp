#include "GpuProjectionProcessing.cuh"
#include "GpuUtmWgsTransform.cuh"
#include "GpuTimer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace winGpu;

__global__ void applyTransformUtmToWgsCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, bool southhemi, double* lon, double* lat)
{
	int h = blockDim.x * blockIdx.x + threadIdx.x;
	int w = blockDim.y * blockIdx.y + threadIdx.y;
	if (height <= h || width <= w)
	{
		return;
	}
	double x = xOrigin + xPixelSize * w;
	double y = yOrigin + yPixelSize * h;
	double newLon = 0.0;
	double newLat = 0.0;

	UtmXYToLatLonGpu(x, y, zone, southhemi, newLon, newLat);

	lon[h * width + w] = newLon;
	lat[h * width + w] = newLat;
}

double winGpu::doTransformUtmToWgsCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, bool southhemi, double* lon, double* lat)
{
	const size_t maxAvaliableCoords = 1500000;
	int countRowsPerIter = maxAvaliableCoords / width;
	int countIter = height / countRowsPerIter + 1;
	const size_t size = width * countRowsPerIter;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(countRowsPerIter / threadsPerBlock.x + 1, width / threadsPerBlock.y + 1);

	double* newLon = new double[size];
	double* newLat = new double[size];
	double* dev_lon = 0;
	double* dev_lat = 0;

	float time;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_lon, size * sizeof(double));
	cudaMalloc((void**)&dev_lat, size * sizeof(double));
	GPU_TIMER_START;
	for (int i = 0; i < countIter; i++)
	{
		double newYOrigin = yOrigin + i * yPixelSize * countRowsPerIter;

		applyTransformUtmToWgsCoordsGpu << <numBlocks, threadsPerBlock >> > (xOrigin, newYOrigin,
			xPixelSize, yPixelSize, countRowsPerIter, width, zone, southhemi, dev_lon, dev_lat);
		cudaDeviceSynchronize();

		cudaMemcpy(newLon, dev_lon, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(newLat, dev_lat, size * sizeof(double), cudaMemcpyDeviceToHost);

		size_t countCoordsForCopy = i != countIter - 1 ? size :
			width * height - countRowsPerIter * width * i;
		for (int j = 0; j < countCoordsForCopy; j++)
		{
			lon[i * size + j] = newLon[j];
			lat[i * size + j] = newLat[j];
		}
	}
	GPU_TIMER_STOP(time);
	cudaFree(dev_lon);
	cudaFree(dev_lat);
	delete[] newLon;
	delete[] newLat;

	return (double)time;
}

__global__ void applyTransformWgsToUtmCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, double* x, double* y)
{
	int h = blockDim.x * blockIdx.x + threadIdx.x;
	int w = blockDim.y * blockIdx.y + threadIdx.y;
	if (height <= h || width <= w)
	{
		return;
	}
	double lon = xOrigin + xPixelSize * w;
	double lat = yOrigin + yPixelSize * h;
	double newX = 0.0;
	double newY = 0.0;

	LatLonToUtmXYGpu(lon, lat, zone, newX, newY);

	x[h * width + w] = newX;
	y[h * width + w] = newY;
}

double winGpu::doTransformWgsToUtmCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, double* x, double* y)
{
	const size_t maxAvaliableCoords = 2000000;
	int countRowsPerIter = maxAvaliableCoords / width;
	int countIter = height / countRowsPerIter + 1;
	const size_t size = width * countRowsPerIter;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(countRowsPerIter / threadsPerBlock.x + 1, width / threadsPerBlock.y + 1);

	double* newX = new double[size];
	double* newY = new double[size];
	double* dev_x = 0;
	double* dev_y = 0;

	float time;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_x, size * sizeof(double));
	cudaMalloc((void**)&dev_y, size * sizeof(double));
	GPU_TIMER_START;
	for (int i = 0; i < countIter; i++)
	{
		double newYOrigin = yOrigin + i * yPixelSize * countRowsPerIter;

		applyTransformWgsToUtmCoordsGpu << <numBlocks, threadsPerBlock >> > (xOrigin, newYOrigin,
			xPixelSize, yPixelSize, countRowsPerIter, width, zone, dev_x, dev_y);
		cudaDeviceSynchronize();

		cudaMemcpy(newX, dev_x, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(newY, dev_y, size * sizeof(double), cudaMemcpyDeviceToHost);

		size_t countCoordsForCopy = i != countIter - 1 ? size :
			width * height - countRowsPerIter * width * i;
		for (int j = 0; j < countCoordsForCopy; j++)
		{
			x[i * size + j] = newX[j];
			y[i * size + j] = newY[j];
		}
	}
	GPU_TIMER_STOP(time);
	cudaFree(dev_x);
	cudaFree(dev_y);
	delete[] newX;
	delete[] newY;

	return (double)time;
}

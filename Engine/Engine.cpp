#include <Engine.h>
#include <TiffWorker.h>
#include <Conversions.h>
#include <CpuUtils.h>
#include <GpuUtils.cuh>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ctime>
#include <chrono>

using namespace win;

double win::testPlusCpu(const double* a, const double* b, double* res, size_t size)
{
	auto start = std::chrono::steady_clock::now();

	winCpu::testPlusCpu(a, b, res, size);

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double time = (double)elapsed.count() / 1000;
	return time;
}

double win::testPlusGpu(const double* a, const double* b, double* res, size_t size)
{
	auto start = std::chrono::steady_clock::now();

	winGpu::testPlusGpu(a, b, res, size);

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double time = (double)elapsed.count() / 1000;
	return time;
}

double win::performFocalOpGpu(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type)
{
	auto start = std::chrono::steady_clock::now();

	GDALAllRegister();
	int typeIn = stringToFocalType(type);
	auto rasterIn = getPixelsFromTiff(pathFrom);
	const int height = std::get<1>(rasterIn);
	const int width = std::get<2>(rasterIn);
	pixel* pixelsOut = new pixel[height * width];
	winGpu::performFocalOpGpu(std::get<0>(rasterIn), height, width, pixelsOut, typeIn);
	copyTiff(pathFrom, pathTo);
	savePixelsToExistTiff(pathTo, pixelsOut);
	delete[] pixelsOut;

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double time = (double)elapsed.count() / 1000;
	return time;
}

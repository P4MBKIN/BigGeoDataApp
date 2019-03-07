#include <Engine.h>
#include <TiffWorker.h>
#include <Conversions.h>
#include <CpuUtils.h>
#include <GpuUtils.cuh>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

using namespace win;

double win::testPlusGpu(const double* a, const double* b, double* res, size_t size)
{
	return winGpu::testPlusGpu(a, b, res, size);
}

double win::performFocalOpGpu(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type)
{
	double time;

	GDALAllRegister();
	int typeIn = stringToFocalType(type);
	auto rasterIn = getPixelsFromTiff(pathFrom);
	const int height = std::get<1>(rasterIn);
	const int width = std::get<2>(rasterIn);
	pixel* pixelsOut = new pixel[height * width];
	time = winGpu::performFocalOpGpu(std::get<0>(rasterIn), height, width, pixelsOut, typeIn);
	copyTiff(pathFrom, pathTo);
	savePixelsToExistTiff(pathTo, pixelsOut);
	delete[] pixelsOut;

	return time;
}

double win::testPlusCpu(const double* a, const double* b, double* res, size_t size)
{
	return winCpu::testPlusCpu(a, b, res, size);
}

double win::performFocalOpCpu(const std::wstring& pathFrom, const std::wstring& type)
{
	double time;

	int typeIn = stringToFocalType(type);
	auto rasterIn = getPixelsFromTiff(pathFrom);
	const int height = std::get<1>(rasterIn);
	const int width = std::get<2>(rasterIn);
	time = winCpu::performFocalOpCpu(std::get<0>(rasterIn), height, width, typeIn);

	return time;
}


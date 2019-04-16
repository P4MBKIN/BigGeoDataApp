#include <Engine.h>
#include <TiffWorker.h>
#include <TiffInfo.h>
#include <Conversions.h>
#include <GeneralUtils.h>
#include <CpuUtils.h>
#include <GpuUtils.cuh>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <algorithm>
#include <iostream>

using namespace win;

double win::testPlusGpu(const double* a, const double* b, double* res, size_t size)
{
	return winGpu::testPlusGpu(a, b, res, size);
}

double win::performFocalOpGpu(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type)
{
	double time;

	GDALAllRegister();
	std::vector<double> matrix = stringToFocalMatrix(type);
	auto rasterIn = getPixelsFromTiff(pathFrom);
	const int height = std::get<1>(rasterIn);
	const int width = std::get<2>(rasterIn);
	pixel* pixelsOut = new pixel[height * width];
	time = winGpu::performFocalOpGpu(std::get<0>(rasterIn), height, width, pixelsOut, matrix);
	copyTiff(pathFrom, pathTo);
	savePixelsToExistTiff(pathTo, pixelsOut);
	delete[] pixelsOut;

	return time;
}

double win::performProjectionOpGpu(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type)
{
	double time = 0;

	GDALAllRegister();
	UtmOrWgsTiff inputInfo = getTiffInfoUtmOrWgs(pathFrom);
	double* newXCoord = new double[inputInfo.height * inputInfo.width];
	double* newYCoord = new double[inputInfo.height * inputInfo.width];
	int outputUtmZone = 0;

	if (inputInfo.isUtm)
	{
		time = winGpu::performTransformUtmToWgsCoordsGpu(inputInfo.xOrigin, inputInfo.yOrigin, inputInfo.xPixelSize, inputInfo.yPixelSize,
			inputInfo.height, inputInfo.width, inputInfo.utmZone, inputInfo.isUtmSouthhemi, newXCoord, newYCoord);
	}
	else
	{
		const std::wstring numberZoneStr = type.substr(3, type.size() - 1);
		outputUtmZone = std::stoi(numberZoneStr);
		time = winGpu::performTransformWgsToUtmCoordsGpu(inputInfo.xOrigin, inputInfo.yOrigin, inputInfo.xPixelSize, inputInfo.yPixelSize,
			inputInfo.height, inputInfo.width, outputUtmZone, newXCoord, newYCoord);
	}

	double minX = *std::min_element(newXCoord, newXCoord + inputInfo.height * inputInfo.width);
	double maxX = *std::max_element(newXCoord, newXCoord + inputInfo.height * inputInfo.width);
	double minY = *std::min_element(newYCoord, newYCoord + inputInfo.height * inputInfo.width);
	double maxY = *std::max_element(newYCoord, newYCoord + inputInfo.height * inputInfo.width);
	double newHeightInCoords = maxY - minY;
	double newWidthInCoords = maxX - minX;
	int newHeight;
	int newWidth;
	double newXPixelSize;
	double newYPixelSize;
	if (newWidthInCoords > newHeightInCoords)
	{
		newWidth = (inputInfo.height > inputInfo.width) ? inputInfo.width : inputInfo.height;
		newHeight = newHeightInCoords / newWidthInCoords * newWidth + 1;
		newXPixelSize = newWidthInCoords / newWidth;
		newYPixelSize = -newXPixelSize;
	}
	else
	{
		newHeight = (inputInfo.height > inputInfo.width) ? inputInfo.width : inputInfo.height;
		newWidth = newWidthInCoords / newHeightInCoords * newHeight + 1;
		newYPixelSize = -newHeightInCoords / newHeight;
		newXPixelSize = -newYPixelSize;
	}
	pixel* pixelsOut = new pixel[newHeight * newWidth];
	auto rasterIn = getPixelsFromTiff(pathFrom);
	replaceNewCoord(minX, maxY, newXPixelSize, newYPixelSize, newHeight, newWidth,
		newXCoord, newYCoord, std::get<1>(rasterIn), std::get<2>(rasterIn), std::get<0>(rasterIn), pixelsOut);

	UtmOrWgsTiff newTiffInfo;
	newTiffInfo.height = newHeight;
	newTiffInfo.width = newWidth;
	newTiffInfo.xOrigin = minX;
	newTiffInfo.yOrigin = maxY;
	newTiffInfo.xPixelSize = newXPixelSize;
	newTiffInfo.yPixelSize = newYPixelSize;
	newTiffInfo.isUtm = false;
	if (!inputInfo.isUtm)
	{
		newTiffInfo.isUtm = true;
		newTiffInfo.utmZone = outputUtmZone;
		newTiffInfo.isUtmSouthhemi = inputInfo.yOrigin < 0;
	}
	createTiffWithData(newTiffInfo, pixelsOut, pathTo);

	delete[] newXCoord;
	delete[] newYCoord;
	return time;
}

double win::testPlusCpu(const double* a, const double* b, double* res, size_t size)
{
	return winCpu::testPlusCpu(a, b, res, size);
}

double win::performFocalOpCpu(const std::wstring& pathFrom, const std::wstring& type)
{
	double time;

	std::vector<double> matrix = stringToFocalMatrix(type);
	auto rasterIn = getPixelsFromTiff(pathFrom);
	const int height = std::get<1>(rasterIn);
	const int width = std::get<2>(rasterIn);
	time = winCpu::performFocalOpCpu(std::get<0>(rasterIn), height, width, matrix);

	return time;
}

double win::performProjectionOpCpu(const std::wstring& pathFrom, const std::wstring& type)
{
	double time = 0;

	GDALAllRegister();
	UtmOrWgsTiff inputInfo = getTiffInfoUtmOrWgs(pathFrom);
	int outputUtmZone = 0;

	if (inputInfo.isUtm)
	{
		time = winCpu::performTransformUtmToWgsCoordsCpu(inputInfo.xOrigin, inputInfo.yOrigin, inputInfo.xPixelSize, inputInfo.yPixelSize,
			inputInfo.height, inputInfo.width, inputInfo.utmZone, inputInfo.isUtmSouthhemi);
	}
	else
	{
		const std::wstring numberZoneStr = type.substr(3, type.size() - 1);
		outputUtmZone = std::stoi(numberZoneStr);
		time = winCpu::performTransformWgsToUtmCoordsCpu(inputInfo.xOrigin, inputInfo.yOrigin, inputInfo.xPixelSize, inputInfo.yPixelSize,
			inputInfo.height, inputInfo.width, outputUtmZone);
	}
	return time;
}

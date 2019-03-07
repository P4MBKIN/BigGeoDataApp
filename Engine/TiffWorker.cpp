#include "TiffWorker.h"
#include <Conversions.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

std::tuple<pixel*, int, int> win::getPixelsFromTiff(const std::wstring& path)
{
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(narrow(path, CP_ACP).c_str(), GA_ReadOnly);
	const int width = GDALGetRasterXSize(poDataset);
	const int height = GDALGetRasterYSize(poDataset);
	const int num = GDALGetRasterCount(poDataset); // == 1
	GDALRasterBandH hBand = GDALGetRasterBand(poDataset, 1);
	GDALDataType hType = GDALGetRasterDataType(hBand);
	int hSize = GDALGetDataTypeSize(hType);

	pixel* data = new pixel[height * width];
	// начинаем посторчное считываение
	for (int i = 0; i < height; i++)
	{
		// Allocate a line buffer
		uint16_t* pData = (uint16_t*)CPLMalloc(hSize * width);
		GDALRasterIO(hBand,
			GF_Read,    // Read from band
			0, i,       // Offs X, Offs Y
			width, 1,	// Exact one line
			pData,      // The target buffer
			width, 1,	// Size of the target buffer
			hType,		// Buffer type
			0, 0);      // Array strides X/Y dir
		for (int j = 0; j < width; j++)
		{
			data[i * width + j] = (pixel)pData[j];
		}
		CPLFree(pData);
	}

	GDALClose(poDataset);

	return std::make_tuple(data, height, width);
}

void win::copyTiff(const std::wstring& pathFrom, const std::wstring& pathTo)
{
	GDALDataset* poDatasetIn = (GDALDataset*)GDALOpen(narrow(pathFrom, CP_ACP).c_str(), GA_ReadOnly);
	GDALDriver* poDriver;
	poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset* poDatasetOut;
	poDatasetOut = poDriver->CreateCopy(narrow(pathTo, CP_ACP).c_str(), poDatasetIn, FALSE,
		NULL, NULL, NULL);
	GDALClose(poDatasetOut);
	GDALClose(poDatasetIn);
}

void win::savePixelsToExistTiff(const std::wstring& path, pixel* data)
{
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(narrow(path, CP_ACP).c_str(), GA_Update);
	const int width = GDALGetRasterXSize(poDataset);
	const int height = GDALGetRasterYSize(poDataset);
	const int num = GDALGetRasterCount(poDataset); // == 1
	GDALRasterBandH hBand = GDALGetRasterBand(poDataset, 1);
	GDALDataType hType = GDALGetRasterDataType(hBand);
	int hSize = GDALGetDataTypeSize(hType);

	for (int i = 0; i < height; i++)
	{
		uint16_t* pData = (uint16_t*)CPLMalloc(hSize * width);
		for (int j = 0; j < width; j++)
		{
			pData[j] = (uint16_t)data[i * width + j];
		}
		GDALRasterIO(hBand,
			GF_Write,      // Write to band
			0, i,       // Offs X, Offs Y
			width, 1, // Exact one line
			pData,        // The target buffer
			width, 1, // Size of the target buffer
			hType,     // Buffer type
			0, 0);       // Array strides X/Y dir
		CPLFree(pData);
	}

	GDALClose(poDataset);
}

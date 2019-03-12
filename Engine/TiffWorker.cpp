#include <TiffWorker.h>
#include <TiffInfo.h>
#include <Conversions.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

using namespace win;

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

UtmOrWgsTiff win::getTiffInfoUtmOrWgs(const std::wstring& path)
{
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(narrow(path, CP_ACP).c_str(), GA_ReadOnly);
	const int width = GDALGetRasterXSize(poDataset);
	const int height = GDALGetRasterYSize(poDataset);
	double adfGeoTransform[6];
	poDataset->GetGeoTransform(adfGeoTransform);
	UtmOrWgsTiff info = getUtmOrWgsInfoFromData(poDataset->GetProjectionRef(), adfGeoTransform, height, width);
	GDALClose(poDataset);
	return info;
}

void win::createTiffWithData(UtmOrWgsTiff info, pixel* data, const std::wstring& path)
{
	GDALDriver* poDriver;
	GDALDataset* newDataset;
	double geoTransform[6] = { info.xOrigin, info.xPixelSize, 0, info.yOrigin, 0, info.yPixelSize };
	OGRSpatialReference oSRS;
	char** rMetadata;
	char* SRS_WKT = NULL;
	GDALRasterBand* band;
	char** ops = NULL;
	GUInt16* rasterArr;
	poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	rMetadata = poDriver->GetMetadata();
	CSLFetchBoolean(rMetadata, GDAL_DCAP_CREATE, FALSE);
	newDataset = poDriver->Create(narrow(path, CP_ACP).c_str(), info.width, info.height, 1, GDT_UInt16, ops);
	rasterArr = (GUInt16*)CPLMalloc(sizeof(GUInt16) * info.width * info.height);
	for (int i = 0; i < info.height; i++)
	{
		for (int j = 0; j < info.width; j++)
		{
			rasterArr[i * info.width + j] = data[i * info.width + j];
		}
	}
	newDataset->SetGeoTransform(geoTransform);
	oSRS.SetWellKnownGeogCS("WGS84");
	oSRS.exportToWkt(&SRS_WKT);
	newDataset->SetProjection(SRS_WKT);
	CPLFree(SRS_WKT);
	band = newDataset->GetRasterBand(1);
	band->RasterIO(GF_Write, 0, 0, info.width, info.height, rasterArr, info.width, info.height, GDT_UInt16, 0, 0);
	CPLFree(rasterArr);
	GDALClose((GDALDatasetH)newDataset);
}

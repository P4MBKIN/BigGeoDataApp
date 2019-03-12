#include <TiffInfo.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

using namespace win;

std::tuple<int, bool> win::getUtmZoneFromPROJCS(const std::string& projcs)
{
	std::string strZone = projcs.substr(18);
	strZone = strZone.substr(0, strZone.size() - 1);
	int zone = std::stoi(strZone);
	bool southhemi = projcs.back() == 'N' ? false : true;
	return std::make_tuple(zone, southhemi);
}

UtmOrWgsTiff win::getUtmOrWgsInfoFromData(const std::string& projRef, const double* geoTransform, int height, int width)
{
	UtmOrWgsTiff infoData;
	OGRSpatialReference oSRS = OGRSpatialReference(projRef.c_str());
	const char* pszProjection = oSRS.GetAttrValue("PROJCS");
	if (pszProjection != NULL)
	{
		infoData.isUtm = true;
		const std::string projcs(pszProjection);
		std::tuple<int, bool> zoneInfo = getUtmZoneFromPROJCS(projcs);
		infoData.utmZone = std::get<0>(zoneInfo);
		infoData.isUtmSouthhemi = std::get<1>(zoneInfo);
	}
	else
	{
		infoData.isUtm = false;
	}
	infoData.height = height;
	infoData.width = width;
	infoData.xOrigin = geoTransform[0];
	infoData.yOrigin = geoTransform[3];
	infoData.xPixelSize = geoTransform[1];
	infoData.yPixelSize = geoTransform[5];

	return infoData;
}

#pragma once
#include <string>
#include <tuple>

namespace win
{
	struct UtmOrWgsTiff
	{
		double xOrigin;
		double yOrigin;
		double xPixelSize;
		double yPixelSize;
		int height;
		int width;
		bool isUtm;

		// для UTM
		int utmZone;
		bool isUtmSouthhemi;
	};

	// строка приходит в виде "WGS 84 / UTM zone 37N"
	std::tuple<int, bool> getUtmZoneFromPROJCS(const std::string& projcs);
	UtmOrWgsTiff getUtmOrWgsInfoFromData(const std::string& projRef, const double* geoTransform, int height, int width);
}

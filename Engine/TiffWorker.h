#pragma once
#include <string>
#include <tuple>
#include <TiffInfo.h>

typedef unsigned short int pixel;

namespace win
{
	std::tuple<pixel*, int, int> getPixelsFromTiff(const std::wstring& path);
	void copyTiff(const std::wstring& pathFrom, const std::wstring& pathTo);
	void savePixelsToExistTiff(const std::wstring& pathTo, pixel* data);
	win::UtmOrWgsTiff getTiffInfoUtmOrWgs(const std::wstring& path);
	void createTiffWithData(win::UtmOrWgsTiff info, pixel* data, const std::wstring& path);
}

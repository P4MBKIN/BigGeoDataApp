#pragma once
#include <string>
#include <tuple>

typedef unsigned short int pixel;

namespace win
{
	std::tuple<pixel*, int, int> getPixelsFromTiff(const std::wstring& path);
	void copyTiff(const std::wstring& pathFrom, const std::wstring& pathTo);
	void savePixelsToExistTiff(const std::wstring& pathTo, pixel* data);
}

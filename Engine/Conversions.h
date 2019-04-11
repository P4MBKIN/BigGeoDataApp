#pragma once

#include <codecvt>
#include <windows.h>
#include <vector>
#include <sstream>
#include <fstream>

/// Преобразует UTF-16 std::wstring в UTF-8 std::string
inline std::string narrow(const std::wstring& wstr, uint32_t codePage = CP_UTF8)
{
	if (wstr.empty())
	{
		return std::string();
	}

	DWORD flags = WC_NO_BEST_FIT_CHARS;
	if (codePage == CP_UTF8 ||
		codePage == CP_UTF7)
	{
		flags = 0;
	}

	DWORD length = ::WideCharToMultiByte(codePage, flags, wstr.c_str(),
		static_cast<int32_t>(wstr.length()), nullptr, 0, nullptr, nullptr);
	if (length > 0)
	{

		std::string buffer(length, 0);
		DWORD returnedChars = ::WideCharToMultiByte(codePage, flags, wstr.c_str(),
			static_cast<int32_t>(wstr.length()), (LPSTR)buffer.data(), static_cast<int32_t>(buffer.size()), nullptr, nullptr);

		if (returnedChars)
		{
			return buffer;
		}
	}
	throw std::system_error(GetLastError(), std::system_category());
}

/// Преобразует UTF-8 и ANSI std::string в UTF-16 std::wstring
inline std::wstring wide(const std::string& str, uint32_t codePage = CP_ACP)
{
	if (str.empty())
	{
		return std::wstring();
	}

	// UTF-8
	try
	{
		return std::wstring(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(str));
	}
	catch (const std::range_error&)
	{
	}

	// ANSI
	size_t length = ::MultiByteToWideChar(codePage, 0, str.data(), (int32_t)str.size(), nullptr, 0);
	if (length)
	{
		std::wstring result(length, 0);

		::MultiByteToWideChar(codePage, 0, str.data(), (int32_t)str.size(),
			const_cast<wchar_t*>(result.data()), (int32_t)result.size());

		return result;
	}
	throw std::system_error(GetLastError(), std::system_category());
}

inline std::vector<double> parseCSV(const std::wstring& path)
{
	std::ifstream data(narrow(path));
	std::string line;
	std::vector<std::vector<std::string> > parsedCsv;
	while (std::getline(data, line))
	{
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<std::string> parsedRow;
		while (std::getline(lineStream, cell, ','))
		{
			parsedRow.push_back(cell);
		}

		parsedCsv.push_back(parsedRow);
	}
	std::vector<double> result;
	for (int i = 0; i < parsedCsv.size(); i++)
	{
		for (int j = 0; j < parsedCsv[i].size(); j++)
		{
			result.push_back(std::stod(parsedCsv[i][j]));
		}
	}
	return result;
};

#define BOX_BLUR_3 \
{ 1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9, \
1.0 / 9, 1.0 / 9, 1.0 / 9 }

#define BOX_BLUR_5 \
{ 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, \
1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25 }

#define BOX_BLUR_7 \
{ 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, \
1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49 }

#define GAUSSIAN_BLUR_3 \
{ 1.0,  2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0 }

#define GAUSSIAN_BLUR_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, 36.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

#define EDGE_DETECTION_3_1 \
{ 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0 }

#define EDGE_DETECTION_3_2 \
{ 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0 }

#define EDGE_DETECTION_3_3 \
{ -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0 }

#define SHARPEN_3 \
{ 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0 }

#define UNSHARP_MASKING_5 \
{ 1.0, 4.0, 6.0, 4.0, 1.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
6.0, 24.0, -476.0, 24.0, 6.0, \
4.0, 16.0, 24.0, 16.0, 4.0, \
1.0, 4.0, 6.0, 4.0, 1.0 }

inline std::vector<double> stringToFocalMatrix(const std::wstring& type)
{
	if (type == L"BoxBlur3")
	{
		return std::vector<double> BOX_BLUR_3;
	}
	else if (type == L"BoxBlur5")
	{
		return std::vector<double> BOX_BLUR_5;
	}
	else if (type == L"BoxBlur7")
	{
		return std::vector<double> BOX_BLUR_7;
	}
	else if (type == L"GaussianBlur3")
	{
		return std::vector<double> GAUSSIAN_BLUR_3;
	}
	else if (type == L"GaussianBlur5")
	{
		return std::vector<double> GAUSSIAN_BLUR_5;
	}
	else if (type == L"EdgeDetection3_1")
	{
		return std::vector<double> EDGE_DETECTION_3_1;
	}
	else if (type == L"EdgeDetection3_2")
	{
		return std::vector<double> EDGE_DETECTION_3_2;
	}
	else if (type == L"EdgeDetection3_3")
	{
		return std::vector<double> EDGE_DETECTION_3_3;
	}
	else if (type == L"Sharpen3")
	{
		return std::vector<double> SHARPEN_3;
	}
	else if (type == L"UnsharpMasking5")
	{
		return std::vector<double> UNSHARP_MASKING_5;
	}
	else
	{
		return parseCSV(type);
	}
}

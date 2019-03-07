#pragma once

#include <codecvt>
#include <windows.h>
#include <vector>

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

inline int stringToFocalType(const std::wstring& type)
{
	if (type == L"BoxBlur3")
	{
		return 0;
	}
	else if (type == L"BoxBlur5")
	{
		return 1;
	}
	else if (type == L"BoxBlur7")
	{
		return 2;
	}
	else if (type == L"GaussianBlur3")
	{
		return 3;
	}
	else if (type == L"GaussianBlur5")
	{
		return 4;
	}
	else if (type == L"EdgeDetection3_1")
	{
		return 5;
	}
	else if (type == L"EdgeDetection3_2")
	{
		return 6;
	}
	else if (type == L"EdgeDetection3_3")
	{
		return 7;
	}
	else if (type == L"Sharpen3")
	{
		return 8;
	}
	else if (type == L"UnsharpMasking5")
	{
		return 9;
	}
	return 0;
}

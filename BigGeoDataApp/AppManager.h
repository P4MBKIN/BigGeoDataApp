#pragma once
#include <string>

class AppManager final
{
public:
	AppManager();
	~AppManager();

	std::wstring test() const;
	std::wstring doFocalOperation(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type, bool isCompare) const;
	std::wstring doProjectionOperation(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type, bool isCompare) const;
};

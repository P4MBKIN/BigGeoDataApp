#include "pch.h"
#include <AppManager.h>
#include <Engine.h>

using namespace win;

AppManager::AppManager()
{
}

AppManager::~AppManager()
{
}

std::wstring AppManager::test() const
{
	//std::wstring result = L"{1.1, 2.2, 3.3, 4.4, 5.5} + {10, 20, 30, 40, 50}\n";
	//const size_t arraySize = 5;
	//const double a[arraySize] = { 1.1, 2.2, 3.3, 4.4, 5.5 };
	//const double b[arraySize] = { 10, 20, 30, 40, 50 };

	//double cGpu[arraySize] = { 0 };
	//result += L"time GPU: " + std::to_wstring(testPlusGpu(a, b, cGpu, arraySize)) + L"\n";
	//result += L"result GPU: " + std::to_wstring(cGpu[0]) + L", " + std::to_wstring(cGpu[1]) + L", " + std::to_wstring(cGpu[2]) + L", "
	//	+ std::to_wstring(cGpu[3]) + L", " + std::to_wstring(cGpu[4]) + L"\n";

	//double cCpu[arraySize] = { 0 };
	//result += L"time CPU: " + std::to_wstring(testPlusCpu(a, b, cCpu, arraySize)) + L"\n";
	//result += L"result CPU: " + std::to_wstring(cCpu[0]) + L", " + std::to_wstring(cCpu[1]) + L", " + std::to_wstring(cCpu[2]) + L", " +
	//	std::to_wstring(cCpu[3]) + L", " + std::to_wstring(cCpu[4]) + L"\n";

	//return result;

	return std::to_wstring(performProjectionOpGpu(L"C:\\University\\Course3Work\\LC81790212015146-SC20150806075046\\LC81790212015146LGN00_sr_band1.tif",
		L"C:\\University\\Course3Work\\LC81790212015146-SC20150806075046\\band1_projection.tif", L""));
}

std::wstring AppManager::doFocalOperation(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type, bool isCompare) const
{
	std::wstring result;
	result += L"time GPU: " + std::to_wstring(performFocalOpGpu(pathFrom, pathTo, type)) + L"\n";
	if (isCompare)
	{
		result += L"time CPU: " + std::to_wstring(performFocalOpCpu(pathFrom, type)) + L"\n";
	}
	return result;
}

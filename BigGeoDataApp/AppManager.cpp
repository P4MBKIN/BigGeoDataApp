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
	std::wstring result = L"{1.1, 2.2, 3.3, 4.4, 5.5} + {10, 20, 30, 40, 50}\n";
	const size_t arraySize = 5;
	const double a[arraySize] = { 1.1, 2.2, 3.3, 4.4, 5.5 };
	const double b[arraySize] = { 10, 20, 30, 40, 50 };

	double cCpu[arraySize] = { 0 };
	result += L"time CPU: " + std::to_wstring(testPlusCpu(a, b, cCpu, arraySize)) + L"\n";
	result += L"result CPU: " + std::to_wstring(cCpu[0]) + L", " + std::to_wstring(cCpu[1]) + L", " + std::to_wstring(cCpu[2]) + L", " +
		std::to_wstring(cCpu[3]) + L", " + std::to_wstring(cCpu[4]) + L"\n";

	double cGpu[arraySize] = { 0 };
	result += L"time GPU: " + std::to_wstring(testPlusGpu(a, b, cGpu, arraySize)) + L"\n";
	result += L"result GPU: " + std::to_wstring(cGpu[0]) + L", " + std::to_wstring(cGpu[1]) + L", " + std::to_wstring(cGpu[2]) + L", "
		+ std::to_wstring(cGpu[3]) + L", " + std::to_wstring(cGpu[4]) + L"\n";

	return result;
}

std::wstring AppManager::doFocalOperation(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type) const
{
	std::wstring result = L"«апускаем фокальную операцию на GPU\n";
	result += L"time GPU: " + std::to_wstring(performFocalOpGpu(pathFrom, pathTo, type)) + L"\n";
	// TODO : сделать дл€ CPU
	return result;
}

#pragma once
#include <string>

namespace win
{
	double testPlusCpu(const double* a, const double* b, double* res, size_t size);


	double testPlusGpu(const double* a, const double* b, double* res, size_t size);
	double performFocalOpGpu(const std::wstring& pathFrom, const std::wstring& pathTo, const std::wstring& type);
}

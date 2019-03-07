#include <CpuUtils.h>

using namespace winCpu;

void winCpu::testPlusCpu(const double* a, const double* b, double* res, size_t size)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = a[i] + b[i];
	}
}
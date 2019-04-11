#include <CpuUtils.h>
#include <CpuTimer.h>
#include <CpuFocalProcessing.h>
#include <ctime>
#include <chrono>

using namespace winCpu;

double winCpu::testPlusCpu(const double* a, const double* b, double* res, size_t size)
{
	double time;

	CPU_TIMER_START;
	for (int i = 0; i < size; i++)
	{
		res[i] = a[i] + b[i];
	}
	CPU_TIMER_STOP(time);

	return time;
}

double winCpu::performFocalOpCpu(pixel* input, int height, int width, std::vector<double> matrix)
{
	return winCpu::doFocalOpCpu(input, height, width, matrix);
}

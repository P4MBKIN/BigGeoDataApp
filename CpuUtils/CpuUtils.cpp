#include <CpuUtils.h>
#include <CpuTimer.h>
#include <CpuFocalProcessing.h>
#include <CpuProjectionProcessing.h>
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

double winCpu::performTransformUtmToWgsCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, bool southhemi)
{
	return doTransformUtmToWgsCoordsCpu(xOrigin, yOrigin, xPixelSize, yPixelSize,
		height, width, zone, southhemi);
}

double winCpu::performTransformWgsToUtmCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone)
{
	return doTransformWgsToUtmCoordsCpu(xOrigin, yOrigin, xPixelSize, yPixelSize,
		height, width, zone);
}

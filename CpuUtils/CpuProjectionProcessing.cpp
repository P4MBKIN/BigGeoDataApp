#include <CpuProjectionProcessing.h>
#include <CpuTimer.h>
#include <CpuUtmWgsTransform.h>
#include <ctime>
#include <chrono>

double winCpu::doTransformUtmToWgsCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone, bool southhemi)
{
	double time;
	CPU_TIMER_START;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			double x = xOrigin + xPixelSize * w;
			double y = yOrigin + yPixelSize * h;
			double newLon = 0.0;
			double newLat = 0.0;
			UtmXYToLatLonCpu(x, y, zone, southhemi, newLon, newLat);
		}
	}
	CPU_TIMER_STOP(time);
	return time;
}

double winCpu::doTransformWgsToUtmCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
	int height, int width, int zone)
{
	double time;
	CPU_TIMER_START;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			double lon = xOrigin + xPixelSize * w;
			double lat = yOrigin + yPixelSize * h;
			double newX = 0.0;
			double newY = 0.0;
			LatLonToUtmXYCpu(lon, lat, zone, newX, newY);
		}
	}
	CPU_TIMER_STOP(time);
	return time;
}

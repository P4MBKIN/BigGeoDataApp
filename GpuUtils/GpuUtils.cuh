#pragma once

typedef unsigned short int pixel;

namespace winGpu
{
	double testPlusGpu(const double* a, const double* b, double* res, size_t size);
	double performFocalOpGpu(pixel* input, int height, int width, pixel* output, int type);
	double performTransformUtmToWgsCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone, bool southhemi, double* lon, double* lat);
}
#pragma once
#include <vector>

typedef short int pixel;

namespace winCpu
{
	double testPlusCpu(const double* a, const double* b, double* res, size_t size);
	double performFocalOpCpu(pixel* input, int height, int width, std::vector<double> matrix);
	double performTransformUtmToWgsCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone, bool southhemi);
	double performTransformWgsToUtmCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone);
}

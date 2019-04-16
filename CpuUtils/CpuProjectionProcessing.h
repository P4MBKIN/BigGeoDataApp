#pragma once
#include <stdio.h>

typedef short int pixel;

namespace winCpu
{
	double doTransformUtmToWgsCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone, bool southhemi);
	double doTransformWgsToUtmCoordsCpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone);
}

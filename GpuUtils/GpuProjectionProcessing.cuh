#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned short int pixel;

namespace winGpu
{
	double doTransformUtmToWgsCoordsGpu(double xOrigin, double yOrigin, double xPixelSize, double yPixelSize,
		int height, int width, int zone, bool southhemi, double* lon, double* lat);
}
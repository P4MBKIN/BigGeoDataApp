#pragma once

typedef short int pixel;

namespace winCpu
{
	double testPlusCpu(const double* a, const double* b, double* res, size_t size);
	double performFocalOpCpu(pixel* input, int height, int width, int type);
}

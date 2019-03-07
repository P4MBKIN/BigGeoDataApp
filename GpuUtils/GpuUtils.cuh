#pragma once

typedef unsigned short int pixel;

namespace winGpu
{
	void testPlusGpu(const double* a, const double* b, double* res, size_t size);
	void performFocalOpGpu(pixel* input, int height, int width, pixel* output, int type);
}
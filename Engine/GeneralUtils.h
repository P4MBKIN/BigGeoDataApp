#pragma once

typedef unsigned short int pixel;

namespace win
{
	void replaceNewCoord(double newXOrigin, double newYOrigin, double newXPixelSize, double newYPixelSize,
		int newHeight, int newWidth, double* x, double* y, int oldHeight, int oldWidth, pixel* input, pixel* output);
}
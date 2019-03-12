#include "GeneralUtils.h"

void win::replaceNewCoord(double newXOrigin, double newYOrigin, double newXPixelSize, double newYPixelSize,
	int newHeight, int newWidth, double* x, double* y, int oldHeight, int oldWidth, pixel* input, pixel* output)
{
	for (int i = 0; i < newHeight; i++)
	{
		for (int j = 0; j < newWidth; j++)
		{
			output[i * newWidth + j] = 55537;
		}
	}
	for (int i = 0; i < oldHeight; i++)
	{
		for (int j = 0; j < oldWidth; j++)
		{
			int h = (y[i * oldWidth + j] - newYOrigin) / newYPixelSize;
			int w = (x[i * oldWidth + j] - newXOrigin) / newXPixelSize;
			output[h * newWidth + w] = input[i * oldWidth + j];
		}
	}
}

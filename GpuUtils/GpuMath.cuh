#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define PI 3.14159265358979
#define EPSILON 1e-15

namespace mathGpu
{
	__device__ double floorGpu(double x)
	{
		double result = (int)x;
		return result;
	}

	__device__ double absGpu(double x)
	{
		bool isNegative = x < 0;
		return isNegative ? -x : x;
	}

	__device__ double newtonGpu(double x, double a)
	{
		if ((absGpu(a*a - x) <= EPSILON))
		{
			return a;
		}
		else
		{
			return newtonGpu(x, (a + x / a) / 2);
		}
	}

	__device__ double sqrtGpu(double x)
	{
		double sqrt = newtonGpu(x, x / 2);
		return sqrt;
	}

	__device__ unsigned long factGpu(int x)
	{
		if (x == 0)
		{
			return 1;
		}
		else
		{
			return x * factGpu(x - 1);
		}
	}

	__device__ double powGpu(double x, int p)
	{
		double result = 1.0;
		for (int i = 1; i <= p; i++)
		{
			result *= x;
		}
		return result;
	}

	__device__ double simplifyRadGpu(double x)
	{
		bool isNegative = x < 0;
		double result = isNegative ? -x : x;
		if (result < 2 * PI)
		{
			return isNegative ? -x : x;
		}
		else
		{
			return simplifyRadGpu(result - 2 * PI);
		}
	}

	__device__ double cosGpu(double x, int n)
	{
		if (n == 0)
		{
			return 1.0;
		}
		double result = (n % 2 == 0 ? 1 : -1);
		result *= powGpu(simplifyRadGpu(x), 2 * n);
		result /= factGpu(2 * n);
		result += cosGpu(x, n - 1);
		return result;
	}

	__device__ double sinGpu(double x, int n)
	{
		if (n == 0)
		{
			return 0.0;
		}
		double result = (n % 2 == 0 ? -1 : 1);
		result *= powGpu(simplifyRadGpu(x), 2 * n - 1);
		result /= factGpu(2 * n - 1);
		result += sinGpu(x, n - 1);
		return result;
	}

	__device__ double tanGpu(double x, int n)
	{
		double sinValue = sinGpu(x, n);
		double cosValue = cosGpu(x, n);
		return sinValue / cosValue;
	}
}
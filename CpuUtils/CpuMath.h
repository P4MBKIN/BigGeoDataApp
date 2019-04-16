#pragma once
#include <stdio.h>

#define PI 3.14159265358979
#define EPSILON 1e-15

namespace mathCpu
{
	inline double floorCpu(double x)
	{
		double result = (int)x;
		return result;
	}

	inline double absCpu(double x)
	{
		bool isNegative = x < 0;
		return isNegative ? -x : x;
	}

	inline double newtonCpu(double x, double a)
	{
		if ((absCpu(a*a - x) <= EPSILON))
		{
			return a;
		}
		else
		{
			return newtonCpu(x, (a + x / a) / 2);
		}
	}

	inline double sqrtCpu(double x)
	{
		double sqrt = newtonCpu(x, x / 2);
		return sqrt;
	}

	inline unsigned long factCpu(int x)
	{
		if (x == 0)
		{
			return 1;
		}
		else
		{
			return x * factCpu(x - 1);
		}
	}

	inline double powCpu(double x, int p)
	{
		double result = 1.0;
		for (int i = 1; i <= p; i++)
		{
			result *= x;
		}
		return result;
	}

	inline double simplifyRadCpu(double x)
	{
		bool isNegative = x < 0;
		double result = isNegative ? -x : x;
		if (result < 2 * PI)
		{
			return isNegative ? -x : x;
		}
		else
		{
			return simplifyRadCpu(result - 2 * PI);
		}
	}

	inline double cosCpu(double x, int n)
	{
		if (n == 0)
		{
			return 1.0;
		}
		double result = (n % 2 == 0 ? 1 : -1);
		result *= powCpu(simplifyRadCpu(x), 2 * n);
		result /= factCpu(2 * n);
		result += cosCpu(x, n - 1);
		return result;
	}

	inline double sinCpu(double x, int n)
	{
		if (n == 0)
		{
			return 0.0;
		}
		double result = (n % 2 == 0 ? -1 : 1);
		result *= powCpu(simplifyRadCpu(x), 2 * n - 1);
		result /= factCpu(2 * n - 1);
		result += sinCpu(x, n - 1);
		return result;
	}

	inline double tanCpu(double x, int n)
	{
		double sinValue = sinCpu(x, n);
		double cosValue = cosCpu(x, n);
		return sinValue / cosValue;
	}
}
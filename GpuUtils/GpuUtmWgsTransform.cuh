#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GpuMath.cuh"
#include <stdio.h>

#define pi 3.14159265358979
#define sm_a 6378137.0
#define sm_b 6356752.314
#define sm_EccSquared 6.69437999013e-03
#define UTMScaleFactor 0.9996

namespace winGpu
{
	__device__ double DegToRadGpu(double deg)
	{
		return (deg / 180.0 * pi);
	}

	__device__ double RadToDegGpu(double rad)
	{
		return (rad / pi * 180.0);
	}

	__device__ double ArcLengthOfMeridianGpu(double phi)
	{
		double alpha, beta, gamma, delta, epsilon, n;
		double result;

		/* Precalculate n */
		n = (sm_a - sm_b) / (sm_a + sm_b);

		/* Precalculate alpha */
		alpha = ((sm_a + sm_b) / 2.0)
			* (1.0 + (mathGpu::powGpu(n, 2) / 4.0) + (mathGpu::powGpu(n, 4) / 64.0));

		/* Precalculate beta */
		beta = (-3.0 * n / 2.0) + (9.0 * mathGpu::powGpu(n, 3) / 16.0)
			+ (-3.0 * mathGpu::powGpu(n, 5) / 32.0);

		/* Precalculate gamma */
		gamma = (15.0 * mathGpu::powGpu(n, 2) / 16.0)
			+ (-15.0 * mathGpu::powGpu(n, 4) / 32.0);

		/* Precalculate delta */
		delta = (-35.0 * mathGpu::powGpu(n, 3) / 48.0)
			+ (105.0 * mathGpu::powGpu(n, 5) / 256.0);

		/* Precalculate epsilon */
		epsilon = (315.0 * mathGpu::powGpu(n, 4) / 512.0);

		/* Now calculate the sum of the series and return */
		result = alpha
			* (phi + (beta * mathGpu::sinGpu(2.0 * phi, 8))
				+ (gamma * mathGpu::sinGpu(4.0 * phi, 8))
				+ (delta * mathGpu::sinGpu(6.0 * phi, 8))
				+ (epsilon * mathGpu::sinGpu(8.0 * phi, 8)));

		return result;
	}

	__device__ double UTMCentralMeridianGpu(int zone)
	{
		double cmeridian;
		cmeridian = DegToRadGpu(-183.0 + (zone * 6.0));

		return cmeridian;
	}

	__device__ double FootpointLatitudeGpu(double y)
	{
		double y_, alpha_, beta_, gamma_, delta_, epsilon_, n;
		double result;

		/* Precalculate n (Eq. 10.18) */
		n = (sm_a - sm_b) / (sm_a + sm_b);

		/* Precalculate alpha_ (Eq. 10.22) */
		/* (Same as alpha in Eq. 10.17) */
		alpha_ = ((sm_a + sm_b) / 2.0)
			* (1 + (mathGpu::powGpu(n, 2) / 4) + (mathGpu::powGpu(n, 4) / 64));

		/* Precalculate y_ (Eq. 10.23) */
		y_ = y / alpha_;

		/* Precalculate beta_ (Eq. 10.22) */
		beta_ = (3.0 * n / 2.0) + (-27.0 * mathGpu::powGpu(n, 3) / 32.0)
			+ (269.0 * mathGpu::powGpu(n, 5) / 512.0);

		/* Precalculate gamma_ (Eq. 10.22) */
		gamma_ = (21.0 * mathGpu::powGpu(n, 2) / 16.0)
			+ (-55.0 * mathGpu::powGpu(n, 4) / 32.0);

		/* Precalculate delta_ (Eq. 10.22) */
		delta_ = (151.0 * mathGpu::powGpu(n, 3) / 96.0)
			+ (-417.0 * mathGpu::powGpu(n, 5) / 128.0);

		/* Precalculate epsilon_ (Eq. 10.22) */
		epsilon_ = (1097.0 * mathGpu::powGpu(n, 4) / 512.0);

		/* Now calculate the sum of the series (Eq. 10.21) */
		result = y_ + (beta_ * mathGpu::sinGpu(2.0 * y_, 8))
			+ (gamma_ * mathGpu::sinGpu(4.0 * y_, 8))
			+ (delta_ * mathGpu::sinGpu(6.0 * y_, 8))
			+ (epsilon_ * mathGpu::sinGpu(8.0 * y_, 8));

		return result;
	}

	__device__ void MapLatLonToXYGpu(double phi, double lambda, double lambda0, double& x, double& y)
	{
		double N, nu2, ep2, t, t2, l;
		double l3coef, l4coef, l5coef, l6coef, l7coef, l8coef;
		//FLOAT tmp; // Unused

		/* Precalculate ep2 */
		ep2 = (mathGpu::powGpu(sm_a, 2) - mathGpu::powGpu(sm_b, 2)) / mathGpu::powGpu(sm_b, 2);

		/* Precalculate nu2 */
		nu2 = ep2 * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 2);

		/* Precalculate N */
		N = mathGpu::powGpu(sm_a, 2) / (sm_b * mathGpu::sqrtGpu(1 + nu2));

		/* Precalculate t */
		t = mathGpu::tanGpu(phi, 8);
		t2 = t * t;
		//tmp = (t2 * t2 * t2) - POW(t, 6.0); // Unused

		/* Precalculate l */
		l = lambda - lambda0;

		/* Precalculate coefficients for l**n in the equations below
		   so a normal human being can read the expressions for easting
		   and northing
		   -- l**1 and l**2 have coefficients of 1.0 */
		l3coef = 1.0 - t2 + nu2;

		l4coef = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2);

		l5coef = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2
			- 58.0 * t2 * nu2;

		l6coef = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2
			- 330.0 * t2 * nu2;

		l7coef = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2);

		l8coef = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2);

		/* Calculate easting (x) */
		x = N * mathGpu::cosGpu(phi, 8) * l
			+ (N / 6.0 * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 3) * l3coef * mathGpu::powGpu(l, 3))
			+ (N / 120.0 * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 5) * l5coef * mathGpu::powGpu(l, 5))
			+ (N / 5040.0 * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 7) * l7coef * mathGpu::powGpu(l, 7));

		/* Calculate northing (y) */
		y = ArcLengthOfMeridianGpu(phi)
			+ (t / 2.0 * N * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 2) * mathGpu::powGpu(l, 2))
			+ (t / 24.0 * N * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 4) * l4coef * mathGpu::powGpu(l, 4))
			+ (t / 720.0 * N * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 6) * l6coef * mathGpu::powGpu(l, 6))
			+ (t / 40320.0 * N * mathGpu::powGpu(mathGpu::cosGpu(phi, 8), 8) * l8coef * mathGpu::powGpu(l, 8));

		return;
	}

	__device__ void MapXYToLatLonGpu(double x, double y, double lambda0, double& phi, double& lambda)
	{
		double phif, Nf, Nfpow, nuf2, ep2, tf, tf2, tf4, cf;
		double x1frac, x2frac, x3frac, x4frac, x5frac, x6frac, x7frac, x8frac;
		double x2poly, x3poly, x4poly, x5poly, x6poly, x7poly, x8poly;

		/* Get the value of phif, the footpoint latitude. */
		phif = FootpointLatitudeGpu(y);

		/* Precalculate ep2 */
		ep2 = (mathGpu::powGpu(sm_a, 2) - mathGpu::powGpu(sm_b, 2))
			/ mathGpu::powGpu(sm_b, 2);

		/* Precalculate cos (phif) */
		cf = mathGpu::cosGpu(phif, 8);

		/* Precalculate nuf2 */
		nuf2 = ep2 * mathGpu::powGpu(cf, 2);

		/* Precalculate Nf and initialize Nfpow */
		Nf = mathGpu::powGpu(sm_a, 2) / (sm_b * mathGpu::sqrtGpu(1 + nuf2));
		Nfpow = Nf;

		/* Precalculate tf */
		tf = mathGpu::tanGpu(phif, 8);
		tf2 = tf * tf;
		tf4 = tf2 * tf2;

		/* Precalculate fractional coefficients for x**n in the equations
		   below to simplify the expressions for latitude and longitude. */
		x1frac = 1.0 / (Nfpow * cf);

		Nfpow *= Nf;   /* now equals Nf**2) */
		x2frac = tf / (2.0 * Nfpow);

		Nfpow *= Nf;   /* now equals Nf**3) */
		x3frac = 1.0 / (6.0 * Nfpow * cf);

		Nfpow *= Nf;   /* now equals Nf**4) */
		x4frac = tf / (24.0 * Nfpow);

		Nfpow *= Nf;   /* now equals Nf**5) */
		x5frac = 1.0 / (120.0 * Nfpow * cf);

		Nfpow *= Nf;   /* now equals Nf**6) */
		x6frac = tf / (720.0 * Nfpow);

		Nfpow *= Nf;   /* now equals Nf**7) */
		x7frac = 1.0 / (5040.0 * Nfpow * cf);

		Nfpow *= Nf;   /* now equals Nf**8) */
		x8frac = tf / (40320.0 * Nfpow);

		/* Precalculate polynomial coefficients for x**n.
		   -- x**1 does not have a polynomial coefficient. */
		x2poly = -1.0 - nuf2;

		x3poly = -1.0 - 2 * tf2 - nuf2;

		x4poly = 5.0 + 3.0 * tf2 + 6.0 * nuf2 - 6.0 * tf2 * nuf2
			- 3.0 * (nuf2 *nuf2) - 9.0 * tf2 * (nuf2 * nuf2);

		x5poly = 5.0 + 28.0 * tf2 + 24.0 * tf4 + 6.0 * nuf2 + 8.0 * tf2 * nuf2;

		x6poly = -61.0 - 90.0 * tf2 - 45.0 * tf4 - 107.0 * nuf2
			+ 162.0 * tf2 * nuf2;

		x7poly = -61.0 - 662.0 * tf2 - 1320.0 * tf4 - 720.0 * (tf4 * tf2);

		x8poly = 1385.0 + 3633.0 * tf2 + 4095.0 * tf4 + 1575 * (tf4 * tf2);

		/* Calculate latitude */
		phi = phif + x2frac * x2poly * (x * x)
			+ x4frac * x4poly * mathGpu::powGpu(x, 4)
			+ x6frac * x6poly * mathGpu::powGpu(x, 6)
			+ x8frac * x8poly * mathGpu::powGpu(x, 8);

		/* Calculate longitude */
		lambda = lambda0 + x1frac * x
			+ x3frac * x3poly * mathGpu::powGpu(x, 3)
			+ x5frac * x5poly * mathGpu::powGpu(x, 5)
			+ x7frac * x7poly * mathGpu::powGpu(x, 7);

		return;
	}

	__device__ int LatLonToUtmXYGpu(double lon, double lat, int zone, double& x, double& y)
	{
		if ((zone < 1) || (zone > 60))
			zone = mathGpu::floorGpu((lon + 180.0) / 6) + 1;

		MapLatLonToXYGpu(DegToRadGpu(lat), DegToRadGpu(lon), UTMCentralMeridianGpu(zone), x, y);

		/* Adjust easting and northing for UTM system. */
		x = x * UTMScaleFactor + 500000.0;
		y = y * UTMScaleFactor;
		if (y < 0.0)
			y = y + 10000000.0;

		return zone;
	}

	__device__ void UtmXYToLatLonGpu(double x, double y, int zone, bool southhemi, double& lon, double& lat)
	{
		double cmeridian;

		x -= 500000.0;
		x /= UTMScaleFactor;

		/* If in southern hemisphere, adjust y accordingly. */
		if (southhemi)
			y -= 10000000.0;

		y /= UTMScaleFactor;

		cmeridian = UTMCentralMeridianGpu(zone);
		MapXYToLatLonGpu(x, y, cmeridian, lat, lon);

		lon = RadToDegGpu(lon);
		lat = RadToDegGpu(lat);

		return;
	}
}
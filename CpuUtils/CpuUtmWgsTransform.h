#pragma once
#include <CpuMath.h>
#include <stdio.h>

#define pi 3.14159265358979
#define sm_a 6378137.0
#define sm_b 6356752.314
#define sm_EccSquared 6.69437999013e-03
#define UTMScaleFactor 0.9996

namespace winCpu
{
	inline double DegToRadCpu(double deg)
	{
		return (deg / 180.0 * pi);
	}

	inline double RadToDegCpu(double rad)
	{
		return (rad / pi * 180.0);
	}

	inline double ArcLengthOfMeridianCpu(double phi)
	{
		double alpha, beta, gamma, delta, epsilon, n;
		double result;

		n = (sm_a - sm_b) / (sm_a + sm_b);

		alpha = ((sm_a + sm_b) / 2.0)
			* (1.0 + (mathCpu::powCpu(n, 2) / 4.0) + (mathCpu::powCpu(n, 4) / 64.0));

		beta = (-3.0 * n / 2.0) + (9.0 * mathCpu::powCpu(n, 3) / 16.0)
			+ (-3.0 * mathCpu::powCpu(n, 5) / 32.0);

		gamma = (15.0 * mathCpu::powCpu(n, 2) / 16.0)
			+ (-15.0 * mathCpu::powCpu(n, 4) / 32.0);

		delta = (-35.0 * mathCpu::powCpu(n, 3) / 48.0)
			+ (105.0 * mathCpu::powCpu(n, 5) / 256.0);

		epsilon = (315.0 * mathCpu::powCpu(n, 4) / 512.0);

		result = alpha
			* (phi + (beta * mathCpu::sinCpu(2.0 * phi, 8))
				+ (gamma * mathCpu::sinCpu(4.0 * phi, 8))
				+ (delta * mathCpu::sinCpu(6.0 * phi, 8))
				+ (epsilon * mathCpu::sinCpu(8.0 * phi, 8)));

		return result;
	}

	inline double UTMCentralMeridianCpu(int zone)
	{
		double cmeridian;
		cmeridian = DegToRadCpu(-183.0 + (zone * 6.0));

		return cmeridian;
	}

	inline double FootpointLatitudeCpu(double y)
	{
		double y_, alpha_, beta_, gamma_, delta_, epsilon_, n;
		double result;

		n = (sm_a - sm_b) / (sm_a + sm_b);

		alpha_ = ((sm_a + sm_b) / 2.0)
			* (1 + (mathCpu::powCpu(n, 2) / 4) + (mathCpu::powCpu(n, 4) / 64));

		y_ = y / alpha_;

		beta_ = (3.0 * n / 2.0) + (-27.0 * mathCpu::powCpu(n, 3) / 32.0)
			+ (269.0 * mathCpu::powCpu(n, 5) / 512.0);

		gamma_ = (21.0 * mathCpu::powCpu(n, 2) / 16.0)
			+ (-55.0 * mathCpu::powCpu(n, 4) / 32.0);

		delta_ = (151.0 * mathCpu::powCpu(n, 3) / 96.0)
			+ (-417.0 * mathCpu::powCpu(n, 5) / 128.0);

		epsilon_ = (1097.0 * mathCpu::powCpu(n, 4) / 512.0);

		result = y_ + (beta_ * mathCpu::sinCpu(2.0 * y_, 8))
			+ (gamma_ * mathCpu::sinCpu(4.0 * y_, 8))
			+ (delta_ * mathCpu::sinCpu(6.0 * y_, 8))
			+ (epsilon_ * mathCpu::sinCpu(8.0 * y_, 8));

		return result;
	}

	inline void MapLatLonToXYCpu(double phi, double lambda, double lambda0, double& x, double& y)
	{
		double N, nu2, ep2, t, t2, l;
		double l3coef, l4coef, l5coef, l6coef, l7coef, l8coef;

		ep2 = (mathCpu::powCpu(sm_a, 2) - mathCpu::powCpu(sm_b, 2)) / mathCpu::powCpu(sm_b, 2);

		nu2 = ep2 * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 2);

		N = mathCpu::powCpu(sm_a, 2) / (sm_b * mathCpu::sqrtCpu(1 + nu2));

		t = mathCpu::tanCpu(phi, 8);
		t2 = t * t;

		l = lambda - lambda0;

		l3coef = 1.0 - t2 + nu2;

		l4coef = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2);

		l5coef = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2
			- 58.0 * t2 * nu2;

		l6coef = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2
			- 330.0 * t2 * nu2;

		l7coef = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2);

		l8coef = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2);

		x = N * mathCpu::cosCpu(phi, 8) * l
			+ (N / 6.0 * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 3) * l3coef * mathCpu::powCpu(l, 3))
			+ (N / 120.0 * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 5) * l5coef * mathCpu::powCpu(l, 5))
			+ (N / 5040.0 * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 7) * l7coef * mathCpu::powCpu(l, 7));

		y = ArcLengthOfMeridianCpu(phi)
			+ (t / 2.0 * N * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 2) * mathCpu::powCpu(l, 2))
			+ (t / 24.0 * N * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 4) * l4coef * mathCpu::powCpu(l, 4))
			+ (t / 720.0 * N * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 6) * l6coef * mathCpu::powCpu(l, 6))
			+ (t / 40320.0 * N * mathCpu::powCpu(mathCpu::cosCpu(phi, 8), 8) * l8coef * mathCpu::powCpu(l, 8));

		return;
	}

	inline void MapXYToLatLonCpu(double x, double y, double lambda0, double& phi, double& lambda)
	{
		double phif, Nf, Nfpow, nuf2, ep2, tf, tf2, tf4, cf;
		double x1frac, x2frac, x3frac, x4frac, x5frac, x6frac, x7frac, x8frac;
		double x2poly, x3poly, x4poly, x5poly, x6poly, x7poly, x8poly;

		phif = FootpointLatitudeCpu(y);

		ep2 = (mathCpu::powCpu(sm_a, 2) - mathCpu::powCpu(sm_b, 2))
			/ mathCpu::powCpu(sm_b, 2);

		cf = mathCpu::cosCpu(phif, 8);

		nuf2 = ep2 * mathCpu::powCpu(cf, 2);

		Nf = mathCpu::powCpu(sm_a, 2) / (sm_b * mathCpu::sqrtCpu(1 + nuf2));
		Nfpow = Nf;

		tf = mathCpu::tanCpu(phif, 8);
		tf2 = tf * tf;
		tf4 = tf2 * tf2;

		x1frac = 1.0 / (Nfpow * cf);

		Nfpow *= Nf;
		x2frac = tf / (2.0 * Nfpow);

		Nfpow *= Nf;
		x3frac = 1.0 / (6.0 * Nfpow * cf);

		Nfpow *= Nf;
		x4frac = tf / (24.0 * Nfpow);

		Nfpow *= Nf;
		x5frac = 1.0 / (120.0 * Nfpow * cf);

		Nfpow *= Nf;
		x6frac = tf / (720.0 * Nfpow);

		Nfpow *= Nf;
		x7frac = 1.0 / (5040.0 * Nfpow * cf);

		Nfpow *= Nf;
		x8frac = tf / (40320.0 * Nfpow);

		x2poly = -1.0 - nuf2;

		x3poly = -1.0 - 2 * tf2 - nuf2;

		x4poly = 5.0 + 3.0 * tf2 + 6.0 * nuf2 - 6.0 * tf2 * nuf2
			- 3.0 * (nuf2 *nuf2) - 9.0 * tf2 * (nuf2 * nuf2);

		x5poly = 5.0 + 28.0 * tf2 + 24.0 * tf4 + 6.0 * nuf2 + 8.0 * tf2 * nuf2;

		x6poly = -61.0 - 90.0 * tf2 - 45.0 * tf4 - 107.0 * nuf2
			+ 162.0 * tf2 * nuf2;

		x7poly = -61.0 - 662.0 * tf2 - 1320.0 * tf4 - 720.0 * (tf4 * tf2);

		x8poly = 1385.0 + 3633.0 * tf2 + 4095.0 * tf4 + 1575 * (tf4 * tf2);

		phi = phif + x2frac * x2poly * (x * x)
			+ x4frac * x4poly * mathCpu::powCpu(x, 4)
			+ x6frac * x6poly * mathCpu::powCpu(x, 6)
			+ x8frac * x8poly * mathCpu::powCpu(x, 8);

		lambda = lambda0 + x1frac * x
			+ x3frac * x3poly * mathCpu::powCpu(x, 3)
			+ x5frac * x5poly * mathCpu::powCpu(x, 5)
			+ x7frac * x7poly * mathCpu::powCpu(x, 7);

		return;
	}

	inline int LatLonToUtmXYCpu(double lon, double lat, int zone, double& x, double& y)
	{
		if ((zone < 1) || (zone > 60))
			zone = mathCpu::floorCpu((lon + 180.0) / 6) + 1;

		MapLatLonToXYCpu(DegToRadCpu(lat), DegToRadCpu(lon), UTMCentralMeridianCpu(zone), x, y);

		x = x * UTMScaleFactor + 500000.0;
		y = y * UTMScaleFactor;
		if (y < 0.0)
			y = y + 10000000.0;

		return zone;
	}

	inline void UtmXYToLatLonCpu(double x, double y, int zone, bool southhemi, double& lon, double& lat)
	{
		double cmeridian;

		x -= 500000.0;
		x /= UTMScaleFactor;

		if (southhemi)
			y -= 10000000.0;

		y /= UTMScaleFactor;

		cmeridian = UTMCentralMeridianCpu(zone);
		MapXYToLatLonCpu(x, y, cmeridian, lat, lon);

		lon = RadToDegCpu(lon);
		lat = RadToDegCpu(lat);

		return;
	}
}
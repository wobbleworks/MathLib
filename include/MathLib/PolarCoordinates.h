///----------------------------------------
///       @file PolarCoordinates.h
///    @ingroup MathLib
///      @brief Conversions between right-handed polar coordinates and @ref Math::Vector directions.
///    @details Convention: longitude about +Y, latitude from the XZ plane, with
///             @c x = -sin(lon)·cos(lat), @c y = sin(lat), @c z = cos(lon)·cos(lat).
///     @author Created by John Stephen.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include "MathLib/Scalar.h"
#include "MathLib/Vector.h"
#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

/// @brief Builds a direction (optionally scaled by @p distance) from polar angles in radians.
template <class T>
[[nodiscard]] Vector<T, 3> vectorFromPolarRad(T lonRad, T latRad, T distance = T(1)) noexcept {
	T cosLatitude = std::cos(latRad);
	return Vector<T, 3>(-std::sin(lonRad) * cosLatitude * distance, std::sin(latRad) * distance, std::cos(lonRad) * cosLatitude * distance);
}

/// @brief Extracts polar angles (radians) from a direction; longitude is wrapped to [0, 2π). Output scalar may differ from the vector's.
template <class T, class TOut>
void polarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad) noexcept {
	T latitude = std::atan2(vector.y, std::sqrt(vector.x * vector.x + vector.z * vector.z));
	T longitude = std::atan2(-vector.x, vector.z);
	if (longitude < 0) {
		longitude += T(2) * std::numbers::pi_v<T>;
	}
	lonRad = static_cast<TOut>(longitude);
	latRad = static_cast<TOut>(latitude);
}

/// @brief Extracts polar angles (radians) and distance (magnitude) from a vector.
template <class T, class TOut>
void polarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad, TOut &distance) noexcept {
	polarFromVectorRad(vector, lonRad, latRad);
	distance = static_cast<TOut>(length(vector));
}

/// @brief Extracts polar angles (radians) from a direction WITHOUT wrapping longitude: longitude is the raw
///        @c atan2 result in (−π, π]. Use @ref polarFromVectorRad when
///        you want longitude normalized to [0, 2π). Output scalar may differ from the vector's.
template <class T, class TOut>
void signedPolarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad) noexcept {
	lonRad = static_cast<TOut>(std::atan2(-vector.x, vector.z));
	latRad = static_cast<TOut>(std::atan2(vector.y, std::sqrt(vector.x * vector.x + vector.z * vector.z)));
}

/// @brief Extracts unwrapped polar angles (radians) and distance (magnitude) from a vector.
template <class T, class TOut>
void signedPolarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad, TOut &distance) noexcept {
	signedPolarFromVectorRad(vector, lonRad, latRad);
	distance = static_cast<TOut>(length(vector));
}

/// @brief Builds a direction from CLOCKWISE polar angles (e.g. azimuth) in radians; +sin(lon) on X.
template <class T>
[[nodiscard]] Vector<T, 3> vectorFromClockwisePolarRad(T lonRad, T latRad, T distance = T(1)) noexcept {
	T cosLatitude = std::cos(latRad);
	return Vector<T, 3>(std::sin(lonRad) * cosLatitude * distance, std::sin(latRad) * distance, std::cos(lonRad) * cosLatitude * distance);
}

/// @brief Extracts CLOCKWISE polar angles (radians) from a direction; longitude wrapped to [0, 2π). Output scalar may differ from the vector's.
template <class T, class TOut>
void clockwisePolarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad) noexcept {
	T latitude = std::atan2(vector.y, std::sqrt(vector.x * vector.x + vector.z * vector.z));
	T longitude = std::atan2(vector.x, vector.z);
	if (longitude < 0) {
		longitude += T(2) * std::numbers::pi_v<T>;
	}
	lonRad = static_cast<TOut>(longitude);
	latRad = static_cast<TOut>(latitude);
}

/// @brief Extracts CLOCKWISE polar angles (radians) and distance (magnitude) from a vector.
template <class T, class TOut>
void clockwisePolarFromVectorRad(const Vector<T, 3> &vector, TOut &lonRad, TOut &latRad, TOut &distance) noexcept {
	T dist = length(vector);
	if (dist == 0) {
		lonRad = latRad = distance = TOut(0);
		return;
	}
	T latitude = std::asin(std::min(std::max(vector.y / dist, T(-1)), T(1)));
	T longitude = std::atan2(vector.x, vector.z);
	if (longitude < 0) {
		longitude += T(2) * std::numbers::pi_v<T>;
	}
	lonRad = static_cast<TOut>(longitude);
	latRad = static_cast<TOut>(latitude);
	distance = static_cast<TOut>(dist);
}

/// @brief Great-circle angular distance (radians) between two (longitude, latitude) points, via the haversine formula.
///        Each vector holds longitude in @c x and latitude in @c y.
template <class T>
[[nodiscard]] T angularDistanceRad(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept {
	const T deltaLon = b.x - a.x;
	const T deltaLat = b.y - a.y;
	const T h = haversine(deltaLat) + std::cos(a.y) * std::cos(b.y) * haversine(deltaLon);
	const T clampedRootH = std::clamp(std::sqrt(h), T(-1), T(1));
	return T(2) * std::asin(clampedRootH);
}

///----------------------------------------
/// @brief Verifies the polar/direction conversions through round-trips and known great-circle distances.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void polarCoordinatesSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-6; };
	const double halfPi = std::numbers::pi / 2;
	
	// vectorFromPolarRad and polarFromVectorRad round-trip for directions clear of the wrap seam.
	{
		const double longitudes[] = {0.5, 2.0, 4.0};
		const double latitudes[] = {0.3, -0.4, 0.2};
		for (int index = 0; index < 3; ++index) {
			const Double3 direction = vectorFromPolarRad<double>(longitudes[index], latitudes[index]);
			check(near(length(direction), 1.0), "unit direction from polar angles");
			double longitude = 0, latitude = 0;
			polarFromVectorRad(direction, longitude, latitude);
			check(near(longitude, longitudes[index]) && near(latitude, latitudes[index]), "polar round-trip");
		}
	}
	
	// signedPolarFromVectorRad leaves longitude unwrapped in (-pi, pi] where polarFromVectorRad wraps to [0, 2pi).
	{
		const Double3 direction = vectorFromPolarRad<double>(5.0, 0.2);
		double wrappedLongitude = 0, wrappedLatitude = 0;
		polarFromVectorRad(direction, wrappedLongitude, wrappedLatitude);
		double signedLongitude = 0, signedLatitude = 0;
		signedPolarFromVectorRad(direction, signedLongitude, signedLatitude);
		check(near(wrappedLongitude, 5.0), "wrapped longitude stays in [0, 2pi)");
		check(near(signedLongitude, 5.0 - 2 * std::numbers::pi), "signed longitude stays unwrapped");
		check(near(signedLatitude, wrappedLatitude), "latitude agrees between variants");
	}
	
	// The clockwise variant flips the X-component handedness and round-trips through its own extractor.
	{
		const double longitude = 1.1, latitude = 0.25;
		const Double3 clockwise = vectorFromClockwisePolarRad<double>(longitude, latitude);
		const Double3 counterClockwise = vectorFromPolarRad<double>(longitude, latitude);
		check(near(clockwise.x, -counterClockwise.x) && near(clockwise.z, counterClockwise.z), "clockwise flips longitude handedness");
		double outLongitude = 0, outLatitude = 0;
		clockwisePolarFromVectorRad(clockwise, outLongitude, outLatitude);
		check(near(outLongitude, longitude) && near(outLatitude, latitude), "clockwise polar round-trip");
	}
	
	// angularDistanceRad (haversine) against known great-circle references; each point is (longitude, latitude).
	{
		check(near(angularDistanceRad(Double2(0.5, 0.3), Double2(0.5, 0.3)), 0.0), "zero distance to self");
		check(near(angularDistanceRad(Double2(0.0, 0.0), Double2(halfPi, 0.0)), halfPi), "quarter turn along the equator");
		check(near(angularDistanceRad(Double2(0.5, 0.0), Double2(1.3, halfPi)), halfPi), "equator to pole is a quarter turn");
		check(near(angularDistanceRad(Double2(0.0, halfPi), Double2(0.0, -halfPi)), std::numbers::pi), "pole to pole is a half turn");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

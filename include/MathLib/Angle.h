///----------------------------------------
/// @file Angle.h
/// @ingroup MathLib
/// @brief Sexagesimal decomposition of angles (radians → hours/degrees, minutes, seconds).
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

/// @brief Hours and decimal minutes. Components are non-negative magnitudes; @c negative carries the sign.
struct HM {
	bool negative = false;
	int hours = 0;
	double minutes = 0;
};

/// @brief Hours, minutes, and decimal seconds. Components are non-negative magnitudes; @c negative carries the sign.
struct HMS {
	bool negative = false;
	int hours = 0;
	int minutes = 0;
	double seconds = 0;
};

/// @brief Degrees and decimal arcminutes. Components are non-negative magnitudes; @c negative carries the sign.
struct DM {
	bool negative = false;
	int degrees = 0;
	double minutes = 0;
};

/// @brief Degrees, arcminutes, and decimal arcseconds. Components are non-negative magnitudes; @c negative carries the sign.
struct DMS {
	bool negative = false;
	int degrees = 0;
	int minutes = 0;
	double seconds = 0;
};

/// @brief Converts radians to hours and decimal minutes.
/// @return The decomposition, or @c std::nullopt if @p radians is NaN.
[[nodiscard]] inline std::optional<HM> radiansToHM(double radians) {
	if (std::isnan(radians)) {
		return std::nullopt;
	}
	
	auto absRadians = std::fabs(radians);
	auto hoursf = absRadians * 12 / std::numbers::pi;
	
	HM result;
	result.negative = radians < 0;
	result.hours = (int)std::floor(hoursf);
	result.minutes = (hoursf - result.hours) * 60;
	
	return result;
}

/// @brief Converts radians to hours, minutes, and decimal seconds.
/// @return The decomposition, or @c std::nullopt if @p radians is NaN.
[[nodiscard]] inline std::optional<HMS> radiansToHMS(double radians) {
	if (std::isnan(radians)) {
		return std::nullopt;
	}
	
	auto absRadians = std::fabs(radians);
	auto hoursf = absRadians * 12 / std::numbers::pi;
	
	HMS result;
	result.negative = radians < 0;
	result.hours = (int)std::floor(hoursf);
	auto minutesf = (hoursf - result.hours) * 60;
	result.minutes = (int)std::floor(minutesf);
	result.seconds = (minutesf - result.minutes) * 60;
	
	return result;
}

/// @brief Converts radians to degrees and decimal arcminutes.
/// @return The decomposition, or @c std::nullopt if @p radians is NaN.
[[nodiscard]] inline std::optional<DM> radiansToDM(double radians) {
	if (std::isnan(radians)) {
		return std::nullopt;
	}
	
	auto absRadians = std::fabs(radians);
	auto degreesf = absRadians * 180 / std::numbers::pi;
	
	DM result;
	result.negative = radians < 0;
	result.degrees = (int)std::floor(degreesf);
	result.minutes = (degreesf - result.degrees) * 60;
	
	return result;
}

/// @brief Converts radians to degrees, arcminutes, and decimal arcseconds.
/// @return The decomposition, or @c std::nullopt if @p radians is NaN.
[[nodiscard]] inline std::optional<DMS> radiansToDMS(double radians) {
	if (std::isnan(radians)) {
		return std::nullopt;
	}
	
	auto absRadians = std::fabs(radians);
	auto degreesf = absRadians * 180 / std::numbers::pi;
	
	DMS result;
	result.negative = radians < 0;
	result.degrees = (int)std::floor(degreesf);
	auto minutesf = (degreesf - result.degrees) * 60;
	result.minutes = (int)std::floor(minutesf);
	result.seconds = (minutesf - result.minutes) * 60;
	
	return result;
}

///----------------------------------------
/// @brief Verifies the sexagesimal decompositions at exact hour/degree boundaries and the sign/NaN paths.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void angleSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// A half-turn is exactly twelve hours in both hour decompositions.
	{
		const auto hm = radiansToHM(std::numbers::pi);
		check(hm.has_value() && !hm->negative && hm->hours == 12 && near(hm->minutes, 0.0), "pi radians is twelve hours");
		
		const auto hms = radiansToHMS(std::numbers::pi);
		check(hms.has_value() && !hms->negative && hms->hours == 12 && hms->minutes == 0 && near(hms->seconds, 0.0),
			"pi radians is twelve hours zero minutes zero seconds");
	}
	
	// A quarter-turn is six hours and ninety degrees.
	{
		const auto hm = radiansToHM(std::numbers::pi / 2);
		check(hm.has_value() && hm->hours == 6 && near(hm->minutes, 0.0), "half pi radians is six hours");
		
		const auto dm = radiansToDM(std::numbers::pi / 2);
		check(dm.has_value() && !dm->negative && dm->degrees == 90 && near(dm->minutes, 0.0), "half pi radians is ninety degrees");
		
		const auto dms = radiansToDMS(std::numbers::pi / 2);
		check(dms.has_value() && dms->degrees == 90 && dms->minutes == 0 && near(dms->seconds, 0.0),
			"half pi radians is ninety degrees zero arcminutes zero arcseconds");
	}
	
	// A negative angle sets the sign flag while the magnitude components stay positive.
	{
		const auto hm = radiansToHM(-std::numbers::pi);
		check(hm.has_value() && hm->negative && hm->hours == 12 && near(hm->minutes, 0.0), "negative half-turn carries the sign");
		
		const auto dms = radiansToDMS(-std::numbers::pi / 2);
		check(dms.has_value() && dms->negative && dms->degrees == 90, "negative quarter-turn carries the sign");
	}
	
	// A NaN input yields no decomposition.
	{
		const auto nan = std::numeric_limits<double>::quiet_NaN();
		check(!radiansToHM(nan).has_value(), "NaN has no hour-minute decomposition");
		check(!radiansToHMS(nan).has_value(), "NaN has no hour-minute-second decomposition");
		check(!radiansToDM(nan).has_value(), "NaN has no degree-minute decomposition");
		check(!radiansToDMS(nan).has_value(), "NaN has no degree-minute-second decomposition");
	}
}
	
} // namespace

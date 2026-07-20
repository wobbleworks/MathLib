///----------------------------------------
/// @file RescalingMap.h
/// @ingroup MathLib
/// @brief Maps between cumulative-distance values and normalized [0,1] factors.
/// @author Created by John Stephen on 7/11/14.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class RescalingMap
/// @brief Maps between cumulative-distance values and normalized [0,1] factors.
/// @details Stores sorted cumulative distances as a private vector.
///----------------------------------------
class RescalingMap final {
///----------------------------------------
	std::vector<double> _values;
	
public:
	RescalingMap() = default;
	
	/// @brief Reserve storage for the given number of entries.
	void reserve(size_t n) { _values.reserve(n); }
	
	/// @brief Append a cumulative distance entry.
	void push_back(double value) { _values.push_back(value); }
	
	///----------------------------------------
	/// @brief Map a cumulative distance value to a normalized [0,1] factor.
	/// @details Performs a binary search through the sorted distance entries and
	///          linearly interpolates between the two surrounding samples.
	/// @param value Cumulative distance to look up.
	/// @return Interpolated factor in [0,1].
	///----------------------------------------
	
	[[nodiscard]] inline double mapValueToFactor(double value) const {
		// Find the first element greater than value
		auto i1 = std::ranges::upper_bound(_values, value);
		
		// Return 0 if before the beginning
		if (i1 == _values.begin()) {
			return 0.0;
		}
		
		// Return 1 if after the end
		if (i1 == _values.end()) {
			return 1.0;
		}
		
		// Calculate the index and interpolation factor
		const auto index1 = static_cast<int>(std::distance(_values.begin(), i1));
		const auto index0 = index1 - 1;
		const auto value0 = i1[-1];
		const auto value1 = i1[0];
		const auto dv = value1 - value0;
		
		// Guard against coincident samples
		if (dv == 0.0) {
			return static_cast<double>(index0) / static_cast<double>(_values.size() - 1);
		}
		
		const auto valueFactor = (value - value0) / dv;
		
		// Add the factor and scale down
		return (index0 + valueFactor) / static_cast<double>(_values.size() - 1);
	}
	
	///----------------------------------------
	/// @brief Map a normalized [0,1] factor back to a cumulative distance value.
	/// @details Expands the factor into the sample index space, then linearly
	///          interpolates between the two surrounding distance entries.
	/// @param factor Normalized factor in [0,1].
	/// @return Interpolated cumulative distance.
	///----------------------------------------
	
	[[nodiscard]] inline double mapFactorToValue(double factor) const {
		// Guard against empty or single-entry maps
		if (_values.size() <= 1) {
			return _values.empty() ? 0.0 : _values.front();
		}
		
		const auto expandedFactor = factor * static_cast<double>(_values.size() - 1);
		const auto index0 = static_cast<int>(std::floor(expandedFactor));
		const auto sampleFactor = expandedFactor - index0;
		const auto index1 = index0 + 1;
		
		// Return the last entry if past the end
		if (index1 >= static_cast<int>(_values.size())) {
			return _values.back();
		}
		
		const auto value0 = _values[index0];
		const auto value1 = _values[index1];
		
		// Linear interpolation
		return std::lerp(value0, value1, sampleFactor);
	}
};

///----------------------------------------
/// @brief Verifies the value/factor mapping endpoints, midpoint interpolation, and round-trip.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void rescalingMapSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// Five evenly spaced cumulative distances span the normalized range.
	RescalingMap map;
	map.push_back(0.0);
	map.push_back(1.0);
	map.push_back(2.0);
	map.push_back(3.0);
	map.push_back(4.0);
	
	// The endpoints map to the extreme factors, and the middle value to one half.
	{
		check(near(map.mapValueToFactor(0.0), 0.0), "first value maps to zero factor");
		check(near(map.mapValueToFactor(4.0), 1.0), "last value maps to unit factor");
		check(near(map.mapValueToFactor(2.0), 0.5), "middle value maps to half factor");
	}
	
	// The factor extremes recover the first and last values, and the midpoint interpolates.
	{
		check(near(map.mapFactorToValue(0.0), 0.0), "zero factor maps to first value");
		check(near(map.mapFactorToValue(1.0), 4.0), "unit factor maps to last value");
		check(near(map.mapFactorToValue(0.5), 2.0), "half factor maps to middle value");
	}
	
	// Mapping a factor to a value and back returns the original factor.
	{
		check(near(map.mapValueToFactor(map.mapFactorToValue(0.375)), 0.375), "factor round-trips through value");
		check(near(map.mapValueToFactor(map.mapFactorToValue(0.25)), 0.25), "quarter factor round-trips");
	}
}
	
} // namespace Math

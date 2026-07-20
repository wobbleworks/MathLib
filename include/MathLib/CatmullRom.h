///----------------------------------------
/// @file CatmullRom.h
/// @ingroup MathLib
/// @brief Single-segment Catmull-Rom spline with uniform, centripetal, and non-uniform modes.
/// @author Created by John Stephen on 7/11/14.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/CubicPolynomial.h"
#include "MathLib/RescalingMap.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class CatmullRom
/// @brief Single-segment Catmull-Rom spline with uniform, centripetal, and non-uniform modes.
/// @details Backed by a @ref CubicPolynomial. The value type may be a scalar (e.g. @c double)
///          or any @ref Math::Vector; chord length and dot product resolve accordingly.
/// @tparam TType Value type (e.g. @c double, @c Math::Double3).
///----------------------------------------
template <class TType>
class CatmullRom final {
///----------------------------------------
	/// Repeated control points would divide by a zero knot interval; clamp to this floor instead.
	static constexpr double minimumKnotInterval = 1.e-100;
	
	CubicPolynomial<TType> _polynomial;
	
	/// @brief Distance between two samples: chord length for vectors, absolute difference for scalars.
	[[nodiscard]] static double chordLength(const TType& first, const TType& second) noexcept {
		if constexpr (std::is_arithmetic_v<TType>) {
			return std::abs(static_cast<double>(second - first));
		} else {
			return Math::distance(first, second);
		}
	}
	
	/// @brief Dot product for the projection solver: component dot for vectors, plain product for scalars.
	[[nodiscard]] static double projectionDot(const TType& first, const TType& second) noexcept {
		if constexpr (std::is_arithmetic_v<TType>) {
			return static_cast<double>(first) * static_cast<double>(second);
		} else {
			return Math::dot(first, second);
		}
	}
	
public:
	CatmullRom() = default;
	
	///----------------------------------------
	/// MARK: Construction
	///----------------------------------------
	
	/// @brief Build a spline using uniform (standard) Catmull-Rom parameterization.
	/// @param p0 First control point.
	/// @param p1 Second control point.
	/// @param p2 Third control point.
	/// @param p3 Fourth control point.
	/// @return The constructed spline.
	[[nodiscard]] static CatmullRom uniform(TType p0, TType p1, TType p2, TType p3) {
		CatmullRom spline;
		spline._polynomial = CubicPolynomial<TType>(p0, p1, p2, p3);
		return spline;
	}
	
	/// @brief Build a spline using centripetal Catmull-Rom parameterization.
	/// @details Knot intervals are the square root of the chord lengths, which avoids the
	///          cusps and self-intersections uniform parameterization can produce.
	/// @param p0 First control point.
	/// @param p1 Second control point.
	/// @param p2 Third control point.
	/// @param p3 Fourth control point.
	/// @return The constructed spline.
	[[nodiscard]] static CatmullRom centripetal(TType p0, TType p1, TType p2, TType p3) {
		const auto knot0 = std::sqrt(chordLength(p0, p1));
		const auto knot1 = std::sqrt(chordLength(p1, p2));
		const auto knot2 = std::sqrt(chordLength(p2, p3));
		return nonuniform(p0, p1, p2, p3, knot0, knot1, knot2);
	}
	
	/// @brief Build a spline using non-uniform Catmull-Rom with explicit knot intervals.
	/// @param p0  First control point.
	/// @param p1  Second control point.
	/// @param p2  Third control point.
	/// @param p3  Fourth control point.
	/// @param dt0 Knot interval between p0 and p1.
	/// @param dt1 Knot interval between p1 and p2.
	/// @param dt2 Knot interval between p2 and p3.
	/// @return The constructed spline.
	[[nodiscard]] static CatmullRom nonuniform(TType p0, TType p1, TType p2, TType p3, double dt0, double dt1, double dt2) {
		// Collapse repeated points onto a valid neighboring interval.
		if (dt1 < minimumKnotInterval) {
			dt1 = minimumKnotInterval;
		}
		if (dt0 < minimumKnotInterval) {
			dt0 = dt1;
		}
		if (dt2 < minimumKnotInterval) {
			dt2 = dt1;
		}
		
		CatmullRom spline;
		spline._polynomial = CubicPolynomial<TType>::nonuniformCatmullRom(p0, p1, p2, p3, dt0, dt1, dt2);
		return spline;
	}
	
	///----------------------------------------
	/// MARK: Evaluation
	///----------------------------------------
	
	/// @brief Evaluate the spline at parameter t in [0,1].
	/// @param t Parameter in [0,1].
	/// @return Evaluated value.
	[[nodiscard]] TType eval(double t) const noexcept { return _polynomial.eval(t); }
	
	/// @brief Evaluate the first derivative at parameter t.
	/// @param t Parameter in [0,1].
	/// @return First derivative value.
	[[nodiscard]] TType evalDerivative(double t) const noexcept { return _polynomial.evalDerivative(t); }
	
	/// @brief Evaluate the second derivative at parameter t.
	/// @param t Parameter in [0,1].
	/// @return Second derivative value.
	[[nodiscard]] TType evalSecondDerivative(double t) const noexcept { return _polynomial.evalSecondDerivative(t); }
	
	/// @brief Find the parameter t whose evaluated value is closest to the target.
	/// @details Scalar curves bisect the parameter interval; vector curves refine the closest
	///          initial sample with Newton's method on the projection condition.
	/// @param target   Target value to search for.
	/// @param maxError Convergence threshold.
	/// @return Parameter t in [0,1] corresponding to the target value.
	[[nodiscard]] double timeForValue(TType target, double maxError) const;
	
	///----------------------------------------
	/// MARK: Linearization
	///----------------------------------------
	
	/// @brief Build a linearization map by sampling the curve and accumulating arc-length.
	/// @param numSamples          Number of samples to take along the curve.
	/// @param[out] totalDistance  Total arc-length measured.
	/// @return A RescalingMap containing cumulative distances at each sample.
	[[nodiscard]] RescalingMap calcLinearizationMap(int numSamples, double& totalDistance) const;
	
	///----------------------------------------
	/// MARK: Sampling
	///----------------------------------------
	
	/// @brief Sample the curve uniformly in parameter space.
	/// @param numSamples Number of evenly spaced samples.
	/// @return Vector of evaluated samples.
	[[nodiscard]] std::vector<TType> getSamples(int numSamples) const;
};

///----------------------------------------
/// MARK: CatmullRom Implementation
///----------------------------------------

template <class TType>
double CatmullRom<TType>::timeForValue(TType target, double maxError) const {
	if constexpr (std::is_arithmetic_v<TType>) {
		// Monotone scalar curve: bisect the parameter interval.
		auto lowerTime = 0.0;
		auto upperTime = 1.0;
		
		for (auto iteration = 0; iteration < 32; ++iteration) {
			const auto midpointTime = 0.5 * (lowerTime + upperTime);
			const auto value = eval(midpointTime);
			
			// Done when within the error
			const auto residual = value - target;
			if (std::abs(residual) < maxError) {
				return midpointTime;
			}
			
			// Adjust the boundary
			if (residual < 0) {
				lowerTime = midpointTime;
			} else {
				upperTime = midpointTime;
			}
		}
		
		return 0.5 * (lowerTime + upperTime);
	} else {
		// Newton's method on the projection condition g(t) = dot(C(t) - target, C'(t)) = 0,
		// which locates the parameter where the curve is closest to the target point.
		
		// Initial guess: sample evenly and pick the closest point
		constexpr auto numInitialSamples = 8;
		auto bestTime = 0.0;
		auto bestDistance = chordLength(eval(0.0), target);
		
		for (auto sampleIndex = 1; sampleIndex <= numInitialSamples; ++sampleIndex) {
			const auto sampleTime = static_cast<double>(sampleIndex) / numInitialSamples;
			const auto distanceToTarget = chordLength(eval(sampleTime), target);
			if (distanceToTarget < bestDistance) {
				bestDistance = distanceToTarget;
				bestTime = sampleTime;
			}
		}
		
		// Early exit if already within tolerance
		if (bestDistance < maxError) {
			return bestTime;
		}
		
		// Refine with Newton iteration
		auto time = bestTime;
		for (auto iteration = 0; iteration < 32; ++iteration) {
			const auto offset = eval(time) - target;
			const auto firstDerivative = evalDerivative(time);
			const auto secondDerivative = evalSecondDerivative(time);
			
			const auto projection = projectionDot(offset, firstDerivative);
			const auto projectionSlope = projectionDot(firstDerivative, firstDerivative) + projectionDot(offset, secondDerivative);
			
			// Guard against a degenerate derivative
			if (std::abs(projectionSlope) < 1e-20) {
				break;
			}
			
			time = std::clamp(time - projection / projectionSlope, 0.0, 1.0);
			
			if (chordLength(eval(time), target) < maxError) {
				return time;
			}
		}
		
		return time;
	}
}

///----------------------------------------
/// MARK: Linearization
///----------------------------------------

template <class TType>
RescalingMap CatmullRom<TType>::calcLinearizationMap(int numSamples, double& totalDistance) const {
	auto linearizationMap = RescalingMap{};
	
	// Leave if there are no samples
	if (numSamples <= 0) {
		return linearizationMap;
	}
	
	// Build cumulative arc-length by evaluating inline (avoids a temporary vector)
	linearizationMap.reserve(numSamples);
	totalDistance = 0.0;
	
	const auto divisor = (numSamples > 1) ? static_cast<double>(numSamples - 1) : 1.0;
	auto priorSample = eval(0.0);
	linearizationMap.push_back(0.0);
	
	for (auto sampleIndex = 1; sampleIndex < numSamples; ++sampleIndex) {
		const auto t = static_cast<double>(sampleIndex) / divisor;
		const auto sample = eval(t);
		
		const auto distanceFromPrior = chordLength(sample, priorSample);
		assert(!std::isnan(distanceFromPrior));
		totalDistance += distanceFromPrior;
		
		linearizationMap.push_back(totalDistance);
		priorSample = sample;
	}
	
	return linearizationMap;
}

///----------------------------------------
/// MARK: Sampling
///----------------------------------------

template <class TType>
std::vector<TType> CatmullRom<TType>::getSamples(int numSamples) const {
	auto samples = std::vector<TType>{};
	samples.reserve(numSamples);
	
	for (auto sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
		const auto t = (numSamples > 1) ? static_cast<double>(sampleIndex) / (numSamples - 1) : 0.0;
		const auto sample = eval(t);
		if constexpr (std::is_arithmetic_v<TType>) {
			assert(!std::isnan(sample));
		} else {
			assert(!std::isnan(sample.x));
		}
		samples.push_back(sample);
	}
	
	return samples;
}

///----------------------------------------
/// @brief Verifies interpolation, inversion, and linearization against known references.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void catmullRomSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// A uniform scalar segment interpolates its interior control points at the ends of the parameter
	// interval and stays linear through evenly spaced, collinear points.
	{
		const auto segment = CatmullRom<double>::uniform(0.0, 1.0, 2.0, 3.0);
		check(near(segment.eval(0.0), 1.0), "eval(0) is the second control point");
		check(near(segment.eval(1.0), 2.0), "eval(1) is the third control point");
		check(near(segment.eval(0.5), 1.5), "collinear midpoint is the linear interpolant");
	}
	
	// The interior control points are reproduced exactly even when the four points are unevenly spaced.
	{
		const auto segment = CatmullRom<double>::uniform(-2.0, 3.0, 8.0, 5.0);
		check(near(segment.eval(0.0), 3.0), "eval(0) reproduces p1 for uneven spacing");
		check(near(segment.eval(1.0), 8.0), "eval(1) reproduces p2 for uneven spacing");
	}
	
	// timeForValue inverts the collinear curve, whose value is 1 + t, so a known on-curve value recovers
	// its parameter. The reference values sit inside the interval, clear of the endpoints.
	{
		const auto segment = CatmullRom<double>::uniform(0.0, 1.0, 2.0, 3.0);
		check(near(segment.timeForValue(1.25, 1.0e-12), 0.25), "timeForValue recovers t = 1/4");
		check(near(segment.timeForValue(1.75, 1.0e-12), 0.75), "timeForValue recovers t = 3/4");
	}
	
	// Arc-length linearization of the collinear scalar curve measures the straight-line span from p1 to
	// p2, its endpoints map to the ends of that span, and the uniform samples run from p1 to p2.
	{
		const auto segment = CatmullRom<double>::uniform(0.0, 1.0, 2.0, 3.0);
		auto totalDistance = 0.0;
		const auto map = segment.calcLinearizationMap(64, totalDistance);
		check(near(totalDistance, 1.0), "linearized length equals the straight-line span");
		check(near(map.mapFactorToValue(0.0), 0.0), "factor 0 maps to distance 0");
		check(near(map.mapFactorToValue(1.0), 1.0), "factor 1 maps to the total distance");
		const auto samples = segment.getSamples(16);
		check(samples.size() == 16, "sample count matches the request");
		check(near(samples.front(), 1.0) && near(samples.back(), 2.0), "samples run from p1 to p2");
	}
	
	// The vector overload behaves the same: a straight, evenly spaced line stays linear, timeForValue
	// projects a point back to its parameter, and the linearized length is the chord from p1 to p2.
	{
		const auto line = CatmullRom<Double2>::uniform(
			Double2(0.0, 0.0), Double2(1.0, 1.0), Double2(2.0, 2.0), Double2(3.0, 3.0));
		check(approxEqual(line.eval(0.0), Double2(1.0, 1.0), 1.0e-9), "vector eval(0) is p1");
		check(approxEqual(line.eval(1.0), Double2(2.0, 2.0), 1.0e-9), "vector eval(1) is p2");
		check(approxEqual(line.eval(0.5), Double2(1.5, 1.5), 1.0e-9), "vector midpoint is the linear interpolant");
		check(near(line.timeForValue(Double2(1.5, 1.5), 1.0e-9), 0.5), "vector timeForValue recovers t = 1/2");
		auto totalDistance = 0.0;
		const auto map = line.calcLinearizationMap(64, totalDistance);
		check(near(totalDistance, std::sqrt(2.0)), "vector linearized length is the chord from p1 to p2");
		check(near(map.mapFactorToValue(1.0), std::sqrt(2.0)), "factor 1 maps to the chord length");
	}
}
	
} // namespace Math

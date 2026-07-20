///----------------------------------------
/// @file SegmentedCatmullRom.h
/// @ingroup MathLib
/// @brief Piecewise Catmull-Rom spline parameterized by arc-length distance.
/// @author Created by John Stephen on 7/11/14.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/CatmullRom.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class SegmentedCatmullRom
/// @brief Piecewise Catmull-Rom spline parameterized by arc-length distance.
/// @details Each segment stores both a 3D point curve and a scalar distance
///          curve, enabling evaluation at arbitrary distances along the path.
///----------------------------------------
class SegmentedCatmullRom final {
///----------------------------------------
	struct Segment {
		CatmullRom<Double3> _pointsCurve;
		CatmullRom<double> _distanceCurve;
		double _startDistance = 0.0;
		double _distanceFromEnd = 0.0;
		double _spannedDistance = 0.0;
	};
	
	using Segments = std::vector<Segment>;
	
	Segments _segments;
	double _totalDistance = 0.0;
	
	/// @brief Resolve a forward distance to its containing segment and local parameter t.
	/// @param distance         Arc-length distance from the start.
	/// @param maxDistanceError Convergence threshold for the distance solver.
	/// @return Pair of (segment reference, t parameter).
	struct SegmentResult { const Segment& segment; double t; };
	
	[[nodiscard]] inline SegmentResult resolveSegment(double distance, double maxDistanceError) const {
		assert(!_segments.empty());
		assert(distance >= 0.0);
		
		// Search for the first segment *after* the requested distance
		auto i = std::ranges::upper_bound(_segments, distance, std::ranges::less{}, &Segment::_startDistance);
		
		// If it's before the first segment, use t=0 for the first segment
		if (i == _segments.begin()) {
			return {*i, 0.0};
		}
		
		// The desired segment is the segment before the one found
		const auto& segment = *(i - 1);
		
		// Find the t value for the distance curve
		const auto localDistance = distance - segment._startDistance;
		const auto t = segment._distanceCurve.timeForValue(localDistance, maxDistanceError);
		assert(t >= 0.0 && t <= 1.0);
		
		return {segment, t};
	}
	
public:
	///----------------------------------------
	/// @brief Initialize the spline from an ordered set of control points.
	/// @details Builds centripetal Catmull-Rom segments for both the 3D point
	///          curve and the scalar distance curve. Populates cumulative start
	///          and end distances for each segment. Endpoints are duplicated to
	///          form the outer control points of the first and last segments.
	/// @param points At least three control points.
	///----------------------------------------
	
	inline void setPoints(std::span<const Double3> points) {
		// Fail if too few points
		if (points.size() < 3) {
			assert(false);
			return;
		}
		
		// Useful values
		const auto numSegments = points.size() - 1;
		const auto maxIndex = points.size() - 1;
		
		// Reserve space
		_segments.clear();
		_segments.resize(numSegments);
		
		// Initial samples
		auto p0 = points[0];
		auto p1 = points[0];
		auto p2 = points[1];
		auto d0 = 0.0;
		auto d1 = 0.0;
		auto d2 = Math::distance(p1, p2);
		
		// Iterate the segments
		auto distanceFromStart = 0.0;
		for (auto segmentIndex = size_t{0}; segmentIndex < numSegments; ++segmentIndex) {
			auto& segment = _segments[segmentIndex];
			
			// Retrieve the fourth sample
			const auto index3 = std::min(segmentIndex + 2, maxIndex);
			const auto p3 = points[index3];
			const auto d3 = d2 + Math::distance(p2, p3);
			
			// Initialize the segment
			segment._startDistance = distanceFromStart;
			segment._pointsCurve = CatmullRom<Double3>::centripetal(p0, p1, p2, p3);
			segment._distanceCurve = CatmullRom<double>::centripetal(d0 - segment._startDistance, 0.0, d2 - segment._startDistance, d3 - segment._startDistance);
			
			// Measure the length using the distance polynomial
			segment._spannedDistance = segment._distanceCurve.eval(1.0);
			distanceFromStart += segment._spannedDistance;
			
			// Shift the samples
			p0 = p1;
			d0 = d1;
			p1 = p2;
			d1 = d2;
			p2 = p3;
			d2 = d3;
		}
		
		// Populate the distances from the end
		auto distanceFromEnd = 0.0;
		for (auto segmentIndex = numSegments; segmentIndex > 0; --segmentIndex) {
			auto& segment = _segments[segmentIndex - 1];
			segment._distanceFromEnd = distanceFromEnd;
			distanceFromEnd += segment._spannedDistance;
		}
		
		// Update total distance
		_totalDistance = distanceFromStart;
	}
	
	/// @brief Return the total arc-length of the spline.
	/// @returns Total distance in meters.
	[[nodiscard]] double totalDistance() const noexcept { return _totalDistance; }
	
	///----------------------------------------
	/// @brief Evaluate the spline at a given forward distance.
	/// @param distance         Arc-length distance from the start.
	/// @param maxDistanceError Convergence threshold for the distance solver.
	/// @return 3D position on the curve.
	///----------------------------------------
	
	[[nodiscard]] inline Double3 eval(double distance, double maxDistanceError) const {
		const auto [segment, t] = resolveSegment(distance, maxDistanceError);
		return segment._pointsCurve.eval(t);
	}
	
	///----------------------------------------
	/// @brief Evaluate the spline at a given distance from the end.
	/// @details Performs a lower-bound search over segments by distance-from-end,
	///          then inverts the distance curve within the found segment.
	/// @param distanceFromEnd  Arc-length distance measured from the end.
	/// @param maxDistanceError Convergence threshold for the distance solver.
	/// @return 3D position on the curve.
	///----------------------------------------
	
	[[nodiscard]] inline Double3 evalAtDistanceFromEnd(double distanceFromEnd, double maxDistanceError) const {
		// Validity check
		assert(!_segments.empty());
		
		// Search for the first segment *after* the requested distance from the end
		auto i = std::ranges::lower_bound(_segments, distanceFromEnd, std::ranges::greater{}, &Segment::_distanceFromEnd);
		
		// If it's after the last segment, return time 1 for the last segment
		if (i == _segments.end()) {
			return _segments.back()._pointsCurve.eval(1.0);
		}
		
		// Find the t value for the distance curve
		const auto& segment = *i;
		const auto localDistanceFromEnd = distanceFromEnd - segment._distanceFromEnd;
		const auto localDistance = std::clamp(segment._spannedDistance - localDistanceFromEnd, 0.0, segment._spannedDistance);
		const auto t = segment._distanceCurve.timeForValue(localDistance, maxDistanceError);
		assert(t >= 0.0 && t <= 1.0);
		
		// Evaluate the point curve
		return segment._pointsCurve.eval(t);
	}
	
	///----------------------------------------
	/// @brief Evaluate the derivative (tangent) at a given forward distance.
	/// @param distance         Arc-length distance from the start.
	/// @param maxDistanceError Convergence threshold for the distance solver.
	/// @return 3D tangent vector at the evaluated point.
	///----------------------------------------
	
	[[nodiscard]] inline Double3 evalDerivative(double distance, double maxDistanceError) const {
		const auto [segment, t] = resolveSegment(distance, maxDistanceError);
		return segment._pointsCurve.evalDerivative(t);
	}
	
	/// @brief Return the start distance of a particular segment.
	/// @param segmentIndex Zero-based segment index.
	/// @return Start distance of the segment.
	[[nodiscard]] double segmentStartDistance(size_t segmentIndex) const noexcept { return _segments[segmentIndex]._startDistance; }
};

///----------------------------------------
/// @brief Verifies arc-length parameterization and endpoint interpolation against known references.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void segmentedCatmullRomSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-6; };
	
	// Four evenly spaced, collinear control points a unit apart along the x axis. Cumulative arc-length
	// at each point equals its index, so the spline spans three units in total.
	const std::array<Double3, 4> points = {
		Double3(0.0, 0.0, 0.0),
		Double3(1.0, 0.0, 0.0),
		Double3(2.0, 0.0, 0.0),
		Double3(3.0, 0.0, 0.0),
	};
	
	SegmentedCatmullRom spline;
	spline.setPoints(points);
	
	constexpr auto maxError = 1.0e-9;
	constexpr auto tolerance = 1.0e-6;
	
	// The total arc-length is the sum of the straight segment lengths.
	{
		check(near(spline.totalDistance(), 3.0), "total distance sums the segment lengths");
	}
	
	// Distance 0 lands on the first control point and the total distance lands on the last.
	{
		check(approxEqual(spline.eval(0.0, maxError), points.front(), tolerance), "distance 0 is the first point");
		check(approxEqual(spline.eval(spline.totalDistance(), maxError), points.back(), tolerance),
			"total distance is the last point");
	}
	
	// The curve passes through each input control point at its cumulative distance.
	{
		check(approxEqual(spline.eval(1.0, maxError), points[1], tolerance), "control point 1 lies at distance 1");
		check(approxEqual(spline.eval(2.0, maxError), points[2], tolerance), "control point 2 lies at distance 2");
	}
	
	// A forward distance and its mirror measured from the end evaluate to the same on-line point.
	{
		const auto forward = spline.eval(1.5, maxError);
		const auto fromEnd = spline.evalAtDistanceFromEnd(spline.totalDistance() - 1.5, maxError);
		check(approxEqual(forward, fromEnd, tolerance), "forward and from-end evaluations agree");
		check(approxEqual(forward, Double3(1.5, 0.0, 0.0), tolerance), "the midspan point lies on the line");
	}
}
	
} // namespace Math

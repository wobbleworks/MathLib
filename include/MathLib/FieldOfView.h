///----------------------------------------
///       @file FieldOfView.h
///    @ingroup MathLib
///      @brief The logarithmic mapping between a field of view and a normalized zoom factor.
///    @details A zoom control wants a linear travel: equal drags should feel like equal zooms whether
///             the view spans a hundred degrees or a fraction of an arcsecond. The angle itself is a
///             terrible parameter for that — the useful range spans nine orders of magnitude and is
///             compressed almost entirely into the last few percent of any linear slider.
///
///             @ref Math::FieldOfViewCurve maps the field of view onto @c [0,1] so that the *tangent*
///             of the half-angle grows exponentially across the travel. Factor @c 0 is the widest
///             field of view and factor @c 1 the narrowest, so the factor reads as "how far zoomed
///             in" — larger is closer.
///
///             The curve is deliberately separate from @ref Math::Camera. Zoom controls, gesture
///             recognizers and level-of-detail thresholds all need this mapping while holding no
///             camera at all, and a stored factor must keep meaning the same field of view across
///             releases.
///       @note The default curve's limits are a persistence contract: a zoom factor written to user
///             preferences is only meaningful against the range it was produced from. Changing
///             @ref Math::minimumFieldOfViewRadians or @ref Math::maximumFieldOfViewRadians silently
///             moves every saved zoom level.
///     @author Created by John Stephen on 7/30/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Numbers.h"
#include "MathLib/SelfTestCheck.h"

#include <algorithm>
#include <cmath>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief The narrowest field of view the default curve reaches, in radians (a ten-thousandth of an arcsecond).
///----------------------------------------

inline constexpr double minimumFieldOfViewRadians = 0.0001 * arcsec2rad;

///----------------------------------------
/// @brief The widest field of view the default curve reaches, in radians.
///----------------------------------------

inline constexpr double maximumFieldOfViewRadians = 120 * deg2rad;

///----------------------------------------
/// @brief A conventional starting field of view, in radians.
///----------------------------------------

inline constexpr double defaultFieldOfViewRadians = 60 * deg2rad;

///----------------------------------------
/// @class FieldOfViewCurve
/// @brief An invertible logarithmic mapping between a field of view and a normalized @c [0,1] zoom factor.
/// @details Factor @c 0 is @ref maximumFieldOfViewRadians and factor @c 1 is
///          @ref minimumFieldOfViewRadians, with the tangent of the half-angle varying exponentially
///          in between. Both directions clamp their input, so the mapping is total: every finite
///          angle yields a factor in @c [0,1] and every factor yields an angle within the range.
/// @note    Not usable with a NaN input; the clamps propagate it rather than rejecting it.
///----------------------------------------

class FieldOfViewCurve final {
///----------------------------------------
public:
	///----------------------------------------
	///   @brief Builds a curve spanning the given field-of-view range.
	///   @param minimumFieldOfView The narrowest field of view, in radians — maps to factor 1.
	///   @param maximumFieldOfView The widest field of view, in radians — maps to factor 0.
	///----------------------------------------
	
	FieldOfViewCurve(double minimumFieldOfView, double maximumFieldOfView) noexcept
		: _minimumFieldOfView(minimumFieldOfView)
		, _maximumFieldOfView(maximumFieldOfView)
		, _minimumHalfAngleTangent(std::tan(minimumFieldOfView / 2))
		, _maximumHalfAngleTangent(std::tan(maximumFieldOfView / 2)) {
		const double tangentRatio = _maximumHalfAngleTangent / _minimumHalfAngleTangent;
		_exponentScale = std::log(tangentRatio + 1);
		_tangentScale = (_maximumHalfAngleTangent - _minimumHalfAngleTangent) / tangentRatio;
	}
	
	/// @brief The narrowest field of view the curve reaches, in radians.
	[[nodiscard]] double minimumFieldOfViewRadians() const noexcept { return _minimumFieldOfView; }
	
	/// @brief The widest field of view the curve reaches, in radians.
	[[nodiscard]] double maximumFieldOfViewRadians() const noexcept { return _maximumFieldOfView; }
	
	///----------------------------------------
	///   @brief The zoom factor for a field of view.
	///   @param fieldOfViewRadians The field of view, in radians; clamped to the curve's range.
	///   @return The factor in @c [0,1] — 0 at the widest field of view, 1 at the narrowest.
	///----------------------------------------
	
	[[nodiscard]] double zoomFactorForFieldOfView(double fieldOfViewRadians) const noexcept {
		const double clampedFieldOfView = std::clamp(fieldOfViewRadians, _minimumFieldOfView, _maximumFieldOfView);
		const double halfAngleTangent = std::tan(clampedFieldOfView / 2);
		const double factor = std::log((halfAngleTangent - _minimumHalfAngleTangent) / _tangentScale + 1) / _exponentScale;
		return 1 - std::clamp(factor, 0.0, 1.0);
	}
	
	///----------------------------------------
	///   @brief The field of view for a zoom factor.
	///   @param zoomFactor The factor; clamped to @c [0,1].
	///   @return The field of view in radians, within the curve's range.
	///----------------------------------------
	
	[[nodiscard]] double fieldOfViewForZoomFactor(double zoomFactor) const noexcept {
		const double clampedFactor = std::clamp(zoomFactor, 0.0, 1.0);
		const double halfAngleTangent = (std::exp((1 - clampedFactor) * _exponentScale) - 1) * _tangentScale + _minimumHalfAngleTangent;
		return 2 * std::atan(halfAngleTangent);
	}
	
private:
	double _minimumFieldOfView;
	double _maximumFieldOfView;
	double _minimumHalfAngleTangent;
	double _maximumHalfAngleTangent;
	double _exponentScale;
	double _tangentScale;
};

///----------------------------------------
/// @brief The shared curve over [@ref minimumFieldOfViewRadians, @ref maximumFieldOfViewRadians].
/// @details A function-local static, so it is initialized on first use rather than during dynamic
///          initialization — no ordering hazard for a caller that maps a field of view from its own
///          static constructor.
///----------------------------------------

[[nodiscard]] inline const FieldOfViewCurve &defaultFieldOfViewCurve() noexcept {
	static const FieldOfViewCurve curve(minimumFieldOfViewRadians, maximumFieldOfViewRadians);
	return curve;
}

///----------------------------------------
/// @brief The zoom factor for a field of view, on the @ref defaultFieldOfViewCurve.
///----------------------------------------

[[nodiscard]] inline double zoomFactorForFieldOfView(double fieldOfViewRadians) noexcept {
	return defaultFieldOfViewCurve().zoomFactorForFieldOfView(fieldOfViewRadians);
}

///----------------------------------------
/// @brief The field of view for a zoom factor, on the @ref defaultFieldOfViewCurve.
///----------------------------------------

[[nodiscard]] inline double fieldOfViewForZoomFactor(double zoomFactor) noexcept {
	return defaultFieldOfViewCurve().fieldOfViewForZoomFactor(zoomFactor);
}

///----------------------------------------
/// @brief Verifies the curve's endpoints, inversion, monotonicity and clamping behavior.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void fieldOfViewSelfTest() {
	using selftest::check;
	const FieldOfViewCurve &curve = defaultFieldOfViewCurve();
	
	// The endpoints anchor the persistence contract: the widest field of view is factor 0 and the
	// narrowest is factor 1. A drifting endpoint silently remaps every stored zoom level.
	{
		check(std::abs(curve.zoomFactorForFieldOfView(maximumFieldOfViewRadians) - 0.0) < 1.0e-12, "the widest field of view is factor 0");
		check(std::abs(curve.zoomFactorForFieldOfView(minimumFieldOfViewRadians) - 1.0) < 1.0e-12, "the narrowest field of view is factor 1");
		check(std::abs(curve.fieldOfViewForZoomFactor(0.0) - maximumFieldOfViewRadians) < 1.0e-12, "factor 0 is the widest field of view");
		check(std::abs(curve.fieldOfViewForZoomFactor(1.0) - minimumFieldOfViewRadians) < 1.0e-15, "factor 1 is the narrowest field of view");
	}
	
	// Round-tripping in both directions. The angle round-trip is checked in relative terms because
	// the range spans nine orders of magnitude and an absolute tolerance would be vacuous at the
	// narrow end and unreachable at the wide one.
	{
		for (double factor = 0.0; factor <= 1.0; factor += 0.05) {
			const double fieldOfView = curve.fieldOfViewForZoomFactor(factor);
			check(std::abs(curve.zoomFactorForFieldOfView(fieldOfView) - factor) < 1.0e-9, "factor to field of view and back");
		}
		for (const double degrees : {0.001, 0.1, 1.0, 10.0, 60.0, 119.0}) {
			const double fieldOfView = degrees * deg2rad;
			const double recovered = curve.fieldOfViewForZoomFactor(curve.zoomFactorForFieldOfView(fieldOfView));
			check(std::abs(recovered - fieldOfView) < 1.0e-9 * fieldOfView, "field of view to factor and back");
		}
	}
	
	// Strictly decreasing: a wider field of view is always a smaller factor. Zooming in must never
	// reverse direction partway along the travel.
	{
		double previousFactor = curve.zoomFactorForFieldOfView(minimumFieldOfViewRadians);
		for (double factor = 0.999; factor > 0.0; factor -= 0.001) {
			const double currentFactor = curve.zoomFactorForFieldOfView(curve.fieldOfViewForZoomFactor(factor));
			check(currentFactor < previousFactor, "the factor decreases as the field of view widens");
			previousFactor = currentFactor;
		}
	}
	
	// Both directions clamp, so out-of-range input saturates rather than escaping the range.
	{
		check(curve.zoomFactorForFieldOfView(1000.0) == 0.0, "an absurdly wide field of view saturates at factor 0");
		check(curve.zoomFactorForFieldOfView(-1.0) == 1.0, "a negative field of view saturates at factor 1");
		check(curve.fieldOfViewForZoomFactor(5.0) == curve.fieldOfViewForZoomFactor(1.0), "a factor above 1 saturates");
		check(curve.fieldOfViewForZoomFactor(-5.0) == curve.fieldOfViewForZoomFactor(0.0), "a factor below 0 saturates");
	}
	
	// A curve over a different range is self-consistent too — the endpoints follow its own limits,
	// not the default ones.
	{
		const FieldOfViewCurve narrowCurve(1 * deg2rad, 45 * deg2rad);
		check(std::abs(narrowCurve.fieldOfViewForZoomFactor(0.0) - 45 * deg2rad) < 1.0e-12, "a custom curve starts at its own maximum");
		check(std::abs(narrowCurve.fieldOfViewForZoomFactor(1.0) - 1 * deg2rad) < 1.0e-12, "a custom curve ends at its own minimum");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

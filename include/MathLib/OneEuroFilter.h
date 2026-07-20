///----------------------------------------
/// @file OneEuroFilter.h
/// @ingroup MathLib
/// @brief Low-latency One-Euro smoothing filter.
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include "MathLib/Numbers.h"
#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class OneEuroFilter
/// @brief Low-latency smoothing filter (One-Euro) for scalar signals.
/// @details Filters a noisy input stream while adapting its cutoff based on
/// the instantaneous derivative. This yields low lag during slow
/// changes and higher responsiveness during fast changes--ideal for
/// interactive controls (e.g., touch/joystick velocity).
///----------------------------------------

struct OneEuroFilter {
	
	/// @brief Numerical floor used when guarding divisions and cutoffs.
	static constexpr float eps = 1e-6f;
	
	/// @brief Base cutoff frequency in Hz (lower = smoother, higher = snappier).
	float minCutoff = 1.0f;
	
	/// @brief Speed sensitivity; increases cutoff with |dx| (0 = fixed cutoff).
	float beta      = 0.2f;
	
	/// @brief Cutoff for the derivative estimator in Hz.
	float dCutoff   = 1.0f;
	
	///----------------------------------------
	/// @brief Filter one sample.
	/// @details Applies derivative-adaptive low-pass filtering to x with step
	/// dt. On the first call (or after reset), seeds the state
	/// with x and returns it unmodified.
	/// @param x  New input sample.
	/// @param dt Time step in seconds since the previous sample.
	/// @return Smoothed output sample.
	///----------------------------------------
	
	[[nodiscard]] float filter(float x, float dt) noexcept;
	
	///----------------------------------------
	/// @brief Reset the filter to an uninitialized state.
	/// @details After a reset, the next call to filter seeds the internal
	/// state with the provided sample and returns it directly.
	///----------------------------------------
	
	void reset() noexcept { hasPrev = false; }
	
private:
	/// @brief Previous filtered value.
	float xPrev = 0.0f;
	
	/// @brief Previous filtered derivative estimate.
	float dxPrev = 0.0f;
	
	/// @brief Whether the filter has been seeded.
	bool hasPrev = false;
	
	///----------------------------------------
	/// @brief Initialize the filter with a starting value.
	/// @param x0 Initial sample used to seed the filter and derivative.
	///----------------------------------------
	
	void reset(float x0) noexcept {
		xPrev = x0;
		dxPrev = 0.0f;
		hasPrev = true;
	}
	
	///----------------------------------------
	/// @brief Compute smoothing factor for a given cutoff and step.
	/// @param fc Cutoff frequency in Hz.
	/// @param dt Time step in seconds.
	/// @return Exponential smoothing factor in (0,1].
	///----------------------------------------
	
	[[nodiscard]] static float alpha(float fc, float dt) noexcept {
		float tau = 1.0f / (2.0f * math::numbers::pi_float * std::max(fc, eps));
		return dt / (dt + tau);
	}
};

///----------------------------------------
/// OneEuroFilter inlines
///----------------------------------------

inline float OneEuroFilter::filter(float x, float dt) noexcept {
	if (!hasPrev) {
		reset(x);
		return x;
	}
	
	// Estimate the derivative
	auto dx = (x - xPrev) / std::max(dt, eps);
	auto ad = alpha(dCutoff, dt);
	auto dxHat = ad * dx + (1.0f - ad) * dxPrev;
	
	// Adapt the cutoff and filter the signal
	auto cutoff = minCutoff + beta * std::abs(dxHat);
	auto a = alpha(cutoff, dt);
	auto xHat = a * x + (1.0f - a) * xPrev;
	
	// Store state
	xPrev = xHat;
	dxPrev = dxHat;
	return xHat;
}

///----------------------------------------
/// @brief Verifies the filter's deterministic seeding, convergence, and cutoff behavior.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void oneEuroFilterSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-6; };
	constexpr float dt = 1.0f / 60.0f;
	
	// The first sample after construction (or reset) seeds the state and is returned unchanged.
	{
		OneEuroFilter filter;
		check(filter.filter(5.0f, dt) == 5.0f, "first sample returns the seed unchanged");
		
		OneEuroFilter reused;
		(void)reused.filter(0.0f, dt);
		reused.reset();
		check(reused.filter(-3.0f, dt) == -3.0f, "reset re-seeds on the next sample");
	}
	
	// A constant input stream stays pinned to that constant: zero derivative, weighted average of equal values.
	{
		OneEuroFilter filter;
		(void)filter.filter(2.0f, dt);
		for (int i = 0; i < 32; ++i) {
			check(near(filter.filter(2.0f, dt), 2.0), "constant input stays constant");
		}
	}
	
	// A step input settles monotonically toward the new value and never overshoots it.
	{
		OneEuroFilter filter;
		(void)filter.filter(0.0f, dt);
		constexpr float slack = 1.0e-6f;
		float previous = 0.0f;
		for (int i = 0; i < 300; ++i) {
			const float current = filter.filter(1.0f, dt);
			check(current >= previous - slack, "step response is monotonically increasing");
			check(current <= 1.0f + slack, "step response never overshoots the target");
			previous = current;
		}
		check(near(previous, 1.0), "step response converges to the target");
	}
	
	// With the speed term disabled the cutoff is fixed, so the first post-seed output equals alpha = dt/(dt+tau).
	{
		OneEuroFilter filter;
		filter.minCutoff = 1.0f;
		filter.beta = 0.0f;
		(void)filter.filter(0.0f, dt);
		const double tau = 1.0 / (2.0 * std::numbers::pi * 1.0);
		const double expectedAlpha = static_cast<double>(dt) / (static_cast<double>(dt) + tau);
		check(near(filter.filter(1.0f, dt), expectedAlpha), "fixed-cutoff alpha matches dt/(dt+tau)");
	}
}
	
} // namespace Math

///----------------------------------------
///      @file Scalar.h
///   @ingroup MathLib
///     @brief Scalar, interpolation, and kernel math utilities.
///   @details Small numeric helpers that complement the standard library:
///            power-of-two queries over @c <bit>, approximate float comparison,
///            interpolation primitives, and Gaussian kernel generation.
///    @author Created by John Stephen on 6/23/26.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <numbers>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "MathLib/Numbers.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// MARK: Integers
///----------------------------------------

///----------------------------------------
///   @brief Determine whether an integer is an exact power of two.
/// @tparam TInteger Any integral type.
///   @param value Value to test.
/// @returns @c true if @p value is a positive power of two.
///----------------------------------------

template <std::integral TInteger>
[[nodiscard]] constexpr bool isPowerOfTwo(TInteger value) noexcept {
	using Unsigned = std::make_unsigned_t<TInteger>;
	return value > TInteger(0) && std::has_single_bit(static_cast<Unsigned>(value));
}

///----------------------------------------
///   @brief Round an integer up to the nearest power of two.
/// @tparam TInteger Any integral type.
///   @param value Value to round; values of one or less yield one.
/// @returns The smallest power of two greater than or equal to @p value.
///----------------------------------------

template <std::integral TInteger>
[[nodiscard]] constexpr TInteger nextPowerOfTwo(TInteger value) noexcept {
	using Unsigned = std::make_unsigned_t<TInteger>;
	if (value <= TInteger(1)) {
		return TInteger(1);
	}
	
	return static_cast<TInteger>(std::bit_ceil(static_cast<Unsigned>(value)));
}

///----------------------------------------
/// MARK: Floats
///----------------------------------------

///----------------------------------------
///   @brief Compare two floating-point values for approximate equality.
/// @tparam TFloat Any floating-point type.
///   @param value1 First value.
///   @param value2 Second value.
///   @param maxDifference Maximum absolute difference treated as equal.
/// @returns @c true if the values differ by less than @p maxDifference.
///----------------------------------------

template <std::floating_point TFloat>
[[nodiscard]] constexpr bool approxEqual(TFloat value1, TFloat value2, TFloat maxDifference = TFloat(0.00001)) noexcept {
	return std::abs(value1 - value2) < maxDifference;
}

///----------------------------------------
///   @brief Compute the haversine, sin²(angle/2), of an angle.
/// @tparam TFloat Any floating-point type.
///   @param angleRad Angle in radians.
/// @returns The haversine of @p angleRad.
///----------------------------------------

template <std::floating_point TFloat>
[[nodiscard]] TFloat haversine(TFloat angleRad) noexcept {
	const auto halfSine = std::sin(TFloat(0.5) * angleRad);
	return halfSine * halfSine;
}

///----------------------------------------
/// MARK: Interpolation
///----------------------------------------

///----------------------------------------
///   @brief Cosine-eased interpolation between two values.
/// @tparam TFloat Any floating-point type.
///   @param value1 Start value, returned at @p factor 0.
///   @param value2 End value, returned at @p factor 1.
///   @param factor Interpolation factor in [0, 1].
/// @returns The eased interpolation between @p value1 and @p value2.
///----------------------------------------

template <std::floating_point TFloat>
[[nodiscard]] TFloat cosInterpolate(TFloat value1, TFloat value2, TFloat factor) noexcept {
	const auto easedFactor = TFloat(0.5) * (TFloat(1) - std::cos(factor * std::numbers::pi_v<TFloat>));
	return value1 + easedFactor * (value2 - value1);
}

///----------------------------------------
///   @brief Bilinearly interpolate the four corner vectors of a grid cell.
/// @details Blends the corners with the horizontal and vertical fractional weights,
///          accelerated by the SIMD vector arithmetic operators.
/// @tparam T Vector scalar type.
/// @tparam N Vector component count.
///   @param topLeft Top-left corner vector.
///   @param topRight Top-right corner vector.
///   @param bottomLeft Bottom-left corner vector.
///   @param bottomRight Bottom-right corner vector.
///   @param rightFactor Horizontal weight toward the right edge, in [0, 1].
///   @param bottomFactor Vertical weight toward the bottom edge, in [0, 1].
/// @returns The bilinearly blended vector.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Vector<T, N> biLinearInterpolate(
	const Vector<T, N>& topLeft,
	const Vector<T, N>& topRight,
	const Vector<T, N>& bottomLeft,
	const Vector<T, N>& bottomRight,
	std::type_identity_t<T> rightFactor,
	std::type_identity_t<T> bottomFactor) noexcept {
	
	// Derive the complementary edge weights
	const auto leftFactor = T(1) - rightFactor;
	const auto topFactor = T(1) - bottomFactor;
	
	// Blend the four corners component-wise
	return leftFactor * topFactor * topLeft
		+ rightFactor * topFactor * topRight
		+ leftFactor * bottomFactor * bottomLeft
		+ rightFactor * bottomFactor * bottomRight;
}

///----------------------------------------
/// MARK: Kernels
///----------------------------------------

///----------------------------------------
///   @brief Generate a normalized 1-D Gaussian blur kernel.
/// @details Produces a symmetric kernel spanning ±⌈radius⌉ samples, normalized to
///          sum to one. Adapted from http://www.jhlabs.com/ip/blurring.html.
///   @param radius Blur radius in samples; the kernel length is 2·⌈radius⌉ + 1.
///   @param kernel Destination buffer receiving the kernel weights.
/// @returns The number of weights written, or @c std::nullopt if @p kernel is too small.
///----------------------------------------

[[nodiscard]] inline std::optional<int> generateGaussianKernel(float radius, std::span<float> kernel) {
	// Determine the kernel extent and total length
	const auto intRadius = static_cast<int>(std::ceil(radius));
	const auto length = intRadius * 2 + 1;
	
	// Fail if the destination cannot hold the kernel
	if (std::cmp_greater(length, kernel.size())) {
		return std::nullopt;
	}
	
	// Precompute the Gaussian coefficients
	const auto sigma = radius / 3.0f;
	const auto twoSigmaSquared = 2.0f * sigma * sigma;
	const auto normalization = std::sqrt(std::numbers::pi_v<float> * 2.0f * sigma);
	const auto radiusSquared = radius * radius;
	
	// Accumulate the unnormalized weights
	auto total = 0.0f;
	auto index = 0;
	for (auto row = -intRadius; row <= intRadius; ++row) {
		const auto distance = static_cast<float>(row * row);
		const auto coefficient = distance > radiusSquared ? 0.0f : std::exp(-distance / twoSigmaSquared) / normalization;
		kernel[index++] = coefficient;
		total += coefficient;
	}
	
	// Normalize so the weights sum to one
	const auto scale = 1.0f / total;
	for (auto i = 0; i < length; ++i) {
		kernel[i] *= scale;
	}
	
	return length;
}

///----------------------------------------
/// @brief Verifies the scalar helpers, interpolators, and Gaussian kernel against known results.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void scalarSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// nextPowerOfTwo rounds up, saturates small inputs to one, and is idempotent on exact powers.
	{
		check(nextPowerOfTwo(1000) == 1024, "next power above 1000");
		check(nextPowerOfTwo(0) == 1, "next power of zero is one");
		check(nextPowerOfTwo(1) == 1, "next power of one is one");
		check(nextPowerOfTwo(16) == 16, "next power of an exact power is itself");
	}
	
	// cosInterpolate returns the endpoints at the extremes and their mean at the midpoint.
	{
		check(near(cosInterpolate(3.0, 7.0, 0.0), 3.0), "cosine interpolation at start");
		check(near(cosInterpolate(3.0, 7.0, 1.0), 7.0), "cosine interpolation at end");
		check(near(cosInterpolate(3.0, 7.0, 0.5), 5.0), "cosine interpolation at midpoint");
	}
	
	// biLinearInterpolate reproduces each corner at its factor pair and averages them at the center.
	{
		const Double2 topLeft(1.0, 2.0);
		const Double2 topRight(3.0, 4.0);
		const Double2 bottomLeft(5.0, 6.0);
		const Double2 bottomRight(7.0, 8.0);
		
		const auto atTopLeft = biLinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, 0.0, 0.0);
		check(near(atTopLeft[0], 1.0) && near(atTopLeft[1], 2.0), "bilinear top-left corner");
		
		const auto atTopRight = biLinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, 1.0, 0.0);
		check(near(atTopRight[0], 3.0) && near(atTopRight[1], 4.0), "bilinear top-right corner");
		
		const auto atBottomLeft = biLinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, 0.0, 1.0);
		check(near(atBottomLeft[0], 5.0) && near(atBottomLeft[1], 6.0), "bilinear bottom-left corner");
		
		const auto atBottomRight = biLinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, 1.0, 1.0);
		check(near(atBottomRight[0], 7.0) && near(atBottomRight[1], 8.0), "bilinear bottom-right corner");
		
		const auto atCenter = biLinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, 0.5, 0.5);
		check(near(atCenter[0], 4.0) && near(atCenter[1], 5.0), "bilinear center average");
	}
	
	// generateGaussianKernel produces a symmetric, normalized kernel of the requested length; too small fails.
	{
		std::array<float, 5> weights{};
		const auto count = generateGaussianKernel(2.0f, weights);
		check(count.has_value() && *count == 5, "gaussian kernel length");
		
		auto total = 0.0;
		for (const auto weight : weights) {
			total += weight;
		}
		check(std::abs(total - 1.0) < 1.0e-6, "gaussian kernel normalized to one");
		check(std::abs(weights[0] - weights[4]) < 1.0e-9, "gaussian kernel symmetric outer weights");
		check(std::abs(weights[1] - weights[3]) < 1.0e-9, "gaussian kernel symmetric inner weights");
		
		std::array<float, 4> tooSmall{};
		check(!generateGaussianKernel(2.0f, tooSmall).has_value(), "gaussian kernel rejects small buffer");
	}
}
	
} // namespace Math

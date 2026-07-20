///----------------------------------------
/// @file Point2d.h
/// @ingroup MathLib
/// @brief A 2D point template with arithmetic and Chebyshev distance.
/// @author Created by John Stephen on 1/1/26.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief A 2D point with x/y coordinates.
/// @tparam TType Arithmetic type for the coordinate components.
///----------------------------------------

///----------------------------------------
template <class TType>
struct point2d {
///----------------------------------------
	TType _x{}, _y{};
	
	constexpr point2d() = default;
	constexpr point2d(TType x, TType y) : _x(x), _y(y) {}
	
	///----------------------------------------
	/// @brief Chebyshev (L-infinity) distance to another point.
	/// @param point The other point.
	/// @return The maximum of the absolute x and y deltas.
	///----------------------------------------
	
	[[nodiscard]] constexpr TType chebyshevDistanceTo(const point2d& point) const { return std::max(std::abs(point._x - _x), std::abs(point._y - _y)); }
	
	constexpr point2d& operator+=(const point2d& point) noexcept { _x += point._x; _y += point._y; return *this; }
	constexpr point2d& operator-=(const point2d& point) noexcept { _x -= point._x; _y -= point._y; return *this; }
	constexpr point2d& operator*=(const point2d& point) noexcept { _x *= point._x; _y *= point._y; return *this; }
	constexpr point2d& operator/=(const point2d& point) noexcept { _x /= point._x; _y /= point._y; return *this; }
	constexpr point2d& operator*=(TType value) noexcept { _x *= value; _y *= value; return *this; }
	constexpr point2d& operator/=(TType value) noexcept { _x /= value; _y /= value; return *this; }
	
	[[nodiscard]] constexpr point2d operator+(const point2d& point) const noexcept { return {_x + point._x, _y + point._y}; }
	[[nodiscard]] constexpr point2d operator-(const point2d& point) const noexcept { return {_x - point._x, _y - point._y}; }
	[[nodiscard]] constexpr point2d operator*(const point2d& point) const noexcept { return {_x * point._x, _y * point._y}; }
	[[nodiscard]] constexpr point2d operator/(const point2d& point) const noexcept { return {_x / point._x, _y / point._y}; }
	[[nodiscard]] constexpr point2d operator*(TType value) const noexcept { return {_x * value, _y * value}; }
	[[nodiscard]] constexpr point2d operator/(TType value) const noexcept { return {_x / value, _y / value}; }
	[[nodiscard]] constexpr point2d operator-() const noexcept { return {-_x, -_y}; }
	
	[[nodiscard]] constexpr bool operator==(const point2d&) const = default;
	[[nodiscard]] constexpr auto operator<=>(const point2d&) const = default;
};

using point2d_int16 = point2d<int16_t>;
using point2d_int32 = point2d<int32_t>;
using point2d_float = point2d<float>;
using point2d_double = point2d<double>;
	
} // namespace Math

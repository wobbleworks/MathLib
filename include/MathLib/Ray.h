///----------------------------------------
///       @file Ray.h
///    @ingroup MathLib
///      @brief A ray: an origin point and a direction vector.
///    @details A small value type pairing an origin with a direction. The direction is
///             not required to be unit length. Built on @ref Math::Vector; carries no
///             intersection behavior of its own.
///     @author Created by John Stephen.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Vector.h"

///----------------------------------------
namespace Math {
///----------------------------------------

/// @brief A ray defined by an origin and a direction.
/// @tparam TFloat Scalar type (float or double).
template <class TFloat>
struct Ray {
	Vector<TFloat, 3> origin;     ///< The ray's starting point.
	Vector<TFloat, 3> direction;  ///< The ray's direction (need not be unit length).
	
	Ray() noexcept = default;
	
	Ray(const Vector<TFloat, 3> &origin, const Vector<TFloat, 3> &direction) noexcept
		: origin(origin), direction(direction) {}
		
	/// @brief The point at parametric distance @p t along the ray: @c origin + @c direction * t.
	[[nodiscard]] Vector<TFloat, 3> pointAt(TFloat t) const noexcept { return origin + direction * t; }
	
	[[nodiscard]] bool operator==(const Ray &) const noexcept = default;
};

using Ray3d = Ray<double>;
using Ray3f = Ray<float>;

///----------------------------------------
} // namespace Math
///----------------------------------------

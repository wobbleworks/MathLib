///----------------------------------------
///       @file Plane.h
///    @ingroup MathLib
///      @brief An infinite plane in Hessian normal form.
///    @details A plane is a unit @ref normal and a signed @ref distance from the origin along that
///             normal, so a point's relationship to it is one dot product:
///             @c signedDistanceToPoint returns how far the point lies along the normal, positive on
///             the side the normal points to and negative behind. Storing the distance rather than a
///             point on the plane is what makes a frustum test cheap — six planes become six dot
///             products with no subtraction.
///       @note The normal is expected to be unit length. @ref signedDistanceToPoint is only a true
///             distance when it is; @ref normalized rescales a plane built from an unnormalized
///             normal. The sign convention — positive in front, along the normal — is what
///             @ref Math::Frustum relies on, where every plane's normal points inward.
///     @author Created by John Stephen (wobbleworks.com)
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

#include <cmath>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class Plane
/// @brief An infinite plane, as a unit normal and a signed distance from the origin.
/// @tparam TFloat Scalar type (float or double).
///----------------------------------------

template <class TFloat>
struct Plane {
	Vector<TFloat, 3> normal;      ///< Unit normal; the plane's front faces this way.
	TFloat distance = TFloat{0};   ///< Signed distance from the origin along @ref normal.
	
	Plane() noexcept = default;
	
	///----------------------------------------
	/// @brief A plane with the given unit normal, @p distance from the origin along it.
	///----------------------------------------
	
	Plane(const Vector<TFloat, 3> &normal, TFloat distance) noexcept
		: normal(normal), distance(distance) {}
		
	///----------------------------------------
	///   @brief The plane with unit normal @p normal passing through @p point.
	/// @details The distance is @c -dot(normal, point), the offset that puts @p point exactly on the
	///          plane. This is the form a frustum uses: each plane is an orientation plus a point it
	///          pivots about, usually the viewer.
	///----------------------------------------
	
	[[nodiscard]] static Plane fromNormalAndPoint(const Vector<TFloat, 3> &normal, const Vector<TFloat, 3> &point) noexcept {
		return Plane(normal, -dot(normal, point));
	}
	
	///----------------------------------------
	///   @brief The plane through three points, wound counter-clockwise as seen from the front.
	/// @details The normal is @c normalize(cross(b - a, c - a)). Collinear points give no plane and
	///          yield a zero normal, which @ref isValid reports.
	///----------------------------------------
	
	[[nodiscard]] static Plane fromPoints(const Vector<TFloat, 3> &a, const Vector<TFloat, 3> &b, const Vector<TFloat, 3> &c) noexcept {
		auto perpendicular = cross(b - a, c - a);
		auto lengthOfPerpendicular = length(perpendicular);
		if (!(lengthOfPerpendicular > TFloat{0})) {
			return Plane(Vector<TFloat, 3>{}, TFloat{0});
		}
		auto unitNormal = perpendicular * (TFloat{1} / lengthOfPerpendicular);
		return fromNormalAndPoint(unitNormal, a);
	}
	
	///----------------------------------------
	///   @brief How far @p point lies in front of the plane, along the normal.
	/// @details Positive in front, negative behind, zero on the plane. With a unit normal this is the
	///          true perpendicular distance.
	///----------------------------------------
	
	[[nodiscard]] TFloat signedDistanceToPoint(const Vector<TFloat, 3> &point) const noexcept {
		return dot(normal, point) + distance;
	}
	
	///----------------------------------------
	/// @brief Whether @p point lies on the normal's side of the plane (or exactly on it).
	///----------------------------------------
	
	[[nodiscard]] bool isPointInFront(const Vector<TFloat, 3> &point) const noexcept {
		return signedDistanceToPoint(point) >= TFloat{0};
	}
	
	///----------------------------------------
	/// @brief The point on the plane closest to @p point.
	///----------------------------------------
	
	[[nodiscard]] Vector<TFloat, 3> closestPointTo(const Vector<TFloat, 3> &point) const noexcept {
		return point - normal * signedDistanceToPoint(point);
	}
	
	///----------------------------------------
	///   @brief The same plane with a unit normal.
	/// @details Rescales both the normal and the distance, so the plane's position is unchanged and
	///          @ref signedDistanceToPoint becomes a true distance. A zero normal is returned as-is.
	///----------------------------------------
	
	[[nodiscard]] Plane normalized() const noexcept {
		auto lengthOfNormal = length(normal);
		if (!(lengthOfNormal > TFloat{0})) {
			return *this;
		}
		auto scale = TFloat{1} / lengthOfNormal;
		return Plane(normal * scale, distance * scale);
	}
	
	///----------------------------------------
	/// @brief The same plane facing the other way.
	///----------------------------------------
	
	[[nodiscard]] Plane flipped() const noexcept {
		return Plane(-normal, -distance);
	}
	
	///----------------------------------------
	/// @brief Whether the plane has a usable (non-degenerate, finite) normal.
	///----------------------------------------
	
	[[nodiscard]] bool isValid() const noexcept {
		return std::isfinite(distance) && lengthSquared(normal) > TFloat{0};
	}
	
	[[nodiscard]] bool operator==(const Plane &) const noexcept = default;
};

using Plane3d = Plane<double>;
using Plane3f = Plane<float>;

///----------------------------------------
/// @brief Validates the plane forms, the sign convention, and the point queries.
///----------------------------------------

inline void planeSelfTest() {
	using selftest::check;
	auto approxEqual = [](double a, double b) noexcept {
		return std::fabs(a - b) <= 1.e-12;
	};
	
	// The canonical plane: z = 0 facing +z, so distance is zero and +z is the front
	auto ground = Plane3d(Double3(0, 0, 1), 0);
	check(approxEqual(ground.signedDistanceToPoint(Double3(0, 0, 5)), 5.0), "a point in front is a positive distance");
	check(approxEqual(ground.signedDistanceToPoint(Double3(0, 0, -5)), -5.0), "a point behind is a negative distance");
	check(approxEqual(ground.signedDistanceToPoint(Double3(9, -3, 0)), 0.0), "a point on the plane is zero, wherever it sits");
	check(ground.isPointInFront(Double3(0, 0, 1)), "in front reads as in front");
	check(!ground.isPointInFront(Double3(0, 0, -1)), "behind does not");
	
	// Offsetting along the normal moves the plane, not its facing
	auto raised = Plane3d::fromNormalAndPoint(Double3(0, 0, 1), Double3(0, 0, 10));
	check(approxEqual(raised.distance, -10.0), "a plane through a point stores the negated offset");
	check(approxEqual(raised.signedDistanceToPoint(Double3(0, 0, 10)), 0.0), "the point it was built through lies on it");
	check(approxEqual(raised.signedDistanceToPoint(Double3(0, 0, 12)), 2.0), "and distances measure from there");
	
	// Three points wound counter-clockwise seen from +z give a +z normal
	auto fromPoints = Plane3d::fromPoints(Double3(0, 0, 0), Double3(1, 0, 0), Double3(0, 1, 0));
	check(approxEqual(fromPoints.normal.z, 1.0), "counter-clockwise winding faces the viewer");
	check(approxEqual(fromPoints.signedDistanceToPoint(Double3(0, 0, 3)), 3.0), "and the plane sits where the points do");
	
	// Collinear points describe no plane
	auto degenerate = Plane3d::fromPoints(Double3(0, 0, 0), Double3(1, 1, 1), Double3(2, 2, 2));
	check(!degenerate.isValid(), "collinear points make no plane");
	
	// The closest point is the perpendicular foot
	auto foot = raised.closestPointTo(Double3(4, -7, 25));
	check(approxEqual(foot.x, 4.0) && approxEqual(foot.y, -7.0) && approxEqual(foot.z, 10.0), "the closest point drops straight onto the plane");
	
	// Normalizing rescales distance with the normal, leaving the plane where it was
	auto unnormalized = Plane3d(Double3(0, 0, 4), -40);
	auto normalizedPlane = unnormalized.normalized();
	check(approxEqual(length(normalizedPlane.normal), 1.0), "normalizing gives a unit normal");
	check(approxEqual(normalizedPlane.signedDistanceToPoint(Double3(0, 0, 10)), 0.0), "and does not move the plane");
	check(approxEqual(normalizedPlane.signedDistanceToPoint(Double3(0, 0, 13)), 3.0), "so distances become true distances");
	
	// Flipping swaps which side is the front without moving the plane
	auto flipped = raised.flipped();
	check(approxEqual(flipped.signedDistanceToPoint(Double3(0, 0, 10)), 0.0), "flipping leaves the plane in place");
	check(approxEqual(flipped.signedDistanceToPoint(Double3(0, 0, 12)), -2.0), "but reverses which side is in front");
	
	// A float instantiation compiles and agrees
	auto groundf = Plane3f(Float3(0, 0, 1), 0);
	check(std::fabs(groundf.signedDistanceToPoint(Float3(0, 0, 2.5f)) - 2.5f) <= 1.e-6f, "the float instantiation matches");
}

///----------------------------------------
} // namespace Math
///----------------------------------------

///----------------------------------------
///       @file Frustum.h
///    @ingroup MathLib
///      @brief The open viewing volume of a camera: an apex and five inward-facing planes.
///    @details A frustum here is a semi-infinite pyramid — an apex and the front, left, right, top
///             and bottom planes that bound what the viewer can see. Every plane passes through the
///             apex and every normal points inward, so a point is visible exactly when it is in
///             front of all five, and @ref Math::Plane's sign convention makes each test one dot
///             product.
///
///             There is deliberately no far plane. Neither caller wanted one: the renderer never
///             applied a far distance, and the star catalog bounds its traversal by magnitude
///             instead, since a star's visibility falls off with brightness rather than with a
///             distance the geometry could know. Adding a far plane would have meant every consumer
///             carrying a distance it had no way to choose.
///       @note The apex invariant is load-bearing. It is why @ref transformedBy can rebuild the
///             planes from a transformed apex and rotated normals rather than transforming plane
///             equations, and why @ref Math::Plane::fromNormalAndPoint is all the construction that
///             is needed. A frustum whose planes were offset from its apex could not be represented.
///     @author Created by John Stephen (wobbleworks.com)
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Matrix.h"
#include "MathLib/Plane.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Transforms.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class Frustum
/// @brief An open viewing volume: an apex and five inward-facing bounding planes.
/// @tparam TFloat Scalar type (float or double).
///----------------------------------------

template <class TFloat>
class Frustum {
///----------------------------------------
public:
	using Vector3 = Vector<TFloat, 3>;
	
	///----------------------------------------
	/// @brief The bounding planes, in storage order.
	///----------------------------------------
	
	enum class Side : size_t {
		front = 0,
		left,
		right,
		top,
		bottom,
		count
	};
	
	///----------------------------------------
	/// @brief How a volume sits relative to the frustum.
	///----------------------------------------
	
	enum class Containment {
		outside,       ///< Entirely outside; nothing within it is visible.
		intersecting,  ///< Straddles a bounding plane; children must be tested individually.
		inside         ///< Entirely inside; everything within it is visible without further testing.
	};
	
	static constexpr auto sideCount = size_t(Side::count);
	
	Frustum() noexcept = default;
	
	///----------------------------------------
	///   @brief A perspective frustum with independent half-angles on each side.
	/// @details @p orientation supplies the view axes as its first three columns — x to the right, y
	///          up, z forward — and each half-angle opens its plane away from the forward axis. The
	///          four angles are independent so an off-centre projection (a split viewport, or a
	///          display whose optical centre is not its geometric centre) is expressible directly.
	///   @param apex The viewpoint every plane passes through.
	///   @param orientation Column-major view basis; only the upper 3×3 is read.
	///   @param leftHalfAngle Angle from the forward axis to the left plane, in radians.
	///   @param rightHalfAngle Angle from the forward axis to the right plane, in radians.
	///   @param topHalfAngle Angle from the forward axis to the top plane, in radians.
	///   @param bottomHalfAngle Angle from the forward axis to the bottom plane, in radians.
	///----------------------------------------
	
	[[nodiscard]] static Frustum perspective(const Vector3 &apex, const Matrix<TFloat, 4> &orientation,
	                                         TFloat leftHalfAngle, TFloat rightHalfAngle,
	                                         TFloat topHalfAngle, TFloat bottomHalfAngle) noexcept {
		auto xAxis = columnOf(orientation, 0);
		auto yAxis = columnOf(orientation, 1);
		auto zAxis = columnOf(orientation, 2);
		
		auto frustum = Frustum();
		frustum._apex = apex;
		frustum.setNormals(zAxis,
			rotateAboutAxis(xAxis, yAxis, -leftHalfAngle),
			rotateAboutAxis(-xAxis, yAxis, rightHalfAngle),
			rotateAboutAxis(-yAxis, xAxis, -topHalfAngle),
			rotateAboutAxis(yAxis, xAxis, bottomHalfAngle));
		return frustum;
	}
	
	///----------------------------------------
	///   @brief A symmetric perspective frustum from a full field of view on each axis.
	/// @details Each field of view is halved and applied to both of its sides.
	///   @param horizontalFieldOfView Total horizontal field of view, in radians.
	///   @param verticalFieldOfView Total vertical field of view, in radians.
	///----------------------------------------
	
	[[nodiscard]] static Frustum perspective(const Vector3 &apex, const Matrix<TFloat, 4> &orientation,
	                                         TFloat horizontalFieldOfView, TFloat verticalFieldOfView) noexcept {
		auto halfHorizontal = horizontalFieldOfView / TFloat{2};
		auto halfVertical = verticalFieldOfView / TFloat{2};
		return perspective(apex, orientation, halfHorizontal, halfHorizontal, halfVertical, halfVertical);
	}
	
	///----------------------------------------
	///   @brief A frustum enclosing the cone of @p halfAngle about @p direction.
	/// @details The four lateral planes are tangent to the cone, so the result is the square pyramid
	///          that circumscribes it — every direction within the cone is inside the frustum, and a
	///          little beyond it near the diagonals. That is what a spatial-index query wants: a cheap
	///          conservative bound, never a false negative.
	///   @param apex The viewpoint every plane passes through.
	///   @param direction Unit forward direction.
	///   @param halfAngle Angle from @p direction to the cone's surface, in radians.
	///----------------------------------------
	
	[[nodiscard]] static Frustum cone(const Vector3 &apex, const Vector3 &direction, TFloat halfAngle) noexcept {
		// Any pair of axes perpendicular to the direction will do; this one degrades gracefully when
		// the direction is vertical, where the natural choice collapses to zero length.
		auto rawRight = Vector3(direction.z, TFloat{0}, -direction.x);
		auto rightLength = length(rawRight);
		auto right = rightLength > TFloat{1.e-30} ? rawRight * (TFloat{1} / rightLength) : Vector3(1, 0, 0);
		auto up = normalize(cross(direction, right));
		
		// Rotating the forward direction most of the way to perpendicular leaves a normal that leans
		// inward by exactly the cone's half-angle.
		auto rotation = TFloat(std::numbers::pi / 2) - halfAngle;
		
		auto frustum = Frustum();
		frustum._apex = apex;
		frustum.setNormals(direction,
			rotateAboutAxis(direction, up, rotation),
			rotateAboutAxis(direction, up, -rotation),
			rotateAboutAxis(direction, right, rotation),
			rotateAboutAxis(direction, right, -rotation));
		frustum._halfAngle = halfAngle;
		return frustum;
	}
	
	///----------------------------------------
	/// @brief Whether the frustum has been built and its planes are usable.
	///----------------------------------------
	
	[[nodiscard]] bool isValid() const noexcept {
		return std::ranges::all_of(_planes, [](const Plane<TFloat> &plane) { return plane.isValid(); });
	}
	
	///----------------------------------------
	/// @brief The viewpoint every plane passes through.
	///----------------------------------------
	
	[[nodiscard]] const Vector3 &apex() const noexcept {
		return _apex;
	}
	
	///----------------------------------------
	/// @brief The forward direction — the front plane's normal.
	///----------------------------------------
	
	[[nodiscard]] const Vector3 &direction() const noexcept {
		return planeOn(Side::front).normal;
	}
	
	///----------------------------------------
	///   @brief The half-angle a @ref cone frustum was built with.
	/// @return The half-angle in radians, or NaN for a frustum that was not built as a cone — a
	///         perspective frustum has four independent half-angles and no single one to report.
	///----------------------------------------
	
	[[nodiscard]] TFloat halfAngle() const noexcept {
		return _halfAngle;
	}
	
	///----------------------------------------
	/// @brief The bounding plane on the given side.
	///----------------------------------------
	
	[[nodiscard]] const Plane<TFloat> &planeOn(Side side) const noexcept {
		return _planes[size_t(side)];
	}
	
	///----------------------------------------
	/// @brief Whether @p point lies within the frustum.
	///----------------------------------------
	
	[[nodiscard]] bool containsPoint(const Vector3 &point) const noexcept {
		return std::ranges::all_of(_planes, [&](const Plane<TFloat> &plane) { return plane.isPointInFront(point); });
	}
	
	///----------------------------------------
	///   @brief How far inside the frustum @p point lies.
	/// @details The smallest signed distance across the five planes: positive inside, and its
	///          magnitude is how far the point could move before crossing the nearest plane. Negative
	///          outside. Comparing against a negated radius is a sphere test.
	///----------------------------------------
	
	[[nodiscard]] TFloat signedDistanceToPoint(const Vector3 &point) const noexcept {
		auto smallest = std::numeric_limits<TFloat>::infinity();
		for (const auto &plane : _planes) {
			smallest = std::min(smallest, plane.signedDistanceToPoint(point));
		}
		return smallest;
	}
	
	///----------------------------------------
	/// @brief How far @p point lies in front of the frustum's front plane.
	///----------------------------------------
	
	[[nodiscard]] TFloat distanceInFrontOfPoint(const Vector3 &point) const noexcept {
		return planeOn(Side::front).signedDistanceToPoint(point);
	}
	
	///----------------------------------------
	///   @brief Whether the sphere at @p center of @p radius touches the frustum.
	/// @details Conservative: a sphere straddling two planes near an edge can be reported as touching
	///          when it does not quite reach the volume. That errs toward drawing, never toward
	///          wrongly culling something visible.
	///----------------------------------------
	
	[[nodiscard]] bool intersectsSphere(const Vector3 &center, TFloat radius) const noexcept {
		return signedDistanceToPoint(center) >= -radius;
	}
	
	///----------------------------------------
	///   @brief Whether the axis-aligned box spanning @p minCorner to @p maxCorner is visible.
	/// @details Tests each plane against the box corner furthest along that plane's normal, and the
	///          one furthest against it. If the furthest-along corner is behind a plane the whole box
	///          is; if the furthest-against corner is in front of every plane the whole box is. That
	///          distinguishes a wholly-contained volume — whose children need no further testing —
	///          from one that straddles an edge.
	///----------------------------------------
	
	[[nodiscard]] Containment containmentOfBox(const Vector3 &minCorner, const Vector3 &maxCorner) const noexcept {
		auto entirelyInside = true;
		for (const auto &plane : _planes) {
			// The corner reaching furthest along the normal, and its opposite
			auto positive = Vector3(
				plane.normal.x >= TFloat{0} ? maxCorner.x : minCorner.x,
				plane.normal.y >= TFloat{0} ? maxCorner.y : minCorner.y,
				plane.normal.z >= TFloat{0} ? maxCorner.z : minCorner.z);
			if (!plane.isPointInFront(positive)) {
				return Containment::outside;
			}
			
			auto negative = Vector3(
				plane.normal.x >= TFloat{0} ? minCorner.x : maxCorner.x,
				plane.normal.y >= TFloat{0} ? minCorner.y : maxCorner.y,
				plane.normal.z >= TFloat{0} ? minCorner.z : maxCorner.z);
			if (!plane.isPointInFront(negative)) {
				entirelyInside = false;
			}
		}
		return entirelyInside ? Containment::inside : Containment::intersecting;
	}
	
	///----------------------------------------
	///   @brief The frustum moved and reoriented by @p transform.
	/// @details The apex is transformed by the full matrix and the normals by its rotation, then the
	///          planes are rebuilt through the new apex — which is exact because every plane passes
	///          through it. A transform with non-uniform scale or shear will not preserve the
	///          normals' perpendicularity to the volume's faces.
	///----------------------------------------
	
	[[nodiscard]] Frustum transformedBy(const Matrix<TFloat, 4> &transform) const noexcept {
		auto xAxis = columnOf(transform, 0);
		auto yAxis = columnOf(transform, 1);
		auto zAxis = columnOf(transform, 2);
		auto translation = columnOf(transform, 3);
		
		auto rotate = [&](const Vector3 &vector) noexcept {
			return xAxis * vector.x + yAxis * vector.y + zAxis * vector.z;
		};
		
		auto result = Frustum();
		result._apex = rotate(_apex) + translation;
		result._halfAngle = _halfAngle;
		for (size_t index = 0; index < sideCount; ++index) {
			result._planes[index] = Plane<TFloat>::fromNormalAndPoint(rotate(_planes[index].normal), result._apex);
		}
		return result;
	}
	
	[[nodiscard]] bool operator==(const Frustum &) const noexcept = default;
	
private:
	///----------------------------------------
	/// @brief Rotates @p vector about the unit @p axis by @p angle radians (Rodrigues' formula).
	///----------------------------------------
	
	[[nodiscard]] static Vector3 rotateAboutAxis(const Vector3 &vector, const Vector3 &axis, TFloat angle) noexcept {
		auto cosine = std::cos(angle);
		auto sine = std::sin(angle);
		return vector * cosine + cross(axis, vector) * sine + axis * (dot(axis, vector) * (TFloat{1} - cosine));
	}
	
	///----------------------------------------
	/// @brief The upper three elements of a matrix column.
	///----------------------------------------
	
	[[nodiscard]] static Vector3 columnOf(const Matrix<TFloat, 4> &matrix, int index) noexcept {
		return Vector3(matrix(0, index), matrix(1, index), matrix(2, index));
	}
	
	///----------------------------------------
	/// @brief Builds the five planes from inward normals, all passing through the apex.
	///----------------------------------------
	
	void setNormals(const Vector3 &front, const Vector3 &left, const Vector3 &right,
	                const Vector3 &top, const Vector3 &bottom) noexcept {
		_planes[size_t(Side::front)] = Plane<TFloat>::fromNormalAndPoint(front, _apex);
		_planes[size_t(Side::left)] = Plane<TFloat>::fromNormalAndPoint(left, _apex);
		_planes[size_t(Side::right)] = Plane<TFloat>::fromNormalAndPoint(right, _apex);
		_planes[size_t(Side::top)] = Plane<TFloat>::fromNormalAndPoint(top, _apex);
		_planes[size_t(Side::bottom)] = Plane<TFloat>::fromNormalAndPoint(bottom, _apex);
	}
	
	Vector3 _apex{};
	std::array<Plane<TFloat>, sideCount> _planes{};
	TFloat _halfAngle = std::numeric_limits<TFloat>::quiet_NaN();
};

using Frustum64 = Frustum<double>;
using Frustum32 = Frustum<float>;

///----------------------------------------
/// @brief Validates frustum construction, the point and volume queries, and transformation.
///----------------------------------------

inline void frustumSelfTest() {
	using selftest::check;
	auto approxEqual = [](double a, double b) noexcept {
		return std::fabs(a - b) <= 1.e-9;
	};
	
	// A 90° symmetric frustum looking down +z from the origin
	auto identity = Double4x4::identity();
	auto quarterTurn = std::numbers::pi / 2;
	auto viewer = Frustum64::perspective(Double3(0, 0, 0), identity, quarterTurn, quarterTurn);
	
	check(viewer.isValid(), "a constructed frustum is valid");
	check(!Frustum64().isValid(), "a default frustum is not");
	check(approxEqual(viewer.direction().z, 1.0), "the forward direction is the view axis");
	
	// Straight ahead is inside; behind and beyond the 45° edges is not
	check(viewer.containsPoint(Double3(0, 0, 10)), "a point straight ahead is inside");
	check(!viewer.containsPoint(Double3(0, 0, -10)), "a point behind the viewer is not");
	check(viewer.containsPoint(Double3(9, 0, 10)), "a point inside the horizontal edge is");
	check(!viewer.containsPoint(Double3(11, 0, 10)), "a point outside it is not");
	check(viewer.containsPoint(Double3(0, 9, 10)), "and the same vertically");
	check(!viewer.containsPoint(Double3(0, 11, 10)), "and outside vertically is not");
	
	// The apex is the one point on every plane at once
	check(viewer.containsPoint(Double3(0, 0, 0)), "the apex itself is inside");
	check(approxEqual(viewer.signedDistanceToPoint(Double3(0, 0, 0)), 0.0), "and sits exactly on the planes");
	
	// Distance is how far the point could move before leaving
	check(viewer.signedDistanceToPoint(Double3(0, 0, 10)) > 0, "an interior point has positive clearance");
	check(viewer.signedDistanceToPoint(Double3(0, 0, -10)) < 0, "an exterior point does not");
	check(approxEqual(viewer.distanceInFrontOfPoint(Double3(3, -4, 10)), 10.0), "front distance measures along the view axis");
	
	// A sphere is visible when it reaches the volume, even with its centre outside
	check(viewer.intersectsSphere(Double3(0, 0, 10), 1.0), "a sphere inside is visible");
	check(!viewer.intersectsSphere(Double3(0, 0, -10), 1.0), "a sphere well behind is not");
	check(viewer.intersectsSphere(Double3(0, 0, -1), 5.0), "one behind but large enough to reach is");
	
	// Box containment separates wholly-inside from straddling, which is what a tree traversal needs
	check(viewer.containmentOfBox(Double3(-1, -1, 10), Double3(1, 1, 12)) == Frustum64::Containment::inside, "a box well inside is inside");
	check(viewer.containmentOfBox(Double3(-1, -1, -12), Double3(1, 1, -10)) == Frustum64::Containment::outside, "a box behind is outside");
	check(viewer.containmentOfBox(Double3(-100, -1, 10), Double3(100, 1, 12)) == Frustum64::Containment::intersecting, "a box crossing the edges straddles");
	
	// The cone form bounds the requested angle: inside it is inside, and the bound is conservative
	auto cone = Frustum64::cone(Double3(0, 0, 0), Double3(0, 0, 1), 0.1);
	check(cone.isValid(), "a cone frustum is valid");
	check(approxEqual(cone.halfAngle(), 0.1), "and reports the angle it was built with");
	check(std::isnan(viewer.halfAngle()), "a perspective frustum reports no single half-angle");
	check(cone.containsPoint(Double3(0, 0, 1)), "the cone axis is inside");
	// A direction just inside the cone must be inside; one well outside must not be
	check(cone.containsPoint(Double3(std::sin(0.09), 0, std::cos(0.09))), "just inside the cone angle is inside");
	check(!cone.containsPoint(Double3(std::sin(0.2), 0, std::cos(0.2))), "well outside it is not");
	
	// Translating moves the volume with the apex
	auto moved = viewer.transformedBy(translationMatrix(Double3(100, 0, 0)));
	check(approxEqual(moved.apex().x, 100.0), "translation carries the apex");
	check(moved.containsPoint(Double3(100, 0, 10)), "and the volume moves with it");
	check(!moved.containsPoint(Double3(0, 0, 10)), "leaving the old location behind");
	check(approxEqual(moved.direction().z, 1.0), "a pure translation leaves the direction alone");
	
	// A float instantiation compiles and agrees
	auto viewerf = Frustum32::perspective(Float3(0, 0, 0), Float4x4::identity(), float(quarterTurn), float(quarterTurn));
	check(viewerf.containsPoint(Float3(0, 0, 10)), "the float instantiation matches");
}

///----------------------------------------
} // namespace Math
///----------------------------------------

///----------------------------------------
///       @file Transforms.h
///    @ingroup MathLib
///      @brief Free-function builders for affine and rotation @ref Math::Matrix transforms.
///    @details @ref Math::Matrix is intentionally minimal (pure linear algebra). The semantic,
///             parameterized transform constructors — translation, scale, axis rotations, basis
///             localization/globalization — live here as free functions in @c namespace Math, mirroring
///             the way @ref Math::dot / @ref Math::normalize accompany @ref Math::Vector. Every builder is
///             column-vector: a matrix @c m maps a point @c p via @c m*p, and translation lives in column 3.
///             The builders produce the standard column-vector matrix forms; these exact values are a
///             compatibility contract and must not change.
///     @author Created by John Stephen on 6/27/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Matrix.h"
#include "MathLib/Vector.h"
#include "MathLib/PolarCoordinates.h"
#include "MathLib/SelfTestCheck.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <type_traits>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief A translation: maps @c p to @c p+offset, with the offset in column 3.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> translationMatrix(const Vector<T, 3>& offset) noexcept {
	return Matrix<T, 4>(
		Vector<T, 4>(1, 0, 0, 0),
		Vector<T, 4>(0, 1, 0, 0),
		Vector<T, 4>(0, 0, 1, 0),
		Vector<T, 4>(offset.x, offset.y, offset.z, 1));
}

///----------------------------------------
/// @brief Translation by @c -origin — moves @p origin to the coordinate origin.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> originMatrix(const Vector<T, 3>& origin) noexcept {
	return translationMatrix<T>(Vector<T, 3>(-origin.x, -origin.y, -origin.z));
}

///----------------------------------------
/// @brief A per-axis scale.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> scaleMatrix(T x, T y, T z) noexcept {
	return Matrix<T, 4>(
		Vector<T, 4>(x, 0, 0, 0),
		Vector<T, 4>(0, y, 0, 0),
		Vector<T, 4>(0, 0, z, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief A uniform scale.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> scaleMatrix(T scale) noexcept {
	return scaleMatrix<T>(scale, scale, scale);
}

///----------------------------------------
/// @brief Rotation by @p radians about the x axis.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> xAxisRotationMatrix(T radians) noexcept {
	const T c = std::cos(radians);
	const T s = std::sin(radians);
	return Matrix<T, 4>(
		Vector<T, 4>(1, 0, 0, 0),
		Vector<T, 4>(0, c, -s, 0),
		Vector<T, 4>(0, s, c, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief Rotation by @p radians about the y axis.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> yAxisRotationMatrix(T radians) noexcept {
	const T c = std::cos(radians);
	const T s = std::sin(radians);
	return Matrix<T, 4>(
		Vector<T, 4>(c, 0, s, 0),
		Vector<T, 4>(0, 1, 0, 0),
		Vector<T, 4>(-s, 0, c, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief Rotation by @p radians about the z axis.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> zAxisRotationMatrix(T radians) noexcept {
	const T c = std::cos(radians);
	const T s = std::sin(radians);
	return Matrix<T, 4>(
		Vector<T, 4>(c, -s, 0, 0),
		Vector<T, 4>(s, c, 0, 0),
		Vector<T, 4>(0, 0, 1, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief Rotation by @p radians about an arbitrary unit @p axis (Rodrigues' formula).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> axisRotationMatrix(const Vector<T, 3>& axis, T radians) noexcept {
	const T s = std::sin(radians);
	const T c = std::cos(radians);
	const T oneMinusCos = 1 - c;
	const T x = axis.x;
	const T y = axis.y;
	const T z = axis.z;
	return Matrix<T, 4>(
		Vector<T, 4>(x * x * oneMinusCos + c, y * x * oneMinusCos - z * s, z * x * oneMinusCos + y * s, 0),
		Vector<T, 4>(x * y * oneMinusCos + z * s, y * y * oneMinusCos + c, z * y * oneMinusCos - x * s, 0),
		Vector<T, 4>(x * z * oneMinusCos - y * s, y * z * oneMinusCos + x * s, z * z * oneMinusCos + c, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief Localization: maps world vectors into the basis (@p x, @p y, @p z) — the basis vectors become the rows.
/// @details World→local. The image of @p x is the x axis, etc. Inverse (transpose) of @ref globalizationMatrix.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 3> localizationMatrix(const Vector<T, 3>& x, const Vector<T, 3>& y, const Vector<T, 3>& z) noexcept {
	return Matrix<T, 3>::fromRows(x, y, z);
}

///----------------------------------------
/// @brief Localization at a center: @ref localizationMatrix with @p center mapped to the origin.
/// @details World→local 4×4; the 3×3 rotation is @ref localizationMatrix (the basis vectors become the rows)
///          and the translation carries @p center to the origin.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> localizationMatrix(const Vector<T, 3>& x, const Vector<T, 3>& y, const Vector<T, 3>& z, const Vector<T, 3>& center) noexcept {
	return Matrix<T, 4>(
		Vector<T, 4>(x.x, y.x, z.x, 0),
		Vector<T, 4>(x.y, y.y, z.y, 0),
		Vector<T, 4>(x.z, y.z, z.z, 0),
		Vector<T, 4>(-dot(x, center), -dot(y, center), -dot(z, center), 1));
}

///----------------------------------------
/// @brief Globalization: maps local vectors out of the basis (@p x, @p y, @p z) — the basis vectors become the columns.
/// @details Local→world. Inverse (transpose) of @ref localizationMatrix.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 3> globalizationMatrix(const Vector<T, 3>& x, const Vector<T, 3>& y, const Vector<T, 3>& z) noexcept {
	return Matrix<T, 3>(x, y, z);
}

///----------------------------------------
/// @brief World→viewer transform for an observer at @p longitudeRad / @p latitudeRad on a sphere of the given @p radius.
/// @details The observer's local axes rotate the world into view space and @c (0, radius, 0) maps to the origin.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> viewerTransform(T longitudeRad, std::type_identity_t<T> latitudeRad, std::type_identity_t<T> radius) noexcept {
	const T sinLon = std::sin(longitudeRad), cosLon = std::cos(longitudeRad);
	const T sinLat = std::sin(latitudeRad), cosLat = std::cos(latitudeRad);
	const Vector<T, 3> xAxis(-cosLon, 0, -sinLon);
	const Vector<T, 3> yAxis(-sinLon * cosLat, sinLat, cosLon * cosLat);
	const Vector<T, 3> zAxis = cross(xAxis, yAxis);
	return Matrix<T, 4>(
		Vector<T, 4>(xAxis.x, yAxis.x, zAxis.x, 0),
		Vector<T, 4>(xAxis.y, yAxis.y, zAxis.y, 0),
		Vector<T, 4>(xAxis.z, yAxis.z, zAxis.z, 0),
		Vector<T, 4>(0, -radius, 0, 1));
}

///----------------------------------------
/// @brief Retargets a perspective projection's depth mapping to a finite [@p nearZ, @p farZ] range.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> withPerspectiveZRange(const Matrix<T, 4>& projection, T nearZ, std::type_identity_t<T> farZ) noexcept {
	Vector<T, 4> column2 = projection.column(2);
	Vector<T, 4> column3 = projection.column(3);
	column2.z = (nearZ + farZ) / (farZ - nearZ);
	column3.z = T(-2) * nearZ * farZ / (farZ - nearZ);
	return Matrix<T, 4>(projection.column(0), projection.column(1), column2, column3);
}

///----------------------------------------
/// @brief Retargets a perspective projection to an infinite far plane with the given @p nearZ.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> withInfinitePerspectiveNearZ(const Matrix<T, 4>& projection, T nearZ) noexcept {
	Vector<T, 4> column2 = projection.column(2);
	Vector<T, 4> column3 = projection.column(3);
	column2.z = 1;
	column3.z = T(-2) * nearZ;
	return Matrix<T, 4>(projection.column(0), projection.column(1), column2, column3);
}

///----------------------------------------
/// @brief A perspective frustum projection with the given clip bounds and near/far planes, with depth mapping enabled.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> frustumMatrix(T left, std::type_identity_t<T> top, std::type_identity_t<T> right, std::type_identity_t<T> bottom, std::type_identity_t<T> nearZ, std::type_identity_t<T> farZ) noexcept {
	const T zFactor = (farZ + nearZ) / (farZ - nearZ);
	const T zOffset = T(-2) * farZ * nearZ / (farZ - nearZ);
	return Matrix<T, 4>(
		Vector<T, 4>(T(2) * nearZ / (right - left), 0, 0, 0),
		Vector<T, 4>(0, T(2) * nearZ / (top - bottom), 0, 0),
		Vector<T, 4>(-(left + right) / (right - left), -(top + bottom) / (top - bottom), zFactor, 1),
		Vector<T, 4>(0, 0, zOffset, 0));
}

///----------------------------------------
/// @brief A perspective frustum projection with an optional @p disableDepth flag that flattens the depth mapping.
///        With @p disableDepth false this matches @ref frustumMatrix.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> frustumMatrix(T left, std::type_identity_t<T> top, std::type_identity_t<T> right, std::type_identity_t<T> bottom, std::type_identity_t<T> nearZ, std::type_identity_t<T> farZ, bool disableDepth) noexcept {
	const T zFactor = (farZ + nearZ) / (farZ - nearZ);
	const T zOffset = T(-2) * farZ * nearZ / (farZ - nearZ);
	return Matrix<T, 4>(
		Vector<T, 4>(T(2) * nearZ / (right - left), 0, 0, 0),
		Vector<T, 4>(0, T(2) * nearZ / (top - bottom), 0, 0),
		Vector<T, 4>(-(left + right) / (right - left), -(top + bottom) / (top - bottom), disableDepth ? T(1) : zFactor, 1),
		Vector<T, 4>(0, 0, disableDepth ? T(0) : zOffset, 0));
}

///----------------------------------------
/// @brief A perspective projection from vertical field of view and aspect ratio.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> perspectiveMatrix(T fovyRad, std::type_identity_t<T> aspect, std::type_identity_t<T> nearZ, std::type_identity_t<T> farZ) noexcept {
	const T tanHalfFovy = std::tan(fovyRad / T(2));
	return Matrix<T, 4>(
		Vector<T, 4>(T(1) / (aspect * tanHalfFovy), 0, 0, 0),
		Vector<T, 4>(0, T(1) / tanHalfFovy, 0, 0),
		Vector<T, 4>(0, 0, (nearZ + farZ) / (nearZ - farZ), T(2) * nearZ * farZ / (nearZ - farZ)),
		Vector<T, 4>(0, 0, -1, 0));
}

///----------------------------------------
/// @brief A perspective projection with an infinite far plane, from vertical field of view and aspect ratio.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> infinitePerspectiveMatrix(T fovyRad, std::type_identity_t<T> aspect, std::type_identity_t<T> nearZ) noexcept {
	const T tanHalfFovy = std::tan(fovyRad / T(2));
	return Matrix<T, 4>(
		Vector<T, 4>(T(1) / (aspect * tanHalfFovy), 0, 0, 0),
		Vector<T, 4>(0, T(1) / tanHalfFovy, 0, 0),
		Vector<T, 4>(0, 0, 1, 1),
		Vector<T, 4>(0, 0, T(-2) * nearZ, 0));
}

///----------------------------------------
/// @brief An orthographic projection mapping the box [@p topLeftFront, @p bottomRightBack] to clip space; @p disableDepth flattens
///        the z mapping. The Y range is inverted to keep the top-left origin convention.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> orthographicMatrix(const Vector<T, 3>& topLeftFront, const Vector<T, 3>& bottomRightBack, bool disableDepth = false) noexcept {
	const T rangeX = bottomRightBack.x - topLeftFront.x;
	const T rangeY = -(bottomRightBack.y - topLeftFront.y);
	const T rangeZ = bottomRightBack.z - topLeftFront.z;
	const T scaleX = T(2) / rangeX;
	const T scaleY = T(2) / rangeY;
	const T scaleZ = T(2) / rangeZ;
	return Matrix<T, 4>(
		Vector<T, 4>(scaleX, 0, 0, 0),
		Vector<T, 4>(0, scaleY, 0, 0),
		Vector<T, 4>(0, 0, disableDepth ? T(1) : -scaleZ, 0),
		Vector<T, 4>(-(topLeftFront.x + bottomRightBack.x) / rangeX, -(topLeftFront.y + bottomRightBack.y) / rangeY, -(topLeftFront.z + bottomRightBack.z) / rangeZ, 1));
}

///----------------------------------------
/// @brief Retargets a perspective projection to an infinite far plane, rewriting only the depth (z/w) terms and preserving any
///        principal-point offset in @c column(2).xy.
/// @note Unlike @ref withInfinitePerspectiveNearZ this also sets @c column(2).w to 1 and @c column(3).w to 0.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> withInfinitePerspectiveDepth(const Matrix<T, 4>& projection, T nearZ) noexcept {
	Vector<T, 4> column2 = projection.column(2);
	Vector<T, 4> column3 = projection.column(3);
	column2.z = 1;
	column2.w = 1;
	column3.z = T(-2) * nearZ;
	column3.w = 0;
	return Matrix<T, 4>(projection.column(0), projection.column(1), column2, column3);
}

///----------------------------------------
/// @brief NDC→viewport (window) transform for the given pixel bounds; maps clip [-1,1] to the bounds, flipping Y.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> viewportMatrix(T left, std::type_identity_t<T> top, std::type_identity_t<T> right, std::type_identity_t<T> bottom) noexcept {
	const T halfWidth = (right - left) / T(2);
	const T halfHeight = (bottom - top) / T(2);
	return Matrix<T, 4>(
		Vector<T, 4>(halfWidth, 0, 0, 0),
		Vector<T, 4>(0, -halfHeight, 0, 0),
		Vector<T, 4>(0, 0, 1, 0),
		Vector<T, 4>(halfWidth + left, halfHeight + top, 0, 1));
}

///----------------------------------------
/// @brief The inverse of a rigid (rotation + translation) transform: transpose the rotation, negate-rotate the translation.
/// @details Cheap exact inverse for an orthonormal @p matrix.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> orthoInverse(const Matrix<T, 4>& matrix) noexcept {
	const Vector<T, 4> c0 = matrix.column(0);
	const Vector<T, 4> c1 = matrix.column(1);
	const Vector<T, 4> c2 = matrix.column(2);
	const Vector<T, 4> t = matrix.column(3);
	const Vector<T, 4> r0(c0.x, c1.x, c2.x, 0);
	const Vector<T, 4> r1(c0.y, c1.y, c2.y, 0);
	const Vector<T, 4> r2(c0.z, c1.z, c2.z, 0);
	const T tx = -(r0.x * t.x + r1.x * t.y + r2.x * t.z);
	const T ty = -(r0.y * t.x + r1.y * t.y + r2.y * t.z);
	const T tz = -(r0.z * t.x + r1.z * t.y + r2.z * t.z);
	return Matrix<T, 4>(r0, r1, r2, Vector<T, 4>(tx, ty, tz, 1));
}

///----------------------------------------
/// @brief The inverse of a uniformly-scaled orthogonal transform: like @ref orthoInverse but dividing the
///        transposed rotation by the squared scale, so it stays correct for a non-unit uniform scale.
///        Asserts (and returns @p matrix) on a singular basis.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> uniformScaleOrthoInverse(const Matrix<T, 4>& matrix) noexcept {
	const Vector<T, 4> c0 = matrix.column(0);
	const Vector<T, 4> c1 = matrix.column(1);
	const Vector<T, 4> c2 = matrix.column(2);
	const Vector<T, 4> t = matrix.column(3);
	Vector<T, 3> row0(c0.x, c1.x, c2.x);
	Vector<T, 3> row1(c0.y, c1.y, c2.y);
	Vector<T, 3> row2(c0.z, c1.z, c2.z);
	const T scaleSquared = dot(row0, row0);
	if (scaleSquared == 0) {
		assert(false);
		return matrix;
	}
	const T invScaleSquared = T(1) / scaleSquared;
	row0 = row0 * invScaleSquared;
	row1 = row1 * invScaleSquared;
	row2 = row2 * invScaleSquared;
	const T tx = -(row0.x * t.x + row1.x * t.y + row2.x * t.z);
	const T ty = -(row0.y * t.x + row1.y * t.y + row2.y * t.z);
	const T tz = -(row0.z * t.x + row1.z * t.y + row2.z * t.z);
	return Matrix<T, 4>(
		Vector<T, 4>(row0.x, row0.y, row0.z, 0),
		Vector<T, 4>(row1.x, row1.y, row1.z, 0),
		Vector<T, 4>(row2.x, row2.y, row2.z, 0),
		Vector<T, 4>(tx, ty, tz, 1));
}

///----------------------------------------
/// @brief Whether @p matrix is a rigid (orthonormal rotation + translation) transform.
/// @details The upper-left 3×3 rows are unit-length within [0.9999, 1.0001], the homogeneous column is
///          clean (no perspective, w==1) and the translation is finite. Intended for debug assertions.
///----------------------------------------

template <class T>
[[nodiscard]] bool isOrtho(const Matrix<T, 4>& matrix) noexcept {
	const Vector<T, 4> c0 = matrix.column(0);
	const Vector<T, 4> c1 = matrix.column(1);
	const Vector<T, 4> c2 = matrix.column(2);
	const Vector<T, 4> t = matrix.column(3);
	const T row0LengthSquared = c0.x * c0.x + c1.x * c1.x + c2.x * c2.x;
	const T row1LengthSquared = c0.y * c0.y + c1.y * c1.y + c2.y * c2.y;
	const T row2LengthSquared = c0.z * c0.z + c1.z * c1.z + c2.z * c2.z;
	return row0LengthSquared > T(0.9999) && row0LengthSquared < T(1.0001)
	    && row1LengthSquared > T(0.9999) && row1LengthSquared < T(1.0001)
	    && row2LengthSquared > T(0.9999) && row2LengthSquared < T(1.0001)
	    && std::isfinite(t.x) && std::isfinite(t.y) && std::isfinite(t.z)
	    && c0.w == 0 && c1.w == 0 && c2.w == 0 && t.w == 1;
}

///----------------------------------------
/// @brief @p matrix with its translation removed — the rotation part with a clean homogeneous column.
/// @details The rotation columns are unchanged and column 3 becomes (0, 0, 0, 1).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> withoutTranslation(const Matrix<T, 4>& matrix) noexcept {
	return Matrix<T, 4>(matrix.column(0), matrix.column(1), matrix.column(2), Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief The translation (position) of a 4×4 transform — the x, y, z of column 3.
///----------------------------------------

template <class T>
[[nodiscard]] Vector<T, 3> translation(const Matrix<T, 4>& matrix) noexcept {
	const Vector<T, 4> column = matrix.column(3);
	return Vector<T, 3>(column.x, column.y, column.z);
}

///----------------------------------------
/// @brief Transforms a point through @p matrix (rotation + translation).
///----------------------------------------

template <class T>
[[nodiscard]] Vector<T, 3> transformPoint(const Matrix<T, 4>& matrix, const Vector<T, 3>& point) noexcept {
	const Vector<T, 4> result = matrix * Vector<T, 4>(point.x, point.y, point.z, 1);
	return Vector<T, 3>(result.x, result.y, result.z);
}

///----------------------------------------
/// @brief Transforms a direction through @p matrix (rotation only, ignoring translation).
///----------------------------------------

template <class T>
[[nodiscard]] Vector<T, 3> transformDirection(const Matrix<T, 4>& matrix, const Vector<T, 3>& direction) noexcept {
	const Vector<T, 4> result = matrix * Vector<T, 4>(direction.x, direction.y, direction.z, 0);
	return Vector<T, 3>(result.x, result.y, result.z);
}

///----------------------------------------
/// @brief @p base with a per-axis scale composed onto it.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> scaled(const Matrix<T, 4>& base, std::type_identity_t<T> x, std::type_identity_t<T> y, std::type_identity_t<T> z) noexcept {
	return base * scaleMatrix<T>(x, y, z);
}

///----------------------------------------
/// @brief @p base with a uniform scale composed onto it.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> scaled(const Matrix<T, 4>& base, std::type_identity_t<T> scale) noexcept {
	return base * scaleMatrix<T>(scale);
}

///----------------------------------------
/// @brief @p base with a rotation about unit @p axis composed onto it.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> rotated(const Matrix<T, 4>& base, std::type_identity_t<T> radians, const Vector<std::type_identity_t<T>, 3>& axis) noexcept {
	return base * axisRotationMatrix<T>(axis, radians);
}

///----------------------------------------
/// @brief @p base with a rotation about unit axis (@p x, @p y, @p z) composed onto it.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> rotated(const Matrix<T, 4>& base, std::type_identity_t<T> radians, std::type_identity_t<T> x, std::type_identity_t<T> y, std::type_identity_t<T> z) noexcept {
	return base * axisRotationMatrix<T>(Vector<T, 3>(x, y, z), radians);
}

///----------------------------------------
/// @brief @p base with a translation composed onto it.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> translated(const Matrix<T, 4>& base, std::type_identity_t<T> x, std::type_identity_t<T> y, std::type_identity_t<T> z) noexcept {
	return translationMatrix<T>(Vector<T, 3>(x, y, z)) * base;
}

///----------------------------------------
/// @brief @p base with a translation composed onto it on the opposite side.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> reverseTranslated(const Matrix<T, 4>& base, std::type_identity_t<T> x, std::type_identity_t<T> y, std::type_identity_t<T> z) noexcept {
	return base * translationMatrix<T>(Vector<T, 3>(x, y, z));
}

///----------------------------------------
/// @brief An orientation from yaw, pitch and roll. The z axis points along (@p yaw, @p pitch); the x axis is
///        90° behind in yaw, rolled about z; the y axis completes the basis. The axes are the matrix columns.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> yawPitchRollMatrix(T yaw, T pitch, T roll) noexcept {
	const Vector<T, 3> zAxis = vectorFromPolarRad<T>(yaw, pitch);
	Vector<T, 3> xAxis = vectorFromPolarRad<T>(yaw - std::numbers::pi_v<T> / 2, T(0));
	const Vector<T, 4> rotatedX = axisRotationMatrix<T>(zAxis, roll) * Vector<T, 4>(xAxis.x, xAxis.y, xAxis.z, 0);
	xAxis = Vector<T, 3>(rotatedX.x, rotatedX.y, rotatedX.z);
	const Vector<T, 3> yAxis = cross(zAxis, xAxis);
	return Matrix<T, 4>(
		Vector<T, 4>(xAxis.x, xAxis.y, xAxis.z, 0),
		Vector<T, 4>(yAxis.x, yAxis.y, yAxis.z, 0),
		Vector<T, 4>(zAxis.x, zAxis.y, zAxis.z, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief An orientation from yaw, pitch and roll using the standard CCW roll convention. The z axis
///        points along (@p yaw, @p pitch); the x axis is 90° behind in yaw, rotated CCW about z by
///        @p roll; y completes the basis. Axes are the columns.
/// @note Differs from @ref yawPitchRollMatrix only in roll handedness (that one rolls the opposite way).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> orientationFromYawPitchRoll(T yaw, T pitch, T roll) noexcept {
	const Vector<T, 3> zAxis = vectorFromPolarRad<T>(yaw, pitch);
	const Vector<T, 3> baseX = vectorFromPolarRad<T>(yaw - std::numbers::pi_v<T> / 2, T(0));
	// Rodrigues rotation of baseX about zAxis by roll; baseX ⊥ zAxis so the parallel term vanishes.
	const Vector<T, 3> xAxis = baseX * std::cos(roll) + cross(zAxis, baseX) * std::sin(roll);
	const Vector<T, 3> yAxis = cross(zAxis, xAxis);
	return Matrix<T, 4>(
		Vector<T, 4>(xAxis.x, xAxis.y, xAxis.z, 0),
		Vector<T, 4>(yAxis.x, yAxis.y, yAxis.z, 0),
		Vector<T, 4>(zAxis.x, zAxis.y, zAxis.z, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief An orientation (local→world) from Euler yaw/pitch/roll, with the composed X/Y/Z basis axes as the matrix
///        columns.
/// @note This is the transpose of @ref eulerRotationMatrix, which places the same axes as rows (world→local).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> orientationFromEuler(T yaw, T pitch, T roll) noexcept {
	const T sinYaw = std::sin(yaw);
	const T cosYaw = std::cos(yaw);
	const T sinPitch = std::sin(pitch);
	const T cosPitch = std::cos(pitch);
	if (roll == 0) {
		return Matrix<T, 4>(
			Vector<T, 4>(cosYaw, 0, sinYaw, 0),
			Vector<T, 4>(sinYaw * sinPitch, cosPitch, -cosYaw * sinPitch, 0),
			Vector<T, 4>(-sinYaw * cosPitch, sinPitch, cosYaw * cosPitch, 0),
			Vector<T, 4>(0, 0, 0, 1));
	}
	const T sinRoll = std::sin(roll);
	const T cosRoll = std::cos(roll);
	const T sinPitchSinRoll = sinPitch * sinRoll;
	const T sinPitchCosRoll = sinPitch * cosRoll;
	return Matrix<T, 4>(
		Vector<T, 4>(cosYaw * cosRoll - sinYaw * sinPitchSinRoll, -cosPitch * sinRoll, sinYaw * cosRoll + cosYaw * sinPitchSinRoll, 0),
		Vector<T, 4>(cosYaw * sinRoll + sinYaw * sinPitchCosRoll, cosPitch * cosRoll, sinYaw * sinRoll - cosYaw * sinPitchCosRoll, 0),
		Vector<T, 4>(-sinYaw * cosPitch, sinPitch, cosYaw * cosPitch, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief @p base with only its translation (column 3) scaled by @p factor — e.g. a units conversion of the
///        translation.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> withScaledTranslation(const Matrix<T, 4>& base, std::type_identity_t<T> factor) noexcept {
	const Vector<T, 4> t = base.column(3);
	return Matrix<T, 4>(base.column(0), base.column(1), base.column(2), Vector<T, 4>(t.x * factor, t.y * factor, t.z * factor, t.w));
}

///----------------------------------------
/// @brief The upper-left 3×3 (rotation) block of a 4×4 transform, with translation and the homogeneous row dropped.
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 3> rotationMatrix3(const Matrix<T, 4>& matrix) noexcept {
	const Vector<T, 4> c0 = matrix.column(0);
	const Vector<T, 4> c1 = matrix.column(1);
	const Vector<T, 4> c2 = matrix.column(2);
	return Matrix<T, 3>(
		Vector<T, 3>(c0.x, c0.y, c0.z),
		Vector<T, 3>(c1.x, c1.y, c1.z),
		Vector<T, 3>(c2.x, c2.y, c2.z));
}

///----------------------------------------
/// @brief A 3×3 rotation embedded as a homogeneous 4×4 transform (rotation in the upper-left, zero translation).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> rotation4d(const Matrix<T, 3>& rotation) noexcept {
	const Vector<T, 3> c0 = rotation.column(0);
	const Vector<T, 3> c1 = rotation.column(1);
	const Vector<T, 3> c2 = rotation.column(2);
	return Matrix<T, 4>(
		Vector<T, 4>(c0.x, c0.y, c0.z, 0),
		Vector<T, 4>(c1.x, c1.y, c1.z, 0),
		Vector<T, 4>(c2.x, c2.y, c2.z, 0),
		Vector<T, 4>(0, 0, 0, 1));
}

///----------------------------------------
/// @brief Rotation from Euler yaw/pitch/roll angles in radians.
/// @details Builds the X/Y/Z basis axes from the angles and localizes onto them (the axes become the rotation's rows).
///----------------------------------------

template <class T>
[[nodiscard]] Matrix<T, 4> eulerRotationMatrix(T yawRad, std::type_identity_t<T> pitchRad, std::type_identity_t<T> rollRad) noexcept {
	const T sinYaw = std::sin(yawRad);
	const T cosYaw = std::cos(yawRad);
	const T sinPitch = std::sin(pitchRad);
	const T cosPitch = std::cos(pitchRad);
	
	Vector<T, 3> xAxis, yAxis, zAxis;
	if (rollRad == 0) {
		xAxis = Vector<T, 3>(cosYaw, 0, sinYaw);
		yAxis = Vector<T, 3>(sinYaw * sinPitch, cosPitch, -cosYaw * sinPitch);
		zAxis = Vector<T, 3>(-sinYaw * cosPitch, sinPitch, cosYaw * cosPitch);
	} else {
		const T sinRoll = std::sin(rollRad);
		const T cosRoll = std::cos(rollRad);
		const T sinPitchSinRoll = sinPitch * sinRoll;
		const T sinPitchCosRoll = sinPitch * cosRoll;
		
		xAxis = Vector<T, 3>(cosYaw * cosRoll - sinYaw * sinPitchSinRoll, -cosPitch * sinRoll, sinYaw * cosRoll + cosYaw * sinPitchSinRoll);
		yAxis = Vector<T, 3>(cosYaw * sinRoll + sinYaw * sinPitchCosRoll, cosPitch * cosRoll, sinYaw * sinRoll - cosYaw * sinPitchCosRoll);
		zAxis = Vector<T, 3>(-sinYaw * cosPitch, sinPitch, cosYaw * cosPitch);
	}
	
	return rotation4d(localizationMatrix(xAxis, yAxis, zAxis));
}

///----------------------------------------
/// @brief Extracts yaw, pitch and roll from an orientation built by @ref yawPitchRollMatrix (inverse of it).
/// @details At the gimbal poles (pitch ±90°) yaw and roll turn about the same vertical axis, so only their
///          combined turn survives in the matrix; it is reported entirely as yaw, with roll zero.
///----------------------------------------

template <class T>
void yawPitchRollFromMatrix(const Matrix<T, 4>& matrix, T& yaw, T& pitch, T& roll) noexcept {
	const Vector<T, 4> xAxis = matrix.column(0);
	const Vector<T, 4> yAxis = matrix.column(1);
	const Vector<T, 4> zAxis = matrix.column(2);
	if (std::fabs(zAxis.y) > T(0.999999)) {
		// Gimbal pole: the z axis is vertical, so the x axis lies in the horizontal plane carrying the
		// combined yaw/roll turn — x = (cos(yaw ± roll), 0, sin(yaw ± roll)).
		pitch = std::copysign(std::numbers::pi_v<T> / 2, zAxis.y);
		yaw = std::atan2(xAxis.z, xAxis.x);
		roll = 0;
		return;
	}
	pitch = std::asin(std::min(std::max(zAxis.y, T(-1)), T(1)));
	yaw = std::atan2(-zAxis.x, zAxis.z);
	roll = std::atan2(-xAxis.y, yAxis.y);
	if (std::fabs(yaw) < T(0.0000000001)) yaw = 0;
	if (std::fabs(pitch) < T(0.0000000001)) pitch = 0;
	if (std::fabs(roll) < T(0.0000000001)) roll = 0;
}

///----------------------------------------
/// @brief Verifies the transform builders through invertibility invariants and known point mappings.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void transformsSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-6; };
	
	// Translation and uniform scale act on a point exactly as their names promise.
	{
		const Double3 point(1.0, 2.0, 3.0);
		const Double3 translated = transformPoint(translationMatrix<double>(Double3(4.0, -5.0, 6.0)), point);
		check(near(translated.x, 5.0) && near(translated.y, -3.0) && near(translated.z, 9.0), "translation offsets a point");
		const Double3 scaled = transformPoint(scaleMatrix<double>(2.0), point);
		check(near(scaled.x, 2.0) && near(scaled.y, 4.0) && near(scaled.z, 6.0), "uniform scale scales a point");
	}
	
	// orthoInverse composed with a rigid (rotation + translation) transform is the identity.
	{
		const Double4x4 rigid = translationMatrix<double>(Double3(3.0, -4.0, 5.0)) * xAxisRotationMatrix<double>(0.4);
		check(approxEqual(orthoInverse(rigid) * rigid, Double4x4::identity(), 1.0e-6), "orthoInverse undoes a rigid transform");
		check(approxEqual(rigid * orthoInverse(rigid), Double4x4::identity(), 1.0e-6), "orthoInverse is a two-sided inverse");
	}
	
	// A perspective projection maps the near/far plane centers to the -1/+1 depth extremes and the near-plane top edge to +1.
	// The builders produce the standard column-vector projection matrices, so the standard
	// clip mapping is the v·M reading — transpose(projection)·v — with a 90 deg vertical FOV making the near-plane half-height 1.
	{
		const Double4x4 clip = transpose(perspectiveMatrix<double>(std::numbers::pi / 2, 1.0, 1.0, 3.0));
		const Double4 nearClip = clip * Double4(0.0, 0.0, -1.0, 1.0);
		const Double4 farClip = clip * Double4(0.0, 0.0, -3.0, 1.0);
		const Double4 topClip = clip * Double4(0.0, 1.0, -1.0, 1.0);
		check(near(nearClip.z / nearClip.w, -1.0), "near plane maps to NDC z -1");
		check(near(farClip.z / farClip.w, 1.0), "far plane maps to NDC z +1");
		check(near(topClip.y / topClip.w, 1.0), "near-plane top edge maps to NDC y +1");
	}
	
	// An orthographic box maps its corners to the [-1, 1] cube, with Y and Z inverted per the top-left, forward-Z convention.
	{
		const Double4x4 ortho = orthographicMatrix<double>(Double3(-2.0, 3.0, 5.0), Double3(2.0, -3.0, -5.0));
		const Double3 lowCorner = transformPoint(ortho, Double3(-2.0, 3.0, 5.0));
		const Double3 highCorner = transformPoint(ortho, Double3(2.0, -3.0, -5.0));
		check(near(lowCorner.x, -1.0) && near(lowCorner.y, 1.0) && near(lowCorner.z, 1.0), "ortho maps top-left-front corner");
		check(near(highCorner.x, 1.0) && near(highCorner.y, -1.0) && near(highCorner.z, -1.0), "ortho maps bottom-right-back corner");
	}
	
	// yawPitchRollFromMatrix inverts yawPitchRollMatrix for a modest angle triple clear of gimbal lock.
	{
		const double yaw = 0.3, pitch = 0.2, roll = 0.1;
		double extractedYaw = 0, extractedPitch = 0, extractedRoll = 0;
		yawPitchRollFromMatrix(yawPitchRollMatrix<double>(yaw, pitch, roll), extractedYaw, extractedPitch, extractedRoll);
		check(near(extractedYaw, yaw) && near(extractedPitch, pitch) && near(extractedRoll, roll), "yaw/pitch/roll round-trip");
	}

	// At the gimbal poles yaw and roll share an axis: the combined turn reports entirely as yaw, and the
	// decomposition must still recompose its matrix. Distinct yaw/roll catch a dropped or halved term.
	for (const double pitch : {std::numbers::pi / 2, -std::numbers::pi / 2}) {
		const Double4x4 pole = yawPitchRollMatrix<double>(0.7, pitch, 0.3);
		double extractedYaw = 0, extractedPitch = 0, extractedRoll = 0;
		yawPitchRollFromMatrix(pole, extractedYaw, extractedPitch, extractedRoll);
		check(near(extractedPitch, pitch), "pole pitch extracts as its pole");
		check(near(extractedRoll, 0.0), "pole roll reports as zero");
		check(approxEqual(yawPitchRollMatrix<double>(extractedYaw, extractedPitch, extractedRoll), pole, 1.0e-6),
			"pole decomposition recomposes its matrix");
	}
}
	
} // namespace Math

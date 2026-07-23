///----------------------------------------
///       @file AppleBackend.h
///    @ingroup MathLib
///      @brief Apple backend for the vector, matrix and quaternion operations that @ref Vector.h,
///             @ref Matrix.h and @ref Quaternion.h are built on.
///    @details Wraps the platform @c <simd/simd.h> library. The native types are the SDK's
///             @c simd_double3 / @c simd_double4x4 / @c simd_quatd (and their float and lower-rank
///             siblings); the @ref native_vector / @ref native_matrix / @ref native_quaternion traits
///             map to them, and @ref Backend forwards its neutral operation vocabulary to the
///             overloaded @c simd_* functions. Templated forwarders suffice because each @c simd_*
///             name is overloaded across the native types. The counterpart @ref detail/GenericBackend.h
///             provides the same interface with plain structs on non-Apple platforms; only one is
///             ever included.
///     @author Created by John Stephen on 7/7/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <simd/simd.h>

#include <cstdint>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief Maps a (scalar, component-count) pair to its SDK vector type.
///----------------------------------------

template <class TScalar, int Count>
struct native_vector;

template <> struct native_vector<double, 2> { using type = simd_double2; };
template <> struct native_vector<double, 3> { using type = simd_double3; };
template <> struct native_vector<double, 4> { using type = simd_double4; };
template <> struct native_vector<float, 2> { using type = simd_float2; };
template <> struct native_vector<float, 3> { using type = simd_float3; };
template <> struct native_vector<float, 4> { using type = simd_float4; };
template <> struct native_vector<uint8_t, 3> { using type = simd_uchar3; };
template <> struct native_vector<uint8_t, 4> { using type = simd_uchar4; };

///----------------------------------------
/// @brief Maps a (scalar, rows, cols) triple to its SDK matrix type and identity constant.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct native_matrix;

template <> struct native_matrix<double, 2, 2> { using type = simd_double2x2; [[nodiscard]] static type identity() noexcept { return matrix_identity_double2x2; } };
template <> struct native_matrix<double, 3, 3> { using type = simd_double3x3; [[nodiscard]] static type identity() noexcept { return matrix_identity_double3x3; } };
template <> struct native_matrix<double, 4, 4> { using type = simd_double4x4; [[nodiscard]] static type identity() noexcept { return matrix_identity_double4x4; } };
template <> struct native_matrix<float, 2, 2> { using type = simd_float2x2; [[nodiscard]] static type identity() noexcept { return matrix_identity_float2x2; } };
template <> struct native_matrix<float, 3, 3> { using type = simd_float3x3; [[nodiscard]] static type identity() noexcept { return matrix_identity_float3x3; } };
template <> struct native_matrix<float, 4, 4> { using type = simd_float4x4; [[nodiscard]] static type identity() noexcept { return matrix_identity_float4x4; } };

///----------------------------------------
/// @brief Maps a scalar type to its SDK quaternion type.
///----------------------------------------

template <class TScalar>
struct native_quaternion;

template <> struct native_quaternion<double> { using type = simd_quatd; };
template <> struct native_quaternion<float>  { using type = simd_quatf; };

///----------------------------------------
/// @struct Backend
/// @brief The Apple operation vocabulary: static methods forwarding to the overloaded @c simd_*
///        functions. The generic backend defines a same-named struct with plain-struct math; only
///        one is ever included.
///----------------------------------------

struct Backend {
	
	///----------------------------------------
	/// @name Vector operations
	///----------------------------------------
	/// @{
	
	template <class V> [[nodiscard]] static auto dot(const V& a, const V& b) noexcept { return simd_dot(a, b); }
	template <class V> [[nodiscard]] static auto cross(const V& a, const V& b) noexcept { return simd_cross(a, b); }
	template <class V> [[nodiscard]] static auto length(const V& v) noexcept { return simd_length(v); }
	template <class V> [[nodiscard]] static auto lengthSquared(const V& v) noexcept { return simd_length_squared(v); }
	template <class V> [[nodiscard]] static auto distance(const V& a, const V& b) noexcept { return simd_distance(a, b); }
	template <class V> [[nodiscard]] static auto distanceSquared(const V& a, const V& b) noexcept { return simd_distance_squared(a, b); }
	template <class V> [[nodiscard]] static auto normalize(const V& v) noexcept { return simd_normalize(v); }
	template <class V> [[nodiscard]] static auto clamp(const V& value, const V& low, const V& high) noexcept { return simd_clamp(value, low, high); }
	template <class V, class TExponent> [[nodiscard]] static auto pow(const V& base, TExponent exponent) noexcept { return simd::pow(base, V(exponent)); }

	/// @}
	
	///----------------------------------------
	/// @name Matrix construction
	///----------------------------------------
	/// @{
	
	template <class... C> [[nodiscard]] static auto matrixWithColumns(const C&... columns) noexcept { return simd_matrix(columns...); }
	template <class... R> [[nodiscard]] static auto matrixWithRows(const R&... rows) noexcept { return simd_matrix_from_rows(rows...); }
	
	/// @}
	
	///----------------------------------------
	/// @name Matrix, vector and quaternion products
	///----------------------------------------
	/// @{
	
	template <class A, class B> [[nodiscard]] static auto multiply(const A& a, const B& b) noexcept { return simd_mul(a, b); }
	template <class M> [[nodiscard]] static auto inverse(const M& m) noexcept { return simd_inverse(m); }
	template <class M> [[nodiscard]] static auto determinant(const M& m) noexcept { return simd_determinant(m); }
	template <class M> [[nodiscard]] static auto transpose(const M& m) noexcept { return simd_transpose(m); }
	
	/// @}
	
	///----------------------------------------
	/// @name Quaternion operations
	///----------------------------------------
	/// @{
	
	template <class... A> [[nodiscard]] static auto makeQuaternion(const A&... args) noexcept { return simd_quaternion(args...); }
	template <class Q> [[nodiscard]] static auto rotationMatrix3x3(const Q& q) noexcept { return simd_matrix3x3(q); }
	template <class Q> [[nodiscard]] static auto rotationMatrix4x4(const Q& q) noexcept { return simd_matrix4x4(q); }
	template <class Q, class V> [[nodiscard]] static auto rotate(const Q& q, const V& v) noexcept { return simd_act(q, v); }
	template <class Q> [[nodiscard]] static auto conjugate(const Q& q) noexcept { return simd_conjugate(q); }
	template <class Q, class T> [[nodiscard]] static auto slerp(const Q& from, const Q& to, T t) noexcept { return simd_slerp(from, to, t); }
	
	/// @}
};
	
} // namespace detail
	
} // namespace Math

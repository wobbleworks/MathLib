///----------------------------------------
///       @file SimdBackend.h
///    @ingroup MathLib
///      @brief Portable hardware-SIMD backend for the vector, matrix and quaternion operations that
///             @ref Vector.h, @ref Matrix.h and @ref Quaternion.h are built on.
///    @details Like @ref detail/GenericBackend.h this backend owns no platform library, but instead of
///             plain scalar-loop structs its native vector type is Clang's @c ext_vector_type compiler
///             vector extension (so it is Clang-only; GCC and others use @ref detail/GenericBackend.h).
///             Element-wise arithmetic and the matrix column combinations therefore lower to hardware
///             SIMD — NEON on ARM, SSE/AVX on x86 — while every convention (column-major storage,
///             @ref Backend::multiply of a matrix and vector as a column linear combination,
///             quaternion @c .vector = @c (ix,iy,iz,real), and the exact numerics) is copied from
///             @ref detail/GenericBackend.h so the MathLib self-tests pass unchanged on every backend.
///             The native types stay standard-layout with the scalars at offset 0, so
///             @ref Math::detail::VectorStorage can overlay them with named @c x/y/z[/w] fields. Only
///             one backend is ever included.
///     @author Created by John Stephen on 7/8/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <cmath>
#include <cstdint>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief A logical @p Count-element vector of @p TScalar laid out as a hardware SIMD register.
/// @details Clang's @c ext_vector_type gives an exact @p Count-element type — a 3-vector is a distinct
///          type from a 4-vector, its storage padded to 4 lanes (matching Apple's @c simd_double3) — so
///          @p TScalar and @p Count stay deducible in the @ref Backend operations, the scalars start at
///          offset 0, and element-wise @c + @c - @c * @c /, scalar broadcast, @c [i] subscripting and
///          braced initialization all lower to hardware SIMD. This backend is Clang-only: GCC's
///          @c vector_size rounds the byte size to a power of two, which would make a 3- and a 4-vector
///          the same type with a lane count no longer deducible, so GCC and every other compiler use
///          @ref detail/GenericBackend.h instead (chosen in @ref detail/BackendSelect.h).
///----------------------------------------

#if !defined(__clang__)
#error "SimdBackend.h requires Clang's ext_vector_type; GCC and other compilers use GenericBackend.h."
#endif

template <class TScalar, int Count>
using SimdVector = TScalar __attribute__((ext_vector_type(Count)));

///----------------------------------------
/// @brief A column-major square matrix: @c columns[c] is column @c c (a native SIMD vector), so
///        @c columns[c][r] is row @c r of column @c c and a @c reinterpret_cast to the scalar type
///        yields the column-major element buffer that GPU upload paths expect. Standard-layout.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct SimdMatrix {
	SimdVector<TScalar, Rows> columns[Cols];
};

///----------------------------------------
/// @brief A unit rotation quaternion. Storage is @c .vector = @c (ix, iy, iz, real).
///----------------------------------------

template <class TScalar>
struct SimdQuaternion {
	SimdVector<TScalar, 4> vector;
};

///----------------------------------------
/// @brief The identity matrix, the value a default-constructed @ref Math::Matrix starts from.
///----------------------------------------

template <class T, int N>
[[nodiscard]] inline SimdMatrix<T, N, N> simdIdentityMatrix() noexcept {
	SimdMatrix<T, N, N> result{};
	for (int i = 0; i < N; ++i) result.columns[i][i] = T(1);
	return result;
}

///----------------------------------------
/// @brief Maps a (scalar, component-count) pair to its native SIMD vector type.
///----------------------------------------

template <class TScalar, int Count>
struct native_vector;

template <> struct native_vector<double, 2> { using type = SimdVector<double, 2>; };
template <> struct native_vector<double, 3> { using type = SimdVector<double, 3>; };
template <> struct native_vector<double, 4> { using type = SimdVector<double, 4>; };
template <> struct native_vector<float, 2> { using type = SimdVector<float, 2>; };
template <> struct native_vector<float, 3> { using type = SimdVector<float, 3>; };
template <> struct native_vector<float, 4> { using type = SimdVector<float, 4>; };
template <> struct native_vector<uint8_t, 3> { using type = SimdVector<uint8_t, 3>; };
template <> struct native_vector<uint8_t, 4> { using type = SimdVector<uint8_t, 4>; };

///----------------------------------------
/// @brief Maps a (scalar, rows, cols) triple to its native SIMD matrix type and identity.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct native_matrix;

template <> struct native_matrix<double, 2, 2> { using type = SimdMatrix<double, 2, 2>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<double, 2>(); } };
template <> struct native_matrix<double, 3, 3> { using type = SimdMatrix<double, 3, 3>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<double, 3>(); } };
template <> struct native_matrix<double, 4, 4> { using type = SimdMatrix<double, 4, 4>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<double, 4>(); } };
template <> struct native_matrix<float, 2, 2> { using type = SimdMatrix<float, 2, 2>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<float, 2>(); } };
template <> struct native_matrix<float, 3, 3> { using type = SimdMatrix<float, 3, 3>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<float, 3>(); } };
template <> struct native_matrix<float, 4, 4> { using type = SimdMatrix<float, 4, 4>; [[nodiscard]] static type identity() noexcept { return simdIdentityMatrix<float, 4>(); } };

///----------------------------------------
/// @brief Maps a scalar type to its native SIMD quaternion type.
///----------------------------------------

template <class TScalar>
struct native_quaternion;

template <> struct native_quaternion<double> { using type = SimdQuaternion<double>; };
template <> struct native_quaternion<float>  { using type = SimdQuaternion<float>; };

///----------------------------------------
/// @struct Backend
/// @brief The hardware-SIMD operation vocabulary: static methods over the @ref SimdVector /
///        @ref SimdMatrix / @ref SimdQuaternion native types. Element-wise work and matrix column
///        combinations use native vector arithmetic (this is what vectorizes); dot products are an
///        explicit lane sum and the determinant, inverse and quaternion math are the scalar formulas
///        copied verbatim from @ref detail/GenericBackend.h. The Apple and generic backends define a
///        same-named struct; only one is ever included.
///----------------------------------------

struct Backend {
	
	///----------------------------------------
	/// @name Vector operations
	///----------------------------------------
	/// @{
	
	template <class T, int N>
	[[nodiscard]] static T dot(const SimdVector<T, N>& a, const SimdVector<T, N>& b) noexcept {
		SimdVector<T, N> product = a * b;
		T sum = T(0);
		for (int i = 0; i < N; ++i) sum += product[i];
		return sum;
	}
	
	template <class T>
	[[nodiscard]] static SimdVector<T, 3> cross(const SimdVector<T, 3>& a, const SimdVector<T, 3>& b) noexcept {
		return SimdVector<T, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
	}
	
	template <class T, int N>
	[[nodiscard]] static T lengthSquared(const SimdVector<T, N>& a) noexcept {
		return dot(a, a);
	}
	
	template <class T, int N>
	[[nodiscard]] static T length(const SimdVector<T, N>& a) noexcept {
		return std::sqrt(lengthSquared(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static SimdVector<T, N> normalize(const SimdVector<T, N>& a) noexcept {
		return a * (T(1) / length(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static T distanceSquared(const SimdVector<T, N>& a, const SimdVector<T, N>& b) noexcept {
		return lengthSquared(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static T distance(const SimdVector<T, N>& a, const SimdVector<T, N>& b) noexcept {
		return length(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static SimdVector<T, N> clamp(const SimdVector<T, N>& value, const SimdVector<T, N>& low, const SimdVector<T, N>& high) noexcept {
		SimdVector<T, N> result = value;
		for (int i = 0; i < N; ++i) {
			T v = value[i];
			if (v < low[i]) v = low[i];
			if (v > high[i]) v = high[i];
			result[i] = v;
		}
		return result;
	}

	template <class T, int N>
	[[nodiscard]] static SimdVector<T, N> pow(const SimdVector<T, N>& base, T exponent) noexcept {
		SimdVector<T, N> result = base;
		for (int i = 0; i < N; ++i) result[i] = std::pow(base[i], exponent);
		return result;
	}

	/// @}
	
	///----------------------------------------
	/// @name Matrix construction
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 2, 2> matrixWithColumns(const SimdVector<T, 2>& c0, const SimdVector<T, 2>& c1) noexcept {
		SimdMatrix<T, 2, 2> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 3, 3> matrixWithColumns(const SimdVector<T, 3>& c0, const SimdVector<T, 3>& c1, const SimdVector<T, 3>& c2) noexcept {
		SimdMatrix<T, 3, 3> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 4, 4> matrixWithColumns(const SimdVector<T, 4>& c0, const SimdVector<T, 4>& c1, const SimdVector<T, 4>& c2, const SimdVector<T, 4>& c3) noexcept {
		SimdMatrix<T, 4, 4> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		result.columns[3] = c3;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 3, 3> matrixWithRows(const SimdVector<T, 3>& r0, const SimdVector<T, 3>& r1, const SimdVector<T, 3>& r2) noexcept {
		SimdMatrix<T, 3, 3> result;
		for (int c = 0; c < 3; ++c) {
			result.columns[c][0] = r0[c];
			result.columns[c][1] = r1[c];
			result.columns[c][2] = r2[c];
		}
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 4, 4> matrixWithRows(const SimdVector<T, 4>& r0, const SimdVector<T, 4>& r1, const SimdVector<T, 4>& r2, const SimdVector<T, 4>& r3) noexcept {
		SimdMatrix<T, 4, 4> result;
		for (int c = 0; c < 4; ++c) {
			result.columns[c][0] = r0[c];
			result.columns[c][1] = r1[c];
			result.columns[c][2] = r2[c];
			result.columns[c][3] = r3[c];
		}
		return result;
	}
	
	/// @}
	
	///----------------------------------------
	/// @name Matrix and vector products
	///----------------------------------------
	/// @{
	
	/// @brief Matrix times column vector as a linear combination of the matrix columns (vectorized).
	template <class T, int N>
	[[nodiscard]] static SimdVector<T, N> multiply(const SimdMatrix<T, N, N>& matrix, const SimdVector<T, N>& vector) noexcept {
		SimdVector<T, N> result = matrix.columns[0] * vector[0];
		for (int c = 1; c < N; ++c) result += matrix.columns[c] * vector[c];
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static SimdMatrix<T, N, N> multiply(const SimdMatrix<T, N, N>& a, const SimdMatrix<T, N, N>& b) noexcept {
		SimdMatrix<T, N, N> result;
		for (int c = 0; c < N; ++c) result.columns[c] = multiply(a, b.columns[c]);
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static SimdMatrix<T, N, N> transpose(const SimdMatrix<T, N, N>& matrix) noexcept {
		SimdMatrix<T, N, N> result;
		for (int c = 0; c < N; ++c)
			for (int r = 0; r < N; ++r)
				result.columns[c][r] = matrix.columns[r][c];
		return result;
	}
	
	/// @}
	
	///----------------------------------------
	/// @name Determinants (@c m(r,c) is @c columns[c][r])
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static T determinant(const SimdMatrix<T, 2, 2>& m) noexcept {
		return m.columns[0][0] * m.columns[1][1] - m.columns[1][0] * m.columns[0][1];
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const SimdMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1))
		     - at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0))
		     + at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const SimdMatrix<T, 4, 4>& matrix) noexcept {
		// Column-major flat buffer m[c*4 + r], matching the OpenGL / Apple element order.
		const T* m = reinterpret_cast<const T*>(&matrix);
		T c0 = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
		T c1 = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
		T c2 = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
		T c3 = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
		return m[0] * c0 + m[4] * c1 + m[8] * c2 + m[12] * c3;
	}
	
	/// @}
	
	///----------------------------------------
	/// @name Inverses (the 4x4 uses the cofactor/adjugate method on the column-major flat buffer)
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 2, 2> inverse(const SimdMatrix<T, 2, 2>& m) noexcept {
		T det = determinant(m);
		T invDet = T(1) / det;
		SimdMatrix<T, 2, 2> result;
		result.columns[0][0] = m.columns[1][1] * invDet;
		result.columns[0][1] = -m.columns[0][1] * invDet;
		result.columns[1][0] = -m.columns[1][0] * invDet;
		result.columns[1][1] = m.columns[0][0] * invDet;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 3, 3> inverse(const SimdMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		T det = determinant(m);
		T invDet = T(1) / det;
		SimdMatrix<T, 3, 3> result;
		// result(r,c) = cofactor(c,r) / det  (adjugate is the transpose of the cofactor matrix).
		auto cof = [&](int r, int c, T value) noexcept { result.columns[c][r] = value * invDet; };
		cof(0, 0, at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1));
		cof(0, 1, at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2));
		cof(0, 2, at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1));
		cof(1, 0, at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2));
		cof(1, 1, at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0));
		cof(1, 2, at(0, 2) * at(1, 0) - at(0, 0) * at(1, 2));
		cof(2, 0, at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
		cof(2, 1, at(0, 1) * at(2, 0) - at(0, 0) * at(2, 1));
		cof(2, 2, at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0));
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 4, 4> inverse(const SimdMatrix<T, 4, 4>& matrix) noexcept {
		const T* m = reinterpret_cast<const T*>(&matrix);
		T inv[16];
		inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
		inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
		inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
		inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
		inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
		inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
		inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
		inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
		inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
		inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
		inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
		inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
		inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
		inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
		inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
		inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
		
		T det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
		T invDet = T(1) / det;
		
		SimdMatrix<T, 4, 4> result;
		T* out = reinterpret_cast<T*>(&result);
		for (int i = 0; i < 16; ++i) out[i] = inv[i] * invDet;
		return result;
	}
	
	/// @}
	
	///----------------------------------------
	/// @name Quaternion operations
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> makeQuaternion(T ix, T iy, T iz, T real) noexcept {
		SimdQuaternion<T> result;
		result.vector = SimdVector<T, 4>{ix, iy, iz, real};
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> makeQuaternion(const SimdVector<T, 4>& vector) noexcept {
		SimdQuaternion<T> result;
		result.vector = vector;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> makeQuaternion(const SimdMatrix<T, 3, 3>& matrix) noexcept {
		auto at = [&](int r, int c) noexcept { return matrix.columns[c][r]; };
		T trace = at(0, 0) + at(1, 1) + at(2, 2);
		T x, y, z, w;
		if (trace > T(0)) {
			T s = std::sqrt(trace + T(1)) * T(2); // s = 4w
			w = T(0.25) * s;
			x = (at(2, 1) - at(1, 2)) / s;
			y = (at(0, 2) - at(2, 0)) / s;
			z = (at(1, 0) - at(0, 1)) / s;
		} else if (at(0, 0) > at(1, 1) && at(0, 0) > at(2, 2)) {
			T s = std::sqrt(T(1) + at(0, 0) - at(1, 1) - at(2, 2)) * T(2); // s = 4x
			w = (at(2, 1) - at(1, 2)) / s;
			x = T(0.25) * s;
			y = (at(0, 1) + at(1, 0)) / s;
			z = (at(0, 2) + at(2, 0)) / s;
		} else if (at(1, 1) > at(2, 2)) {
			T s = std::sqrt(T(1) + at(1, 1) - at(0, 0) - at(2, 2)) * T(2); // s = 4y
			w = (at(0, 2) - at(2, 0)) / s;
			x = (at(0, 1) + at(1, 0)) / s;
			y = T(0.25) * s;
			z = (at(1, 2) + at(2, 1)) / s;
		} else {
			T s = std::sqrt(T(1) + at(2, 2) - at(0, 0) - at(1, 1)) * T(2); // s = 4z
			w = (at(1, 0) - at(0, 1)) / s;
			x = (at(0, 2) + at(2, 0)) / s;
			y = (at(1, 2) + at(2, 1)) / s;
			z = T(0.25) * s;
		}
		return makeQuaternion(x, y, z, w);
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 3, 3> rotationMatrix3x3(const SimdQuaternion<T>& q) noexcept {
		T x = q.vector[0], y = q.vector[1], z = q.vector[2], w = q.vector[3];
		T xx = x * x, yy = y * y, zz = z * z;
		T xy = x * y, xz = x * z, yz = y * z;
		T wx = w * x, wy = w * y, wz = w * z;
		SimdMatrix<T, 3, 3> result;
		result.columns[0] = SimdVector<T, 3>{T(1) - T(2) * (yy + zz), T(2) * (xy + wz), T(2) * (xz - wy)};
		result.columns[1] = SimdVector<T, 3>{T(2) * (xy - wz), T(1) - T(2) * (xx + zz), T(2) * (yz + wx)};
		result.columns[2] = SimdVector<T, 3>{T(2) * (xz + wy), T(2) * (yz - wx), T(1) - T(2) * (xx + yy)};
		return result;
	}
	
	template <class T>
	[[nodiscard]] static SimdMatrix<T, 4, 4> rotationMatrix4x4(const SimdQuaternion<T>& q) noexcept {
		SimdMatrix<T, 3, 3> rotation = rotationMatrix3x3(q);
		SimdMatrix<T, 4, 4> result = simdIdentityMatrix<T, 4>();
		for (int c = 0; c < 3; ++c)
			for (int r = 0; r < 3; ++r)
				result.columns[c][r] = rotation.columns[c][r];
		return result;
	}
	
	/// @brief Rotates a 3-vector by a quaternion: @c v + 2u×(u×v + w·v) with @c u the imaginary part.
	template <class T>
	[[nodiscard]] static SimdVector<T, 3> rotate(const SimdQuaternion<T>& q, const SimdVector<T, 3>& v) noexcept {
		SimdVector<T, 3> u{q.vector[0], q.vector[1], q.vector[2]};
		T w = q.vector[3];
		SimdVector<T, 3> t = T(2) * cross(u, v);
		return v + w * t + cross(u, t);
	}
	
	template <class T>
	[[nodiscard]] static T dot(const SimdQuaternion<T>& a, const SimdQuaternion<T>& b) noexcept {
		return dot(a.vector, b.vector);
	}
	
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> conjugate(const SimdQuaternion<T>& q) noexcept {
		return makeQuaternion(-q.vector[0], -q.vector[1], -q.vector[2], q.vector[3]);
	}
	
	/// @brief The Hamilton product (apply @p b, then @p a).
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> multiply(const SimdQuaternion<T>& a, const SimdQuaternion<T>& b) noexcept {
		T ax = a.vector[0], ay = a.vector[1], az = a.vector[2], aw = a.vector[3];
		T bx = b.vector[0], by = b.vector[1], bz = b.vector[2], bw = b.vector[3];
		T x = aw * bx + ax * bw + ay * bz - az * by;
		T y = aw * by - ax * bz + ay * bw + az * bx;
		T z = aw * bz + ax * by - ay * bx + az * bw;
		T w = aw * bw - ax * bx - ay * by - az * bz;
		return makeQuaternion(x, y, z, w);
	}
	
	/// @brief Shortest-arc spherical linear interpolation.
	template <class T>
	[[nodiscard]] static SimdQuaternion<T> slerp(const SimdQuaternion<T>& from, const SimdQuaternion<T>& to, T t) noexcept {
		SimdVector<T, 4> a = from.vector;
		SimdVector<T, 4> b = to.vector;
		T cosTheta = dot(a, b);
		if (cosTheta < T(0)) {
			b = -b;
			cosTheta = -cosTheta;
		}
		
		if (cosTheta > T(0.9995)) {
			// Nearly parallel: normalized linear interpolation avoids the small-angle blow-up.
			SimdVector<T, 4> result = a + (b - a) * t;
			return makeQuaternion(result * (T(1) / length(result)));
		}
		
		T theta = std::acos(cosTheta);
		T sinTheta = std::sin(theta);
		T weightFrom = std::sin((T(1) - t) * theta) / sinTheta;
		T weightTo = std::sin(t * theta) / sinTheta;
		return makeQuaternion(a * weightFrom + b * weightTo);
	}
	
	/// @}
};
	
} // namespace detail
	
} // namespace Math

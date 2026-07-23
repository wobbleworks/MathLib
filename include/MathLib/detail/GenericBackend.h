///----------------------------------------
///       @file GenericBackend.h
///    @ingroup MathLib
///      @brief Portable (non-Apple) backend for the vector, matrix and quaternion operations that
///             @ref Vector.h, @ref Matrix.h and @ref Quaternion.h are built on.
///    @details Defined entirely with plain standard-layout structs — no @c <simd/simd.h>, no compiler
///             vector extensions (@c ext_vector_type / @c vector_size) — so it builds on Clang, GCC and
///             MSVC. Everything lives in @ref Math::detail: the native types are @ref GenericVector /
///             @ref GenericMatrix / @ref GenericQuaternion, the @ref native_vector / @ref native_matrix /
///             @ref native_quaternion traits map to them, and @ref Backend exposes the neutral
///             operation vocabulary as static methods. Every convention (column-major storage,
///             @ref Backend::multiply of a matrix and vector as a column linear combination,
///             @ref Backend::matrixWithColumns from columns, @ref Backend::matrixWithRows from rows,
///             quaternion @c .vector = @c (ix,iy,iz,real)) mirrors the Apple backend's semantics so the
///             MathLib self-tests pass unchanged on both paths.
///     @author Created by John Stephen on 7/7/26.
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
/// @brief A fixed-size numeric vector: standard-layout, scalars at offset 0, no leading padding, so
///        @ref Math::detail::VectorStorage can overlay it with named @c x/y/z[/w] fields.
///----------------------------------------

template <class TScalar, int Count>
struct GenericVector;

template <class TScalar>
struct GenericVector<TScalar, 2> {
	union {
		TScalar e[2];
		struct { TScalar x, y; };
	};
	GenericVector() noexcept : e{TScalar(0), TScalar(0)} {}
	GenericVector(TScalar splat) noexcept : e{splat, splat} {}
	GenericVector(TScalar x_, TScalar y_) noexcept : e{x_, y_} {}
	[[nodiscard]] TScalar& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] const TScalar& operator[](int index) const noexcept { return e[index]; }
};

template <class TScalar>
struct GenericVector<TScalar, 3> {
	union {
		TScalar e[3];
		struct { TScalar x, y, z; };
	};
	GenericVector() noexcept : e{TScalar(0), TScalar(0), TScalar(0)} {}
	GenericVector(TScalar splat) noexcept : e{splat, splat, splat} {}
	GenericVector(TScalar x_, TScalar y_, TScalar z_) noexcept : e{x_, y_, z_} {}
	[[nodiscard]] TScalar& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] const TScalar& operator[](int index) const noexcept { return e[index]; }
};

template <class TScalar>
struct GenericVector<TScalar, 4> {
	union {
		TScalar e[4];
		struct { TScalar x, y, z, w; };
	};
	GenericVector() noexcept : e{TScalar(0), TScalar(0), TScalar(0), TScalar(0)} {}
	GenericVector(TScalar splat) noexcept : e{splat, splat, splat, splat} {}
	GenericVector(TScalar x_, TScalar y_, TScalar z_, TScalar w_) noexcept : e{x_, y_, z_, w_} {}
	[[nodiscard]] TScalar& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] const TScalar& operator[](int index) const noexcept { return e[index]; }
};

///----------------------------------------
/// @brief Element-wise vector arithmetic and scalar broadcasts, found by ADL from the wrapper bodies.
///----------------------------------------

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator+(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] + b[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator-(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] - b[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator*(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] * b[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator/(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] / b[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator-(const GenericVector<T, N>& a) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = -a[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator*(const GenericVector<T, N>& a, T scale) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] * scale;
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator*(T scale, const GenericVector<T, N>& a) noexcept {
	return a * scale;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator/(const GenericVector<T, N>& a, T scale) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] / scale;
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator/(T scale, const GenericVector<T, N>& a) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = scale / a[i];
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator+(const GenericVector<T, N>& a, T scalar) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] + scalar;
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator+(T scalar, const GenericVector<T, N>& a) noexcept {
	return a + scalar;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator-(const GenericVector<T, N>& a, T scalar) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] - scalar;
	return result;
}

template <class T, int N>
[[nodiscard]] inline GenericVector<T, N> operator-(T scalar, const GenericVector<T, N>& a) noexcept {
	GenericVector<T, N> result;
	for (int i = 0; i < N; ++i) result[i] = scalar - a[i];
	return result;
}

template <class T, int N>
inline GenericVector<T, N>& operator+=(GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	for (int i = 0; i < N; ++i) a[i] += b[i];
	return a;
}

template <class T, int N>
inline GenericVector<T, N>& operator-=(GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
	for (int i = 0; i < N; ++i) a[i] -= b[i];
	return a;
}

template <class T, int N>
inline GenericVector<T, N>& operator*=(GenericVector<T, N>& a, T scale) noexcept {
	for (int i = 0; i < N; ++i) a[i] *= scale;
	return a;
}

template <class T, int N>
inline GenericVector<T, N>& operator/=(GenericVector<T, N>& a, T scale) noexcept {
	for (int i = 0; i < N; ++i) a[i] /= scale;
	return a;
}

///----------------------------------------
/// @brief A column-major square matrix: @c columns[c][r] is row @c r of column @c c, laid out as
///        @c C contiguous @c R-scalar columns so @c reinterpret_cast to the scalar type yields the
///        column-major element buffer that GPU upload paths expect.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct GenericMatrix {
	GenericVector<TScalar, Rows> columns[Cols];
};

///----------------------------------------
/// @brief A unit rotation quaternion. Storage is @c .vector = @c (ix, iy, iz, real).
///----------------------------------------

template <class TScalar>
struct GenericQuaternion {
	GenericVector<TScalar, 4> vector;
};

///----------------------------------------
/// @brief The identity matrix, the value a default-constructed @ref Math::Matrix starts from.
///----------------------------------------

template <class T, int N>
[[nodiscard]] inline GenericMatrix<T, N, N> genericIdentityMatrix() noexcept {
	GenericMatrix<T, N, N> result;
	for (int i = 0; i < N; ++i) result.columns[i][i] = T(1);
	return result;
}

///----------------------------------------
/// @brief Maps a (scalar, component-count) pair to its portable vector type.
///----------------------------------------

template <class TScalar, int Count>
struct native_vector;

template <> struct native_vector<double, 2> { using type = GenericVector<double, 2>; };
template <> struct native_vector<double, 3> { using type = GenericVector<double, 3>; };
template <> struct native_vector<double, 4> { using type = GenericVector<double, 4>; };
template <> struct native_vector<float, 2> { using type = GenericVector<float, 2>; };
template <> struct native_vector<float, 3> { using type = GenericVector<float, 3>; };
template <> struct native_vector<float, 4> { using type = GenericVector<float, 4>; };
template <> struct native_vector<uint8_t, 3> { using type = GenericVector<uint8_t, 3>; };
template <> struct native_vector<uint8_t, 4> { using type = GenericVector<uint8_t, 4>; };

///----------------------------------------
/// @brief Maps a (scalar, rows, cols) triple to its portable matrix type and identity.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct native_matrix;

template <> struct native_matrix<double, 2, 2> { using type = GenericMatrix<double, 2, 2>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<double, 2>(); } };
template <> struct native_matrix<double, 3, 3> { using type = GenericMatrix<double, 3, 3>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<double, 3>(); } };
template <> struct native_matrix<double, 4, 4> { using type = GenericMatrix<double, 4, 4>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<double, 4>(); } };
template <> struct native_matrix<float, 2, 2> { using type = GenericMatrix<float, 2, 2>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<float, 2>(); } };
template <> struct native_matrix<float, 3, 3> { using type = GenericMatrix<float, 3, 3>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<float, 3>(); } };
template <> struct native_matrix<float, 4, 4> { using type = GenericMatrix<float, 4, 4>; [[nodiscard]] static type identity() noexcept { return genericIdentityMatrix<float, 4>(); } };

///----------------------------------------
/// @brief Maps a scalar type to its portable quaternion type.
///----------------------------------------

template <class TScalar>
struct native_quaternion;

template <> struct native_quaternion<double> { using type = GenericQuaternion<double>; };
template <> struct native_quaternion<float>  { using type = GenericQuaternion<float>; };

///----------------------------------------
/// @struct Backend
/// @brief The portable operation vocabulary: static methods over the @ref GenericVector /
///        @ref GenericMatrix / @ref GenericQuaternion native types. The Apple backend defines a
///        same-named struct that forwards to the platform library; only one is ever included.
///----------------------------------------

struct Backend {
	
	///----------------------------------------
	/// @name Vector operations
	///----------------------------------------
	/// @{
	
	template <class T, int N>
	[[nodiscard]] static T dot(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
		T sum = T(0);
		for (int i = 0; i < N; ++i) sum += a[i] * b[i];
		return sum;
	}
	
	template <class T>
	[[nodiscard]] static GenericVector<T, 3> cross(const GenericVector<T, 3>& a, const GenericVector<T, 3>& b) noexcept {
		return GenericVector<T, 3>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}
	
	template <class T, int N>
	[[nodiscard]] static T lengthSquared(const GenericVector<T, N>& a) noexcept {
		return dot(a, a);
	}
	
	template <class T, int N>
	[[nodiscard]] static T length(const GenericVector<T, N>& a) noexcept {
		return std::sqrt(lengthSquared(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static GenericVector<T, N> normalize(const GenericVector<T, N>& a) noexcept {
		return a * (T(1) / length(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static T distanceSquared(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
		return lengthSquared(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static T distance(const GenericVector<T, N>& a, const GenericVector<T, N>& b) noexcept {
		return length(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static GenericVector<T, N> clamp(const GenericVector<T, N>& value, const GenericVector<T, N>& low, const GenericVector<T, N>& high) noexcept {
		GenericVector<T, N> result;
		for (int i = 0; i < N; ++i) {
			T v = value[i];
			if (v < low[i]) v = low[i];
			if (v > high[i]) v = high[i];
			result[i] = v;
		}
		return result;
	}

	template <class T, int N>
	[[nodiscard]] static GenericVector<T, N> pow(const GenericVector<T, N>& base, T exponent) noexcept {
		GenericVector<T, N> result;
		for (int i = 0; i < N; ++i) result[i] = std::pow(base[i], exponent);
		return result;
	}

	/// @}
	
	///----------------------------------------
	/// @name Matrix construction
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 2, 2> matrixWithColumns(const GenericVector<T, 2>& c0, const GenericVector<T, 2>& c1) noexcept {
		GenericMatrix<T, 2, 2> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 3, 3> matrixWithColumns(const GenericVector<T, 3>& c0, const GenericVector<T, 3>& c1, const GenericVector<T, 3>& c2) noexcept {
		GenericMatrix<T, 3, 3> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 4, 4> matrixWithColumns(const GenericVector<T, 4>& c0, const GenericVector<T, 4>& c1, const GenericVector<T, 4>& c2, const GenericVector<T, 4>& c3) noexcept {
		GenericMatrix<T, 4, 4> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		result.columns[3] = c3;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 3, 3> matrixWithRows(const GenericVector<T, 3>& r0, const GenericVector<T, 3>& r1, const GenericVector<T, 3>& r2) noexcept {
		GenericMatrix<T, 3, 3> result;
		for (int c = 0; c < 3; ++c) {
			result.columns[c][0] = r0[c];
			result.columns[c][1] = r1[c];
			result.columns[c][2] = r2[c];
		}
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 4, 4> matrixWithRows(const GenericVector<T, 4>& r0, const GenericVector<T, 4>& r1, const GenericVector<T, 4>& r2, const GenericVector<T, 4>& r3) noexcept {
		GenericMatrix<T, 4, 4> result;
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
	
	template <class T, int N>
	[[nodiscard]] static GenericVector<T, N> multiply(const GenericMatrix<T, N, N>& matrix, const GenericVector<T, N>& vector) noexcept {
		GenericVector<T, N> result;
		for (int r = 0; r < N; ++r) {
			T sum = T(0);
			for (int c = 0; c < N; ++c) sum += matrix.columns[c][r] * vector[c];
			result[r] = sum;
		}
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static GenericMatrix<T, N, N> multiply(const GenericMatrix<T, N, N>& a, const GenericMatrix<T, N, N>& b) noexcept {
		GenericMatrix<T, N, N> result;
		for (int c = 0; c < N; ++c) result.columns[c] = multiply(a, b.columns[c]);
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static GenericMatrix<T, N, N> transpose(const GenericMatrix<T, N, N>& matrix) noexcept {
		GenericMatrix<T, N, N> result;
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
	[[nodiscard]] static T determinant(const GenericMatrix<T, 2, 2>& m) noexcept {
		return m.columns[0][0] * m.columns[1][1] - m.columns[1][0] * m.columns[0][1];
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const GenericMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1))
		     - at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0))
		     + at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const GenericMatrix<T, 4, 4>& matrix) noexcept {
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
	[[nodiscard]] static GenericMatrix<T, 2, 2> inverse(const GenericMatrix<T, 2, 2>& m) noexcept {
		T det = determinant(m);
		T invDet = T(1) / det;
		GenericMatrix<T, 2, 2> result;
		result.columns[0][0] = m.columns[1][1] * invDet;
		result.columns[0][1] = -m.columns[0][1] * invDet;
		result.columns[1][0] = -m.columns[1][0] * invDet;
		result.columns[1][1] = m.columns[0][0] * invDet;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 3, 3> inverse(const GenericMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		T det = determinant(m);
		T invDet = T(1) / det;
		GenericMatrix<T, 3, 3> result;
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
	[[nodiscard]] static GenericMatrix<T, 4, 4> inverse(const GenericMatrix<T, 4, 4>& matrix) noexcept {
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
		
		GenericMatrix<T, 4, 4> result;
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
	[[nodiscard]] static GenericQuaternion<T> makeQuaternion(T ix, T iy, T iz, T real) noexcept {
		GenericQuaternion<T> result;
		result.vector = GenericVector<T, 4>(ix, iy, iz, real);
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericQuaternion<T> makeQuaternion(const GenericVector<T, 4>& vector) noexcept {
		GenericQuaternion<T> result;
		result.vector = vector;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericQuaternion<T> makeQuaternion(const GenericMatrix<T, 3, 3>& matrix) noexcept {
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
	[[nodiscard]] static GenericMatrix<T, 3, 3> rotationMatrix3x3(const GenericQuaternion<T>& q) noexcept {
		T x = q.vector.x, y = q.vector.y, z = q.vector.z, w = q.vector.w;
		T xx = x * x, yy = y * y, zz = z * z;
		T xy = x * y, xz = x * z, yz = y * z;
		T wx = w * x, wy = w * y, wz = w * z;
		GenericMatrix<T, 3, 3> result;
		result.columns[0] = GenericVector<T, 3>(T(1) - T(2) * (yy + zz), T(2) * (xy + wz), T(2) * (xz - wy));
		result.columns[1] = GenericVector<T, 3>(T(2) * (xy - wz), T(1) - T(2) * (xx + zz), T(2) * (yz + wx));
		result.columns[2] = GenericVector<T, 3>(T(2) * (xz + wy), T(2) * (yz - wx), T(1) - T(2) * (xx + yy));
		return result;
	}
	
	template <class T>
	[[nodiscard]] static GenericMatrix<T, 4, 4> rotationMatrix4x4(const GenericQuaternion<T>& q) noexcept {
		GenericMatrix<T, 3, 3> rotation = rotationMatrix3x3(q);
		GenericMatrix<T, 4, 4> result = genericIdentityMatrix<T, 4>();
		for (int c = 0; c < 3; ++c)
			for (int r = 0; r < 3; ++r)
				result.columns[c][r] = rotation.columns[c][r];
		return result;
	}
	
	/// @brief Rotates a 3-vector by a quaternion: @c v + 2u×(u×v + w·v) with @c u the imaginary part.
	template <class T>
	[[nodiscard]] static GenericVector<T, 3> rotate(const GenericQuaternion<T>& q, const GenericVector<T, 3>& v) noexcept {
		GenericVector<T, 3> u(q.vector.x, q.vector.y, q.vector.z);
		T w = q.vector.w;
		GenericVector<T, 3> t = T(2) * cross(u, v);
		return v + w * t + cross(u, t);
	}
	
	template <class T>
	[[nodiscard]] static T dot(const GenericQuaternion<T>& a, const GenericQuaternion<T>& b) noexcept {
		return dot(a.vector, b.vector);
	}
	
	template <class T>
	[[nodiscard]] static GenericQuaternion<T> conjugate(const GenericQuaternion<T>& q) noexcept {
		return makeQuaternion(-q.vector.x, -q.vector.y, -q.vector.z, q.vector.w);
	}
	
	/// @brief The Hamilton product (apply @p b, then @p a).
	template <class T>
	[[nodiscard]] static GenericQuaternion<T> multiply(const GenericQuaternion<T>& a, const GenericQuaternion<T>& b) noexcept {
		T ax = a.vector.x, ay = a.vector.y, az = a.vector.z, aw = a.vector.w;
		T bx = b.vector.x, by = b.vector.y, bz = b.vector.z, bw = b.vector.w;
		T x = aw * bx + ax * bw + ay * bz - az * by;
		T y = aw * by - ax * bz + ay * bw + az * bx;
		T z = aw * bz + ax * by - ay * bx + az * bw;
		T w = aw * bw - ax * bx - ay * by - az * bz;
		return makeQuaternion(x, y, z, w);
	}
	
	/// @brief Shortest-arc spherical linear interpolation.
	template <class T>
	[[nodiscard]] static GenericQuaternion<T> slerp(const GenericQuaternion<T>& from, const GenericQuaternion<T>& to, T t) noexcept {
		GenericVector<T, 4> a = from.vector;
		GenericVector<T, 4> b = to.vector;
		T cosTheta = dot(a, b);
		if (cosTheta < T(0)) {
			b = -b;
			cosTheta = -cosTheta;
		}
		
		if (cosTheta > T(0.9995)) {
			// Nearly parallel: normalized linear interpolation avoids the small-angle blow-up.
			GenericVector<T, 4> result = a + (b - a) * t;
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

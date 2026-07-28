///----------------------------------------
///       @file IntelBackend.h
///    @ingroup MathLib
///      @brief x86 hardware-SIMD backend for the vector, matrix and quaternion operations that
///             @ref Vector.h, @ref Matrix.h and @ref Quaternion.h are built on.
///    @details Like @ref detail/GenericBackend.h and @ref detail/SimdBackend.h this backend owns no
///             platform library, but instead of scalar-loop structs or Clang's @c ext_vector_type it
///             lays each native vector out over SSE2 intrinsics (@c <immintrin.h>) — the SAME intrinsic
///             API on MSVC, GCC and Clang — so MSVC and GCC on x86 get real hardware SIMD (the
///             ext_vector backend is Clang-only). Only baseline SSE2 is used (no AVX / @c _mm256_*), so
///             it runs on every x86-64 CPU (which guarantees SSE2) and the default MSVC @c /arch. Every
///             convention (column-major storage, @ref Backend::multiply of a matrix and vector as a
///             column linear combination, quaternion @c .vector = @c (ix,iy,iz,real), and the exact
///             numerics) is copied from @ref detail/GenericBackend.h so the MathLib self-tests pass
///             unchanged on every backend. The native types stay standard-layout with the scalars at
///             offset 0, so @ref Math::detail::VectorStorage can overlay them with named @c x/y/z[/w]
///             fields. Only one backend is ever included.
///     @author Created by John Stephen on 7/8/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#if !(defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#error "IntelBackend.h requires x86 (SSE2); other targets use GenericBackend.h."
#endif

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief A logical @p Count-element vector of @p TScalar laid out over an SSE2 register, standard-layout
///        with the scalars at offset 0 (no leading padding) so @ref Math::detail::VectorStorage can
///        overlay it with named @c x/y/z[/w] fields. The lane count @p Count is carried in the type, so a
///        3-vector is a distinct type from a 4-vector and @p TScalar / @p Count stay deducible in the
///        @ref Backend operations. Each specialization holds an intrinsic register unioned with a scalar
///        array @c e for @c [] access; padding lanes are kept zero and are never read by the operations.
/// @details The constructors are @c constexpr so @ref Math::Vector values can be built during constant
///          evaluation, where the intrinsics — ordinary runtime functions — cannot run. Each one
///          initializes the scalar array @c e (the union member a constant expression may then read)
///          and, at run time only, overwrites it with the equivalent register form; the array store is
///          dead there and the optimizer drops it, so the emitted code is unchanged.
///----------------------------------------

template <class TScalar, int Count>
struct IntelVector;

///----------------------------------------
/// @brief @c float vectors: one @c __m128 (four packed floats); the 2- and 3-vectors use the low lanes
///        with the unused high lanes held at zero.
///----------------------------------------

template <>
struct IntelVector<float, 2> {
	union {
		__m128 m;
		float e[4];
		struct { float x, y; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = _mm_setzero_ps(); }
	constexpr IntelVector(float splat) noexcept : e{splat, splat} { if (!std::is_constant_evaluated()) m = _mm_set_ps(0.0f, 0.0f, splat, splat); }
	constexpr IntelVector(float x_, float y_) noexcept : e{x_, y_} { if (!std::is_constant_evaluated()) m = _mm_set_ps(0.0f, 0.0f, y_, x_); }
	[[nodiscard]] constexpr float& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const float& operator[](int index) const noexcept { return e[index]; }
};

template <>
struct IntelVector<float, 3> {
	union {
		__m128 m;
		float e[4];
		struct { float x, y, z; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = _mm_setzero_ps(); }
	constexpr IntelVector(float splat) noexcept : e{splat, splat, splat} { if (!std::is_constant_evaluated()) m = _mm_set_ps(0.0f, splat, splat, splat); }
	constexpr IntelVector(float x_, float y_, float z_) noexcept : e{x_, y_, z_} { if (!std::is_constant_evaluated()) m = _mm_set_ps(0.0f, z_, y_, x_); }
	[[nodiscard]] constexpr float& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const float& operator[](int index) const noexcept { return e[index]; }
};

template <>
struct IntelVector<float, 4> {
	union {
		__m128 m;
		float e[4];
		struct { float x, y, z, w; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = _mm_setzero_ps(); }
	constexpr IntelVector(float splat) noexcept : e{splat, splat, splat, splat} { if (!std::is_constant_evaluated()) m = _mm_set1_ps(splat); }
	constexpr IntelVector(float x_, float y_, float z_, float w_) noexcept : e{x_, y_, z_, w_} { if (!std::is_constant_evaluated()) m = _mm_set_ps(w_, z_, y_, x_); }
	[[nodiscard]] constexpr float& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const float& operator[](int index) const noexcept { return e[index]; }
};

///----------------------------------------
/// @brief @c double 2-vector: one @c __m128d (two packed doubles).
///----------------------------------------

template <>
struct IntelVector<double, 2> {
	union {
		__m128d m;
		double e[2];
		struct { double x, y; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = _mm_setzero_pd(); }
	constexpr IntelVector(double splat) noexcept : e{splat, splat} { if (!std::is_constant_evaluated()) m = _mm_set1_pd(splat); }
	constexpr IntelVector(double x_, double y_) noexcept : e{x_, y_} { if (!std::is_constant_evaluated()) m = _mm_set_pd(y_, x_); }
	[[nodiscard]] constexpr double& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const double& operator[](int index) const noexcept { return e[index]; }
};

///----------------------------------------
/// @brief @c double 3- and 4-vectors: a pair of @c __m128d (@c lo = @c e[0],e[1]; @c hi = @c e[2],e[3]),
///        so four contiguous doubles; the 3-vector holds @c e[3] at zero.
///----------------------------------------

template <>
struct IntelVector<double, 3> {
	union {
		struct { __m128d lo, hi; } m;
		double e[4];
		struct { double x, y, z; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = {_mm_setzero_pd(), _mm_setzero_pd()}; }
	constexpr IntelVector(double splat) noexcept : e{splat, splat, splat} { if (!std::is_constant_evaluated()) m = {_mm_set1_pd(splat), _mm_set_pd(0.0, splat)}; }
	constexpr IntelVector(double x_, double y_, double z_) noexcept : e{x_, y_, z_} { if (!std::is_constant_evaluated()) m = {_mm_set_pd(y_, x_), _mm_set_pd(0.0, z_)}; }
	[[nodiscard]] constexpr double& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const double& operator[](int index) const noexcept { return e[index]; }
};

template <>
struct IntelVector<double, 4> {
	union {
		struct { __m128d lo, hi; } m;
		double e[4];
		struct { double x, y, z, w; };
	};
	constexpr IntelVector() noexcept : e{} { if (!std::is_constant_evaluated()) m = {_mm_setzero_pd(), _mm_setzero_pd()}; }
	constexpr IntelVector(double splat) noexcept : e{splat, splat, splat, splat} { if (!std::is_constant_evaluated()) m = {_mm_set1_pd(splat), _mm_set1_pd(splat)}; }
	constexpr IntelVector(double x_, double y_, double z_, double w_) noexcept : e{x_, y_, z_, w_} { if (!std::is_constant_evaluated()) m = {_mm_set_pd(y_, x_), _mm_set_pd(w_, z_)}; }
	[[nodiscard]] constexpr double& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const double& operator[](int index) const noexcept { return e[index]; }
};

///----------------------------------------
/// @brief @c uint8_t 3- and 4-vectors: plain scalar storage (packed colour, not a SIMD hot path) with the
///        same operation surface implemented per-component.
///----------------------------------------

template <>
struct IntelVector<uint8_t, 3> {
	union {
		uint8_t e[3];
	};
	constexpr IntelVector() noexcept : e{0, 0, 0} {}
	constexpr IntelVector(uint8_t splat) noexcept : e{splat, splat, splat} {}
	constexpr IntelVector(uint8_t x_, uint8_t y_, uint8_t z_) noexcept : e{x_, y_, z_} {}
	[[nodiscard]] constexpr uint8_t& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const uint8_t& operator[](int index) const noexcept { return e[index]; }
};

template <>
struct IntelVector<uint8_t, 4> {
	union {
		uint8_t e[4];
	};
	constexpr IntelVector() noexcept : e{0, 0, 0, 0} {}
	constexpr IntelVector(uint8_t splat) noexcept : e{splat, splat, splat, splat} {}
	constexpr IntelVector(uint8_t x_, uint8_t y_, uint8_t z_, uint8_t w_) noexcept : e{x_, y_, z_, w_} {}
	[[nodiscard]] constexpr uint8_t& operator[](int index) noexcept { return e[index]; }
	[[nodiscard]] constexpr const uint8_t& operator[](int index) const noexcept { return e[index]; }
};

///----------------------------------------
/// @brief Applies @p operation to each pair of logical lanes of @p a and @p b — the path the operators
///        below take during constant evaluation, where the intrinsics cannot run.
/// @details Only the @p Count logical lanes are written; a 3-vector's padding lane keeps the zero its
///          constructor left there, where the register path lets arithmetic garbage land in it. Neither
///          is ever read, so the two paths agree on every value the vector actually carries.
///----------------------------------------

template <class TScalar, int Count, class TOperation>
[[nodiscard]] constexpr IntelVector<TScalar, Count> applyToLanes(const IntelVector<TScalar, Count>& a, const IntelVector<TScalar, Count>& b, TOperation operation) noexcept {
	IntelVector<TScalar, Count> result;
	for (int lane = 0; lane < Count; ++lane) result[lane] = operation(a[lane], b[lane]);
	return result;
}

///----------------------------------------
/// @brief Applies @p operation to each logical lane of @p a — the unary and vector-with-scalar form of
///        the constant-evaluation path.
///----------------------------------------

template <class TScalar, int Count, class TOperation>
[[nodiscard]] constexpr IntelVector<TScalar, Count> applyToLanes(const IntelVector<TScalar, Count>& a, TOperation operation) noexcept {
	IntelVector<TScalar, Count> result;
	for (int lane = 0; lane < Count; ++lane) result[lane] = operation(a[lane]);
	return result;
}

///----------------------------------------
/// @brief Element-wise arithmetic and scalar broadcasts, found by ADL from the wrapper bodies. The
///        @c float family lowers to @c _mm_*_ps on the packed register, the @c double 3/4-vectors apply
///        the @c _mm_*_pd op to both halves, the @c double 2-vector to its single register, and the
///        @c uint8_t vectors run per-component. Padding lanes may pick up arithmetic garbage but are
///        never read by @ref Backend (dot is restricted to the logical lanes; the flat-buffer matrix
///        paths touch only the fully-packed 4-wide types). Every operator is @c constexpr: at run time
///        it is the register form below, and during constant evaluation — where no intrinsic can run —
///        it is the lane-by-lane form of @ref applyToLanes.
///----------------------------------------

///----------------------------------------
/// @name float vector operators (SSE2 packed-single)
///----------------------------------------
/// @{

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator+(const IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](float x, float y) { return x + y; });
	IntelVector<float, N> result;
	result.m = _mm_add_ps(a.m, b.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator-(const IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](float x, float y) { return x - y; });
	IntelVector<float, N> result;
	result.m = _mm_sub_ps(a.m, b.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator*(const IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](float x, float y) { return x * y; });
	IntelVector<float, N> result;
	result.m = _mm_mul_ps(a.m, b.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator/(const IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](float x, float y) { return x / y; });
	IntelVector<float, N> result;
	result.m = _mm_div_ps(a.m, b.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator-(const IntelVector<float, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [](float x) { return -x; });
	IntelVector<float, N> result;
	result.m = _mm_sub_ps(_mm_setzero_ps(), a.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator*(const IntelVector<float, N>& a, float scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](float x) { return x * scale; });
	IntelVector<float, N> result;
	result.m = _mm_mul_ps(a.m, _mm_set1_ps(scale));
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator*(float scale, const IntelVector<float, N>& a) noexcept {
	return a * scale;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator/(const IntelVector<float, N>& a, float scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](float x) { return x / scale; });
	IntelVector<float, N> result;
	result.m = _mm_div_ps(a.m, _mm_set1_ps(scale));
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator/(float scale, const IntelVector<float, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](float x) { return scale / x; });
	IntelVector<float, N> result;
	result.m = _mm_div_ps(_mm_set1_ps(scale), a.m);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator+(const IntelVector<float, N>& a, float scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](float x) { return x + scalar; });
	IntelVector<float, N> result;
	result.m = _mm_add_ps(a.m, _mm_set1_ps(scalar));
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator+(float scalar, const IntelVector<float, N>& a) noexcept {
	return a + scalar;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator-(const IntelVector<float, N>& a, float scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](float x) { return x - scalar; });
	IntelVector<float, N> result;
	result.m = _mm_sub_ps(a.m, _mm_set1_ps(scalar));
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<float, N> operator-(float scalar, const IntelVector<float, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](float x) { return scalar - x; });
	IntelVector<float, N> result;
	result.m = _mm_sub_ps(_mm_set1_ps(scalar), a.m);
	return result;
}

template <int N>
constexpr IntelVector<float, N>& operator+=(IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] += b[lane]; return a; }
	a.m = _mm_add_ps(a.m, b.m);
	return a;
}

template <int N>
constexpr IntelVector<float, N>& operator-=(IntelVector<float, N>& a, const IntelVector<float, N>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] -= b[lane]; return a; }
	a.m = _mm_sub_ps(a.m, b.m);
	return a;
}

template <int N>
constexpr IntelVector<float, N>& operator*=(IntelVector<float, N>& a, float scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] *= scale; return a; }
	a.m = _mm_mul_ps(a.m, _mm_set1_ps(scale));
	return a;
}

template <int N>
constexpr IntelVector<float, N>& operator/=(IntelVector<float, N>& a, float scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] /= scale; return a; }
	a.m = _mm_div_ps(a.m, _mm_set1_ps(scale));
	return a;
}

/// @}

///----------------------------------------
/// @name double 2-vector operators (SSE2 packed-double, single register)
///----------------------------------------
/// @{

[[nodiscard]] constexpr IntelVector<double, 2> operator+(const IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x + y; });
	IntelVector<double, 2> result;
	result.m = _mm_add_pd(a.m, b.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator-(const IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x - y; });
	IntelVector<double, 2> result;
	result.m = _mm_sub_pd(a.m, b.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator*(const IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x * y; });
	IntelVector<double, 2> result;
	result.m = _mm_mul_pd(a.m, b.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator/(const IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x / y; });
	IntelVector<double, 2> result;
	result.m = _mm_div_pd(a.m, b.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator-(const IntelVector<double, 2>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [](double x) { return -x; });
	IntelVector<double, 2> result;
	result.m = _mm_sub_pd(_mm_setzero_pd(), a.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator*(const IntelVector<double, 2>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return x * scale; });
	IntelVector<double, 2> result;
	result.m = _mm_mul_pd(a.m, _mm_set1_pd(scale));
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator*(double scale, const IntelVector<double, 2>& a) noexcept {
	return a * scale;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator/(const IntelVector<double, 2>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return x / scale; });
	IntelVector<double, 2> result;
	result.m = _mm_div_pd(a.m, _mm_set1_pd(scale));
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator/(double scale, const IntelVector<double, 2>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return scale / x; });
	IntelVector<double, 2> result;
	result.m = _mm_div_pd(_mm_set1_pd(scale), a.m);
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator+(const IntelVector<double, 2>& a, double scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return x + scalar; });
	IntelVector<double, 2> result;
	result.m = _mm_add_pd(a.m, _mm_set1_pd(scalar));
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator+(double scalar, const IntelVector<double, 2>& a) noexcept {
	return a + scalar;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator-(const IntelVector<double, 2>& a, double scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return x - scalar; });
	IntelVector<double, 2> result;
	result.m = _mm_sub_pd(a.m, _mm_set1_pd(scalar));
	return result;
}

[[nodiscard]] constexpr IntelVector<double, 2> operator-(double scalar, const IntelVector<double, 2>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return scalar - x; });
	IntelVector<double, 2> result;
	result.m = _mm_sub_pd(_mm_set1_pd(scalar), a.m);
	return result;
}

constexpr IntelVector<double, 2>& operator+=(IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < 2; ++lane) a[lane] += b[lane]; return a; }
	a.m = _mm_add_pd(a.m, b.m);
	return a;
}

constexpr IntelVector<double, 2>& operator-=(IntelVector<double, 2>& a, const IntelVector<double, 2>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < 2; ++lane) a[lane] -= b[lane]; return a; }
	a.m = _mm_sub_pd(a.m, b.m);
	return a;
}

constexpr IntelVector<double, 2>& operator*=(IntelVector<double, 2>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < 2; ++lane) a[lane] *= scale; return a; }
	a.m = _mm_mul_pd(a.m, _mm_set1_pd(scale));
	return a;
}

constexpr IntelVector<double, 2>& operator/=(IntelVector<double, 2>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < 2; ++lane) a[lane] /= scale; return a; }
	a.m = _mm_div_pd(a.m, _mm_set1_pd(scale));
	return a;
}

/// @}

///----------------------------------------
/// @name double 3/4-vector operators (SSE2 packed-double, both halves)
///----------------------------------------
/// @{

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator+(const IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x + y; });
	IntelVector<double, N> result;
	result.m.lo = _mm_add_pd(a.m.lo, b.m.lo);
	result.m.hi = _mm_add_pd(a.m.hi, b.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator-(const IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x - y; });
	IntelVector<double, N> result;
	result.m.lo = _mm_sub_pd(a.m.lo, b.m.lo);
	result.m.hi = _mm_sub_pd(a.m.hi, b.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator*(const IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x * y; });
	IntelVector<double, N> result;
	result.m.lo = _mm_mul_pd(a.m.lo, b.m.lo);
	result.m.hi = _mm_mul_pd(a.m.hi, b.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator/(const IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, b, [](double x, double y) { return x / y; });
	IntelVector<double, N> result;
	result.m.lo = _mm_div_pd(a.m.lo, b.m.lo);
	result.m.hi = _mm_div_pd(a.m.hi, b.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator-(const IntelVector<double, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [](double x) { return -x; });
	IntelVector<double, N> result;
	result.m.lo = _mm_sub_pd(_mm_setzero_pd(), a.m.lo);
	result.m.hi = _mm_sub_pd(_mm_setzero_pd(), a.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator*(const IntelVector<double, N>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return x * scale; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scale);
	result.m.lo = _mm_mul_pd(a.m.lo, s);
	result.m.hi = _mm_mul_pd(a.m.hi, s);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator*(double scale, const IntelVector<double, N>& a) noexcept {
	return a * scale;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator/(const IntelVector<double, N>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return x / scale; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scale);
	result.m.lo = _mm_div_pd(a.m.lo, s);
	result.m.hi = _mm_div_pd(a.m.hi, s);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator/(double scale, const IntelVector<double, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scale](double x) { return scale / x; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scale);
	result.m.lo = _mm_div_pd(s, a.m.lo);
	result.m.hi = _mm_div_pd(s, a.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator+(const IntelVector<double, N>& a, double scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return x + scalar; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scalar);
	result.m.lo = _mm_add_pd(a.m.lo, s);
	result.m.hi = _mm_add_pd(a.m.hi, s);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator+(double scalar, const IntelVector<double, N>& a) noexcept {
	return a + scalar;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator-(const IntelVector<double, N>& a, double scalar) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return x - scalar; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scalar);
	result.m.lo = _mm_sub_pd(a.m.lo, s);
	result.m.hi = _mm_sub_pd(a.m.hi, s);
	return result;
}

template <int N> requires (N == 3 || N == 4)
[[nodiscard]] constexpr IntelVector<double, N> operator-(double scalar, const IntelVector<double, N>& a) noexcept {
	if (std::is_constant_evaluated()) return applyToLanes(a, [scalar](double x) { return scalar - x; });
	IntelVector<double, N> result;
	__m128d s = _mm_set1_pd(scalar);
	result.m.lo = _mm_sub_pd(s, a.m.lo);
	result.m.hi = _mm_sub_pd(s, a.m.hi);
	return result;
}

template <int N> requires (N == 3 || N == 4)
constexpr IntelVector<double, N>& operator+=(IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] += b[lane]; return a; }
	a.m.lo = _mm_add_pd(a.m.lo, b.m.lo);
	a.m.hi = _mm_add_pd(a.m.hi, b.m.hi);
	return a;
}

template <int N> requires (N == 3 || N == 4)
constexpr IntelVector<double, N>& operator-=(IntelVector<double, N>& a, const IntelVector<double, N>& b) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] -= b[lane]; return a; }
	a.m.lo = _mm_sub_pd(a.m.lo, b.m.lo);
	a.m.hi = _mm_sub_pd(a.m.hi, b.m.hi);
	return a;
}

template <int N> requires (N == 3 || N == 4)
constexpr IntelVector<double, N>& operator*=(IntelVector<double, N>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] *= scale; return a; }
	__m128d s = _mm_set1_pd(scale);
	a.m.lo = _mm_mul_pd(a.m.lo, s);
	a.m.hi = _mm_mul_pd(a.m.hi, s);
	return a;
}

template <int N> requires (N == 3 || N == 4)
constexpr IntelVector<double, N>& operator/=(IntelVector<double, N>& a, double scale) noexcept {
	if (std::is_constant_evaluated()) { for (int lane = 0; lane < N; ++lane) a[lane] /= scale; return a; }
	__m128d s = _mm_set1_pd(scale);
	a.m.lo = _mm_div_pd(a.m.lo, s);
	a.m.hi = _mm_div_pd(a.m.hi, s);
	return a;
}

/// @}

///----------------------------------------
/// @name uint8_t vector operators (per-component)
///----------------------------------------
/// @{

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator+(const IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] + b[i];
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator-(const IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] - b[i];
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator*(const IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] * b[i];
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator/(const IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] / b[i];
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator-(const IntelVector<uint8_t, N>& a) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = uint8_t(-a[i]);
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator*(const IntelVector<uint8_t, N>& a, uint8_t scale) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] * scale;
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator*(uint8_t scale, const IntelVector<uint8_t, N>& a) noexcept {
	return a * scale;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator/(const IntelVector<uint8_t, N>& a, uint8_t scale) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] / scale;
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator/(uint8_t scale, const IntelVector<uint8_t, N>& a) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = scale / a[i];
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator+(const IntelVector<uint8_t, N>& a, uint8_t scalar) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] + scalar;
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator+(uint8_t scalar, const IntelVector<uint8_t, N>& a) noexcept {
	return a + scalar;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator-(const IntelVector<uint8_t, N>& a, uint8_t scalar) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = a[i] - scalar;
	return result;
}

template <int N>
[[nodiscard]] constexpr IntelVector<uint8_t, N> operator-(uint8_t scalar, const IntelVector<uint8_t, N>& a) noexcept {
	IntelVector<uint8_t, N> result;
	for (int i = 0; i < N; ++i) result[i] = scalar - a[i];
	return result;
}

template <int N>
constexpr IntelVector<uint8_t, N>& operator+=(IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	for (int i = 0; i < N; ++i) a[i] += b[i];
	return a;
}

template <int N>
constexpr IntelVector<uint8_t, N>& operator-=(IntelVector<uint8_t, N>& a, const IntelVector<uint8_t, N>& b) noexcept {
	for (int i = 0; i < N; ++i) a[i] -= b[i];
	return a;
}

template <int N>
constexpr IntelVector<uint8_t, N>& operator*=(IntelVector<uint8_t, N>& a, uint8_t scale) noexcept {
	for (int i = 0; i < N; ++i) a[i] *= scale;
	return a;
}

template <int N>
constexpr IntelVector<uint8_t, N>& operator/=(IntelVector<uint8_t, N>& a, uint8_t scale) noexcept {
	for (int i = 0; i < N; ++i) a[i] /= scale;
	return a;
}

/// @}

///----------------------------------------
/// @brief A column-major square matrix: @c columns[c] is column @c c (a native SSE2-backed vector), so
///        @c columns[c][r] is row @c r of column @c c and a @c reinterpret_cast to the scalar type yields
///        the column-major element buffer that GPU upload paths (and the 4x4 determinant/inverse) expect.
///        Standard-layout; the 4-wide columns hold exactly @c R contiguous scalars.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct IntelMatrix {
	IntelVector<TScalar, Rows> columns[Cols];
};

///----------------------------------------
/// @brief A unit rotation quaternion. Storage is @c .vector = @c (ix, iy, iz, real).
///----------------------------------------

template <class TScalar>
struct IntelQuaternion {
	IntelVector<TScalar, 4> vector;
};

///----------------------------------------
/// @brief The identity matrix, the value a default-constructed @ref Math::Matrix starts from.
///----------------------------------------

template <class T, int N>
[[nodiscard]] inline IntelMatrix<T, N, N> intelIdentityMatrix() noexcept {
	IntelMatrix<T, N, N> result;
	for (int i = 0; i < N; ++i) result.columns[i][i] = T(1);
	return result;
}

///----------------------------------------
/// @brief Maps a (scalar, component-count) pair to its native SSE2 vector type.
///----------------------------------------

template <class TScalar, int Count>
struct native_vector;

template <> struct native_vector<double, 2> { using type = IntelVector<double, 2>; };
template <> struct native_vector<double, 3> { using type = IntelVector<double, 3>; };
template <> struct native_vector<double, 4> { using type = IntelVector<double, 4>; };
template <> struct native_vector<float, 2> { using type = IntelVector<float, 2>; };
template <> struct native_vector<float, 3> { using type = IntelVector<float, 3>; };
template <> struct native_vector<float, 4> { using type = IntelVector<float, 4>; };
template <> struct native_vector<uint8_t, 3> { using type = IntelVector<uint8_t, 3>; };
template <> struct native_vector<uint8_t, 4> { using type = IntelVector<uint8_t, 4>; };

///----------------------------------------
/// @brief Maps a (scalar, rows, cols) triple to its native SSE2 matrix type and identity.
///----------------------------------------

template <class TScalar, int Rows, int Cols>
struct native_matrix;

template <> struct native_matrix<double, 2, 2> { using type = IntelMatrix<double, 2, 2>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<double, 2>(); } };
template <> struct native_matrix<double, 3, 3> { using type = IntelMatrix<double, 3, 3>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<double, 3>(); } };
template <> struct native_matrix<double, 4, 4> { using type = IntelMatrix<double, 4, 4>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<double, 4>(); } };
template <> struct native_matrix<float, 2, 2> { using type = IntelMatrix<float, 2, 2>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<float, 2>(); } };
template <> struct native_matrix<float, 3, 3> { using type = IntelMatrix<float, 3, 3>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<float, 3>(); } };
template <> struct native_matrix<float, 4, 4> { using type = IntelMatrix<float, 4, 4>; [[nodiscard]] static type identity() noexcept { return intelIdentityMatrix<float, 4>(); } };

///----------------------------------------
/// @brief Maps a scalar type to its native SSE2 quaternion type.
///----------------------------------------

template <class TScalar>
struct native_quaternion;

template <> struct native_quaternion<double> { using type = IntelQuaternion<double>; };
template <> struct native_quaternion<float>  { using type = IntelQuaternion<float>; };

///----------------------------------------
/// @struct Backend
/// @brief The x86 SSE2 operation vocabulary: static methods over the @ref IntelVector / @ref IntelMatrix
///        / @ref IntelQuaternion native types. Element-wise work and matrix column combinations use the
///        native vector arithmetic (this is what vectorizes); dot products are an explicit lane sum and
///        the determinant, inverse and quaternion math are the scalar formulas copied verbatim from
///        @ref detail/GenericBackend.h. The other backends define a same-named struct; only one is ever
///        included.
///----------------------------------------

struct Backend {
	
	///----------------------------------------
	/// @name Vector operations
	///----------------------------------------
	/// @{
	
	template <class T, int N>
	[[nodiscard]] static T dot(const IntelVector<T, N>& a, const IntelVector<T, N>& b) noexcept {
		T sum = T(0);
		for (int i = 0; i < N; ++i) sum += a[i] * b[i];
		return sum;
	}
	
	template <class T>
	[[nodiscard]] static IntelVector<T, 3> cross(const IntelVector<T, 3>& a, const IntelVector<T, 3>& b) noexcept {
		return IntelVector<T, 3>(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
	}
	
	template <class T, int N>
	[[nodiscard]] static T lengthSquared(const IntelVector<T, N>& a) noexcept {
		return dot(a, a);
	}
	
	template <class T, int N>
	[[nodiscard]] static T length(const IntelVector<T, N>& a) noexcept {
		return std::sqrt(lengthSquared(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static IntelVector<T, N> normalize(const IntelVector<T, N>& a) noexcept {
		return a * (T(1) / length(a));
	}
	
	template <class T, int N>
	[[nodiscard]] static T distanceSquared(const IntelVector<T, N>& a, const IntelVector<T, N>& b) noexcept {
		return lengthSquared(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static T distance(const IntelVector<T, N>& a, const IntelVector<T, N>& b) noexcept {
		return length(a - b);
	}
	
	template <class T, int N>
	[[nodiscard]] static IntelVector<T, N> clamp(const IntelVector<T, N>& value, const IntelVector<T, N>& low, const IntelVector<T, N>& high) noexcept {
		IntelVector<T, N> result = value;
		for (int i = 0; i < N; ++i) {
			T v = value[i];
			if (v < low[i]) v = low[i];
			if (v > high[i]) v = high[i];
			result[i] = v;
		}
		return result;
	}

	template <class T, int N>
	[[nodiscard]] static IntelVector<T, N> pow(const IntelVector<T, N>& base, T exponent) noexcept {
		IntelVector<T, N> result = base;
		for (int i = 0; i < N; ++i) result[i] = std::pow(base[i], exponent);
		return result;
	}

	/// @}
	
	///----------------------------------------
	/// @name Matrix construction
	///----------------------------------------
	/// @{
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 2, 2> matrixWithColumns(const IntelVector<T, 2>& c0, const IntelVector<T, 2>& c1) noexcept {
		IntelMatrix<T, 2, 2> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 3, 3> matrixWithColumns(const IntelVector<T, 3>& c0, const IntelVector<T, 3>& c1, const IntelVector<T, 3>& c2) noexcept {
		IntelMatrix<T, 3, 3> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 4, 4> matrixWithColumns(const IntelVector<T, 4>& c0, const IntelVector<T, 4>& c1, const IntelVector<T, 4>& c2, const IntelVector<T, 4>& c3) noexcept {
		IntelMatrix<T, 4, 4> result;
		result.columns[0] = c0;
		result.columns[1] = c1;
		result.columns[2] = c2;
		result.columns[3] = c3;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 3, 3> matrixWithRows(const IntelVector<T, 3>& r0, const IntelVector<T, 3>& r1, const IntelVector<T, 3>& r2) noexcept {
		IntelMatrix<T, 3, 3> result;
		for (int c = 0; c < 3; ++c) {
			result.columns[c][0] = r0[c];
			result.columns[c][1] = r1[c];
			result.columns[c][2] = r2[c];
		}
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 4, 4> matrixWithRows(const IntelVector<T, 4>& r0, const IntelVector<T, 4>& r1, const IntelVector<T, 4>& r2, const IntelVector<T, 4>& r3) noexcept {
		IntelMatrix<T, 4, 4> result;
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
	[[nodiscard]] static IntelVector<T, N> multiply(const IntelMatrix<T, N, N>& matrix, const IntelVector<T, N>& vector) noexcept {
		IntelVector<T, N> result = matrix.columns[0] * vector[0];
		for (int c = 1; c < N; ++c) result += matrix.columns[c] * vector[c];
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static IntelMatrix<T, N, N> multiply(const IntelMatrix<T, N, N>& a, const IntelMatrix<T, N, N>& b) noexcept {
		IntelMatrix<T, N, N> result;
		for (int c = 0; c < N; ++c) result.columns[c] = multiply(a, b.columns[c]);
		return result;
	}
	
	template <class T, int N>
	[[nodiscard]] static IntelMatrix<T, N, N> transpose(const IntelMatrix<T, N, N>& matrix) noexcept {
		IntelMatrix<T, N, N> result;
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
	[[nodiscard]] static T determinant(const IntelMatrix<T, 2, 2>& m) noexcept {
		return m.columns[0][0] * m.columns[1][1] - m.columns[1][0] * m.columns[0][1];
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const IntelMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1))
		     - at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0))
		     + at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
	}
	
	template <class T>
	[[nodiscard]] static T determinant(const IntelMatrix<T, 4, 4>& matrix) noexcept {
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
	[[nodiscard]] static IntelMatrix<T, 2, 2> inverse(const IntelMatrix<T, 2, 2>& m) noexcept {
		T det = determinant(m);
		T invDet = T(1) / det;
		IntelMatrix<T, 2, 2> result;
		result.columns[0][0] = m.columns[1][1] * invDet;
		result.columns[0][1] = -m.columns[0][1] * invDet;
		result.columns[1][0] = -m.columns[1][0] * invDet;
		result.columns[1][1] = m.columns[0][0] * invDet;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 3, 3> inverse(const IntelMatrix<T, 3, 3>& m) noexcept {
		auto at = [&](int r, int c) noexcept { return m.columns[c][r]; };
		T det = determinant(m);
		T invDet = T(1) / det;
		IntelMatrix<T, 3, 3> result;
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
	[[nodiscard]] static IntelMatrix<T, 4, 4> inverse(const IntelMatrix<T, 4, 4>& matrix) noexcept {
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
		
		IntelMatrix<T, 4, 4> result;
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
	[[nodiscard]] static IntelQuaternion<T> makeQuaternion(T ix, T iy, T iz, T real) noexcept {
		IntelQuaternion<T> result;
		result.vector = IntelVector<T, 4>(ix, iy, iz, real);
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelQuaternion<T> makeQuaternion(const IntelVector<T, 4>& vector) noexcept {
		IntelQuaternion<T> result;
		result.vector = vector;
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelQuaternion<T> makeQuaternion(const IntelMatrix<T, 3, 3>& matrix) noexcept {
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
	[[nodiscard]] static IntelMatrix<T, 3, 3> rotationMatrix3x3(const IntelQuaternion<T>& q) noexcept {
		T x = q.vector[0], y = q.vector[1], z = q.vector[2], w = q.vector[3];
		T xx = x * x, yy = y * y, zz = z * z;
		T xy = x * y, xz = x * z, yz = y * z;
		T wx = w * x, wy = w * y, wz = w * z;
		IntelMatrix<T, 3, 3> result;
		result.columns[0] = IntelVector<T, 3>(T(1) - T(2) * (yy + zz), T(2) * (xy + wz), T(2) * (xz - wy));
		result.columns[1] = IntelVector<T, 3>(T(2) * (xy - wz), T(1) - T(2) * (xx + zz), T(2) * (yz + wx));
		result.columns[2] = IntelVector<T, 3>(T(2) * (xz + wy), T(2) * (yz - wx), T(1) - T(2) * (xx + yy));
		return result;
	}
	
	template <class T>
	[[nodiscard]] static IntelMatrix<T, 4, 4> rotationMatrix4x4(const IntelQuaternion<T>& q) noexcept {
		IntelMatrix<T, 3, 3> rotation = rotationMatrix3x3(q);
		IntelMatrix<T, 4, 4> result = intelIdentityMatrix<T, 4>();
		for (int c = 0; c < 3; ++c)
			for (int r = 0; r < 3; ++r)
				result.columns[c][r] = rotation.columns[c][r];
		return result;
	}
	
	/// @brief Rotates a 3-vector by a quaternion: @c v + 2u×(u×v + w·v) with @c u the imaginary part.
	template <class T>
	[[nodiscard]] static IntelVector<T, 3> rotate(const IntelQuaternion<T>& q, const IntelVector<T, 3>& v) noexcept {
		IntelVector<T, 3> u(q.vector[0], q.vector[1], q.vector[2]);
		T w = q.vector[3];
		IntelVector<T, 3> t = T(2) * cross(u, v);
		return v + w * t + cross(u, t);
	}
	
	template <class T>
	[[nodiscard]] static T dot(const IntelQuaternion<T>& a, const IntelQuaternion<T>& b) noexcept {
		return dot(a.vector, b.vector);
	}
	
	template <class T>
	[[nodiscard]] static IntelQuaternion<T> conjugate(const IntelQuaternion<T>& q) noexcept {
		return makeQuaternion(-q.vector[0], -q.vector[1], -q.vector[2], q.vector[3]);
	}
	
	/// @brief The Hamilton product (apply @p b, then @p a).
	template <class T>
	[[nodiscard]] static IntelQuaternion<T> multiply(const IntelQuaternion<T>& a, const IntelQuaternion<T>& b) noexcept {
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
	[[nodiscard]] static IntelQuaternion<T> slerp(const IntelQuaternion<T>& from, const IntelQuaternion<T>& to, T t) noexcept {
		IntelVector<T, 4> a = from.vector;
		IntelVector<T, 4> b = to.vector;
		T cosTheta = dot(a, b);
		if (cosTheta < T(0)) {
			b = -b;
			cosTheta = -cosTheta;
		}
		
		if (cosTheta > T(0.9995)) {
			// Nearly parallel: normalized linear interpolation avoids the small-angle blow-up.
			IntelVector<T, 4> result = a + (b - a) * t;
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

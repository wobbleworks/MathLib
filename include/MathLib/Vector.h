///----------------------------------------
///       @file Vector.h
///    @ingroup MathLib
///      @brief Zero-overhead value wrapper over the backend vector type.
///    @details A thin, header-only abstraction whose only data member is the backend vector,
///             so it is trivially copyable and lowers to that type with no overhead. The public
///             API never names a backend type: it is chosen by the
///             @ref detail::native_vector trait and reached only through @ref Vector::native, the
///             single escape hatch, so the implementation can be retargeted to another platform.
///     @author Created by John Stephen on 6/18/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/detail/BackendSelect.h"

///----------------------------------------
/// @brief When 1 (the default), @ref Math::Vector and @ref Math::Matrix convert implicitly to and
///        from their backend type, so they interoperate with the raw backend types at API boundaries.
///        Define it to 0 to enforce explicit @c native() / @c fromNative at every boundary.
///----------------------------------------

#ifndef MATH_IMPLICIT_NATIVE
#define MATH_IMPLICIT_NATIVE 1
#endif

#include "MathLib/SelfTestCheck.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief Alias to the backend's native vector type for a (scalar, component-count) pair.
/// @details The @ref native_vector trait is declared and specialized by the selected backend
///          (@ref detail/AppleBackend.h or @ref detail/GenericBackend.h).
///----------------------------------------

template <class TScalar, int Count>
using native_vector_t = typename native_vector<TScalar, Count>::type;

///----------------------------------------
/// @brief Per-dimension storage: an anonymous union overlaying the backend vector with
///        named, mutable @c x/y/z[/w] fields (the same bytes). @ref Vector inherits this so callers
///        can read and write components directly while the operations use @c _native.
///----------------------------------------

template <class TScalar, int Count>
struct VectorStorage;

template <class TScalar>
struct VectorStorage<TScalar, 2> {
	using Native = native_vector_t<TScalar, 2>;
	union {
		Native _native;
		struct { TScalar x, y; };
	};
	constexpr VectorStorage() noexcept : _native{} {}
	constexpr VectorStorage(const Native& native) noexcept : _native(native) {}
};

template <class TScalar>
struct VectorStorage<TScalar, 3> {
	using Native = native_vector_t<TScalar, 3>;
	union {
		Native _native;
		struct { TScalar x, y, z; };
		struct { TScalar r, g, b; };
		struct { TScalar h, s, v; };
	};
	constexpr VectorStorage() noexcept : _native{} {}
	constexpr VectorStorage(const Native& native) noexcept : _native(native) {}
};

template <class TScalar>
struct VectorStorage<TScalar, 4> {
	using Native = native_vector_t<TScalar, 4>;
	union {
		Native _native;
		struct { TScalar x, y, z, w; };
		struct { TScalar r, g, b, a; };
		struct { TScalar h, s, v, l; };
	};
	constexpr VectorStorage() noexcept : _native{} {}
	constexpr VectorStorage(const Native& native) noexcept : _native(native) {}
};
	
} // namespace detail

///----------------------------------------
/// @class Vector
/// @brief A fixed-size numeric vector backed by the native backend type.
/// @tparam TScalar Component scalar type (@c float or @c double).
/// @tparam Count Number of components (2, 3, or 4).
///----------------------------------------

template <class TScalar, int Count>
class Vector : public detail::VectorStorage<TScalar, Count> {
	static_assert(Count >= 2 && Count <= 4, "Vector supports 2 to 4 components");
	
public:
	using Scalar = TScalar;
	using Native = detail::native_vector_t<TScalar, Count>;
	using Base = detail::VectorStorage<TScalar, Count>;
	using Base::_native;
	
	static constexpr int count = Count;
	
	///----------------------------------------
	/// @brief Constructs the zero vector.
	///----------------------------------------
	
	constexpr Vector() noexcept = default;
	
	///----------------------------------------
	/// @brief Constructs a 2-component vector.
	///----------------------------------------
	
	constexpr Vector(Scalar x, Scalar y) noexcept requires (Count == 2) : Base(Native{x, y}) {}
	
	///----------------------------------------
	/// @brief Constructs a 3-component vector.
	///----------------------------------------
	
	constexpr Vector(Scalar x, Scalar y, Scalar z) noexcept requires (Count == 3) : Base(Native{x, y, z}) {}
	
	///----------------------------------------
	/// @brief Constructs a 4-component vector.
	///----------------------------------------
	
	constexpr Vector(Scalar x, Scalar y, Scalar z, Scalar w) noexcept requires (Count == 4) : Base(Native{x, y, z, w}) {}
	
	///----------------------------------------
	/// @brief Wraps a native backend vector.
	///----------------------------------------
	
	[[nodiscard]] constexpr static Vector fromNative(const Native& native) noexcept {
		Vector result;
		result._native = native;
		return result;
	}
	
#if MATH_IMPLICIT_NATIVE
	///----------------------------------------
	/// @brief Implicitly wraps a native backend vector (interop with raw backend types, gated by @c MATH_IMPLICIT_NATIVE).
	///----------------------------------------
	
	constexpr Vector(const Native& native) noexcept : Base(native) {}
	
	///----------------------------------------
	/// @brief Implicitly converts to the native backend vector (interop with raw backend types).
	///----------------------------------------
	
	[[nodiscard]] constexpr operator const Native&() const noexcept { return _native; }
#endif
	
	/// @brief The underlying backend vector (the platform escape hatch).
	[[nodiscard]] constexpr const Native& native() const noexcept { return _native; }
	
	/// @brief The underlying backend vector, mutable.
	[[nodiscard]] constexpr Native& native() noexcept { return _native; }
	
	/// @brief The component at @p index.
	[[nodiscard]] constexpr Scalar operator[](int index) const noexcept { return _native[index]; }
	
	///----------------------------------------
	/// @brief Exact component-wise equality.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool operator==(const Vector& other) const noexcept {
		for (int i = 0; i < Count; ++i) {
			if (_native[i] != other._native[i]) {
				return false;
			}
		}
		return true;
	}
	
	/// @brief Adds @p other component-wise.
	constexpr Vector& operator+=(const Vector& other) noexcept { _native += other._native; return *this; }
	
	/// @brief Subtracts @p other component-wise.
	constexpr Vector& operator-=(const Vector& other) noexcept { _native -= other._native; return *this; }
	
	/// @brief Scales by @p scale.
	constexpr Vector& operator*=(Scalar scale) noexcept { _native *= scale; return *this; }
	
	/// @brief Divides by @p scale.
	constexpr Vector& operator/=(Scalar scale) noexcept { _native /= scale; return *this; }
};

///----------------------------------------
/// @brief Component-wise sum.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator+(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return Vector<T, N>::fromNative(a.native() + b.native());
}

///----------------------------------------
/// @brief Component-wise difference.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator-(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return Vector<T, N>::fromNative(a.native() - b.native());
}

///----------------------------------------
/// @brief Component-wise product (Hadamard).
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator*(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return Vector<T, N>::fromNative(a.native() * b.native());
}

///----------------------------------------
/// @brief Component-wise quotient.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator/(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return Vector<T, N>::fromNative(a.native() / b.native());
}

///----------------------------------------
/// @brief Negation.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator-(const Vector<T, N>& vector) noexcept {
	return Vector<T, N>::fromNative(-vector.native());
}

///----------------------------------------
/// @brief Scales a vector by a scalar.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator*(const Vector<T, N>& vector, std::type_identity_t<T> scale) noexcept {
	return Vector<T, N>::fromNative(vector.native() * scale);
}

///----------------------------------------
/// @brief Scales a vector by a scalar.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator*(std::type_identity_t<T> scale, const Vector<T, N>& vector) noexcept {
	return vector * scale;
}

///----------------------------------------
/// @brief Divides a vector by a scalar.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator/(const Vector<T, N>& vector, std::type_identity_t<T> scale) noexcept {
	return Vector<T, N>::fromNative(vector.native() / scale);
}

///----------------------------------------
/// @brief Divides a scalar by every component.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator/(std::type_identity_t<T> scale, const Vector<T, N>& vector) noexcept {
	return Vector<T, N>::fromNative(scale / vector.native());
}

///----------------------------------------
/// @brief Adds a scalar to every component.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator+(const Vector<T, N>& vector, std::type_identity_t<T> scalar) noexcept {
	return Vector<T, N>::fromNative(vector.native() + scalar);
}

///----------------------------------------
/// @brief Adds a scalar to every component.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator+(std::type_identity_t<T> scalar, const Vector<T, N>& vector) noexcept {
	return vector + scalar;
}

///----------------------------------------
/// @brief Subtracts a scalar from every component.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator-(const Vector<T, N>& vector, std::type_identity_t<T> scalar) noexcept {
	return Vector<T, N>::fromNative(vector.native() - scalar);
}

///----------------------------------------
/// @brief Subtracts every component from a scalar.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> operator-(std::type_identity_t<T> scalar, const Vector<T, N>& vector) noexcept {
	return Vector<T, N>::fromNative(scalar - vector.native());
}

///----------------------------------------
/// @brief Dot product.
///----------------------------------------

template <class T, int N>
[[nodiscard]] T dot(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return detail::Backend::dot(a.native(), b.native());
}

///----------------------------------------
/// @brief Cross product of two 3-component vectors.
///----------------------------------------

template <class T>
[[nodiscard]] Vector<T, 3> cross(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept {
	return Vector<T, 3>::fromNative(detail::Backend::cross(a.native(), b.native()));
}

///----------------------------------------
/// @brief Euclidean length.
///----------------------------------------

template <class T, int N>
[[nodiscard]] T length(const Vector<T, N>& vector) noexcept {
	return detail::Backend::length(vector.native());
}

///----------------------------------------
/// @brief Unit vector in the same direction.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Vector<T, N> normalize(const Vector<T, N>& vector) noexcept {
	return Vector<T, N>::fromNative(detail::Backend::normalize(vector.native()));
}

///----------------------------------------
/// @brief Whether two vectors agree to within a tolerance, component-wise.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr bool approxEqual(const Vector<T, N>& a, const Vector<T, N>& b, T tolerance) noexcept {
	for (int i = 0; i < N; ++i) {
		if (std::abs(a[i] - b[i]) > tolerance) {
			return false;
		}
	}
	return true;
}

///----------------------------------------
/// @brief Squared Euclidean length (avoids the square root).
///----------------------------------------

template <class T, int N>
[[nodiscard]] T lengthSquared(const Vector<T, N>& vector) noexcept {
	return detail::Backend::lengthSquared(vector.native());
}

///----------------------------------------
/// @brief Distance between two points.
///----------------------------------------

template <class T, int N>
[[nodiscard]] T distance(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return detail::Backend::distance(a.native(), b.native());
}

///----------------------------------------
/// @brief Squared distance between two points (avoids the square root).
///----------------------------------------

template <class T, int N>
[[nodiscard]] T distanceSquared(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return detail::Backend::distanceSquared(a.native(), b.native());
}

///----------------------------------------
/// @brief Linear interpolation @c a + (b - a) * t.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> lerp(const Vector<T, N>& a, const Vector<T, N>& b, T t) noexcept {
	return a + (b - a) * t;
}

///----------------------------------------
/// @brief Midpoint of two vectors.
///----------------------------------------

template <class T, int N>
[[nodiscard]] constexpr Vector<T, N> midpoint(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return (a + b) * T(0.5);
}

///----------------------------------------
/// @brief Component-wise clamp of each element into the inclusive [@p min, @p max] range.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Vector<T, N> clamp(const Vector<T, N>& value, const Vector<T, N>& min, const Vector<T, N>& max) noexcept {
	return Vector<T, N>::fromNative(detail::Backend::clamp(value.native(), min.native(), max.native()));
}

///----------------------------------------
/// @brief Dot product clamped to [-1, 1] (safe for @c acos of unit vectors).
///----------------------------------------

template <class T, int N>
[[nodiscard]] T clampedDot(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
	return std::clamp(dot(a, b), T(-1), T(1));
}

/// @brief True if every component is finite (neither NaN nor infinity).
template <class T, int N>
[[nodiscard]] bool isfinite(const Vector<T, N>& vector) noexcept {
	for (int index = 0; index < N; ++index) {
		if (!std::isfinite(vector[index])) {
			return false;
		}
	}
	return true;
}

/// @brief True if any component is NaN.
template <class T, int N>
[[nodiscard]] bool isnan(const Vector<T, N>& vector) noexcept {
	for (int index = 0; index < N; ++index) {
		if (std::isnan(vector[index])) {
			return true;
		}
	}
	return false;
}

///----------------------------------------
/// @brief Raises each component of @p base to the scalar @p exponent (component-wise @c std::pow).
///----------------------------------------

template <class T, int N>
[[nodiscard]] Vector<T, N> pow(const Vector<T, N>& base, std::type_identity_t<T> exponent) noexcept {
	return Vector<T, N>::fromNative(detail::Backend::pow(base.native(), exponent));
}

/// @brief Returns @p vector rotated about the X axis by @p radians (y' = y·cos + z·sin, z' = z·cos − y·sin).
template <class T>
[[nodiscard]] Vector<T, 3> rotatedAroundX(const Vector<T, 3>& vector, T radians) noexcept {
	T s = std::sin(radians);
	T c = std::cos(radians);
	return Vector<T, 3>(vector.x, vector.y * c + vector.z * s, vector.z * c - vector.y * s);
}

/// @brief Returns @p vector rotated about the Y axis by @p radians (x' = x·cos − z·sin, z' = z·cos + x·sin).
template <class T>
[[nodiscard]] Vector<T, 3> rotatedAroundY(const Vector<T, 3>& vector, T radians) noexcept {
	T s = std::sin(radians);
	T c = std::cos(radians);
	return Vector<T, 3>(vector.x * c - vector.z * s, vector.y, vector.z * c + vector.x * s);
}

/// @brief Returns @p vector rotated about the Z axis by @p radians (x' = x·cos + y·sin, y' = y·cos − x·sin).
template <class T>
[[nodiscard]] Vector<T, 3> rotatedAroundZ(const Vector<T, 3>& vector, T radians) noexcept {
	T s = std::sin(radians);
	T c = std::cos(radians);
	return Vector<T, 3>(vector.x * c + vector.y * s, vector.y * c - vector.x * s, vector.z);
}

using Double2 = Vector<double, 2>;
using Double3 = Vector<double, 3>;
using Double4 = Vector<double, 4>;
using Float2 = Vector<float, 2>;
using Float3 = Vector<float, 3>;
using Float4 = Vector<float, 4>;

using RGBColor = Float3;
using RGBAColor = Float4;

///----------------------------------------
/// @brief Verifies the hand-written axis rotations against known basis-vector turns.
/// @details Only @ref rotatedAroundX / @ref rotatedAroundY / @ref rotatedAroundZ are hand-written;
///          every other operation delegates to the backend and is out of scope. Throws
///          @ref Math::selftest::Failure on the first violation.
///----------------------------------------
// Compile-time verification of constexpr initialization and operations
static_assert([] {
	constexpr Float3 v{1.0f, 2.0f, 3.0f};
	constexpr Float3 w{4.0f, 5.0f, 6.0f};
	constexpr Float3 sum = v + w;
	constexpr Float3 scaled = v * 2.0f;
	return sum[0] == 5.0f && sum[1] == 7.0f && sum[2] == 9.0f &&
	       scaled[0] == 2.0f && scaled[1] == 4.0f && scaled[2] == 6.0f;
}(), "Vector must support constexpr initialization and basic arithmetic");

inline void vectorSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	constexpr double quarterTurn = std::numbers::pi / 2.0;
	
	// rotatedAroundX by 90° sends +Y to −Z; a zero angle leaves the vector unchanged.
	{
		const Double3 rotated = rotatedAroundX(Double3(0.0, 1.0, 0.0), quarterTurn);
		check(near(rotated.x, 0.0) && near(rotated.y, 0.0) && near(rotated.z, -1.0), "rotatedAroundX sends +Y to -Z");
		check(approxEqual(rotatedAroundX(Double3(0.0, 1.0, 0.0), 0.0), Double3(0.0, 1.0, 0.0), 1.0e-12), "rotatedAroundX by zero is identity");
	}
	
	// rotatedAroundY by 90° sends +Z to −X.
	{
		const Double3 rotated = rotatedAroundY(Double3(0.0, 0.0, 1.0), quarterTurn);
		check(near(rotated.x, -1.0) && near(rotated.y, 0.0) && near(rotated.z, 0.0), "rotatedAroundY sends +Z to -X");
		check(approxEqual(rotatedAroundY(Double3(0.0, 0.0, 1.0), 0.0), Double3(0.0, 0.0, 1.0), 1.0e-12), "rotatedAroundY by zero is identity");
	}
	
	// rotatedAroundZ by 90° sends +X to −Y.
	{
		const Double3 rotated = rotatedAroundZ(Double3(1.0, 0.0, 0.0), quarterTurn);
		check(near(rotated.x, 0.0) && near(rotated.y, -1.0) && near(rotated.z, 0.0), "rotatedAroundZ sends +X to -Y");
		check(approxEqual(rotatedAroundZ(Double3(1.0, 0.0, 0.0), 0.0), Double3(1.0, 0.0, 0.0), 1.0e-12), "rotatedAroundZ by zero is identity");
	}
}
	
} // namespace Math

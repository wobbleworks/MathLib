///----------------------------------------
///       @file Matrix.h
///    @ingroup MathLib
///      @brief Zero-overhead value wrapper over the backend matrix type.
///    @details A thin, header-only abstraction whose only data member is the backend matrix,
///             so it is trivially copyable and lowers to that type with no overhead. Storage is
///             column-major; @c a*b applies @c b then @c a, and @c m*v transforms a column vector.
///             The public API never names a backend type: it is chosen by the
///             @ref detail::native_matrix trait and reached only through @ref Matrix::native, the
///             single escape hatch, so the implementation can be retargeted to another platform.
///     @author Created by John Stephen on 6/18/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/detail/BackendSelect.h"

#include "MathLib/Vector.h"
#include "MathLib/SelfTestCheck.h"

#include <cmath>
#include <numbers>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief Alias to the backend's native matrix type for a (scalar, rows, cols) triple.
/// @details The @ref native_matrix trait (with its @c type alias and @c identity()) is declared and
///          specialized by the selected backend (@ref detail/AppleBackend.h or @ref detail/GenericBackend.h).
///----------------------------------------

template <class TScalar, int Rows, int Cols>
using native_matrix_t = typename native_matrix<TScalar, Rows, Cols>::type;
	
} // namespace detail

///----------------------------------------
/// @class Matrix
/// @brief A fixed-size, column-major matrix backed by the native backend type.
/// @tparam TScalar Element scalar type (@c float or @c double).
/// @tparam Rows Number of rows (2, 3, or 4).
/// @tparam Cols Number of columns; defaults to @p Rows. Only square matrices are supported.
///----------------------------------------

template <class TScalar, int Rows, int Cols = Rows>
class Matrix {
	static_assert(Rows == Cols, "Only square matrices are supported");
	
public:
	using Scalar = TScalar;
	using Native = detail::native_matrix_t<TScalar, Rows, Cols>;
	using Column = Vector<TScalar, Rows>;
	using Row = Vector<TScalar, Cols>;
	
	static constexpr int rows = Rows;
	static constexpr int cols = Cols;
	
	///----------------------------------------
	/// @brief Constructs the identity matrix.
	///----------------------------------------
	
	Matrix() noexcept : _native(detail::native_matrix<TScalar, Rows, Cols>::identity()) {}
	
	///----------------------------------------
	/// @brief The identity matrix.
	///----------------------------------------
	
	[[nodiscard]] static Matrix identity() noexcept { return Matrix(); }
	
	///----------------------------------------
	/// @brief Constructs a 2-column matrix from its columns.
	///----------------------------------------
	
	Matrix(Column c0, Column c1) noexcept requires (Cols == 2)
		: _native(detail::Backend::matrixWithColumns(c0.native(), c1.native())) {}
		
	///----------------------------------------
	/// @brief Constructs a 3-column matrix from its columns.
	///----------------------------------------
	
	Matrix(Column c0, Column c1, Column c2) noexcept requires (Cols == 3)
		: _native(detail::Backend::matrixWithColumns(c0.native(), c1.native(), c2.native())) {}
		
	///----------------------------------------
	/// @brief Constructs a 4-column matrix from its columns.
	///----------------------------------------
	
	Matrix(Column c0, Column c1, Column c2, Column c3) noexcept requires (Cols == 4)
		: _native(detail::Backend::matrixWithColumns(c0.native(), c1.native(), c2.native(), c3.native())) {}
		
	///----------------------------------------
	/// @brief Builds a 3-row matrix from its rows.
	///----------------------------------------
	
	[[nodiscard]] static Matrix fromRows(Row r0, Row r1, Row r2) noexcept requires (Rows == 3) {
		return fromNative(detail::Backend::matrixWithRows(r0.native(), r1.native(), r2.native()));
	}
	
	///----------------------------------------
	/// @brief Builds a 4-row matrix from its rows.
	///----------------------------------------
	
	[[nodiscard]] static Matrix fromRows(Row r0, Row r1, Row r2, Row r3) noexcept requires (Rows == 4) {
		return fromNative(detail::Backend::matrixWithRows(r0.native(), r1.native(), r2.native(), r3.native()));
	}
	
	///----------------------------------------
	/// @brief Builds a rotation of @p radians about the unit @p axis (Rodrigues' rotation formula).
	/// @details The 3×3 rotation occupies the upper-left block, with an identity fourth row and column.
	///          @p axis is assumed to be normalized.
	///----------------------------------------
	
	[[nodiscard]] static Matrix axisRotation(const Vector<TScalar, 3>& axis, TScalar radians) noexcept requires (Rows == 4) {
		const TScalar sine = std::sin(radians);
		const TScalar cosine = std::cos(radians);
		const TScalar oneMinusCosine = TScalar(1) - cosine;
		const TScalar x = axis.x;
		const TScalar y = axis.y;
		const TScalar z = axis.z;
		return Matrix(
			Column(x * x * oneMinusCosine + cosine,    y * x * oneMinusCosine - z * sine,   z * x * oneMinusCosine + y * sine,   TScalar(0)),
			Column(x * y * oneMinusCosine + z * sine,  y * y * oneMinusCosine + cosine,     z * y * oneMinusCosine - x * sine,   TScalar(0)),
			Column(x * z * oneMinusCosine - y * sine,  y * z * oneMinusCosine + x * sine,   z * z * oneMinusCosine + cosine,     TScalar(0)),
			Column(TScalar(0), TScalar(0), TScalar(0), TScalar(1)));
	}
	
	///----------------------------------------
	/// @brief Wraps a native backend matrix.
	///----------------------------------------
	
	[[nodiscard]] static Matrix fromNative(const Native& native) noexcept {
		Matrix result;
		result._native = native;
		return result;
	}
	
#if MATH_IMPLICIT_NATIVE
	///----------------------------------------
	/// @brief Implicitly wraps a native backend matrix (interop with raw backend types, gated by @c MATH_IMPLICIT_NATIVE).
	///----------------------------------------
	
	Matrix(const Native& native) noexcept : _native(native) {}
	
	///----------------------------------------
	/// @brief Implicitly converts to the native backend matrix (interop with raw backend types).
	///----------------------------------------
	
	[[nodiscard]] operator const Native&() const noexcept { return _native; }
#endif
	
	/// @brief The underlying backend matrix (the platform escape hatch).
	[[nodiscard]] const Native& native() const noexcept { return _native; }
	
	/// @brief The underlying backend matrix, mutable.
	[[nodiscard]] Native& native() noexcept { return _native; }
	
	/// @brief The column at @p index.
	[[nodiscard]] Column column(int index) const noexcept { return Column::fromNative(_native.columns[index]); }
	
	/// @brief The column at @p index.
	[[nodiscard]] Column operator[](int index) const noexcept { return column(index); }
	
	/// @brief The element at the given row and column.
	[[nodiscard]] Scalar operator()(int row, int col) const noexcept { return _native.columns[col][row]; }
	
	/// @brief A pointer to the column-major element storage (@c Rows*Cols contiguous scalars), for handing to GPU upload APIs.
	[[nodiscard]] const Scalar* data() const noexcept { return reinterpret_cast<const Scalar*>(&_native); }
	
	///----------------------------------------
	/// @brief Exact element-wise equality.
	///----------------------------------------
	
	[[nodiscard]] bool operator==(const Matrix& other) const noexcept {
		for (int col = 0; col < Cols; ++col) {
			for (int row = 0; row < Rows; ++row) {
				if (_native.columns[col][row] != other._native.columns[col][row]) {
					return false;
				}
			}
		}
		return true;
	}
	
private:
	Native _native;
};

///----------------------------------------
/// @brief Matrix product. The result applies @p b then @p a.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Matrix<T, N, N> operator*(const Matrix<T, N, N>& a, const Matrix<T, N, N>& b) noexcept {
	return Matrix<T, N, N>::fromNative(detail::Backend::multiply(a.native(), b.native()));
}

///----------------------------------------
/// @brief Element-wise scalar-precision cast between matrices (e.g. @c Double4x4 → @c Float4x4).
/// @details The cross-precision counterpart to the same-precision native conversions; converts
///          each element from @c TIn to @c TOut. Use at the GPU/storage edge that needs a narrower
///          (or wider) element type than the working matrix.
///----------------------------------------

template <class TOut, class TIn, int Rows, int Cols>
[[nodiscard]] Matrix<TOut, Rows, Cols> matrixCast(const Matrix<TIn, Rows, Cols>& matrix) noexcept {
	Matrix<TOut, Rows, Cols> result;
	for (int col = 0; col < Cols; ++col)
		for (int row = 0; row < Rows; ++row)
			result.native().columns[col][row] = TOut(matrix.native().columns[col][row]);
	return result;
}

///----------------------------------------
/// @brief Transforms a column vector by a matrix.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Vector<T, N> operator*(const Matrix<T, N, N>& matrix, const Vector<T, N>& vector) noexcept {
	return Vector<T, N>::fromNative(detail::Backend::multiply(matrix.native(), vector.native()));
}

///----------------------------------------
/// @brief Transpose.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Matrix<T, N, N> transpose(const Matrix<T, N, N>& matrix) noexcept {
	return Matrix<T, N, N>::fromNative(detail::Backend::transpose(matrix.native()));
}

///----------------------------------------
/// @brief Inverse.
///----------------------------------------

template <class T, int N>
[[nodiscard]] Matrix<T, N, N> inverse(const Matrix<T, N, N>& matrix) noexcept {
	return Matrix<T, N, N>::fromNative(detail::Backend::inverse(matrix.native()));
}

///----------------------------------------
/// @brief Determinant.
///----------------------------------------

template <class T, int N>
[[nodiscard]] T determinant(const Matrix<T, N, N>& matrix) noexcept {
	return detail::Backend::determinant(matrix.native());
}

///----------------------------------------
/// @brief Whether two matrices agree to within a tolerance, element-wise.
///----------------------------------------

template <class T, int N>
[[nodiscard]] bool approxEqual(const Matrix<T, N, N>& a, const Matrix<T, N, N>& b, T tolerance) noexcept {
	for (int col = 0; col < N; ++col) {
		for (int row = 0; row < N; ++row) {
			if (std::abs(a.native().columns[col][row] - b.native().columns[col][row]) > tolerance) {
				return false;
			}
		}
	}
	return true;
}

using Double2x2 = Matrix<double, 2>;
using Double3x3 = Matrix<double, 3>;
using Double4x4 = Matrix<double, 4>;
using Float2x2 = Matrix<float, 2>;
using Float3x3 = Matrix<float, 3>;
using Float4x4 = Matrix<float, 4>;

///----------------------------------------
/// @brief Verifies the hand-written @ref Matrix::axisRotation against known rotations.
/// @details Only @c axisRotation is hand-written; every other operation delegates to the backend and
///          is out of scope. Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void matrixSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	constexpr double quarterTurn = std::numbers::pi / 2.0;
	
	// axisRotation about +Z by 90° maps +X to −Y (the Rodrigues block is transposed relative to a
	// right-handed active rotation, so the sign is pinned by reading the formula).
	{
		const Double4x4 rotation = Double4x4::axisRotation(Double3(0.0, 0.0, 1.0), quarterTurn);
		const Double4 rotatedX = rotation * Double4(1.0, 0.0, 0.0, 0.0);
		check(near(rotatedX[0], 0.0) && near(rotatedX[1], -1.0) && near(rotatedX[2], 0.0), "axisRotation Z sends +X to -Y");
	}
	
	// axisRotation about +X by 90° maps +Y to −Z, consistent with the same convention.
	{
		const Double4x4 rotation = Double4x4::axisRotation(Double3(1.0, 0.0, 0.0), quarterTurn);
		const Double4 rotatedY = rotation * Double4(0.0, 1.0, 0.0, 0.0);
		check(near(rotatedY[0], 0.0) && near(rotatedY[1], 0.0) && near(rotatedY[2], -1.0), "axisRotation X sends +Y to -Z");
	}
	
	// A zero-angle rotation is the identity matrix.
	{
		const Double4x4 rotation = Double4x4::axisRotation(Double3(0.0, 0.0, 1.0), 0.0);
		check(approxEqual(rotation, Double4x4::identity(), 1.0e-12), "zero-angle axisRotation is identity");
	}
}
	
} // namespace Math

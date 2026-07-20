///----------------------------------------
/// @file Tridiagonal.h
/// @ingroup MathLib
/// @brief Thomas algorithm for tridiagonal systems (header-only).
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "MathLib/Blast.h"
#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

/// @brief Solve A x = rhs for a tridiagonal A by the Thomas algorithm.
/// @details Shared implementation behind the @c float and @c double overloads of @ref solveTridiagonal.
/// A is the n x n matrix whose bands are @p subDiagonal (below the diagonal), @p diagonal, and
/// @p superDiagonal (above it); n is @c diagonal.size(). The two off-diagonals each hold n-1
/// coefficients, indexed naturally: @c subDiagonal[i] and @c superDiagonal[i] flank @c diagonal[i+1]
/// and @c diagonal[i] respectively. Every operand is a @ref Blast::StridedSpan, so a band or the result
/// may address one column of a grid or one field of an array of structs in place, with no copy. The
/// forward and back sweeps are a serial recurrence and so run scalar; the strided operands are the
/// point, not vectorization. Fails (without completing @p result) if a pivot reaches zero, which happens
/// when A is singular or would need row pivoting the Thomas algorithm does not perform.
/// @tparam TFloat Floating-point element type.
/// @return True when the solution was written; false on a zero pivot.
template <class TFloat>
[[nodiscard]] bool solveTridiagonal(Blast::StridedSpan<const TFloat> subDiagonal, Blast::StridedSpan<const TFloat> diagonal,
                                    Blast::StridedSpan<const TFloat> superDiagonal, Blast::StridedSpan<const TFloat> rhs,
                                    Blast::StridedSpan<TFloat> result) noexcept {
	const std::size_t n = diagonal.size();
	assert(rhs.size() >= n);
	assert(result.size() >= n);
	assert(n < 2 || (subDiagonal.size() >= n - 1 && superDiagonal.size() >= n - 1));
	
	if (n == 0) {
		return true;
	}
	
	// Normalized super-diagonal from the forward sweep; small systems stay on the stack to keep the
	// per-solve cost allocation-free, which matters when many systems are solved each frame.
	constexpr std::size_t stackCapacity = 1024;
	TFloat stackScratch[stackCapacity];
	std::vector<TFloat> heapScratch;
	TFloat* gamma = stackScratch;
	if (n > stackCapacity) {
		heapScratch.resize(n);
		gamma = heapScratch.data();
	}
	
	TFloat beta = diagonal[0];
	if (beta == 0) {
		return false;
	}
	result[0] = rhs[0] / beta;
	
	// Forward elimination.
	for (std::size_t i = 1; i < n; ++i) {
		gamma[i] = superDiagonal[i - 1] / beta;
		beta = diagonal[i] - subDiagonal[i - 1] * gamma[i];
		if (beta == 0) {
			return false;
		}
		result[i] = (rhs[i] - subDiagonal[i - 1] * result[i - 1]) / beta;
	}
	
	// Back substitution.
	for (std::size_t i = n - 1; i-- > 0; ) {
		result[i] -= gamma[i + 1] * result[i + 1];
	}
	
	return true;
}

///----------------------------------------
} // namespace detail
///----------------------------------------

///----------------------------------------
/// @brief Solve the tridiagonal system A x = rhs by the Thomas algorithm.
/// @details A is the n x n matrix with bands @p subDiagonal (below the diagonal), @p diagonal, and
/// @p superDiagonal (above it), where n is @c diagonal.size(). Each off-diagonal holds n-1 coefficients
/// in natural order: @c subDiagonal[i] and @c superDiagonal[i] are the entries flanking @c diagonal[i+1]
/// and @c diagonal[i]. Operands are @ref Blast::StridedSpan views, so a @c std::vector, @c std::array, or
/// @c std::span converts implicitly as a contiguous run, while a strided view addresses one grid column
/// or struct field in place without copying. The solution is written to @p result.
/// @param subDiagonal   The n-1 coefficients below the diagonal.
/// @param diagonal      The n diagonal coefficients; its size sets n.
/// @param superDiagonal The n-1 coefficients above the diagonal.
/// @param rhs           The n right-hand-side values.
/// @param result        Output solution view (at least n elements).
/// @return True when the solution was written; false on a zero pivot (A singular or needing pivoting).
///----------------------------------------
[[nodiscard]] inline bool solveTridiagonal(Blast::StridedSpan<const double> subDiagonal, Blast::StridedSpan<const double> diagonal,
                                           Blast::StridedSpan<const double> superDiagonal, Blast::StridedSpan<const double> rhs,
                                           Blast::StridedSpan<double> result) noexcept {
	return detail::solveTridiagonal<double>(subDiagonal, diagonal, superDiagonal, rhs, result);
}

/// @copydoc solveTridiagonal(Blast::StridedSpan<const double>, Blast::StridedSpan<const double>, Blast::StridedSpan<const double>, Blast::StridedSpan<const double>, Blast::StridedSpan<double>)
[[nodiscard]] inline bool solveTridiagonal(Blast::StridedSpan<const float> subDiagonal, Blast::StridedSpan<const float> diagonal,
                                           Blast::StridedSpan<const float> superDiagonal, Blast::StridedSpan<const float> rhs,
                                           Blast::StridedSpan<float> result) noexcept {
	return detail::solveTridiagonal<float>(subDiagonal, diagonal, superDiagonal, rhs, result);
}

///----------------------------------------
/// @brief Verifies the Thomas solver against a known system, the strided path, and edge cases.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void tridiagonalSelfTest() {
	using selftest::check;
	
	// The 1D Poisson matrix tridiag(-1, 2, -1) applied to [1, 1, 1, 1] gives [1, 0, 0, 1], so that
	// right-hand side has the exact solution [1, 1, 1, 1].
	const std::vector<double> subDiagonal{-1.0, -1.0, -1.0};
	const std::vector<double> diagonal{2.0, 2.0, 2.0, 2.0};
	const std::vector<double> superDiagonal{-1.0, -1.0, -1.0};
	const std::vector<double> rhs{1.0, 0.0, 0.0, 1.0};
	
	{
		std::vector<double> solution(4);
		const bool solved = solveTridiagonal(subDiagonal, diagonal, superDiagonal, rhs, solution);
		check(solved, "Poisson solve succeeds");
		check(std::abs(solution[0] - 1.0) < 1.0e-12 && std::abs(solution[1] - 1.0) < 1.0e-12 &&
			std::abs(solution[2] - 1.0) < 1.0e-12 && std::abs(solution[3] - 1.0) < 1.0e-12,
			"Poisson solution is all ones");
	}
	
	// The solution may be scattered straight into one field of an array of structs, with no copy.
	{
		struct Point { double x, y, z; };
		std::array<Point, 4> points{};
		auto ys = Blast::field(points, &Point::y);
		const bool solved = solveTridiagonal(Blast::StridedSpan<const double>(subDiagonal),
			Blast::StridedSpan<const double>(diagonal), Blast::StridedSpan<const double>(superDiagonal),
			Blast::StridedSpan<const double>(rhs), ys);
		check(solved && std::abs(points[1].y - 1.0) < 1.0e-12 && std::abs(points[3].y - 1.0) < 1.0e-12,
			"strided result view is written in place");
	}
	
	// A zero leading pivot is reported rather than dividing by zero.
	{
		const std::vector<double> singularDiagonal{0.0, 2.0};
		const std::vector<double> offDiagonal{-1.0};
		std::vector<double> solution(2);
		check(!solveTridiagonal(offDiagonal, singularDiagonal, offDiagonal,
			std::vector<double>{1.0, 1.0}, solution), "zero pivot returns false");
	}
	
	// The float overload resolves and solves the same system.
	{
		const std::vector<float> subF{-1.0f, -1.0f, -1.0f};
		const std::vector<float> diagF{2.0f, 2.0f, 2.0f, 2.0f};
		const std::vector<float> superF{-1.0f, -1.0f, -1.0f};
		const std::vector<float> rhsF{1.0f, 0.0f, 0.0f, 1.0f};
		std::vector<float> solution(4);
		check(solveTridiagonal(subF, diagF, superF, rhsF, solution) &&
			std::abs(solution[0] - 1.0f) < 1.0e-5f, "float overload solves");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

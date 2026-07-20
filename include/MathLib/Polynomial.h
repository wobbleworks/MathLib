///----------------------------------------
/// @file Polynomial.h
/// @ingroup MathLib
/// @brief Closed-form real-root solvers for linear, quadratic, and cubic polynomials.
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief A small, fixed-capacity set of real roots with a known count.
/// @details Indexable and iterable up to @ref count (entries beyond it are
/// unspecified). Returned by the polynomial solvers. Kept an aggregate so a
/// solver can fill @ref value and @ref count directly.
/// @tparam TScalar  Root value type.
/// @tparam Capacity Maximum number of roots the source polynomial can have.
///----------------------------------------
template <class TScalar, int Capacity>
struct RealRoots final {
	std::array<TScalar, Capacity> value{};
	int count = 0;
	
	[[nodiscard]] constexpr int size() const noexcept {return count;}
	[[nodiscard]] constexpr bool empty() const noexcept {return count == 0;}
	[[nodiscard]] constexpr TScalar operator[](int index) const noexcept {return value[index];}
	[[nodiscard]] constexpr const TScalar *begin() const noexcept {return value.data();}
	[[nodiscard]] constexpr const TScalar *end() const noexcept {return value.data() + count;}
	
	/// @brief The same roots reordered ascending.
	/// @details Counts of two and three use a compare-exchange network, the only
	/// sizes the solvers in this header produce; anything larger falls back to a
	/// general sort.
	/// @return A copy with the live roots sorted.
	[[nodiscard]] constexpr RealRoots sorted() const noexcept {
		RealRoots result = *this;
		if constexpr (Capacity >= 2) {
			auto& roots = result.value;
			
			// One comparison feeding both selections, so an exchange costs a single
			// test and compiles to conditional moves rather than a branch.
			const auto testAndExchange = [&roots](int lower, int upper) noexcept {
				const TScalar left = roots[lower];
				const TScalar right = roots[upper];
				const bool ordered = left <= right;
				roots[lower] = ordered ? left : right;
				roots[upper] = ordered ? right : left;
			};

			if constexpr (Capacity >= 3) {
				if (result.count == 3) {
					testAndExchange(0, 1);
					testAndExchange(1, 2);
					testAndExchange(0, 1);
					return result;
				}
			}
			if (result.count == 2) {
				testAndExchange(0, 1);
				return result;
			}
			if constexpr (Capacity > 3) {
				if (result.count > 3) {
					std::sort(roots.begin(), roots.begin() + result.count);
				}
			}
		}
		return result;
	}
};

///----------------------------------------
/// @brief Real roots of the linear polynomial a*x + b.
/// @details A degenerate a == 0 has no root and reports none (a nonzero
/// constant genuinely has none; the identically-zero case is not distinguished).
/// @param a Linear coefficient.
/// @param b Constant coefficient.
/// @return Zero or one real root.
///----------------------------------------
template <class TType>
[[nodiscard]] RealRoots<TType, 1> solveLinear(TType a, TType b) noexcept {
	RealRoots<TType, 1> result;
	if (a != 0) {
		result.value[0] = -b / a;
		result.count = 1;
	}
	return result;
}

///----------------------------------------
/// @brief Real roots of the quadratic polynomial a*x^2 + b*x + c, in solver order.
/// @details A degenerate a == 0 falls back to @ref solveLinear. The two-root
/// case uses the numerically stable Citardauq form to avoid cancellation.
/// @param a Quadratic coefficient.
/// @param b Linear coefficient.
/// @param c Constant coefficient.
/// @return Zero, one, or two real roots.
///----------------------------------------
template <class TType>
[[nodiscard]] RealRoots<TType, 2> solveQuadratic(TType a, TType b, TType c) noexcept {
	RealRoots<TType, 2> result;
	
	if (a == 0) {
		auto linear = solveLinear(b, c);
		result.count = linear.count;
		result.value[0] = linear.value[0];
		return result;
	}
	
	auto discriminant = b * b - 4 * a * c;
	if (discriminant < 0) {
		return result;
	}
	if (discriminant == 0) {
		result.value[0] = -b / (2 * a);
		result.count = 1;
		return result;
	}
	
	auto q = (b + std::copysign(std::sqrt(discriminant), b)) / -2;
	result.value[0] = q / a;
	result.value[1] = c / q;
	result.count = 2;
	return result;
}

///----------------------------------------
/// @brief Real roots of the cubic polynomial a*x^3 + b*x^2 + c*x + d, in solver order.
/// @details A degenerate a == 0 falls back to @ref solveQuadratic. The
/// three-real-root branch uses the trigonometric form; the single-real-root
/// branch uses Cardano.
/// @param a Cubic coefficient.
/// @param b Quadratic coefficient.
/// @param c Linear coefficient.
/// @param d Constant coefficient.
/// @return One, two (degenerate), or three real roots.
///----------------------------------------
template <class TType>
[[nodiscard]] RealRoots<TType, 3> solveCubic(TType a, TType b, TType c, TType d) noexcept {
	RealRoots<TType, 3> result;
	
	if (a == 0) {
		auto quadratic = solveQuadratic(b, c, d);
		result.count = quadratic.count;
		for (int i = 0; i < quadratic.count; ++i) {
			result.value[i] = quadratic.value[i];
		}
		return result;
	}
	
	// Reduce to x^3 + A x^2 + B x + C by dividing through by a.
	auto A = b / a;
	auto B = c / a;
	auto C = d / a;
	
	auto Q = (A * A - 3 * B) / 9;
	auto R = (2 * A * A * A - 9 * A * B + 27 * C) / 54;
	auto discriminant = R * R - Q * Q * Q;
	
	if (discriminant < 0) {
		// Three distinct real roots.
		constexpr auto twoPi = 2 * std::numbers::pi;
		auto theta = std::acos(R / std::sqrt(Q * Q * Q));
		auto oneThirdA = A / 3;
		auto minusTwoSqrtQ = -2 * std::sqrt(Q);
		result.value[0] = minusTwoSqrtQ * std::cos(theta / 3) - oneThirdA;
		result.value[1] = minusTwoSqrtQ * std::cos((theta + twoPi) / 3) - oneThirdA;
		result.value[2] = minusTwoSqrtQ * std::cos((theta - twoPi) / 3) - oneThirdA;
		result.count = 3;
	} else {
		// One real root.
		auto S = -std::copysign(std::pow(std::abs(R) + std::sqrt(discriminant), TType(1) / TType(3)), R);
		auto T = S == 0 ? TType(0) : (Q / S);
		result.value[0] = S + T - A / 3;
		result.count = 1;
	}
	return result;
}

///----------------------------------------
/// @brief Verifies the closed-form solvers against polynomials with known roots.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void polynomialSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// Linear: 2t - 4 = 0 has the single root t = 2; a degenerate leading coefficient has no root.
	{
		const auto roots = solveLinear(2.0, -4.0);
		check(roots.count == 1 && near(roots[0], 2.0), "linear root");
		check(solveLinear(0.0, 5.0).count == 0, "degenerate linear reports no root");
	}
	
	// Quadratic: distinct, repeated, complex, and degenerate-to-linear cases.
	{
		const auto distinct = solveQuadratic(1.0, -3.0, 2.0).sorted();
		check(distinct.count == 2 && near(distinct[0], 1.0) && near(distinct[1], 2.0), "two real roots");
		const auto repeated = solveQuadratic(1.0, -2.0, 1.0);
		check(repeated.count == 1 && near(repeated[0], 1.0), "repeated root");
		check(solveQuadratic(1.0, 0.0, 1.0).count == 0, "negative discriminant has no real root");
		const auto asLinear = solveQuadratic(0.0, 2.0, -4.0);
		check(asLinear.count == 1 && near(asLinear[0], 2.0), "quadratic degenerates to linear");
	}
	
	// Cubic: three real roots (trigonometric branch), one real root (Cardano branch), and degenerate.
	{
		const auto three = solveCubic(1.0, -6.0, 11.0, -6.0).sorted();
		check(three.count == 3 && near(three[0], 1.0) && near(three[1], 2.0) && near(three[2], 3.0),
			"three real roots");
		const auto one = solveCubic(1.0, 0.0, 0.0, -8.0);
		check(one.count == 1 && near(one[0], 2.0), "single real cube root of 8");
		const auto asQuadratic = solveCubic(0.0, 1.0, -3.0, 2.0).sorted();
		check(asQuadratic.count == 2 && near(asQuadratic[0], 1.0) && near(asQuadratic[1], 2.0),
			"cubic degenerates to quadratic");
	}
	
	// sorted() reorders the live roots ascending without disturbing the count.
	{
		RealRoots<double, 3> unsorted;
		unsorted.value = {3.0, 1.0, 2.0};
		unsorted.count = 3;
		const auto ordered = unsorted.sorted();
		check(ordered[0] == 1.0 && ordered[1] == 2.0 && ordered[2] == 3.0, "sorted ascending");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

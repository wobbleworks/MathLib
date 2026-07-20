///----------------------------------------
/// @file CubicPolynomial.h
/// @ingroup MathLib
/// @brief Cubic polynomial with evaluation and root-finding.
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Polynomial.h"
#include "MathLib/SelfTestCheck.h"

#include <cassert>
#include <cmath>
#include <type_traits>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @class CubicPolynomial
/// @brief A cubic polynomial of the form a3*t^3 + a2*t^2 + a1*t + a0.
/// @details Provides evaluation, derivative evaluation, and root-finding.
/// The four-argument constructor interpolates four control points as a
/// uniform Catmull-Rom segment; @ref hermite and @ref nonuniformCatmullRom
/// build the other common cubic forms. Root-finding is a scalar-only
/// concern (roots are parameter values) and is constrained accordingly.
/// @tparam TType   Value type for coefficients.
/// @tparam TScalar Scalar type for the parameter (default: double).
///----------------------------------------
template <class TType, class TScalar = double>
class CubicPolynomial final {
///----------------------------------------
	TType _a0{}, _a1{}, _a2{}, _a3{};
public:
	CubicPolynomial() = default;
	
	/// @brief Interpolate four control points as a uniform Catmull-Rom segment.
	/// @details The curve runs from @p p2 to @p p3 over t in [0, 1], with @p p1 and
	///          @p p4 shaping the endpoint tangents (standard Catmull-Rom basis).
	/// @param p1 First control point.
	/// @param p2 Second control point (curve start).
	/// @param p3 Third control point (curve end).
	/// @param p4 Fourth control point.
	constexpr CubicPolynomial(TType p1, TType p2, TType p3, TType p4) noexcept {
		_a0 = p2;
		_a1 = TType(-0.5) * p1 + TType(0.5) * p3;
		_a2 = p1 - TType(2.5) * p2 + TType(2.0) * p3 - TType(0.5) * p4;
		_a3 = TType(-0.5) * p1 + TType(1.5) * p2 - TType(1.5) * p3 + TType(0.5) * p4;
	}
	
	/// @brief Create a polynomial directly from its coefficients a3, a2, a1, a0.
	/// @param a3 Cubic coefficient.
	/// @param a2 Quadratic coefficient.
	/// @param a1 Linear coefficient.
	/// @param a0 Constant coefficient.
	/// @return Constructed polynomial.
	[[nodiscard]] static constexpr CubicPolynomial fromCoefficients(TType a3, TType a2, TType a1, TType a0) noexcept {
		CubicPolynomial cubicPolynomial;
		cubicPolynomial._a3 = a3;
		cubicPolynomial._a2 = a2;
		cubicPolynomial._a1 = a1;
		cubicPolynomial._a0 = a0;
		return cubicPolynomial;
	}
	
	/// @brief Create a Hermite polynomial from endpoint values and tangents.
	/// @param x0 Value at t=0.
	/// @param x1 Value at t=1.
	/// @param t0 Tangent at t=0.
	/// @param t1 Tangent at t=1.
	/// @return Hermite polynomial.
	[[nodiscard]] static constexpr CubicPolynomial hermite(TType x0, TType x1, TType t0, TType t1) noexcept {
		CubicPolynomial cubicPolynomial;
		cubicPolynomial._a0 = x0;
		cubicPolynomial._a1 = t0;
		cubicPolynomial._a2 = x0 * -3 + x1 * 3 - t0 * 2 - t1;
		cubicPolynomial._a3 = x0 * 2 - x1 * 2 + t0 + t1;
		return cubicPolynomial;
	}
	
	/// @brief Create a non-uniform Catmull-Rom polynomial from four control points and knot intervals.
	/// @param x0  First control point.
	/// @param x1  Second control point.
	/// @param x2  Third control point.
	/// @param x3  Fourth control point.
	/// @param dt0 Knot interval between x0 and x1.
	/// @param dt1 Knot interval between x1 and x2.
	/// @param dt2 Knot interval between x2 and x3.
	/// @return Non-uniform Catmull-Rom polynomial.
	[[nodiscard]] static CubicPolynomial nonuniformCatmullRom(TType x0, TType x1, TType x2, TType x3, double dt0, double dt1, double dt2) noexcept {
		assert(!std::isnan(dt0));
		assert(!std::isnan(dt1));
		assert(!std::isnan(dt2));
		
		TType t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
		TType t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
		t1 *= dt1;
		t2 *= dt1;
		return hermite(x1, x2, t1, t2);
	}
	
	/// @brief The real roots of the first derivative (up to two), in solver order.
	/// @return The derivative's real roots.
	[[nodiscard]] RealRoots<TType, 2> derivativeRoots() const requires std::is_arithmetic_v<TType> {
		// derivative of a3 t^3 + a2 t^2 + a1 t + a0 is 3 a3 t^2 + 2 a2 t + a1
		return Math::solveQuadratic(3 * _a3, 2 * _a2, _a1);
	}
	
	/// @brief The derivative's real roots that lie in [0, 1), order preserved.
	/// @return The in-range derivative roots.
	[[nodiscard]] RealRoots<TType, 2> derivativeRootsInUnitInterval() const requires std::is_arithmetic_v<TType> {
		return unitIntervalSubset(derivativeRoots());
	}
	
	/// @brief The real roots of the polynomial (up to three), in ascending order.
	/// @return The polynomial's real roots.
	[[nodiscard]] RealRoots<TType, 3> roots() const requires std::is_arithmetic_v<TType> {
		return Math::solveCubic(_a3, _a2, _a1, _a0).sorted();
	}
	
	/// @brief The polynomial's real roots that lie in [0, 1), in ascending order.
	/// @return The in-range roots.
	[[nodiscard]] RealRoots<TType, 3> rootsInUnitInterval() const requires std::is_arithmetic_v<TType> {
		return unitIntervalSubset(roots());
	}
	
	/// @brief Evaluate the polynomial at parameter t.
	/// @param t Parameter value.
	/// @return Polynomial value at t.
	[[nodiscard]] constexpr TType eval(TScalar t) const noexcept {
		return t * (t * (t * _a3 + _a2) + _a1) + _a0;
	}
	
	/// @brief Evaluate the first derivative at parameter t.
	/// @param t Parameter value.
	/// @return Derivative value at t.
	[[nodiscard]] constexpr TType evalDerivative(TScalar t) const noexcept {
		// f (t) = a3 t^3 + a2 t^2 + a1 t + a0
		// f'(t) = 3 a3 t^2 + 2 a2 t + a1
		return _a1 + _a2 * 2 * t + _a3 * 3 * t * t;
	}
	
	/// @brief Evaluate the second derivative at parameter t.
	/// @param t Parameter value.
	/// @return Second derivative value at t.
	[[nodiscard]] constexpr TType evalSecondDerivative(TScalar t) const noexcept {
		// f  (t) = a3 t^3 + a2 t^2 + a1 t + a0
		// f''(t) = 6 a3 t + 2 a2
		return _a2 * 2 + _a3 * 6 * t;
	}
	
private:
	/// @brief The subset of @p all lying in [0, 1), order preserved.
	template <int Capacity>
	[[nodiscard]] static RealRoots<TType, Capacity> unitIntervalSubset(const RealRoots<TType, Capacity> &all) noexcept {
		RealRoots<TType, Capacity> result;
		for (int i = 0; i < all.count; ++i) {
			if (all.value[i] >= 0 && all.value[i] < 1) {
				result.value[result.count++] = all.value[i];
			}
		}
		return result;
	}
	
};

///----------------------------------------
/// @brief Verifies construction, evaluation, and root-finding against known references.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void cubicPolynomialSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// A uniform Catmull-Rom segment passes through its inner control points and stays linear for
	// collinear points.
	{
		const CubicPolynomial<double> segment(0.0, 1.0, 2.0, 3.0);
		check(near(segment.eval(0.0), 1.0), "Catmull-Rom eval(0) is the second control point");
		check(near(segment.eval(1.0), 2.0), "Catmull-Rom eval(1) is the third control point");
		check(near(segment.eval(0.5), 1.5), "Catmull-Rom is linear through collinear points");
	}
	
	// Hermite reproduces its endpoint values and tangents.
	{
		const auto curve = CubicPolynomial<double>::hermite(0.0, 1.0, 0.0, 0.0);
		check(near(curve.eval(0.0), 0.0) && near(curve.eval(1.0), 1.0), "Hermite endpoints");
		check(near(curve.evalDerivative(0.0), 0.0) && near(curve.evalDerivative(1.0), 0.0),
			"Hermite endpoint tangents");
	}
	
	// Coefficient form: t^3 and its first two derivatives at t = 2.
	{
		const auto cube = CubicPolynomial<double>::fromCoefficients(1.0, 0.0, 0.0, 0.0);
		check(near(cube.eval(2.0), 8.0), "t^3 evaluates to 8 at t = 2");
		check(near(cube.evalDerivative(2.0), 12.0), "d/dt t^3 = 3t^2 is 12 at t = 2");
		check(near(cube.evalSecondDerivative(2.0), 12.0), "d2/dt2 t^3 = 6t is 12 at t = 2");
	}
	
	// roots() delegates to the polynomial solver and returns ascending values.
	{
		const auto poly = CubicPolynomial<double>::fromCoefficients(1.0, -6.0, 11.0, -6.0);
		const auto roots = poly.roots();
		check(roots.count == 3 && near(roots[0], 1.0) && near(roots[1], 2.0) && near(roots[2], 3.0),
			"cubic roots ascending");
	}
	
	// Only the roots inside [0, 1) survive the unit-interval filter: (t - 1/4)(t - 3/4)(t - 2) keeps
	// {1/4, 3/4} and drops the root at 2. Roots are kept clear of the boundaries so the half-open
	// test is unambiguous.
	{
		const auto poly = CubicPolynomial<double>::fromCoefficients(1.0, -3.0, 2.1875, -0.375);
		const auto inUnit = poly.rootsInUnitInterval();
		check(inUnit.count == 2 && near(inUnit[0], 0.25) && near(inUnit[1], 0.75), "roots in [0, 1)");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

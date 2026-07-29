///----------------------------------------
///       @file SelfTest.h
///    @ingroup MathLib
///      @brief Registers the MathLib self-tests with the shared self-test registry.
///    @details Including this header in a debug build registers each MathLib module's inline
///             @c selfTest with @ref Core::selftest::Registry so it runs as part of
///             @ref Core::selftest::runAll. This is the only MathLib header that depends on CoreLib;
///             every module asserts through its own @ref Math::selftest::check (see SelfTestCheck.h),
///             so including a module header pulls in no test framework. Outside DEBUG builds the
///             registrars compile to no-ops.
///     @author Created by John Stephen on 6/18/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Angle.h"
#include "MathLib/BlackbodyColor.h"
#include "MathLib/Blast.h"
#include "MathLib/CatmullRom.h"
#include "MathLib/Color.h"
#include "MathLib/CubicPolynomial.h"
#include "MathLib/HalfFloat.h"
#include "MathLib/Matrix.h"
#include "MathLib/Plane.h"
#include "MathLib/Frustum.h"
#include "MathLib/OneEuroFilter.h"
#include "MathLib/PerlinNoise.h"
#include "MathLib/PolarCoordinates.h"
#include "MathLib/Polynomial.h"
#include "MathLib/Quaternion.h"
#include "MathLib/Random.h"
#include "MathLib/Rect.h"
#include "MathLib/RescalingMap.h"
#include "MathLib/Scalar.h"
#include "MathLib/SegmentedCatmullRom.h"
#include "MathLib/Spectrum.h"
#include "MathLib/Transforms.h"
#include "MathLib/Tridiagonal.h"
#include "MathLib/Vector.h"
#include "MathLib/XYZColor.h"

#include "CoreLib/SelfTestRegistry.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief Registers @ref Blast::Tests::selfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::Registrar<Blast::Tests> _blastSelfTest("Blast");

///----------------------------------------
/// @brief Registers @ref HalfFloat::selfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::Registrar<HalfFloat> _halfFloatSelfTest("HalfFloat");

///----------------------------------------
/// @brief Registers @ref PerlinNoise::selfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::Registrar<PerlinNoise> _perlinNoiseSelfTest("PerlinNoise");

///----------------------------------------
/// @brief Registers @ref Spectrum::selfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::Registrar<Spectrum> _spectrumSelfTest("Spectrum");

///----------------------------------------
/// @brief Registers @ref XYZColor::selfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::Registrar<XYZColor> _xyzColorSelfTest("XYZColor");

///----------------------------------------
/// @brief Registers @ref polynomialSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _polynomialSelfTest("Polynomial", polynomialSelfTest);

///----------------------------------------
/// @brief Registers @ref cubicPolynomialSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _cubicPolynomialSelfTest("CubicPolynomial", cubicPolynomialSelfTest);

///----------------------------------------
/// @brief Registers @ref tridiagonalSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _tridiagonalSelfTest("Tridiagonal", tridiagonalSelfTest);

///----------------------------------------
/// @brief Registers @ref quaternionSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _quaternionSelfTest("Quaternion", quaternionSelfTest);

///----------------------------------------
/// @brief Registers @ref matrixSelfTest (the hand-written axisRotation); runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _matrixSelfTest("Matrix", matrixSelfTest);

///----------------------------------------
/// @brief Registers @ref planeSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _planeSelfTest("Plane", planeSelfTest);

///----------------------------------------
/// @brief Registers @ref frustumSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _frustumSelfTest("Frustum", frustumSelfTest);

///----------------------------------------
/// @brief Registers @ref vectorSelfTest (the hand-written axis rotations); runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _vectorSelfTest("Vector", vectorSelfTest);

///----------------------------------------
/// @brief Registers @ref transformsSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _transformsSelfTest("Transforms", transformsSelfTest);

///----------------------------------------
/// @brief Verifies that composed Euler orientations survive frame changes through a single
///        matrix→quaternion extraction, including at the gimbal pole.
/// @details The camera path composes @ref yawPitchRollMatrix with a frame rotation and extracts the
///          quaternion once, conjugated (the camera stores the inverse rotation). Ground truth is the
///          quaternion's own Euler convention with its axes re-expressed through the frame change. A
///          yaw/pitch/roll decompose-recompose in the middle fails this at pitch ±90° — the classic
///          gimbal-lock flip — so the direct path must match to near machine precision and stay
///          O(ε)-continuous through the pole. Spans Quaternion and
///          Transforms, hence its home in the aggregator.
///----------------------------------------

inline void orientationFrameChangeSelfTest() {
	using selftest::check;
	constexpr double equivalenceTolerance = 1.0e-12;
	constexpr double halfPi = std::numbers::pi / 2;

	const auto axesOf = [](const Quaternion64 &quaternion) {
		Double3 x, y, z;
		quaternion.getLocalAxes(x, y, z);
		return Double3x3(x, y, z);
	};
	const auto maxAbsDifference = [](const Double3x3 &a, const Double3x3 &b) {
		double result = 0;
		for (int column = 0; column < 3; ++column) {
			for (int row = 0; row < 3; ++row) {
				result = std::max(result, std::abs(a[column][row] - b[column][row]));
			}
		}
		return result;
	};
	const auto composedOrientation = [](const Double4x4 &frame, double yaw, double pitch, double roll) {
		Quaternion64 quaternion;
		quaternion.setRotation(rotationMatrix3(frame * yawPitchRollMatrix(yaw, pitch, roll)));
		return quaternion.conjugate();
	};

	const Double4x4 frames[] = {
		Double4x4(),
		axisRotationMatrix<double>(Double3(0, 1, 0), 0.9),
		axisRotationMatrix<double>(normalize(Double3(0.3, 0.8, -0.5)), 0.9),
	};
	const double yaws[] = {0.0, 0.7, -0.7, 2.9};
	const double rolls[] = {0.0, 0.3, -0.3};
	const double epsilons[] = {1.0e-3, 1.0e-6, 1.0e-9};

	for (const Double4x4 &frame : frames) {
		Double3x3 frameRotation = rotationMatrix3(frame);
		for (double yaw : yaws) {
			for (double roll : rolls) {
				// Equivalence against ground truth at benign, near-pole and exact-pole pitches
				const double pitches[] = {0.3, -halfPi + 1.0e-3, -halfPi - 1.0e-3, -halfPi + 1.0e-6, -halfPi - 1.0e-6, -halfPi + 1.0e-9, -halfPi - 1.0e-9, -halfPi};
				for (double pitch : pitches) {
					Quaternion64 groundTruth;
					groundTruth.setRotationRad(yaw, pitch, roll);
					double error = maxAbsDifference(axesOf(composedOrientation(frame, yaw, pitch, roll)), frameRotation * axesOf(groundTruth));
					check(error < equivalenceTolerance, "a composed orientation matches the frame-changed Euler ground truth");
				}

				// Continuity: the pole is not a branch point for the direct path
				Double3x3 poleAxes = axesOf(composedOrientation(frame, yaw, -halfPi, roll));
				for (double epsilon : epsilons) {
					for (double sign : {1.0, -1.0}) {
						double difference = maxAbsDifference(axesOf(composedOrientation(frame, yaw, -halfPi + sign * epsilon, roll)), poleAxes);
						check(difference <= 10.0 * epsilon + 1.0e-12, "orientation axes are continuous through the gimbal pole");
					}
				}
			}
		}
	}
}

///----------------------------------------
/// @brief Registers @ref orientationFrameChangeSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _orientationFrameChangeSelfTest("OrientationFrameChange", orientationFrameChangeSelfTest);

///----------------------------------------
/// @brief Registers @ref polarCoordinatesSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _polarCoordinatesSelfTest("PolarCoordinates", polarCoordinatesSelfTest);

///----------------------------------------
/// @brief Registers @ref scalarSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _scalarSelfTest("Scalar", scalarSelfTest);

///----------------------------------------
/// @brief Registers @ref angleSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _angleSelfTest("Angle", angleSelfTest);

///----------------------------------------
/// @brief Registers @ref rectSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _rectSelfTest("Rect", rectSelfTest);

///----------------------------------------
/// @brief Registers @ref rescalingMapSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _rescalingMapSelfTest("RescalingMap", rescalingMapSelfTest);

///----------------------------------------
/// @brief Registers @ref catmullRomSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _catmullRomSelfTest("CatmullRom", catmullRomSelfTest);

///----------------------------------------
/// @brief Registers @ref segmentedCatmullRomSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _segmentedCatmullRomSelfTest("SegmentedCatmullRom", segmentedCatmullRomSelfTest);

///----------------------------------------
/// @brief Registers @ref colorSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _colorSelfTest("Color", colorSelfTest);

///----------------------------------------
/// @brief Registers @ref blackbodyColorSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _blackbodyColorSelfTest("BlackbodyColor", blackbodyColorSelfTest);

///----------------------------------------
/// @brief Registers @ref oneEuroFilterSelfTest; runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _oneEuroFilterSelfTest("OneEuroFilter", oneEuroFilterSelfTest);

///----------------------------------------
/// @brief Registers @ref math::randomSelfTest (Random.h is in the lowercase @c math namespace); runs once in a debug build.
///----------------------------------------

inline Core::selftest::FunctionTestRegistrar _randomSelfTest("Random", math::randomSelfTest);

///----------------------------------------
} // namespace Math
///----------------------------------------

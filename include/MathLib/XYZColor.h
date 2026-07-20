///----------------------------------------
///       @file XYZColor.h
///    @ingroup MathLib
///      @brief CIE 1931 XYZ and xyY color value types and their conversions.
///    @details Defines @ref XYZColor (tristimulus) and @ref xyYColor (chromaticity plus
///             luminance) as distinct SIMD-backed value types, with conversions between the
///             two spaces, to and from linear RGB (Rec. 709 primaries, D65 white point), and
///             the CIE 1931 2° standard-observer color-matching table, which is shared with
///             @ref Spectrum::toXYZ.
///     @author Created by John Stephen on 7/6/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/Vector.h"
#include "MathLib/SelfTestCheck.h"

#include <algorithm>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
///  @struct XYZColor
///   @brief A CIE 1931 XYZ tristimulus color.
/// @details The named components overlay a SIMD-backed @ref Float3, so component access and
///          vector math read the same bytes. Tristimulus values are linear, so accumulating
///          and scaling them is physically meaningful.
///----------------------------------------

struct XYZColor final {
	union {
		Float3 vec;                  ///< The tristimulus values as a vector, for SIMD math.
		struct { float X, Y, Z; };     ///< The CIE tristimulus components.
	};
	
	/// @brief Constructs black (all components zero).
	XYZColor() noexcept : vec{} {}
	
	/// @brief Constructs from tristimulus components.
	XYZColor(float X_, float Y_, float Z_) noexcept : vec(X_, Y_, Z_) {}
	
	/// @brief Wraps a vector of tristimulus values.
	explicit XYZColor(const Float3& vec_) noexcept : vec(vec_) {}
	
	/// @brief Adds another color component-wise.
	XYZColor& operator+=(const XYZColor& other) noexcept {
		vec += other.vec;
		return *this;
	}
	
	/// @brief Scales all components.
	XYZColor& operator*=(float scale) noexcept {
		vec *= scale;
		return *this;
	}
	
	///----------------------------------------
	/// @brief Validates the conversions and the color-matching table.
	///----------------------------------------
	
	static void selfTest();
};

///----------------------------------------
///  @struct xyYColor
///   @brief A CIE xyY color: chromaticity (x, y) plus luminance (Y).
/// @details The named components overlay a SIMD-backed @ref Float3. Chromaticity coordinates
///          are not linear, so the type deliberately provides no arithmetic — convert to
///          @ref XYZColor for math.
///----------------------------------------

struct xyYColor final {
	union {
		Float3 vec;                  ///< The components as a vector.
		struct { float x, y, Y; };     ///< The chromaticity (x, y) and luminance (Y) components.
	};
	
	/// @brief Constructs black (all components zero).
	xyYColor() noexcept : vec{} {}
	
	/// @brief Constructs from chromaticity and luminance.
	xyYColor(float x_, float y_, float Y_) noexcept : vec(x_, y_, Y_) {}
	
	/// @brief Wraps a vector of (x, y, Y) components.
	explicit xyYColor(const Float3& vec_) noexcept : vec(vec_) {}
};

///----------------------------------------
namespace detail {
///----------------------------------------

/// @brief One wavelength's CIE 1931 2° standard-observer color-matching weights.
struct ColorMatchingWeights {
	float X, Y, Z;
};

/// @brief First wavelength of @ref cieStandardObserver, in nanometers.
inline constexpr int cieMinWavelength = 380;

/// @brief Wavelength spacing of @ref cieStandardObserver, in nanometers.
inline constexpr int cieWavelengthStep = 5;

/// @brief CIE 1931 2° standard-observer color-matching functions, 380–780 nm in 5 nm steps.
inline constexpr ColorMatchingWeights cieStandardObserver[81] = {
	{0.0014f, 0.0000f, 0.0065f}, {0.0022f, 0.0001f, 0.0105f}, {0.0042f, 0.0001f, 0.0201f},
	{0.0076f, 0.0002f, 0.0362f}, {0.0143f, 0.0004f, 0.0679f}, {0.0232f, 0.0006f, 0.1102f},
	{0.0435f, 0.0012f, 0.2074f}, {0.0776f, 0.0022f, 0.3713f}, {0.1344f, 0.0040f, 0.6456f},
	{0.2148f, 0.0073f, 1.0391f}, {0.2839f, 0.0116f, 1.3856f}, {0.3285f, 0.0168f, 1.6230f},
	{0.3483f, 0.0230f, 1.7471f}, {0.3481f, 0.0298f, 1.7826f}, {0.3362f, 0.0380f, 1.7721f},
	{0.3187f, 0.0480f, 1.7441f}, {0.2908f, 0.0600f, 1.6692f}, {0.2511f, 0.0739f, 1.5281f},
	{0.1954f, 0.0910f, 1.2876f}, {0.1421f, 0.1126f, 1.0419f}, {0.0956f, 0.1390f, 0.8130f},
	{0.0580f, 0.1693f, 0.6162f}, {0.0320f, 0.2080f, 0.4652f}, {0.0147f, 0.2586f, 0.3533f},
	{0.0049f, 0.3230f, 0.2720f}, {0.0024f, 0.4073f, 0.2123f}, {0.0093f, 0.5030f, 0.1582f},
	{0.0291f, 0.6082f, 0.1117f}, {0.0633f, 0.7100f, 0.0782f}, {0.1096f, 0.7932f, 0.0573f},
	{0.1655f, 0.8620f, 0.0422f}, {0.2257f, 0.9149f, 0.0298f}, {0.2904f, 0.9540f, 0.0203f},
	{0.3597f, 0.9803f, 0.0134f}, {0.4334f, 0.9950f, 0.0087f}, {0.5121f, 1.0000f, 0.0057f},
	{0.5945f, 0.9950f, 0.0039f}, {0.6784f, 0.9786f, 0.0027f}, {0.7621f, 0.9520f, 0.0021f},
	{0.8425f, 0.9154f, 0.0018f}, {0.9163f, 0.8700f, 0.0017f}, {0.9786f, 0.8163f, 0.0014f},
	{1.0263f, 0.7570f, 0.0011f}, {1.0567f, 0.6949f, 0.0010f}, {1.0622f, 0.6310f, 0.0008f},
	{1.0456f, 0.5668f, 0.0006f}, {1.0026f, 0.5030f, 0.0003f}, {0.9384f, 0.4412f, 0.0002f},
	{0.8544f, 0.3810f, 0.0002f}, {0.7514f, 0.3210f, 0.0001f}, {0.6424f, 0.2650f, 0.0000f},
	{0.5419f, 0.2170f, 0.0000f}, {0.4479f, 0.1750f, 0.0000f}, {0.3608f, 0.1382f, 0.0000f},
	{0.2835f, 0.1070f, 0.0000f}, {0.2187f, 0.0816f, 0.0000f}, {0.1649f, 0.0610f, 0.0000f},
	{0.1212f, 0.0446f, 0.0000f}, {0.0874f, 0.0320f, 0.0000f}, {0.0636f, 0.0232f, 0.0000f},
	{0.0468f, 0.0170f, 0.0000f}, {0.0329f, 0.0119f, 0.0000f}, {0.0227f, 0.0082f, 0.0000f},
	{0.0158f, 0.0057f, 0.0000f}, {0.0114f, 0.0041f, 0.0000f}, {0.0081f, 0.0029f, 0.0000f},
	{0.0058f, 0.0021f, 0.0000f}, {0.0041f, 0.0015f, 0.0000f}, {0.0029f, 0.0010f, 0.0000f},
	{0.0020f, 0.0007f, 0.0000f}, {0.0014f, 0.0005f, 0.0000f}, {0.0010f, 0.0004f, 0.0000f},
	{0.0007f, 0.0002f, 0.0000f}, {0.0005f, 0.0002f, 0.0000f}, {0.0003f, 0.0001f, 0.0000f},
	{0.0002f, 0.0001f, 0.0000f}, {0.0002f, 0.0001f, 0.0000f}, {0.0001f, 0.0000f, 0.0000f},
	{0.0001f, 0.0000f, 0.0000f}, {0.0001f, 0.0000f, 0.0000f}, {0.0000f, 0.0000f, 0.0000f}
};

///----------------------------------------
} // namespace detail
///----------------------------------------

///----------------------------------------
///   @brief Converts a chromaticity-plus-luminance color to XYZ tristimulus values.
/// @details Zero luminance converts to black; negative X or Z from out-of-gamut
///          chromaticities are clamped to zero.
///   @param color The xyY color to convert.
///  @return The equivalent XYZ color.
///----------------------------------------

[[nodiscard]] inline XYZColor toXYZ(const xyYColor& color) noexcept {
	// Zero luminance is black
	if (color.Y == 0.0f) {
		return {};
	}
	
	// Recover the tristimulus values from the chromaticity coordinates
	XYZColor result(color.x * color.Y / color.y, color.Y, (1.0f - color.x - color.y) * color.Y / color.y);
	result.X = std::max(result.X, 0.0f);
	result.Z = std::max(result.Z, 0.0f);
	return result;
}

///----------------------------------------
///   @brief Converts XYZ tristimulus values to chromaticity plus luminance.
/// @details The chromaticity of black (all components zero) is undefined.
///   @param color The XYZ color to convert.
///  @return The equivalent xyY color.
///----------------------------------------

[[nodiscard]] inline xyYColor toxyY(const XYZColor& color) noexcept {
	auto sum = color.X + color.Y + color.Z;
	return {color.X / sum, color.Y / sum, color.Y};
}

///----------------------------------------
///  @brief Converts XYZ tristimulus values to linear RGB (Rec. 709 primaries, D65 white point).
///  @param color The XYZ color to convert.
/// @return The equivalent linear RGB color; components may fall outside [0, 1] for
///         out-of-gamut colors.
///----------------------------------------

[[nodiscard]] inline RGBColor toRGB(const XYZColor& color) noexcept {
	return {
		 3.240479f * color.X - 1.53715f  * color.Y - 0.498535f * color.Z,
		-0.969256f * color.X + 1.875991f * color.Y + 0.041556f * color.Z,
		 0.055648f * color.X - 0.204043f * color.Y + 1.057311f * color.Z};
}

///----------------------------------------
///  @brief Converts linear RGB (Rec. 709 primaries, D65 white point) to XYZ tristimulus values.
///  @param color The linear RGB color to convert.
/// @return The equivalent XYZ color.
///----------------------------------------

[[nodiscard]] inline XYZColor toXYZ(const RGBColor& color) noexcept {
	return {
		0.412453f * color.r + 0.357580f * color.g + 0.180423f * color.b,
		0.212671f * color.r + 0.715160f * color.g + 0.072169f * color.b,
		0.019334f * color.r + 0.119193f * color.g + 0.950227f * color.b};
}

///----------------------------------------
///   @brief The CIE 1931 2° standard-observer color-matching weights at a wavelength.
/// @details The weights are tabulated at 5 nm steps; intermediate wavelengths use the step
///          below them.
///   @param wavelength The wavelength, in nanometers.
///  @return The (x̄, ȳ, z̄) weights as an @ref XYZColor; zero outside 380–780 nm.
///----------------------------------------

[[nodiscard]] inline XYZColor colorMatchingWeights(int wavelength) noexcept {
	// Wavelengths outside the tabulated range contribute nothing
	auto offset = wavelength - detail::cieMinWavelength;
	if (offset < 0 || offset > 400) {
		return {};
	}
	
	// Look up the enclosing table entry
	const auto& weights = detail::cieStandardObserver[offset / detail::cieWavelengthStep];
	return {weights.X, weights.Y, weights.Z};
}

///----------------------------------------
/// MARK: XYZColor
///----------------------------------------

inline void XYZColor::selfTest() {
	auto approxEqual = [](float a, float b, float tolerance) { return std::fabs(a - b) <= tolerance; };
	
	// The D65 white point round-trips through XYZ and back
	xyYColor whitePoint(0.3127f, 0.3290f, 1.0f);
	auto roundTrip = toxyY(toXYZ(whitePoint));
	Math::selftest::check(approxEqual(roundTrip.x, whitePoint.x, 1.e-5f) && approxEqual(roundTrip.y, whitePoint.y, 1.e-5f) && roundTrip.Y == whitePoint.Y, "xyY-XYZ round trip");
	
	// Zero luminance converts to black
	auto black = toXYZ(xyYColor(0.3127f, 0.3290f, 0.0f));
	Math::selftest::check(black.X == 0.0f && black.Y == 0.0f && black.Z == 0.0f, "zero luminance is black");
	
	// The RGB conversion matrices are mutual inverses
	auto rgbRoundTrip = toRGB(toXYZ(RGBColor(0.25f, 0.5f, 0.75f)));
	Math::selftest::check(approxEqual(rgbRoundTrip.r, 0.25f, 1.e-5f) && approxEqual(rgbRoundTrip.g, 0.5f, 1.e-5f) && approxEqual(rgbRoundTrip.b, 0.75f, 1.e-5f), "RGB-XYZ round trip");
	
	// The color-matching table peaks at 555 nm with unit luminous efficiency
	Math::selftest::check(colorMatchingWeights(555).Y == 1.0f, "observer peak at 555 nm");
	
	// Wavelengths outside the table contribute nothing
	auto below = colorMatchingWeights(300);
	auto above = colorMatchingWeights(800);
	Math::selftest::check(below.X == 0.0f && below.Y == 0.0f && below.Z == 0.0f && above.X == 0.0f && above.Y == 0.0f && above.Z == 0.0f, "out-of-range weights are zero");
	
	// Accumulation and scaling operate component-wise through the vector overlay
	XYZColor accumulated(1.0f, 2.0f, 3.0f);
	accumulated += XYZColor(0.5f, 0.25f, 0.125f);
	accumulated *= 2.0f;
	Math::selftest::check(accumulated.X == 3.0f && accumulated.Y == 4.5f && accumulated.Z == 6.25f, "component-wise accumulation");
}

///----------------------------------------
} // namespace Math
///----------------------------------------

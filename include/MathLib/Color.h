///----------------------------------------
///       @file Color.h
///    @ingroup MathLib
///      @brief RGB/RGBA color types and conversion utilities over the Math::Vector color aliases.
///    @details Core color math (RGB↔HSV, blend, luminance) operating on @ref Math::RGBColor /
///             @ref Math::RGBAColor, plus the 8-bit integer color types @ref Math::RGBColor8 (a
///             packed 3-byte struct) and @ref Math::RGBAColor8 (a @ref Math::Vector alias). HSV is
///             carried as a plain @ref Math::Float3 (hue in x, saturation in y, value in z).
///     @author Created by John Stephen (wobbleworks.com)
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/SelfTestCheck.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

//----------------------------------------
namespace Math {
//----------------------------------------

//----------------------------------------
// Color math helpers
//----------------------------------------

/// @brief Linear interpolation: @p min at factor 0, @p max at factor 1.
[[nodiscard]] constexpr float mix(float factor, float min, float max) {
	return (1 - factor) * min + factor * max;
}

/// @brief The level of a single channel for a hue, peaking at @p channelHue.
[[nodiscard]] constexpr float channelLevelForHue(float hue, float channelHue) {
	hue -= channelHue;
	hue += hue < 0 ? 0.5f : -0.5f;
	if (hue < 0) {
		hue = -hue;
	}
	return std::clamp(6 * hue - 1, 0.0f, 1.0f);
}

/// @brief Red channel level for a fully-saturated hue.
[[nodiscard]] constexpr float redValueForHue(float hue) {
	return channelLevelForHue(hue, 0);
}

/// @brief Green channel level for a fully-saturated hue.
[[nodiscard]] constexpr float greenValueForHue(float hue) {
	return channelLevelForHue(hue, 1.f / 3.f);
}

/// @brief Blue channel level for a fully-saturated hue.
[[nodiscard]] constexpr float blueValueForHue(float hue) {
	return channelLevelForHue(hue, 2.f / 3.f);
}

/// @brief Red channel level for a hue at the given saturation.
[[nodiscard]] constexpr float redValueForHueSaturation(float hue, float saturation) {
	return mix(saturation, 1, channelLevelForHue(hue, 0));
}

/// @brief Green channel level for a hue at the given saturation.
[[nodiscard]] constexpr float greenValueForHueSaturation(float hue, float saturation) {
	return mix(saturation, 1, channelLevelForHue(hue, 1 / 3.f));
}

/// @brief Blue channel level for a hue at the given saturation.
[[nodiscard]] constexpr float blueValueForHueSaturation(float hue, float saturation) {
	return mix(saturation, 1, channelLevelForHue(hue, 2 / 3.f));
}

//----------------------------------------
// RGB <-> HSV (HSV carried as a Float3: hue in x, saturation in y, value in z)
//----------------------------------------

/// @brief Converts an RGB color to HSV (hue in degrees [0,360), saturation/value in [0,1]).
[[nodiscard]] inline Float3 rgb2hsv(const RGBColor &rgb) {
	// Based on an example at http://lolengine.net (jjs)
	float K, min, mid, max;
	if (rgb.g < rgb.b) {
		min = rgb.g;
		mid = rgb.b;
		K = -1.f;
	} else {
		mid = rgb.g;
		min = rgb.b;
		K = 0.f;
	}
	if (rgb.r < mid) {
		max = mid;
		mid = rgb.r;
		K = -2.f / 6.f - K;
	} else {
		max = rgb.r;
	}
	auto chroma = max - std::min(min, mid);
	auto h = std::fabs(K + (mid - min) / (6.f * chroma + 1e-6f)) * 360;
	auto s = chroma / (max + 1e-6f);
	auto v = max;
	
	return {h, s, v};
}

/// @brief Converts an HSV triple (hue, saturation, value) to an RGB color.
[[nodiscard]] inline RGBColor hsv2rgb(float h, float s, float v) {
	// Based on an HLSL example at http://chilliant.blogspot.com (jjs)
	auto Hp = h * (1.f / 60.f);
	auto rgb = Math::Float3{std::fabs(Hp - 3) - 1, 2 - std::fabs(Hp - 2), 2 - std::fabs(Hp - 4)};
	rgb = Math::clamp(rgb, Math::Float3{0, 0, 0}, Math::Float3{1, 1, 1});
	return ((rgb - 1.f) * s + 1.f) * v;
}

/// @brief Converts an HSV color (hue in x, saturation in y, value in z) to an RGB color.
[[nodiscard]] inline RGBColor hsv2rgb(const Float3 &hsv) {
	return hsv2rgb(hsv.x, hsv.y, hsv.z);
}

//----------------------------------------
// Blending and luminance
//----------------------------------------

/// @brief Linearly blends two RGB colors by @p factor (0 → @p color1, 1 → @p color2).
[[nodiscard]] inline RGBColor blend(const RGBColor &color1, const RGBColor &color2, float factor) {
	return color1 + factor * (color2 - color1);
}

/// @brief Linearly blends two RGBA colors by @p factor (0 → @p color1, 1 → @p color2).
[[nodiscard]] inline RGBAColor blend(const RGBAColor &color1, const RGBAColor &color2, float factor) {
	return color1 + factor * (color2 - color1);
}

/// @brief The CIE relative luminance of an RGB color.
[[nodiscard]] inline float cieLuminance(const RGBColor &color) {
	return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

/// @brief Whether all channels are zero.
[[nodiscard]] inline bool isBlack(const RGBColor &color) {
	return color.r == 0 && color.g == 0 && color.b == 0;
}

/// @brief Whether all channels are one.
[[nodiscard]] inline bool isWhite(const RGBColor &color) {
	return color.r == 1 && color.g == 1 && color.b == 1;
}

//----------------------------------------
/// @brief A packed 8-bit RGB color. The 3-byte layout is a contract for color tables and pixel buffers.
//----------------------------------------
struct RGBColor8 {
	uint8_t r, g, b;
	
	RGBColor8() = default;
	constexpr RGBColor8(uint8_t red, uint8_t green, uint8_t blue) noexcept : r(red), g(green), b(blue) {}
	
	/// @brief Unpacks the low three bytes (red = byte 0, green = byte 1, blue = byte 2).
	constexpr explicit RGBColor8(uint32_t packed) noexcept
	: r(uint8_t(packed)), g(uint8_t(packed >> 8)), b(uint8_t(packed >> 16)) {}
	
	/// @brief Quantizes a float color (each channel scaled by 255 and rounded).
	explicit RGBColor8(const RGBColor &color) noexcept
	: r(uint8_t(color.r * 255.f + 0.5f)), g(uint8_t(color.g * 255.f + 0.5f)), b(uint8_t(color.b * 255.f + 0.5f)) {}
	
	void set(uint8_t red, uint8_t green, uint8_t blue) noexcept {r = red; g = green; b = blue;}
	
	[[nodiscard]] uint8_t operator[](int i) const noexcept {return (&r)[i];}
	[[nodiscard]] uint8_t &operator[](int i) noexcept {return (&r)[i];}
	
	[[nodiscard]] constexpr bool isBlack() const noexcept {return r == 0 && g == 0 && b == 0;}
	[[nodiscard]] constexpr bool isWhite() const noexcept {return r == 255 && g == 255 && b == 255;}
	
	/// @brief Promotes to a float color (each channel divided by 255).
	[[nodiscard]] operator RGBColor() const noexcept {return RGBColor(r / 255.f, g / 255.f, b / 255.f);}
	
	[[nodiscard]] constexpr bool operator==(const RGBColor8 &color) const noexcept = default;
	[[nodiscard]] constexpr auto operator<=>(const RGBColor8 &color) const noexcept = default;
};

static_assert(sizeof(RGBColor8) == 3, "RGBColor8 must stay 3 bytes (a packed layout contract)");

//----------------------------------------
// 8-bit RGBA color (4 bytes, backed by the Vector uint8 alias)
//----------------------------------------

/// @brief A packed 8-bit RGBA color.
using RGBAColor8 = Vector<uint8_t, 4>;

/// @brief Packs an RGBA8 color into a uint32 (red = byte 0, green = byte 1, blue = byte 2, alpha = byte 3).
[[nodiscard]] inline uint32_t toUint32(const RGBAColor8 &color) noexcept {
	return uint32_t(color.r) | (uint32_t(color.g) << 8) | (uint32_t(color.b) << 16) | (uint32_t(color.a) << 24);
}

/// @brief Unpacks a uint32 into an RGBA8 color (red = byte 0, green = byte 1, blue = byte 2, alpha = byte 3).
[[nodiscard]] inline RGBAColor8 rgbaColor8FromUint32(uint32_t packed) noexcept {
	return RGBAColor8(uint8_t(packed), uint8_t(packed >> 8), uint8_t(packed >> 16), uint8_t(packed >> 24));
}

/// @brief Quantizes a float RGBA color to 8-bit (each channel scaled by 255 and rounded).
[[nodiscard]] inline RGBAColor8 toRGBAColor8(const RGBAColor &color) noexcept {
	return RGBAColor8(uint8_t(color.r * 255.f + 0.5f), uint8_t(color.g * 255.f + 0.5f), uint8_t(color.b * 255.f + 0.5f), uint8_t(color.a * 255.f + 0.5f));
}

/// @brief Promotes an 8-bit RGBA color to float (each channel divided by 255).
[[nodiscard]] inline RGBAColor toRGBAColor(const RGBAColor8 &color) noexcept {
	return RGBAColor(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f);
}

///----------------------------------------
/// @brief Verifies the color conversions against named-hue references and byte-exact round trips.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void colorSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-4; };
	
	// Named fully-saturated primaries map to their canonical hues (degrees), with full saturation and value.
	{
		const auto red = rgb2hsv(RGBColor(1.f, 0.f, 0.f));
		check(near(red.x, 0.0), "pure red maps to hue 0 degrees");
		check(near(red.y, 1.0) && near(red.z, 1.0), "pure red is fully saturated at full value");
		
		const auto green = rgb2hsv(RGBColor(0.f, 1.f, 0.f));
		check(near(green.x, 120.0), "pure green maps to hue 120 degrees");
		
		const auto blue = rgb2hsv(RGBColor(0.f, 0.f, 1.f));
		check(near(blue.x, 240.0), "pure blue maps to hue 240 degrees");
	}
	
	// A neutral gray has no saturation, and its value equals the shared channel level.
	{
		const auto gray = rgb2hsv(RGBColor(0.5f, 0.5f, 0.5f));
		check(near(gray.y, 0.0), "gray has zero saturation");
		check(near(gray.z, 0.5), "gray value equals the channel level");
	}
	
	// RGB -> HSV -> RGB recovers a mid-gamut color clear of the primary boundaries.
	{
		const RGBColor original(0.6f, 0.3f, 0.2f);
		const auto roundTrip = hsv2rgb(rgb2hsv(original));
		check(near(roundTrip.r, original.r) && near(roundTrip.g, original.g) && near(roundTrip.b, original.b),
			"RGB survives an HSV round trip");
	}
	
	// Rec.709 relative luminance: white is unity and pure green carries the green weight.
	{
		check(near(cieLuminance(RGBColor(1.f, 1.f, 1.f)), 1.0), "white luminance is one");
		check(near(cieLuminance(RGBColor(0.f, 1.f, 0.f)), 0.7152), "pure green luminance is the Rec.709 green weight");
	}
	
	// The 8-bit RGBA pack/unpack preserves every byte through uint32 (exact integer values).
	{
		const RGBAColor8 color(0x12, 0x34, 0x56, 0x78);
		const uint32_t packed = toUint32(color);
		check(packed == 0x78563412u, "RGBA8 packs low-to-high red, green, blue, alpha");
		const auto unpacked = rgbaColor8FromUint32(packed);
		check(unpacked.r == 0x12 && unpacked.g == 0x34 && unpacked.b == 0x56 && unpacked.a == 0x78,
			"uint32 unpacks back to the original bytes");
	}
	
	// The packed 3-byte RGB type unpacks the low three bytes and stays a byte-exact round trip.
	{
		const RGBColor8 color(0xAB, 0xCD, 0xEF);
		const RGBColor8 fromPacked(uint32_t(0x00EFCDAB));
		check(fromPacked == color, "RGBColor8 unpacks the low three bytes");
	}
}
	
} // namespace Math

///----------------------------------------
///       @file HalfFloat.h
///    @ingroup MathLib
///      @brief Half-precision (IEEE 754 binary16) floating-point storage type.
///    @details A trivially copyable, two-byte value type holding a number in IEEE 754 half
///             precision. Conversions to and from @c float use the compiler's native @c _Float16
///             type when available (a single hardware-backed instruction on arm64 and x86 F16C
///             targets) and fall back to a portable software implementation otherwise. The software
///             path is correctly rounded (round-to-nearest, ties to even) and handles signed zero,
///             subnormals, infinities, and NaN. @ref HalfFloat is implicitly convertible to and
///             from @c float, so it can stand in wherever a floating value is expected while
///             occupying half the storage.
///     @author Created by John Stephen on 6/18/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
/// @brief When 1, conversions use the compiler's native @c _Float16 type; when 0, the portable
///        software fallback is used. Auto-detected from @c __FLT16_MANT_DIG__ (defined by Clang and
///        GCC when @c _Float16 is supported as an arithmetic type). Define it explicitly before
///        including this header to force a particular backend.
///----------------------------------------

#if !defined(MATH_HALF_NATIVE)
	#if defined(__FLT16_MANT_DIG__)
		#define MATH_HALF_NATIVE 1
	#else
		#define MATH_HALF_NATIVE 0
	#endif
#endif

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

#if MATH_HALF_NATIVE

///----------------------------------------
/// @brief Converts a @c float to its half-precision bit pattern using the native @c _Float16 type.
/// @param value Source single-precision value.
/// @return IEEE 754 binary16 bit pattern, rounded to nearest.
///----------------------------------------

[[nodiscard]] constexpr uint16_t floatToHalfBits(float value) noexcept {
	return std::bit_cast<uint16_t>(static_cast<_Float16>(value));
}

///----------------------------------------
/// @brief Converts a half-precision bit pattern to a @c float using the native @c _Float16 type.
/// @param bits IEEE 754 binary16 bit pattern.
/// @return Decoded single-precision value.
///----------------------------------------

[[nodiscard]] constexpr float halfBitsToFloat(uint16_t bits) noexcept {
	return static_cast<float>(std::bit_cast<_Float16>(bits));
}

#else

///----------------------------------------
/// @brief Converts a @c float to its half-precision bit pattern in software.
/// @details Round-to-nearest, ties to even. Saturates to infinity on overflow and preserves a
///          non-zero NaN payload.
/// @param value Source single-precision value.
/// @return IEEE 754 binary16 bit pattern.
///----------------------------------------

[[nodiscard]] constexpr uint16_t floatToHalfBits(float value) noexcept {
	// Decode the float into sign, biased exponent, and mantissa
	const uint32_t f = std::bit_cast<uint32_t>(value);
	const uint16_t sign = static_cast<uint16_t>((f >> 16) & 0x8000u);
	const uint32_t exponentField = (f >> 23) & 0xffu;
	const uint32_t mantissa = f & 0x7fffffu;
	
	// Pass through infinities and NaN, keeping a non-zero NaN payload non-zero
	if (exponentField == 0xffu) {
		return static_cast<uint16_t>(sign | 0x7c00u | (mantissa != 0u ? 0x0200u : 0u));
	}
	
	// Re-bias the exponent from float (127) to half (15)
	const int32_t exponent = static_cast<int32_t>(exponentField) - 127 + 15;
	
	// Anything above the half range saturates to infinity
	if (exponent >= 0x1f) {
		return static_cast<uint16_t>(sign | 0x7c00u);
	}
	
	// A zero or subnormal result needs the mantissa shifted into the subnormal field
	if (exponent <= 0) {
		// Values below half of the smallest subnormal round to zero
		if (exponent < -10) {
			return sign;
		}
		
		// Restore the implicit leading one and align it to the subnormal field
		const uint32_t significand = mantissa | 0x800000u;
		const int32_t shift = 14 - exponent;
		uint32_t result = significand >> shift;
		
		// Round to nearest, ties to even
		const uint32_t remainder = significand & ((1u << shift) - 1u);
		const uint32_t halfway = 1u << (shift - 1);
		if (remainder > halfway || (remainder == halfway && (result & 1u) != 0u)) {
			++result;
		}
		
		return static_cast<uint16_t>(sign | result);
	}
	
	// Normal result: keep the top ten mantissa bits
	uint16_t result = static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | static_cast<uint16_t>(mantissa >> 13));
	
	// Round to nearest, ties to even; a mantissa carry propagates correctly into the exponent
	const uint32_t remainder = mantissa & 0x1fffu;
	if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0u)) {
		++result;
	}
	
	return result;
}

///----------------------------------------
/// @brief Converts a half-precision bit pattern to a @c float in software.
/// @details The conversion is always exact, since every half value is representable as a @c float.
/// @param bits IEEE 754 binary16 bit pattern.
/// @return Decoded single-precision value.
///----------------------------------------

[[nodiscard]] constexpr float halfBitsToFloat(uint16_t bits) noexcept {
	// Decode the half into sign, exponent, and mantissa
	const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
	const uint32_t exponentField = (bits >> 10) & 0x1fu;
	const uint32_t mantissa = bits & 0x3ffu;
	
	uint32_t f = 0;
	
	// A zero exponent field means zero or a subnormal
	if (exponentField == 0u) {
		if (mantissa == 0u) {
			// Signed zero
			f = sign;
		} else {
			// Normalize the subnormal by shifting until the implicit bit surfaces
			int32_t exponent = -1;
			uint32_t significand = mantissa;
			do {
				++exponent;
				significand <<= 1;
			} while ((significand & 0x400u) == 0u);
			
			// Drop the now-implicit leading bit and assemble the normalized float
			significand &= 0x3ffu;
			f = sign | (static_cast<uint32_t>(127 - 15 - exponent) << 23) | (significand << 13);
		}
	} else if (exponentField == 0x1fu) {
		// Infinity or NaN, preserving the payload
		f = sign | 0x7f800000u | (mantissa << 13);
	} else {
		// Normal: re-bias the exponent from half (15) to float (127)
		f = sign | ((exponentField - 15u + 127u) << 23) | (mantissa << 13);
	}
	
	return std::bit_cast<float>(f);
}

#endif

///----------------------------------------
} // namespace detail
///----------------------------------------

///----------------------------------------
///   @class HalfFloat
///   @brief A two-byte IEEE 754 half-precision floating-point value.
/// @details Stores the 16-bit pattern in @ref bits and converts to and from @c float on demand,
///          using the native @c _Float16 type when @ref MATH_HALF_NATIVE is set and a software
///          fallback otherwise. Implicitly convertible to and from @c float, so comparisons and
///          arithmetic operate on the decoded value: @c +0 and @c -0 compare equal and NaN is
///          unordered, matching @c float semantics.
///----------------------------------------

struct HalfFloat final {
	/// @brief The raw IEEE 754 binary16 bit pattern.
	uint16_t bits = 0;
	
	///----------------------------------------
	/// @brief Constructs a positive-zero half.
	///----------------------------------------
	
	constexpr HalfFloat() noexcept = default;
	
	///----------------------------------------
	/// @brief Constructs a half from a @c float, rounding to the nearest representable value.
	/// @param value Source single-precision value.
	///----------------------------------------
	
	constexpr HalfFloat(float value) noexcept : bits(detail::floatToHalfBits(value)) {}
	
	///----------------------------------------
	/// @brief Decodes this half to a @c float.
	/// @return The single-precision value.
	///----------------------------------------
	
	[[nodiscard]] constexpr operator float() const noexcept {
		return detail::halfBitsToFloat(bits);
	}
	
	///----------------------------------------
	/// @brief Constructs a half directly from a raw 16-bit pattern, performing no conversion.
	/// @param rawBits IEEE 754 binary16 bit pattern.
	/// @return Half wrapping @p rawBits.
	///----------------------------------------
	
	[[nodiscard]] static constexpr HalfFloat fromBits(uint16_t rawBits) noexcept {
		HalfFloat half;
		half.bits = rawBits;
		return half;
	}
	
	///----------------------------------------
	/// @brief Validates the conversions against known encodings and a full round trip.
	/// @details Throws @ref Math::selftest::Failure on the first violation.
	///----------------------------------------
	
	static void selfTest();
};

static_assert(sizeof(HalfFloat) == 2, "HalfFloat must be exactly two bytes");
static_assert(std::is_trivially_copyable_v<HalfFloat>, "HalfFloat must be trivially copyable");

///----------------------------------------
/// MARK: Self-test
///----------------------------------------

inline void HalfFloat::selfTest() {
	// Known IEEE 754 binary16 encodings convert and decode exactly in both directions
	struct Case {
		float value;
		uint16_t bits;
	};
	constexpr Case cases[] = {
		{.value = 0.0f, .bits = 0x0000},            // positive zero
		{.value = -0.0f, .bits = 0x8000},           // negative zero
		{.value = 1.0f, .bits = 0x3c00},            // one
		{.value = -2.0f, .bits = 0xc000},           // negative power of two
		{.value = 0.5f, .bits = 0x3800},            // fraction
		{.value = 65504.0f, .bits = 0x7bff},        // largest finite half
		{.value = 6.103515625e-05f, .bits = 0x0400},// smallest normal (2^-14)
		{.value = 5.9604645e-08f, .bits = 0x0001}   // smallest subnormal (2^-24)
	};
	for (const auto& [value, bits] : cases) {
		Math::selftest::check(HalfFloat(value).bits == bits, "encoding");
		Math::selftest::check(static_cast<float>(HalfFloat::fromBits(bits)) == value, "decoding");
	}
	
	// Overflow saturates to infinity, and infinities decode back to infinity
	Math::selftest::check(HalfFloat(70000.0f).bits == 0x7c00, "overflow to infinity");
	Math::selftest::check(HalfFloat(std::numeric_limits<float>::infinity()).bits == 0x7c00, "infinity encoding");
	Math::selftest::check(std::isinf(static_cast<float>(HalfFloat::fromBits(0xfc00))), "infinity decoding");
	
	// NaN stays NaN: exponent all ones, a non-zero payload, and a NaN float on decode
	auto notANumber = HalfFloat(std::numeric_limits<float>::quiet_NaN());
	Math::selftest::check((notANumber.bits & 0x7c00u) == 0x7c00u && (notANumber.bits & 0x03ffu) != 0u, "NaN encoding");
	Math::selftest::check(std::isnan(static_cast<float>(notANumber)), "NaN decoding");
	
	// Half of the smallest subnormal sits on a tie and rounds to even, which is zero
	Math::selftest::check(HalfFloat(2.9802322e-08f).bits == 0x0000, "round half to even");
	
	// Every representable value but NaN survives a half -> float -> half round trip
	for (int pattern = 0; pattern < 0x10000; ++pattern) {
		auto bits = static_cast<uint16_t>(pattern);
		auto isNaN = (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0u;
		if (isNaN) {
			continue;
		}
		
		Math::selftest::check(HalfFloat(static_cast<float>(HalfFloat::fromBits(bits))).bits == bits, "round trip");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

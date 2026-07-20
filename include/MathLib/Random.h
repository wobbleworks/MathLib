///----------------------------------------
/// @file Random.h
/// @ingroup MathLib
/// @brief Thread-safe random number generation utilities.
/// @author Created by John Stephen on 9/16/25.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <limits>
#include <cassert>
#include <cmath>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace math {
///----------------------------------------

///----------------------------------------
/// @brief Shared per-thread RNG engine (Mersenne Twister).
/// @returns Reference to a thread-local std::mt19937.
/// @note Each thread gets its own engine seeded from std::random_device.
///       Call initializeRandomSeed() on each thread if you want deterministic runs.
///----------------------------------------

inline std::mt19937 &sharedRandomGenerator() noexcept {
	static thread_local std::mt19937 gen(std::random_device{}());
	return gen;
}

///----------------------------------------
/// @brief Seed the current thread's RNG deterministically.
/// @param seed 32-bit seed value.
/// @note Only affects the calling thread's engine.
///----------------------------------------

inline void initializeRandomSeed(const uint32_t seed) noexcept {
	sharedRandomGenerator().seed(seed);
}

///----------------------------------------
/// @brief Random unsigned 32-bit integer over the full range.
/// @returns U32 in [0, 2^32-1].
///----------------------------------------

[[nodiscard]] inline uint32_t randomUInt32() noexcept {
	static thread_local std::uniform_int_distribution<uint32_t> dist(
		std::numeric_limits<uint32_t>::min(),
		std::numeric_limits<uint32_t>::max()
	);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random signed int over the full range.
/// @returns int in [INT_MIN, INT_MAX].
///----------------------------------------

[[nodiscard]] inline int randomInt() noexcept {
	static thread_local std::uniform_int_distribution<int> dist(
		std::numeric_limits<int>::min(),
		std::numeric_limits<int>::max()
	);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random signed int in [0, max).
/// @param max Exclusive upper bound (must be > 0).
/// @returns int in [0, max-1].
/// @note Uses uniform_int_distribution to avoid modulo bias.
///----------------------------------------

[[nodiscard]] inline int randomInt(const int max) {
	assert(max > 0);
	std::uniform_int_distribution<int> dist(0, max - 1);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Convenience: maximum representable int.
///----------------------------------------

static inline constexpr int maxRandomInt = std::numeric_limits<int>::max();

///----------------------------------------
/// @brief Random signed 16-bit integer over the full range.
/// @returns int16_t in [INT16_MIN, INT16_MAX].
///----------------------------------------

[[nodiscard]] inline int16_t randomInt16() noexcept {
	static thread_local std::uniform_int_distribution<int16_t> dist(
		std::numeric_limits<int16_t>::min(),
		std::numeric_limits<int16_t>::max()
	);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random signed 32-bit integer over the full range.
/// @returns int32_t in [INT32_MIN, INT32_MAX].
///----------------------------------------

[[nodiscard]] inline int32_t randomInt32() noexcept {
	static thread_local std::uniform_int_distribution<int32_t> dist(
		std::numeric_limits<int32_t>::min(),
		std::numeric_limits<int32_t>::max()
	);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Convenience: maximum representable int32_t.
///----------------------------------------

static inline constexpr int32_t maxRandomInt32 = std::numeric_limits<int32_t>::max();

///----------------------------------------
/// @brief Random float in [0, 1].
/// @returns Value in [0, 1], inclusive of 1 by widening with nextafter().
/// @note Distribution is [0, 1+e) so 1.0f can occur; never exceeds 1.0f.
///----------------------------------------

[[nodiscard]] inline float randomFloat() noexcept {
	static const float one = std::nextafter(1.0f, std::numeric_limits<float>::infinity());
	static thread_local std::uniform_real_distribution<float> dist(0.0f, one);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random float in [0, 1).
/// @returns Value in [0, 1) (upper bound exclusive).
///----------------------------------------

[[nodiscard]] inline float randomFloatExclusive() noexcept {
	static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random float in [-1, 1].
/// @returns Value in [-1, 1) (upper bound exclusive).
///----------------------------------------

[[nodiscard]] inline float randomFloatPlusMinus() noexcept {
	static const float plus_one = std::nextafter(1.0f, std::numeric_limits<float>::infinity());
	static thread_local std::uniform_real_distribution<float> dist(-1.0f, plus_one);
	return dist(sharedRandomGenerator());
}

///----------------------------------------
/// @brief Random float in [0, max].
/// @param max Scale factor for the inclusive variant.
/// @returns Value in [0, max].
///----------------------------------------

[[nodiscard]] inline float randomFloat(const float max) noexcept {
	return randomFloat() * max;
}

///----------------------------------------
/// @brief Random float in [0, max).
/// @param max Scale factor for the exclusive variant.
/// @returns Value in [0, max).
///----------------------------------------

[[nodiscard]] inline float randomFloatExclusive(const float max) noexcept {
	return randomFloatExclusive() * max;
}

///----------------------------------------
/// @brief Random float in [-max, max].
/// @param max Positive bound.
/// @returns Value in [-max, max].
///----------------------------------------

[[nodiscard]] inline float randomFloatPlusMinus(const float max) noexcept {
	return randomFloat(2.0f * max) - max;
}

///----------------------------------------
/// @brief Verifies reproducible seeding and the documented bounds of the draw functions.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void randomSelfTest() {
	using Math::selftest::check;
	
	// Seeding the thread-local engine makes the draw sequence exactly reproducible.
	{
		initializeRandomSeed(12345u);
		std::array<uint32_t, 16> first{};
		for (auto &value : first) {
			value = randomUInt32();
		}
		
		initializeRandomSeed(12345u);
		std::array<uint32_t, 16> second{};
		for (auto &value : second) {
			value = randomUInt32();
		}
		
		check(first == second, "identical seed reproduces the sequence");
	}
	
	// Every generator stays inside its documented range across many draws.
	{
		initializeRandomSeed(2024u);
		constexpr float bound = 7.5f;
		bool floatInUnit = true;
		bool exclusiveBelowOne = true;
		bool plusMinusInRange = true;
		bool intInRange = true;
		
		for (int i = 0; i < 10000; ++i) {
			const float unit = randomFloat();
			floatInUnit = floatInUnit && unit >= 0.0f && unit <= 1.0f;
			
			const float exclusive = randomFloatExclusive();
			exclusiveBelowOne = exclusiveBelowOne && exclusive >= 0.0f && exclusive < 1.0f;
			
			const float plusMinus = randomFloatPlusMinus(bound);
			plusMinusInRange = plusMinusInRange && plusMinus >= -bound && plusMinus <= bound;
			
			const int bounded = randomInt(100);
			intInRange = intInRange && bounded >= 0 && bounded < 100;
		}
		
		check(floatInUnit, "randomFloat stays within [0, 1]");
		check(exclusiveBelowOne, "randomFloatExclusive stays within [0, 1)");
		check(plusMinusInRange, "randomFloatPlusMinus(max) stays within [-max, max]");
		check(intInRange, "randomInt(max) stays within [0, max)");
	}
}
	
} // namespace math

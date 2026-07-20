///----------------------------------------
///       @file PerlinNoise.h
///    @ingroup MathLib
///      @brief Seedable gradient (Perlin) noise in one, two, and three dimensions.
///    @details A small, header-only noise generator offering both of Ken Perlin's gradient-noise
///             algorithms behind one interface. The 1985 @em classic algorithm pairs a table of
///             pseudo-random unit gradients with the cubic S-curve @f$3t^2-2t^3@f$. The 2002
///             @em improved algorithm swaps in the quintic fade @f$6t^5-15t^4+10t^3@f$ (which has a
///             continuous second derivative, removing the visible grid artifacts of the classic
///             curve) and selects gradients from a fixed direction set by hashing, avoiding the
///             gradient-table lookup entirely. Pick the variant at the call site with the
///             @ref NoiseQuality template argument; both share one permutation table and seed. Raw
///             output lands in approximately @f$[-1, 1]@f$ — use @ref PerlinNoise::toUnit to remap to
///             @f$[0, 1]@f$. @ref PerlinNoise::fractal sums octaves for fractional Brownian motion.
///     @author Created by John Stephen on 7/7/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>

#include "MathLib/Vector.h"

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
///   @enum NoiseQuality
///   @brief Selects which gradient-noise algorithm @ref PerlinNoise evaluates.
/// @details @c classic is Perlin's original 1985 formulation (random gradients, cubic fade).
///          @c improved is the 2002 revision (fixed hashed gradients, quintic fade) and is the
///          default everywhere a quality is not named — it is smoother and slightly cheaper.
///----------------------------------------

enum class NoiseQuality : uint8_t {
	classic,
	improved
};

///----------------------------------------
///   @struct FractalOptions
///   @brief Parameters controlling a fractional-Brownian-motion octave sum.
/// @details Each successive octave multiplies the sampling frequency by @ref lacunarity and the
///          amplitude by @ref gain. @ref PerlinNoise::fractal normalizes by the total amplitude so
///          the result stays in the same range as a single octave.
///----------------------------------------

struct FractalOptions final {
	int octaves = 4;             ///< Number of noise layers summed (>= 1).
	float frequency = 1.0f;      ///< Sampling frequency of the first octave.
	float lacunarity = 2.0f;     ///< Frequency multiplier applied per octave.
	float gain = 0.5f;           ///< Amplitude multiplier applied per octave.
};

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief Floor toward negative infinity, returned as an @c int.
/// @param value Value to floor.
/// @return Largest integer not greater than @p value.
/// @note Faster than @c std::floor for the small magnitudes noise lattices use, and always integral.
///----------------------------------------

[[nodiscard]] inline constexpr int floorToInt(float value) noexcept {
	const int truncated = static_cast<int>(value);
	return value < static_cast<float>(truncated) ? truncated - 1 : truncated;
}

///----------------------------------------
/// @brief Cubic S-curve @f$3t^2-2t^3@f$ used by the classic algorithm.
/// @param t Interpolation parameter in [0, 1].
/// @return Eased parameter in [0, 1] with a continuous first derivative.
///----------------------------------------

[[nodiscard]] inline constexpr float fadeCubic(float t) noexcept {
	return t * t * (3.0f - 2.0f * t);
}

///----------------------------------------
/// @brief Quintic fade @f$6t^5-15t^4+10t^3@f$ used by the improved algorithm.
/// @param t Interpolation parameter in [0, 1].
/// @return Eased parameter in [0, 1] with continuous first and second derivatives.
///----------------------------------------

[[nodiscard]] inline constexpr float fadeQuintic(float t) noexcept {
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

///----------------------------------------
/// @brief Improved-algorithm 1-D gradient: a signed integer slope from a hash.
/// @param hash Permutation-table value selecting the slope.
/// @param x Offset from the lattice node.
/// @return The gradient dotted with @p x.
///----------------------------------------

[[nodiscard]] inline constexpr float improvedGradient(int hash, float x) noexcept {
	const int h = hash & 15;
	float slope = 1.0f + static_cast<float>(h & 7);
	if (h & 8) {
		slope = -slope;
	}
	return slope * x;
}

///----------------------------------------
/// @brief Improved-algorithm 2-D gradient: one of the four square-diagonal directions.
/// @param hash Permutation-table value selecting the direction.
/// @param offset Offset from the lattice node.
/// @return The gradient dotted with @p offset.
///----------------------------------------

[[nodiscard]] inline float improvedGradient(int hash, const Float2& offset) noexcept {
	const int h = hash & 3;
	const float u = (h & 1) ? -offset.x : offset.x;
	const float v = (h & 2) ? -offset.y : offset.y;
	return u + v;
}

///----------------------------------------
/// @brief Improved-algorithm 3-D gradient: one of the twelve cube-edge directions.
/// @param hash Permutation-table value selecting the direction.
/// @param offset Offset from the lattice node.
/// @return The gradient dotted with @p offset.
/// @note This is Ken Perlin's 2002 branch-selected edge-vector scheme, which replaces the
///       classic random-gradient table lookup with a hashed pick from a fixed direction set.
///----------------------------------------

[[nodiscard]] inline float improvedGradient(int hash, const Float3& offset) noexcept {
	const int h = hash & 15;
	const float u = h < 8 ? offset.x : offset.y;
	const float v = h < 4 ? offset.y : (h == 12 || h == 14 ? offset.x : offset.z);
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}
	
} // namespace detail

///----------------------------------------
///   @class PerlinNoise
///   @brief Seedable coherent gradient noise, evaluable with either Perlin algorithm.
/// @details Owns a 256-entry permutation table (duplicated to 512 to elide index wrapping) and, for
///          the classic algorithm, matching tables of random unit gradients. Everything is derived
///          deterministically from a 32-bit seed, so two instances built from the same seed produce
///          identical fields. The noise and @ref fractal calls are @c const and free of shared
///          mutable state, so one instance may be sampled concurrently from many threads. Choose the
///          algorithm per call through the @ref NoiseQuality template argument, which defaults to
///          @ref NoiseQuality::improved.
///    @note Raw values are approximately in @f$[-1, 1]@f$; the bound is not hard. Clamp if a caller
///          requires a strict range.
///----------------------------------------

class PerlinNoise final {
public:
	///----------------------------------------
	/// @brief Seed applied by the default constructor.
	///----------------------------------------
	
	static constexpr uint32_t defaultSeed = 0x9E3779B9u;
	
	///----------------------------------------
	/// @brief Constructs a generator using @ref defaultSeed.
	///----------------------------------------
	
	PerlinNoise() noexcept {
		reseed(defaultSeed);
	}
	
	///----------------------------------------
	/// @brief Constructs a generator from an explicit seed.
	/// @param seed 32-bit value that deterministically fixes the permutation and gradient tables.
	///----------------------------------------
	
	explicit PerlinNoise(uint32_t seed) noexcept {
		reseed(seed);
	}
	
	///----------------------------------------
	/// @brief Rebuilds the permutation and gradient tables from a new seed.
	/// @param seed 32-bit value that deterministically fixes the tables.
	///----------------------------------------
	
	void reseed(uint32_t seed) noexcept {
		std::mt19937 engine(seed);
		
		// Fill the low half of the permutation with 0..255 and shuffle it.
		std::array<uint8_t, 256> shuffled;
		std::iota(shuffled.begin(), shuffled.end(), uint8_t(0));
		std::shuffle(shuffled.begin(), shuffled.end(), engine);
		
		// Duplicate the shuffled block so any index in [0, 511] reads without a wrap.
		for (int i = 0; i < 256; ++i) {
			_permutation[i] = shuffled[i];
			_permutation[i + 256] = shuffled[i];
		}
		
		// Build the classic algorithm's random unit-gradient tables.
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
		for (int i = 0; i < 256; ++i) {
			_classicGradient1[i] = normalizedSlope(distribution(engine));
			_classicGradient2[i] = randomUnitVector2(engine, distribution);
			_classicGradient3[i] = randomUnitVector3(engine, distribution);
		}
	}
	
	///----------------------------------------
	/// @name Raw noise
	///----------------------------------------
	///@{
	
	///----------------------------------------
	/// @brief One-dimensional gradient noise.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float noise(float x) const noexcept {
		// Locate the containing lattice cell and the offset within it.
		const int X = detail::floorToInt(x) & 255;
		const float fx = x - static_cast<float>(detail::floorToInt(x));
		
		// Ease the offset with the algorithm's fade curve.
		const float u = fade<Quality>(fx);
		
		// Blend the two node gradients.
		const float value = std::lerp(
			gradient1<Quality>(_permutation[X], fx),
			gradient1<Quality>(_permutation[X + 1], fx - 1.0f),
			u);
			
		return value * normalization1<Quality>();
	}
	
	///----------------------------------------
	/// @brief Two-dimensional gradient noise.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float noise(float x, float y) const noexcept {
		// Locate the containing lattice cell and the offsets within it.
		const int X = detail::floorToInt(x) & 255;
		const int Y = detail::floorToInt(y) & 255;
		const float fx = x - static_cast<float>(detail::floorToInt(x));
		const float fy = y - static_cast<float>(detail::floorToInt(y));
		
		// Ease each offset with the algorithm's fade curve.
		const float u = fade<Quality>(fx);
		const float v = fade<Quality>(fy);
		
		// Hash the four cell corners.
		const int A = _permutation[X] + Y;
		const int B = _permutation[X + 1] + Y;
		
		// Offsets from each corner, carried as SIMD vectors for the gradient dot products.
		const Float2 offset00(fx, fy);
		const Float2 offset10(fx - 1.0f, fy);
		const Float2 offset01(fx, fy - 1.0f);
		const Float2 offset11(fx - 1.0f, fy - 1.0f);
		
		// Blend the four corner gradients: interpolate along x, then along y.
		const float lower = std::lerp(
			gradient2<Quality>(_permutation[A], offset00),
			gradient2<Quality>(_permutation[B], offset10),
			u);
		const float upper = std::lerp(
			gradient2<Quality>(_permutation[A + 1], offset01),
			gradient2<Quality>(_permutation[B + 1], offset11),
			u);
			
		return std::lerp(lower, upper, v) * normalization2<Quality>();
	}
	
	///----------------------------------------
	/// @brief Three-dimensional gradient noise.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @param z Sample coordinate along z.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float noise(float x, float y, float z) const noexcept {
		// Locate the containing lattice cell and the offsets within it.
		const int X = detail::floorToInt(x) & 255;
		const int Y = detail::floorToInt(y) & 255;
		const int Z = detail::floorToInt(z) & 255;
		const float fx = x - static_cast<float>(detail::floorToInt(x));
		const float fy = y - static_cast<float>(detail::floorToInt(y));
		const float fz = z - static_cast<float>(detail::floorToInt(z));
		
		// Ease each offset with the algorithm's fade curve.
		const float u = fade<Quality>(fx);
		const float v = fade<Quality>(fy);
		const float w = fade<Quality>(fz);
		
		// Hash the eight cell corners.
		const int A = _permutation[X] + Y;
		const int AA = _permutation[A] + Z;
		const int AB = _permutation[A + 1] + Z;
		const int B = _permutation[X + 1] + Y;
		const int BA = _permutation[B] + Z;
		const int BB = _permutation[B + 1] + Z;
		
		// Offsets from each corner, carried as SIMD vectors for the gradient dot products.
		const float x1 = fx - 1.0f;
		const float y1 = fy - 1.0f;
		const float z1 = fz - 1.0f;
		
		// Blend the eight corner gradients: interpolate along x, then y, then z.
		const float lowerNear = std::lerp(
			gradient3<Quality>(_permutation[AA], Float3(fx, fy, fz)),
			gradient3<Quality>(_permutation[BA], Float3(x1, fy, fz)),
			u);
		const float upperNear = std::lerp(
			gradient3<Quality>(_permutation[AB], Float3(fx, y1, fz)),
			gradient3<Quality>(_permutation[BB], Float3(x1, y1, fz)),
			u);
		const float lowerFar = std::lerp(
			gradient3<Quality>(_permutation[AA + 1], Float3(fx, fy, z1)),
			gradient3<Quality>(_permutation[BA + 1], Float3(x1, fy, z1)),
			u);
		const float upperFar = std::lerp(
			gradient3<Quality>(_permutation[AB + 1], Float3(fx, y1, z1)),
			gradient3<Quality>(_permutation[BB + 1], Float3(x1, y1, z1)),
			u);
			
		const float blendNear = std::lerp(lowerNear, upperNear, v);
		const float blendFar = std::lerp(lowerFar, upperFar, v);
		return std::lerp(blendNear, blendFar, w) * normalization3<Quality>();
	}
	
	///----------------------------------------
	/// @brief Two-dimensional gradient noise sampled at a vector coordinate.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param point Sample coordinate.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float noise(const Float2& point) const noexcept {
		return noise<Quality>(point.x, point.y);
	}
	
	///----------------------------------------
	/// @brief Three-dimensional gradient noise sampled at a vector coordinate.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param point Sample coordinate.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float noise(const Float3& point) const noexcept {
		return noise<Quality>(point.x, point.y, point.z);
	}
	
	///@}
	
	///----------------------------------------
	/// @name Named algorithm variants
	/// @brief Non-template entry points that name the algorithm in the method rather than a template
	///        argument, for callers that cannot supply one — notably Swift, whose C++ interop does not
	///        import function templates. Each simply forwards to @ref noise with the matching quality.
	///----------------------------------------
	///@{
	
	///----------------------------------------
	/// @brief One-dimensional classic (1985) gradient noise.
	/// @param x Sample coordinate.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float classicNoise(float x) const noexcept {
		return noise<NoiseQuality::classic>(x);
	}
	
	///----------------------------------------
	/// @brief Two-dimensional classic (1985) gradient noise.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float classicNoise(float x, float y) const noexcept {
		return noise<NoiseQuality::classic>(x, y);
	}
	
	///----------------------------------------
	/// @brief Three-dimensional classic (1985) gradient noise.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @param z Sample coordinate along z.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float classicNoise(float x, float y, float z) const noexcept {
		return noise<NoiseQuality::classic>(x, y, z);
	}
	
	///----------------------------------------
	/// @brief One-dimensional improved (2002) gradient noise.
	/// @param x Sample coordinate.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float improvedNoise(float x) const noexcept {
		return noise<NoiseQuality::improved>(x);
	}
	
	///----------------------------------------
	/// @brief Two-dimensional improved (2002) gradient noise.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float improvedNoise(float x, float y) const noexcept {
		return noise<NoiseQuality::improved>(x, y);
	}
	
	///----------------------------------------
	/// @brief Three-dimensional improved (2002) gradient noise.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @param z Sample coordinate along z.
	/// @return Noise value in approximately [-1, 1].
	///----------------------------------------
	
	[[nodiscard]] float improvedNoise(float x, float y, float z) const noexcept {
		return noise<NoiseQuality::improved>(x, y, z);
	}
	
	///@}
	
	///----------------------------------------
	/// @name Fractal (fractional Brownian motion)
	///----------------------------------------
	///@{
	
	///----------------------------------------
	/// @brief One-dimensional fractal noise summed over octaves.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate.
	/// @param options Octave count and per-octave frequency/amplitude progression.
	/// @return Amplitude-normalized noise in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float fractal(float x, const FractalOptions& options = {}) const noexcept {
		float frequency = options.frequency;
		float amplitude = 1.0f;
		float sum = 0.0f;
		float totalAmplitude = 0.0f;
		
		// Accumulate each octave at rising frequency and falling amplitude.
		for (int octave = 0; octave < options.octaves; ++octave) {
			sum += amplitude * noise<Quality>(x * frequency);
			totalAmplitude += amplitude;
			frequency *= options.lacunarity;
			amplitude *= options.gain;
		}
		
		return totalAmplitude > 0.0f ? sum / totalAmplitude : 0.0f;
	}
	
	///----------------------------------------
	/// @brief Two-dimensional fractal noise summed over octaves.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @param options Octave count and per-octave frequency/amplitude progression.
	/// @return Amplitude-normalized noise in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float fractal(float x, float y, const FractalOptions& options = {}) const noexcept {
		float frequency = options.frequency;
		float amplitude = 1.0f;
		float sum = 0.0f;
		float totalAmplitude = 0.0f;
		
		// Accumulate each octave at rising frequency and falling amplitude.
		for (int octave = 0; octave < options.octaves; ++octave) {
			sum += amplitude * noise<Quality>(x * frequency, y * frequency);
			totalAmplitude += amplitude;
			frequency *= options.lacunarity;
			amplitude *= options.gain;
		}
		
		return totalAmplitude > 0.0f ? sum / totalAmplitude : 0.0f;
	}
	
	///----------------------------------------
	/// @brief Three-dimensional fractal noise summed over octaves.
	/// @tparam Quality Algorithm variant; defaults to @ref NoiseQuality::improved.
	/// @param x Sample coordinate along x.
	/// @param y Sample coordinate along y.
	/// @param z Sample coordinate along z.
	/// @param options Octave count and per-octave frequency/amplitude progression.
	/// @return Amplitude-normalized noise in approximately [-1, 1].
	///----------------------------------------
	
	template <NoiseQuality Quality = NoiseQuality::improved>
	[[nodiscard]] float fractal(float x, float y, float z, const FractalOptions& options = {}) const noexcept {
		float frequency = options.frequency;
		float amplitude = 1.0f;
		float sum = 0.0f;
		float totalAmplitude = 0.0f;
		
		// Accumulate each octave at rising frequency and falling amplitude.
		for (int octave = 0; octave < options.octaves; ++octave) {
			sum += amplitude * noise<Quality>(x * frequency, y * frequency, z * frequency);
			totalAmplitude += amplitude;
			frequency *= options.lacunarity;
			amplitude *= options.gain;
		}
		
		return totalAmplitude > 0.0f ? sum / totalAmplitude : 0.0f;
	}
	
	///@}
	
	///----------------------------------------
	/// @brief Remaps a raw noise value from approximately [-1, 1] to approximately [0, 1].
	/// @param value Raw noise value.
	/// @return @p value shifted and scaled into the unit range.
	///----------------------------------------
	
	[[nodiscard]] static constexpr float toUnit(float value) noexcept {
		return 0.5f * value + 0.5f;
	}
	
	///----------------------------------------
	/// @brief Validates determinism, continuity, range, and the two algorithms' distinctness.
	/// @details Throws @ref Math::selftest::Failure on the first violation.
	///----------------------------------------
	
	static void selfTest();
	
private:
	std::array<uint8_t, 512> _permutation{};      ///< Shuffled 0..255, duplicated to 512.
	std::array<float, 256> _classicGradient1{};   ///< Classic 1-D random unit slopes.
	std::array<Float2, 256> _classicGradient2{}; ///< Classic 2-D random unit gradients.
	std::array<Float3, 256> _classicGradient3{}; ///< Classic 3-D random unit gradients.
	
	///----------------------------------------
	/// @brief Selects the fade curve for the requested algorithm at compile time.
	///----------------------------------------
	
	template <NoiseQuality Quality>
	[[nodiscard]] static constexpr float fade(float t) noexcept {
		if constexpr (Quality == NoiseQuality::improved) {
			return detail::fadeQuintic(t);
		} else {
			return detail::fadeCubic(t);
		}
	}
	
	///----------------------------------------
	/// @brief 1-D gradient contribution for the requested algorithm.
	///----------------------------------------
	
	template <NoiseQuality Quality>
	[[nodiscard]] float gradient1(int hash, float x) const noexcept {
		if constexpr (Quality == NoiseQuality::improved) {
			return detail::improvedGradient(hash, x);
		} else {
			return _classicGradient1[hash & 255] * x;
		}
	}
	
	///----------------------------------------
	/// @brief 2-D gradient contribution for the requested algorithm.
	/// @details The classic branch dots a stored random unit gradient with the offset through the
	///          SIMD @ref Math::dot; the improved branch selects a fixed direction by hash.
	///----------------------------------------
	
	template <NoiseQuality Quality>
	[[nodiscard]] float gradient2(int hash, const Float2& offset) const noexcept {
		if constexpr (Quality == NoiseQuality::improved) {
			return detail::improvedGradient(hash, offset);
		} else {
			return dot(_classicGradient2[hash & 255], offset);
		}
	}
	
	///----------------------------------------
	/// @brief 3-D gradient contribution for the requested algorithm.
	/// @details The classic branch dots a stored random unit gradient with the offset through the
	///          SIMD @ref Math::dot; the improved branch selects a fixed cube-edge direction by hash.
	///----------------------------------------
	
	template <NoiseQuality Quality>
	[[nodiscard]] float gradient3(int hash, const Float3& offset) const noexcept {
		if constexpr (Quality == NoiseQuality::improved) {
			return detail::improvedGradient(hash, offset);
		} else {
			return dot(_classicGradient3[hash & 255], offset);
		}
	}
	
	///----------------------------------------
	/// @brief Per-dimension scale that maps each algorithm's raw output toward [-1, 1].
	/// @details The classic variant uses unit gradients, whose lattice-noise supremum is the classic
	///          @f$\sqrt{n}/2@f$ result, so its scale is @f$2/\sqrt{n}@f$. The improved variant's
	///          edge-vector gradients already keep the 2-D and 3-D output within [-1, 1] (identity
	///          scale); only its 1-D slope, which reaches 8, needs a pullback, and 1/4 keeps it bounded.
	///----------------------------------------
	///@{
	
	template <NoiseQuality Quality>
	[[nodiscard]] static constexpr float normalization1() noexcept {
		return Quality == NoiseQuality::improved ? 0.25f : 2.0f;
	}
	
	template <NoiseQuality Quality>
	[[nodiscard]] static constexpr float normalization2() noexcept {
		return Quality == NoiseQuality::improved ? 1.0f : 1.41421356f;
	}
	
	template <NoiseQuality Quality>
	[[nodiscard]] static constexpr float normalization3() noexcept {
		return Quality == NoiseQuality::improved ? 1.0f : 1.15470054f;
	}
	
	///@}
	
	///----------------------------------------
	/// @brief Maps a raw random sample to a signed unit slope for the classic 1-D table.
	///----------------------------------------
	
	[[nodiscard]] static float normalizedSlope(float sample) noexcept {
		return sample < 0.0f ? -1.0f : 1.0f;
	}
	
	///----------------------------------------
	/// @brief Draws a random 2-D unit vector by rejection sampling the unit disc.
	///----------------------------------------
	
	template <typename Engine, typename Distribution>
	[[nodiscard]] static Float2 randomUnitVector2(Engine& engine, Distribution& distribution) noexcept {
		// Reject samples outside the unit disc (and the degenerate origin) before normalizing.
		while (true) {
			const Float2 candidate(distribution(engine), distribution(engine));
			const float magnitudeSquared = lengthSquared(candidate);
			if (magnitudeSquared > 1.0e-6f && magnitudeSquared <= 1.0f) {
				return normalize(candidate);
			}
		}
	}
	
	///----------------------------------------
	/// @brief Draws a random 3-D unit vector by rejection sampling the unit ball.
	///----------------------------------------
	
	template <typename Engine, typename Distribution>
	[[nodiscard]] static Float3 randomUnitVector3(Engine& engine, Distribution& distribution) noexcept {
		// Reject samples outside the unit ball (and the degenerate origin) before normalizing.
		while (true) {
			const Float3 candidate(distribution(engine), distribution(engine), distribution(engine));
			const float magnitudeSquared = lengthSquared(candidate);
			if (magnitudeSquared > 1.0e-6f && magnitudeSquared <= 1.0f) {
				return normalize(candidate);
			}
		}
	}
};

///----------------------------------------
/// MARK: Self-test
///----------------------------------------

inline void PerlinNoise::selfTest() {
	using Math::selftest::check;
	
	const PerlinNoise noise(12345u);
	
	// Two generators from the same seed agree everywhere; a different seed diverges.
	const PerlinNoise sameSeed(12345u);
	const PerlinNoise otherSeed(999u);
	check(noise.noise(1.5f, 2.5f, 3.5f) == sameSeed.noise(1.5f, 2.5f, 3.5f), "seed is deterministic");
	check(noise.noise(1.5f, 2.5f, 3.5f) != otherSeed.noise(1.5f, 2.5f, 3.5f), "distinct seeds differ");
	
	// Noise vanishes at integer lattice nodes: the offset-weighted gradients are all zero there.
	check(std::abs(noise.noise(4.0f)) < 1.0e-5f, "1-D zero at lattice node");
	check(std::abs(noise.noise(3.0f, -2.0f)) < 1.0e-5f, "2-D zero at lattice node");
	check(std::abs(noise.noise(1.0f, 5.0f, -7.0f)) < 1.0e-5f, "3-D zero at lattice node");
	
	// Scan a grid in every dimension and for both algorithms: values stay bounded and continuous.
	constexpr float step = 0.013f;
	constexpr float bound = 1.01f;
	constexpr float maxSlope = 0.2f;
	
	// 1-D scan for both algorithms.
	float previousImproved1 = noise.noise<NoiseQuality::improved>(-8.0f);
	float previousClassic1 = noise.noise<NoiseQuality::classic>(-8.0f);
	for (float x = -8.0f; x < 8.0f; x += step) {
		const float improved = noise.noise<NoiseQuality::improved>(x);
		const float classic = noise.noise<NoiseQuality::classic>(x);
		check(std::abs(improved) <= bound, "1-D improved in range");
		check(std::abs(classic) <= bound, "1-D classic in range");
		check(std::abs(improved - previousImproved1) < maxSlope, "1-D improved continuous");
		check(std::abs(classic - previousClassic1) < maxSlope, "1-D classic continuous");
		previousImproved1 = improved;
		previousClassic1 = classic;
	}
	
	// 2-D and 3-D scan for both algorithms: range only (continuity is covered by the 1-D scan).
	for (float x = -6.0f; x < 6.0f; x += step) {
		const float y = x * 0.5f + 1.25f;
		const float z = -x * 0.25f + 0.5f;
		check(std::abs(noise.noise<NoiseQuality::improved>(x, y)) <= bound, "2-D improved in range");
		check(std::abs(noise.noise<NoiseQuality::classic>(x, y)) <= bound, "2-D classic in range");
		check(std::abs(noise.noise<NoiseQuality::improved>(x, y, z)) <= bound, "3-D improved in range");
		check(std::abs(noise.noise<NoiseQuality::classic>(x, y, z)) <= bound, "3-D classic in range");
	}
	
	// The two algorithms are genuinely different away from the lattice nodes.
	check(noise.noise<NoiseQuality::improved>(0.4f, 0.7f, 0.2f) != noise.noise<NoiseQuality::classic>(0.4f, 0.7f, 0.2f),
		  "algorithms differ");
		  
	// Vector overloads agree with their scalar counterparts.
	check(noise.noise(Float2(2.3f, 4.1f)) == noise.noise(2.3f, 4.1f), "2-D vector overload");
	check(noise.noise(Float3(2.3f, 4.1f, 0.6f)) == noise.noise(2.3f, 4.1f, 0.6f), "3-D vector overload");
	
	// Fractal output stays within a single octave's range and honors the identity of one octave.
	const FractalOptions singleOctave{.octaves = 1};
	check(std::abs(noise.fractal(1.7f, 2.9f) ) <= bound, "fractal in range");
	check(noise.fractal(1.7f, 2.9f, singleOctave) == noise.noise(1.7f, 2.9f), "one octave equals raw noise");
	
	// The unit remap sends [-1, 1] onto [0, 1] at the endpoints and midpoint.
	check(std::abs(PerlinNoise::toUnit(-1.0f) - 0.0f) < 1.0e-6f, "toUnit lower bound");
	check(std::abs(PerlinNoise::toUnit(0.0f) - 0.5f) < 1.0e-6f, "toUnit midpoint");
	check(std::abs(PerlinNoise::toUnit(1.0f) - 1.0f) < 1.0e-6f, "toUnit upper bound");
}

///----------------------------------------
} // namespace Math
///----------------------------------------

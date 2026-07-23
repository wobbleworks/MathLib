///----------------------------------------
///      @file BlackbodyColor.h
///   @ingroup MathLib
///     @brief Planck spectral radiance and two-band blackbody temperature estimation.
///   @details @ref Math::planckSpectralRadiance evaluates Planck's law for a single wavelength, and
///            the @ref Math::BlackbodyColor class template converts between photometric band
///            magnitudes, color indices, and temperature for a fixed pair of wavelengths. Depends
///            only on the standard library and @ref Numbers.h.
///    @author Created by John Stephen on 7/7/26.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <cmath>

#include "MathLib/Numbers.h"
#include "MathLib/Vector.h"
#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// MARK: Planck radiance
///----------------------------------------

///----------------------------------------
///   @brief Planck spectral radiance for a single wavelength and temperature.
///   @param wavelengthNm Wavelength in nanometres.
///   @param temperatureK Blackbody temperature in kelvin.
/// @returns The spectral radiance in W·sr⁻¹·m⁻³.
///----------------------------------------

[[nodiscard]] inline float planckSpectralRadiance(float wavelengthNm, float temperatureK) noexcept {
	// Work in metres; raise to the fifth power by multiplication rather than std::pow
	const float wavelengthMeters = 1.0e-9f * wavelengthNm;
	const double meters = wavelengthMeters;
	const double wavelengthToFifth = meters * meters * meters * meters * meters;
	
	const double exponent = (math::numbers::planck_h * math::numbers::einstein_c)
		/ (wavelengthMeters * math::numbers::boltzmann_k * temperatureK);
	const double radiance = (2.0 * math::numbers::planck_h * math::numbers::einstein_c * math::numbers::einstein_c)
		/ (wavelengthToFifth * (std::exp(exponent) - 1.0));
		
	return static_cast<float>(radiance);
}

///----------------------------------------
///   @brief Fraction of a blackbody's total emission radiated between two wavelengths.
/// @details Integrates @ref planckSpectralRadiance across the band with composite Simpson's rule
///          and divides by the Stefan-Boltzmann total radiance σT⁴/π, with σ derived from the
///          same physical constants the integrand uses.
///   @param shortWavelengthNm Short edge of the band, in nanometres.
///   @param longWavelengthNm Long edge of the band, in nanometres.
///   @param temperatureK Blackbody temperature in kelvin.
/// @returns The fraction of total emission falling within the band, in (0, 1).
///----------------------------------------

[[nodiscard]] inline double blackbodyBandFluxFraction(double shortWavelengthNm, double longWavelengthNm, double temperatureK) noexcept {
	// Integrate the spectral radiance across the band with composite Simpson's rule
	constexpr auto numIntervals = 128; // must be even
	const double stepNm = (longWavelengthNm - shortWavelengthNm) / numIntervals;
	double bandRadiance = planckSpectralRadiance(static_cast<float>(shortWavelengthNm), static_cast<float>(temperatureK))
		+ planckSpectralRadiance(static_cast<float>(longWavelengthNm), static_cast<float>(temperatureK));
	for (int interval = 1; interval < numIntervals; ++interval) {
		const double wavelengthNm = shortWavelengthNm + interval * stepNm;
		const double weight = (interval % 2 == 1)? 4.0 : 2.0;
		bandRadiance += weight * planckSpectralRadiance(static_cast<float>(wavelengthNm), static_cast<float>(temperatureK));
	}
	bandRadiance *= stepNm * 1.0e-9 / 3.0; // wavelength step in metres
	
	// Total radiance over all wavelengths is σT⁴/π, with σ = 2π⁵k⁴ / (15h³c²)
	constexpr double pi = std::numbers::pi;
	constexpr double piToFifth = pi * pi * pi * pi * pi;
	constexpr double boltzmannToFourth = math::numbers::boltzmann_k * math::numbers::boltzmann_k
		* math::numbers::boltzmann_k * math::numbers::boltzmann_k;
	constexpr double planckToThird = math::numbers::planck_h * math::numbers::planck_h * math::numbers::planck_h;
	constexpr double stefanBoltzmann = 2.0 * piToFifth * boltzmannToFourth
		/ (15.0 * planckToThird * math::numbers::einstein_c * math::numbers::einstein_c);
	const double temperatureSquared = temperatureK * temperatureK;
	const double totalRadiance = stefanBoltzmann * temperatureSquared * temperatureSquared / pi;
	
	// The band's share of the total
	return bandRadiance / totalRadiance;
}

[[nodiscard]] inline RGBColor blackbodyLinearRGB(float temperatureK) {
	// Representative RGB wavelengths
	static constexpr int wavelength_r = 611; // nm
	static constexpr int wavelength_g = 548; // nm
	static constexpr int wavelength_b = 454; // nm

	/// Compute a linear (radiometric) max-normalized blackbody RGB chromaticity for a temperature in
	/// Kelvin. This applies no display gamma, so it is suitable as the color term of a linear-space
	/// HDR multiply (the magnitude is carried separately).
	temperatureK = std::max(temperatureK, 500.0f);
	auto radiance_r = planckSpectralRadiance(wavelength_r, temperatureK);
	auto radiance_g = planckSpectralRadiance(wavelength_g, temperatureK);
	auto radiance_b = planckSpectralRadiance(wavelength_b, temperatureK);
	auto max_radiance = std::max(std::max(radiance_r, radiance_g), radiance_b);
	return RGBColor{radiance_r, radiance_g, radiance_b} / max_radiance;
}

[[nodiscard]] inline RGBColor blackbodyRGB(float temperatureK) {
	auto linearRGB = blackbodyLinearRGB(temperatureK);
	constexpr float gamma = 1.0f / 2.2f;
	return RGBColor{std::powf(linearRGB.r, gamma), std::powf(linearRGB.g, gamma), std::powf(linearRGB.b, gamma)};
}

///----------------------------------------
namespace detail {
///----------------------------------------

/// @brief Natural log of the flux ratio for a one-magnitude difference (Pogson): ln(100^0.2).
inline const double logFluxRatioPerMagnitude = 0.2 * std::log(100.0);

///----------------------------------------
} // namespace detail
///----------------------------------------

///----------------------------------------
/// MARK: Two-band estimator
///----------------------------------------

///----------------------------------------
///   @brief Blackbody temperature and color-index estimator for a fixed pair of photometric bands.
/// @details Band A is the longer (redder) wavelength and band B the shorter (bluer). The inverse
///          direction fits the color-index/temperature relation with a two-term analytic model; the
///          forward direction predicts one band's magnitude from the other via Planck radiance.
/// @tparam TWavelengthNmA Band A wavelength in nanometres.
/// @tparam TWavelengthNmB Band B wavelength in nanometres.
///----------------------------------------

template <int TWavelengthNmA, int TWavelengthNmB>
class BlackbodyColor final {
///----------------------------------------
public:
	///   @brief Estimate temperature in kelvin from the two band magnitudes.
	[[nodiscard]] static double temperatureFromMagnitudes(double magnitudeA, double magnitudeB) noexcept {
		return analyticTemperature(magnitudeB - magnitudeA);
	}
	
	///   @brief Estimate temperature in kelvin from a color index (band B minus band A).
	[[nodiscard]] static double temperatureFromColorIndex(double colorIndex) noexcept {
		static_assert(TWavelengthNmA > TWavelengthNmB, "color-index inversion requires band A redder than band B");
		return analyticTemperature(colorIndex);
	}
	
	///   @brief Predict the band-B magnitude from the band-A magnitude and a temperature in kelvin.
	[[nodiscard]] static double magnitudeBForTemperature(double magnitudeA, double temperatureK) noexcept {
		return magnitudeA + colorIndexForTemperature(temperatureK);
	}
	
///----------------------------------------
private:
	///----------------------------------------
	/// @brief Per-instantiation constants for the analytic color-index inversion.
	///----------------------------------------
	struct Coefficients final {
		double referenceTemperature;
		double logRatioLong;
		double logRatioShort;
	};
	
	///   @brief The instantiation's coefficients, evaluated once on first use.
	[[nodiscard]] static const Coefficients& coefficients() noexcept {
		static const Coefficients coefficients = computeCoefficients();
		return coefficients;
	}
	
	///   @brief Derive the analytic-fit coefficients from the two band wavelengths.
	[[nodiscard]] static Coefficients computeCoefficients() noexcept {
		constexpr double secondRadiationConstant =
			math::numbers::planck_h * math::numbers::einstein_c / math::numbers::boltzmann_k;
		constexpr double thermalScaleA = secondRadiationConstant / (TWavelengthNmA * 1.0e-9);
		constexpr double thermalScaleB = secondRadiationConstant / (TWavelengthNmB * 1.0e-9);
		
		const double logWavelengthRatio = std::log(thermalScaleB / thermalScaleA);
		const double gammaExponent = 5.0 - 9.0 * std::pow(std::fabs(logWavelengthRatio) + 3.0, -0.252);
		
		return Coefficients{
			.referenceTemperature = 1.6 * (thermalScaleB - thermalScaleA) / 2.0,
			.logRatioLong = 4.0 * logWavelengthRatio,
			.logRatioShort = (5.0 - gammaExponent) * logWavelengthRatio,
		};
	}
	
	///   @brief Two-term analytic inversion of a color index to a temperature in kelvin.
	[[nodiscard]] static double analyticTemperature(double colorIndex) noexcept {
		const auto& coefficient = coefficients();
		const auto scaledIndex = detail::logFluxRatioPerMagnitude * colorIndex;
		return coefficient.referenceTemperature
			* (1.0 / (scaledIndex + coefficient.logRatioLong) + 1.0 / (scaledIndex + coefficient.logRatioShort));
	}
	
	///   @brief Color index (band B minus band A) predicted for a temperature via Planck radiance.
	[[nodiscard]] static double colorIndexForTemperature(double temperatureK) noexcept {
		const auto radianceA = planckSpectralRadiance(TWavelengthNmA, temperatureK);
		const auto radianceB = planckSpectralRadiance(TWavelengthNmB, temperatureK);
		constexpr double pogsonLogFactor = 0.921034037197618;
		return -pogsonLogFactor * std::log(radianceB / radianceA);
	}
};

///----------------------------------------
/// @brief Verifies Planck radiance monotonicity, the Wien peak shift, and the color-index inversion.
/// @details Throws @ref Math::selftest::Failure on the first violation. Tests robust invariants rather
///          than transcribed physical constants, over a normal stellar temperature range.
///----------------------------------------
inline void blackbodyColorSelfTest() {
	using selftest::check;
	
	// Planck radiance is strictly positive for a warm body.
	{
		check(planckSpectralRadiance(500.0f, 6000.0f) > 0.0f, "radiance is positive at 6000 K");
		check(planckSpectralRadiance(500.0f, 3000.0f) > 0.0f, "radiance is positive at 3000 K");
	}
	
	// At a fixed wavelength, a hotter body radiates more.
	{
		const float cool = planckSpectralRadiance(500.0f, 3000.0f);
		const float warm = planckSpectralRadiance(500.0f, 6000.0f);
		check(warm > cool, "radiance rises with temperature at fixed wavelength");
	}
	
	// The spectral peak shifts to shorter wavelengths as temperature rises: a hotter body is
	// relatively bluer, so its short/long radiance ratio exceeds that of a cooler body.
	{
		const double coolRatio = planckSpectralRadiance(400.0f, 3000.0f) / planckSpectralRadiance(800.0f, 3000.0f);
		const double warmRatio = planckSpectralRadiance(400.0f, 6000.0f) / planckSpectralRadiance(800.0f, 6000.0f);
		check(warmRatio > coolRatio, "hotter bodies weight shorter wavelengths more (Wien shift)");
	}
	
	// The forward color index (band-B magnitude predicted from a zero band-A magnitude) grows redder
	// as temperature falls: a cooler star has a larger, more positive color index than a hotter one.
	{
		using Band = BlackbodyColor<550, 440>;
		const double coolIndex = Band::magnitudeBForTemperature(0.0, 3000.0);
		const double warmIndex = Band::magnitudeBForTemperature(0.0, 6000.0);
		check(coolIndex > warmIndex, "cooler bodies have a redder (larger) color index");
	}
	
	// The analytic inversion is monotonic and stays in a sane stellar range: a redder color index
	// yields a cooler temperature, and both temperatures are positive.
	{
		using Band = BlackbodyColor<550, 440>;
		const double redderTemperature = Band::temperatureFromColorIndex(1.0);
		const double bluerTemperature = Band::temperatureFromColorIndex(0.0);
		check(redderTemperature > 0.0 && bluerTemperature > 0.0, "inverted temperatures are positive");
		check(bluerTemperature > redderTemperature, "a bluer color index inverts to a hotter temperature");
	}
	
	// The two entry points agree: a temperature from two band magnitudes matches the temperature from
	// their color index (band B minus band A).
	{
		using Band = BlackbodyColor<550, 440>;
		const double fromMagnitudes = Band::temperatureFromMagnitudes(6.2, 6.7);
		const double fromColorIndex = Band::temperatureFromColorIndex(6.7 - 6.2);
		check(std::abs(fromMagnitudes - fromColorIndex) < 1.0e-6, "magnitude and color-index entry points agree");
	}
	
	// The band flux fraction is a proper fraction, and a broad optical band captures more of a
	// sunlike star's emission than of a cool star's (which leaks into the infrared) or a hot
	// star's (which leaks into the ultraviolet).
	{
		const double cool = blackbodyBandFluxFraction(330.0, 1050.0, 3000.0);
		const double sunlike = blackbodyBandFluxFraction(330.0, 1050.0, 5778.0);
		const double hot = blackbodyBandFluxFraction(330.0, 1050.0, 30000.0);
		check(cool > 0.0 && cool < 1.0, "cool-star band fraction is a proper fraction");
		check(sunlike > 0.0 && sunlike < 1.0, "sunlike band fraction is a proper fraction");
		check(hot > 0.0 && hot < 1.0, "hot-star band fraction is a proper fraction");
		check(sunlike > cool, "an optical band captures more of a sunlike star than a cool star");
		check(sunlike > hot, "an optical band captures more of a sunlike star than a hot star");
	}
	
	// A wider band captures a larger share of the emission.
	{
		const double narrow = blackbodyBandFluxFraction(400.0, 700.0, 5778.0);
		const double wide = blackbodyBandFluxFraction(330.0, 1050.0, 5778.0);
		check(wide > narrow, "a wider band captures a larger flux fraction");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

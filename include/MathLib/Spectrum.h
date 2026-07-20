///----------------------------------------
///       @file Spectrum.h
///    @ingroup MathLib
///      @brief A sampled spectral power distribution over a wavelength range.
///    @details Stores uniformly-spaced sample values across a wavelength interval (in nanometers)
///             and provides linear interpolation (@ref valueAt) plus the CIE 1931 2° standard-observer
///             tristimulus integral (@ref toXYZ). A self-owning value type — copyable and movable.
///     @author Created by John Stephen on 6/24/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/XYZColor.h"
#include "MathLib/SelfTestCheck.h"

#include <span>
#include <vector>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
///   @class Spectrum
///   @brief A sampled spectral power distribution over a wavelength range, in nanometers.
/// @details The samples are uniformly spaced from @ref minWavelength to @ref maxWavelength,
///          inclusive. @ref valueAt interpolates between them; @ref toXYZ integrates the
///          distribution against the CIE 1931 standard observer to give a tristimulus color.
///----------------------------------------

class Spectrum {
public:
	Spectrum() = default;
	
	///----------------------------------------
	///   @brief Constructs a spectrum from uniformly-spaced sample values.
	///   @param minWavelength Wavelength of the first sample, in nanometers.
	///   @param maxWavelength Wavelength of the last sample, in nanometers.
	///   @param samples The sample values, evenly spaced across the inclusive range.
	///----------------------------------------
	
	Spectrum(float minWavelength, float maxWavelength, std::span<const float> samples);
	
	///----------------------------------------
	///   @brief The linearly interpolated value at a wavelength.
	///   @param wavelength The wavelength, in nanometers.
	/// @return The interpolated value, or 0 if the wavelength is outside the sampled range.
	///----------------------------------------
	
	[[nodiscard]] float valueAt(float wavelength) const noexcept;
	
	///----------------------------------------
	/// @brief The CIE 1931 2° standard-observer tristimulus (X, Y, Z) of the distribution.
	///----------------------------------------
	
	[[nodiscard]] XYZColor toXYZ() const noexcept;
	
	/// @brief The sample values.
	[[nodiscard]] std::span<const float> samples() const noexcept { return _samples; }
	
	/// @brief Wavelength of the first sample, in nanometers.
	[[nodiscard]] float minWavelength() const noexcept { return _minWavelength; }
	
	/// @brief Wavelength of the last sample, in nanometers.
	[[nodiscard]] float maxWavelength() const noexcept { return _maxWavelength; }
	
	/// @brief Whether the spectrum holds no samples.
	[[nodiscard]] bool isEmpty() const noexcept { return _samples.empty(); }
	
	///----------------------------------------
	/// @brief Validates interpolation and the tristimulus integral.
	///----------------------------------------
	
	static void selfTest();
	
private:
	std::vector<float> _samples;
	float _minWavelength = 0;
	float _maxWavelength = 0;
};

inline Spectrum::Spectrum(float minWavelength, float maxWavelength, std::span<const float> samples)
:
	_samples(samples.begin(), samples.end()),
	_minWavelength(minWavelength),
	_maxWavelength(maxWavelength) {
}

inline float Spectrum::valueAt(float wavelength) const noexcept {
	// Outside the sampled range contributes nothing
	if (_samples.size() < 2 || wavelength < _minWavelength || wavelength > _maxWavelength) {
		return 0;
	}
	
	// Locate the enclosing sample interval
	auto sampleWidth = (_maxWavelength - _minWavelength) / static_cast<float>(_samples.size() - 1);
	auto position = (wavelength - _minWavelength) / sampleWidth;
	auto index = static_cast<std::size_t>(position);
	
	// Clamp the upper edge to the last sample
	if (index >= _samples.size() - 1) {
		return _samples.back();
	}
	
	// Interpolate within the interval
	return std::lerp(_samples[index], _samples[index + 1], position - static_cast<float>(index));
}

inline XYZColor Spectrum::toXYZ() const noexcept {
	// Weighted sum of the distribution against the CIE 1931 standard observer
	auto tristimulus = XYZColor{};
	auto wavelength = static_cast<float>(detail::cieMinWavelength);
	for (const auto& weights : detail::cieStandardObserver) {
		tristimulus.vec += valueAt(wavelength) * Float3(weights.X, weights.Y, weights.Z);
		wavelength += static_cast<float>(detail::cieWavelengthStep);
	}
	
	return tristimulus;
}

inline void Spectrum::selfTest() {
	auto approxEqual = [](float a, float b) { return std::fabs(a - b) <= 1.e-5f; };
	
	// A flat unit spectrum across the visible range
	std::array<float, 81> flatSamples;
	flatSamples.fill(1.0f);
	Spectrum flat(380.0f, 780.0f, flatSamples);
	
	// Interpolation yields the sample value inside the range and zero outside it
	Math::selftest::check(flat.valueAt(380.0f) == 1.0f && flat.valueAt(580.0f) == 1.0f && flat.valueAt(780.0f) == 1.0f, "flat spectrum samples");
	Math::selftest::check(flat.valueAt(379.0f) == 0.0f && flat.valueAt(781.0f) == 0.0f, "out-of-range samples are zero");
	
	// A flat unit spectrum integrates to the summed observer weights (the Y sum is ~21.37)
	auto white = flat.toXYZ();
	Math::selftest::check(white.Y > 21.0f && white.Y < 22.0f, "flat-spectrum luminance integral");
	Math::selftest::check(white.X > 0.0f && white.Z > 0.0f, "flat-spectrum X and Z are positive");
	
	// Linear interpolation between samples
	const std::array<float, 3> rampSamples = {0.0f, 1.0f, 2.0f}; // 400, 500, 600 nm
	Spectrum ramp(400.0f, 600.0f, rampSamples);
	Math::selftest::check(approxEqual(ramp.valueAt(450.0f), 0.5f) && approxEqual(ramp.valueAt(550.0f), 1.5f), "linear interpolation");
}

///----------------------------------------
} // namespace Math
///----------------------------------------

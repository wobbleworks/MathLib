///----------------------------------------
/// @file Numbers.h
/// @ingroup MathLib
/// @brief Mathematical and unit-conversion constants.
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <numbers>

///----------------------------------------
namespace math::numbers {
///----------------------------------------

// Pi variants

static constexpr double two_pi = 2 * std::numbers::pi;
static constexpr double half_pi = std::numbers::pi / 2;

static constexpr float pi_float = std::numbers::pi_v<float>;
static constexpr float two_pi_float = 2 * pi_float;
static constexpr float half_pi_float = pi_float / 2;

static constexpr float e_float = std::numbers::e_v<float>;

// Degree / radian conversion

static constexpr double deg2rad = std::numbers::pi / 180;
static constexpr double rad2deg = 180 / std::numbers::pi;
static constexpr float deg2radf = std::numbers::pi / 180;
static constexpr float rad2degf = 180 / std::numbers::pi;

// Arcsecond / degree conversion

static constexpr double arcsec2deg = 1.0 / 3600.0;
static constexpr double deg2arcsec = 3600.0;
static constexpr float arcsec2degf = 1.0f / 3600.0f;
static constexpr float deg2arcsecf = 3600.0f;

// Milliarcsecond / degree conversion

static constexpr double mas2deg = 1.0 / 3600000.0;
static constexpr double deg2mas = 3600000.0;
static constexpr float mas2degf = 1.0f / 3600000.0f;
static constexpr float deg2masf = 3600000.0f;

// Milliarcsecond / radian conversion

static constexpr double mas2rad = std::numbers::pi / (3600000 * 180);
static constexpr double rad2mas = (3600000 * 180) / std::numbers::pi;
static constexpr float mas2radf = std::numbers::pi / (3600000 * 180);
static constexpr float rad2masf = (3600000 * 180) / std::numbers::pi;

// Arcsecond / radian conversion

static constexpr double arcsec2rad = arcsec2deg * deg2rad;
static constexpr double rad2arcsec = rad2deg * deg2arcsec;
static constexpr float arcsec2radf = arcsec2degf * deg2radf;
static constexpr float rad2arcsecf = rad2degf * deg2arcsecf;

// Arcminute / degree conversion

static constexpr double arcmin2deg = 1.0 / 60.0;
static constexpr double deg2arcmin = 60.0;
static constexpr float arcmin2degf = 1.0f / 60.0f;
static constexpr float deg2arcminf = 60.0f;

// Arcminute / radian conversion

static constexpr double arcmin2rad = arcmin2deg * deg2rad;
static constexpr double rad2arcmin = rad2deg * deg2arcmin;
static constexpr float arcmin2radf = arcmin2degf * deg2radf;
static constexpr float rad2arcminf = rad2degf * deg2arcminf;

// Degree / hour-angle conversion

static constexpr double deg2hours = 12.0 / 180;
static constexpr double hours2deg = 180.0 / 12;
static constexpr float deg2hoursf = 12.0f / 180;
static constexpr float hours2degf = 180.0f / 12;

// Radian / hour-angle conversion

static constexpr double rad2hours = 12 / std::numbers::pi;
static constexpr double hours2rad = std::numbers::pi / 12;
static constexpr float rad2hoursf = 12 / std::numbers::pi;
static constexpr float hours2radf = std::numbers::pi / 12;

// Metric / imperial length conversion

static constexpr double meters2feet = 3.28084;
static constexpr double feet2meters = 1.0 / 3.28084;
static constexpr float meters2feetf = 3.28084f;
static constexpr float feet2metersf = 1.0f / 3.28084f;

static constexpr double meters2inches = meters2feet * 12.0;
static constexpr double inches2meters = feet2meters / 12.0;
static constexpr float meters2inchesf = meters2feetf * 12.0f;
static constexpr float inches2metersf = feet2metersf / 12.0f;

static constexpr double miles2feet = 5280.0;
static constexpr double feet2miles = 1.0 / 5280.0;
static constexpr float miles2feetf = 5280.0f;
static constexpr float feet2milesf = 1.0f / 5280.0f;

static constexpr double meters2miles = meters2feet * feet2miles;
static constexpr double miles2meters = miles2feet * feet2meters;
static constexpr float meters2milesf = meters2feetf * feet2milesf;
static constexpr float miles2metersf = miles2feetf * feet2metersf;

// Astronomical unit conversion

static constexpr double meters2au = 1.0 / 149597870700.0;
static constexpr double au2meters = 149597870700.0;
static constexpr float meters2auf = 1.0f / 149597870700.0f;
static constexpr float au2metersf = 149597870700.0f;

// Light-year conversion

static constexpr double meters2ly = 1.0 / 9460536207068016.0;
static constexpr double ly2meters = 9460536207068016.0;
static constexpr float meters2lyf = 1.0f / 9460536207068016.0f;
static constexpr float ly2metersf = 9460536207068016.0f;

// Light-day conversion

static constexpr double meters2lightdays = meters2ly * 365.25;
static constexpr double lightdays2meters = ly2meters / 365.25;
static constexpr float meters2lightdaysf = meters2lyf * 365.25f;
static constexpr float lightdays2metersf = ly2metersf / 365.25f;

// Light-second conversion

static constexpr double meters2lightseconds = meters2lightdays * 86400.0;
static constexpr double lightseconds2meters = lightdays2meters / 86400.0;
static constexpr float meters2lightsecondsf = meters2lightdaysf * 86400.0f;
static constexpr float lightseconds2metersf = lightdays2metersf / 86400.0f;

// Kilometer / AU conversion

static constexpr double km2au = 1.0 / 149597870.7;
static constexpr double au2km = 149597870.7;
static constexpr float km2auf = 1.0f / 149597870.7f;
static constexpr float au2kmf = 149597870.7f;

// Parsec conversion

static constexpr double parsecs2au = 648000.0 / std::numbers::pi;
static constexpr double au2parsecs = std::numbers::pi / 648000.0;
static constexpr float parsecs2auf = 648000.0f / std::numbers::pi;
static constexpr float au2parsecsf = std::numbers::pi / 648000.0f;

static constexpr double kpc2au = 648000000.0 / std::numbers::pi;
static constexpr double au2kpc = std::numbers::pi / 648000000.0;
static constexpr float kpc2auf = 648000000.0f / std::numbers::pi;
static constexpr float au2kpcf = std::numbers::pi / 648000000.0f;

static constexpr double parsecs2meters = parsecs2au * au2meters;
static constexpr double meters2parsecs = meters2au * au2parsecs;
static constexpr float parsecs2metersf = parsecs2auf * au2metersf;
static constexpr float meters2parsecsf = meters2auf * au2parsecsf;

static constexpr double parsecs2ly = parsecs2meters * meters2ly;
static constexpr double ly2parsecs = ly2meters * meters2parsecs;
static constexpr float parsecs2lyf = parsecs2metersf * meters2lyf;
static constexpr float ly2parsecsf = ly2metersf * meters2parsecsf;

// AU / light-year cross-conversion

static constexpr double au2ly = au2meters * meters2ly;
static constexpr double ly2au = ly2meters * meters2au;
static constexpr float au2lyf = au2metersf * meters2lyf;
static constexpr float ly2auf = ly2metersf * meters2auf;

static constexpr double au2lightdays = au2meters * meters2lightdays;
static constexpr double lightdays2au = lightdays2meters * meters2au;
static constexpr float au2lightdaysf = au2metersf * meters2lightdaysf;
static constexpr float lightdays2auf = lightdays2metersf * meters2auf;

// Velocity conversion

static constexpr double auPerDayToMetersPerSecond = au2meters / 86400;

// Physical constants
static constexpr double einstein_c = 2.99792458e8; // m/s
static constexpr double planck_h = 6.6260695729e-34; // Planck, J-s
static constexpr double boltzmann_k = 1.3806488e-23; // Boltzmann, J/K
	
} // namespace math::numbers

// Also expose these constants at file scope for unqualified use.
using namespace math::numbers;

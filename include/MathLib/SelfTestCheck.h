///----------------------------------------
/// @file SelfTestCheck.h
/// @ingroup MathLib
/// @brief Assertion primitive for MathLib's in-header self-tests.
/// @details Provides @ref Math::selftest::check, the building block each type's @c selfTest body
///          uses to assert a property. A violation throws @ref Math::selftest::Failure carrying a
///          message prefixed with the failing call site. Keeping this primitive inside MathLib lets
///          a client include any MathLib header without pulling in CoreLib or any external test
///          framework; the registration that runs these tests lives in the opt-in @ref SelfTest.h,
///          which is the only MathLib header that depends on CoreLib.
/// @author John Stephen
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <format>
#include <source_location>
#include <stdexcept>
#include <string_view>

///----------------------------------------
namespace Math::selftest {
///----------------------------------------

///----------------------------------------
/// @struct Failure
/// @brief Exception thrown by @ref check when a self-test condition does not hold.
/// @details Carries a message prefixed with the failing call site, which the registry that runs the
///          test reports alongside the test name. Derives from @c std::runtime_error so the shared
///          runner catches it as a plain @c std::exception regardless of which library raised it.
///----------------------------------------

struct Failure final : std::runtime_error {
	using std::runtime_error::runtime_error;
};

///----------------------------------------
/// @brief Throws @ref Failure unless a self-test condition holds.
/// @details The canonical building block for a @c selfTest body: assert the property that should be
///          true, give it a short label, and let a violation propagate to the test runner.
/// @param condition Property that must be true for the test to pass.
/// @param message Short description of what was being checked.
/// @param location Failing call site; captured automatically.
///----------------------------------------

inline void check(bool condition, std::string_view message = {},
                  std::source_location location = std::source_location::current()) {
	// Nothing to report while the property holds.
	if (condition) {
		return;
	}
	
	// Surface the failing call site and label as the exception message.
	throw Failure(std::format("{}:{}: {}", location.file_name(), location.line(), message));
}
	
} // namespace Math::selftest

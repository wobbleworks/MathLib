///----------------------------------------
///       @file mathlib_selftest_main.cpp
///    @ingroup MathLib
///      @brief Standalone runner that executes every registered MathLib self-test with a pass/fail exit code.
///    @details Not part of the library archive. Including @ref MathLib/SelfTest.h registers every
///             self-tested module with @ref Core::selftest::Registry, so this runner does not name them
///             individually — it asks the registry to run them all and maps the failure count to the process
///             exit code that a continuous-integration job needs. The registry is active only in a @c DEBUG
///             build, so the test target defines @c DEBUG; @ref Core::selftest::testCount guards against a
///             misconfigured build passing vacuously with no tests registered.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#include "MathLib/SelfTest.h"

#include "CoreLib/SelfTestRegistry.h"

#include <cstdio>

///----------------------------------------
/// @brief Runs every registered MathLib self-test and returns a non-zero exit code if any failed.
///----------------------------------------

int main() {
	// With no test registered, fail rather than report success without running anything.
	const auto total = Core::selftest::testCount();
	if (total == 0) {
		std::printf("No self-tests registered.\n");
		return 1;
	}

	// Run them all; the registry reports each result and returns the number that failed.
	const auto failures = Core::selftest::runAll();

	// Summarize and fail the process if any test did not pass.
	std::printf("\n%zu of %zu self-tests passed, %zu skipped.\n",
		total - failures, total, Core::selftest::skippedCount());
	return failures == 0 ? 0 : 1;
}

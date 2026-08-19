#----------------------------------------
# MathLib
#
# Vector / matrix / quaternion math, color and spectrum types, Perlin noise,
# Catmull-Rom splines, and closed-form numerical solvers (polynomials, tridiagonal
# systems, strided-array "Blast" primitives) for the wobbleworks stack.
#
# The implementation is header-only (C++23) and portable. The vector-math types select a
# backend at compile time (include/MathLib/detail/BackendSelect.h): Apple's <simd/simd.h>,
# Clang's vector extensions, baseline SSE2 intrinsics, or plain portable structs — so the
# library builds on Clang, GCC, and MSVC alike. Blast.h uses Accelerate (vDSP / vForce) on
# Apple and a scalar path elsewhere. A single empty anchor source (src/MathLib.cpp) is
# compiled so the target still produces a static archive.
#----------------------------------------

cmake_minimum_required(VERSION 3.20)

# The released version, following Semantic Versioning: the major component changes when the
# public API breaks, the minor when it grows compatibly, the patch for compatible fixes.
# CHANGELOG.md records what each release contains. It is set here, unconditionally and ahead of
# project(), because the install rules below also run in a superbuild where MathLib's own
# project() never executes — reading PROJECT_VERSION there would stamp the package version file
# with the enclosing project's version instead of this library's. The release workflow parses
# this line and rejects a tag that disagrees with it.
set(MATHLIB_VERSION 1.0.0)

# Declare a project only at top level; when added via add_subdirectory() the
# parent project() and toolchain govern. The version is what find_package(MathLib 1.0)
# negotiates against; it is carried here rather than in a tag so a source checkout and an
# installed package agree without consulting git.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	project(MathLib VERSION ${MATHLIB_VERSION} LANGUAGES CXX)
endif()

# GNUInstallDirs defines CMAKE_INSTALL_INCLUDEDIR / _LIBDIR, which the public include
# directories below reference through their install-tree generator expression.
include(GNUInstallDirs)

#----------------------------------------
# Primary build
#
# True when this library owns its install rules and test targets: either it is the top-level
# project, or the repo-root superbuild has declared it a deliberate member of a multi-library
# build rather than somebody's private subdirectory. A sibling that pulls this library in with
# add_subdirectory gets neither, which is what keeps its headers out of the wrong install prefix.
#----------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR WW_SUPERBUILD)
	set(MATHLIB_PRIMARY_BUILD ON)
else()
	set(MATHLIB_PRIMARY_BUILD OFF)
endif()

add_library(MathLib)

#----------------------------------------
# Namespaced alias
#
# Consumers spell the dependency MathLib::MathLib whether they picked it up through
# add_subdirectory() or find_package(); the double colon also makes CMake fail at configure
# time on a misspelling rather than passing the name through to the linker as a bare -l.
#----------------------------------------
add_library(MathLib::MathLib ALIAS MathLib)

#----------------------------------------
# Public API
#
# Headers live under include/MathLib/, so the include path is `include` and
# consumers keep writing  #include "MathLib/Vector.h".
#
# The two generator expressions describe the same headers in the two places they can live: the
# source tree while MathLib is being built, and the install prefix once it has been packaged.
# An exported target may not name a path inside the source directory, so a plain `include`
# here would make install(EXPORT) below fail outright.
#----------------------------------------
target_include_directories(MathLib PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#----------------------------------------
# Companion-library header dependency
#
# MathLib itself has none. The opt-in SelfTest.h aggregator includes "CoreLib/SelfTestRegistry.h"
# to register the inline module self-tests with CoreLib's runner, and it is the only MathLib
# header that names CoreLib at all — nothing in MathLib includes SelfTest.h, and the library's
# sole compiled source is an empty anchor translation unit. So the MathLib *target* needs no
# CoreLib include path, and the exported package declares no dependency on CoreLib: a consumer
# who wants Vector.h is not made to go and find CoreLib first.
#
# What that leaves is a consumer-side contract rather than a build-side one: a translation unit
# that includes MathLib/SelfTest.h must supply CoreLib's headers itself, exactly as it must
# already supply CoreLib's self-test runner to do anything with them. The cache variable below
# stays because the self-test runner at the bottom of this file genuinely needs it.
#----------------------------------------
get_filename_component(MATHLIB_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(CORELIB_INCLUDE_DIR "${MATHLIB_PARENT_DIR}/CoreLib/include"
	CACHE PATH "Directory containing the CoreLib/ header tree (used by the self-test runner)")

target_compile_features(MathLib PUBLIC cxx_std_23)

#----------------------------------------
# Accelerate framework (transitive, Apple non-watchOS)
#
# Blast.h and Tridiagonal.h are public headers whose fast paths call vDSP / vForce from
# Accelerate. Because MathLib is header-only for Blast (the anchor TU is empty), those
# symbols are emitted in each *consumer's* translation units, not here — so the link
# requirement must propagate. Carry Accelerate as an INTERFACE dependency: it links into
# whoever includes the headers, never into MathLib's own archive. The guard matches
# Blast.h's MATH_BLAST_HAS_ACCELERATE (Apple, but not watchOS, where Blast compiles its
# scalar fallback and Accelerate is unavailable) so we never force-link it onto a target
# with no vDSP references.
#----------------------------------------
#
# find_library serves as the probe that the framework is present in the active SDK, but what it
# returns is an absolute path into a specific Xcode — exporting that would bake this machine's
# toolchain location into the installed package. The link uses the relocatable -framework flag
# instead, kept as a single list item so CMake does not read "Accelerate" as a library of its own.
#----------------------------------------
if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "watchOS")
	find_library(ACCELERATE_FRAMEWORK Accelerate)
	target_link_libraries(MathLib INTERFACE "-framework Accelerate")
endif()

# Header-only implementation; this empty anchor TU only exists to produce an archive.
target_sources(MathLib PRIVATE
	src/MathLib.cpp)

#----------------------------------------
# Optional SelfTest component
#
# MathLib/SelfTest.h is the one header that reaches outside MathLib: it includes
# "CoreLib/SelfTestRegistry.h" so the inline module self-tests register with CoreLib's runner.
# Nothing in MathLib includes it, so CoreLib is not a dependency of the library — it is an
# opt-in add-on, and this is how a consumer opts in:
#
#   find_package(MathLib COMPONENTS SelfTest)
#   target_link_libraries(my_tests PRIVATE MathLib::SelfTest)
#
# Deliberately not part of MathLibTargets: an exported target may not link a target CMake cannot
# resolve at export time, and more importantly the two worlds reach CoreLib differently. Here it
# is a sibling checkout addressed through CORELIB_INCLUDE_DIR; for an installed consumer
# MathLibConfig.cmake builds the equivalent target out of the real CoreLib::CoreLib package.
# Both spell the same name, so a consumer's target_link_libraries line is identical either way.
#----------------------------------------
add_library(MathLib_SelfTest INTERFACE)
add_library(MathLib::SelfTest ALIAS MathLib_SelfTest)
target_link_libraries(MathLib_SelfTest INTERFACE MathLib)
target_include_directories(MathLib_SelfTest INTERFACE "${CORELIB_INCLUDE_DIR}")

#----------------------------------------
# Install the static archive to lib/, the public headers to include/, and a find_package-able
# CMake package to lib/cmake/MathLib/.
#
# The package is what a consumer outside this repository resolves MathLib through — a Windows or
# Android build, or anyone who picked the library up from its public release. It declares no
# dependencies, because MathLib has none (see the companion-library note above); Accelerate rides
# along on the exported target as a plain link flag on the platforms that have it.
#
# These rules are deliberately not wrapped in a top-level guard the way CoreLib's are: nothing
# add_subdirectory()s MathLib, so there is no parent prefix for them to leak into. That changes
# when the Stage 1 superbuild lands, and the fix then is an explicit install option rather than
# a guard that would stop the superbuild installing anything at all.
#
# The Apple xcframework pipeline reads only lib/libMathLib.a and include/ out of this prefix, so
# the package files added here are inert as far as that artifact is concerned.
#----------------------------------------
include(CMakePackageConfigHelpers)

set(MATHLIB_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/MathLib")

# The headers are installed by the install(DIRECTORY) below rather than through the target, so
# no INCLUDES DESTINATION here — the install-tree include path is already carried by the
# $<INSTALL_INTERFACE:> expression on the target, and naming it twice would duplicate it.
install(TARGETS MathLib
	EXPORT MathLibTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Headers only. The include tree also carries .DS_Store files, which would otherwise be packaged
# into the xcframework's Headers directory and shipped. An allow-list rather than a
# PATTERN ... EXCLUDE blacklist, so no new kind of stray file can leak in later.
install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	FILES_MATCHING PATTERN "*.h")

install(EXPORT MathLibTargets
	FILE MathLibTargets.cmake
	NAMESPACE MathLib::
	DESTINATION "${MATHLIB_INSTALL_CMAKEDIR}")

# configure_package_config_file (rather than a plain configure_file) is what makes the installed
# package relocatable: it resolves paths relative to the config file's own location, so the prefix
# may be moved, mounted, or unpacked anywhere after the fact.
configure_package_config_file(
	cmake/MathLibConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake"
	INSTALL_DESTINATION "${MATHLIB_INSTALL_CMAKEDIR}")

# SameMajorVersion: the public headers carry the ABI, so a consumer pinned to 1.x accepts any
# later 1.y but never a 2.0 that may have moved something out from under it.
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake"
	VERSION ${MATHLIB_VERSION}
	COMPATIBILITY SameMajorVersion)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake"
	DESTINATION "${MATHLIB_INSTALL_CMAKEDIR}")

#----------------------------------------
# Tests
#
# Each self-tested module ships its self-test inline in its header; the standalone
# runner under test/ registers them through CoreLib's self-test registry and maps the
# failure count to the process exit code a continuous-integration job needs. Built and
# registered with CTest whenever MathLib is the top-level project, on every platform:
# the vector-math types use whichever backend BackendSelect.h picks for the target, and
# Blast.h uses Accelerate on Apple and its scalar fallback elsewhere. DEBUG is defined so
# the registry (CoreLib/SelfTestRegistry.h) is active.
#----------------------------------------
if(MATHLIB_PRIMARY_BUILD)
	set(MATHLIB_IS_TOP_LEVEL ON)
else()
	set(MATHLIB_IS_TOP_LEVEL OFF)
endif()

# BUILD_TESTING is the conventional CMake flag; an unset value (a bare configure, or CI) counts as
# enabled, while a packaging build can pass -DBUILD_TESTING=OFF to skip the test runner.
if(MATHLIB_IS_TOP_LEVEL AND (NOT DEFINED BUILD_TESTING OR BUILD_TESTING))
	enable_testing()

	# Links the SelfTest component rather than MathLib plus a hand-carried CoreLib include path:
	# the runner is the archetypal consumer of that component, so wiring it this way makes the
	# self-test double as the component's own verification.
	add_executable(mathlib_selftest test/mathlib_selftest_main.cpp)
	target_link_libraries(mathlib_selftest PRIVATE MathLib::SelfTest)
	target_compile_definitions(mathlib_selftest PRIVATE DEBUG=1)

	# The runner includes Blast.h and so emits vDSP / vForce references on Apple; it needs no explicit
	# Accelerate link because MathLib carries it as an INTERFACE dependency (see above) and the runner
	# links MathLib. That transitive link is exactly what an external consumer relies on, so the
	# self-test doubles as its verification.

	add_test(NAME MathLibSelfTest COMMAND mathlib_selftest)
endif()

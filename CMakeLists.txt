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

# Declare a project only at top level; when added via add_subdirectory() the
# parent project() and toolchain govern.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	project(MathLib CXX)
endif()

add_library(MathLib)

#----------------------------------------
# Public API
#
# Headers live under include/MathLib/, so the include path is `include` and
# consumers keep writing  #include "MathLib/Vector.h".
#----------------------------------------
target_include_directories(MathLib PUBLIC include)

#----------------------------------------
# Companion-library header dependency
#
# The opt-in SelfTest.h aggregator includes "CoreLib/SelfTestRegistry.h" to register
# the inline module self-tests with CoreLib's runner; no other MathLib header touches
# CoreLib. Expose the CoreLib include root; override this cache variable to point at
# CoreLib's location.
#----------------------------------------
get_filename_component(MATHLIB_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(CORELIB_INCLUDE_DIR "${MATHLIB_PARENT_DIR}/CoreLib/include"
	CACHE PATH "Directory containing the CoreLib/ header tree")
target_include_directories(MathLib PRIVATE "${CORELIB_INCLUDE_DIR}")

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
if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "watchOS")
	find_library(ACCELERATE_FRAMEWORK Accelerate)
	target_link_libraries(MathLib INTERFACE ${ACCELERATE_FRAMEWORK})
endif()

# Header-only implementation; this empty anchor TU only exists to produce an archive.
target_sources(MathLib PRIVATE
	src/MathLib.cpp)

#----------------------------------------
# Install the static archive to lib/ and the public headers to include/.
#----------------------------------------
include(GNUInstallDirs)

install(TARGETS MathLib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	set(MATHLIB_IS_TOP_LEVEL ON)
else()
	set(MATHLIB_IS_TOP_LEVEL OFF)
endif()

# BUILD_TESTING is the conventional CMake flag; an unset value (a bare configure, or CI) counts as
# enabled, while a packaging build can pass -DBUILD_TESTING=OFF to skip the test runner.
if(MATHLIB_IS_TOP_LEVEL AND (NOT DEFINED BUILD_TESTING OR BUILD_TESTING))
	enable_testing()

	add_executable(mathlib_selftest test/mathlib_selftest_main.cpp)
	target_link_libraries(mathlib_selftest PRIVATE MathLib)
	target_include_directories(mathlib_selftest PRIVATE "${CORELIB_INCLUDE_DIR}")
	target_compile_definitions(mathlib_selftest PRIVATE DEBUG=1)

	# The runner includes Blast.h and so emits vDSP / vForce references on Apple; it needs no explicit
	# Accelerate link because MathLib carries it as an INTERFACE dependency (see above) and the runner
	# links MathLib. That transitive link is exactly what an external consumer relies on, so the
	# self-test doubles as its verification.

	add_test(NAME MathLibSelfTest COMMAND mathlib_selftest)
endif()

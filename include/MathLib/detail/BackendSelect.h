///----------------------------------------
///       @file BackendSelect.h
///    @ingroup MathLib
///      @brief The single point at which @ref Vector.h, @ref Matrix.h and @ref Quaternion.h choose and
///             include their backend.
///    @details Four backends implement the same @ref Math::detail interface (the @c native_vector /
///             @c native_matrix / @c native_quaternion trait specializations plus @c struct @c Backend):
///             @ref detail/AppleBackend.h (over @c <simd/simd.h>), @ref detail/SimdBackend.h (over Clang's
///             @c ext_vector_type compiler extension, lowering to NEON/SSE/AVX), @ref detail/IntelBackend.h
///             (over baseline SSE2 @c <immintrin.h> intrinsics — the same API on MSVC, GCC and Clang, so
///             MSVC and GCC on x86 get real hardware SIMD) and @ref detail/GenericBackend.h (portable plain
///             structs). This header defines the selection constants first, then, unless @c MATH_BACKEND
///             was already supplied, auto-detects one, and finally includes exactly the chosen backend.
///             Because the constants and the auto-detect precede the dispatch, a command-line override such
///             as @c -DMATH_BACKEND=MATH_BACKEND_INTEL selects the backend without editing any source. On
///             Clang the auto-detect keeps the @c ext_vector SIMD backend; @c MATH_BACKEND_INTEL is the
///             x86/SSE2 path chosen for MSVC and GCC. Valid overrides: @c MATH_BACKEND_APPLE,
///             @c MATH_BACKEND_SIMD, @c MATH_BACKEND_INTEL, @c MATH_BACKEND_GENERIC.
///     @author Created by John Stephen on 7/8/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

///----------------------------------------
/// @brief The backend selection constants. Defined before the auto-detect and the dispatch so a
///        command-line @c -DMATH_BACKEND=MATH_BACKEND_SIMD (or @c _APPLE / @c _GENERIC) overrides the
///        default without touching source.
///----------------------------------------

#define MATH_BACKEND_APPLE   1
#define MATH_BACKEND_SIMD    2
#define MATH_BACKEND_GENERIC 3
#define MATH_BACKEND_INTEL   4

///----------------------------------------
/// @brief When the backend was not chosen on the command line, auto-detect it: Apple's @c <simd/simd.h>
///        where available, else Clang's compiler vector extensions, else — on x86 with SSE2 (which every
///        x86-64 CPU guarantees) — the SSE2 intrinsics backend, else the portable plain-struct backend.
///        Only Clang is selected for the vector-extension backend: it alone keeps the vector element count
///        deducible in the operations (see @ref detail/SimdBackend.h). MSVC and GCC on x86 therefore take
///        @ref detail/IntelBackend.h for real hardware SIMD, and any remaining target uses the generic
///        backend, whose scalar loops the optimizer still vectorizes.
///----------------------------------------

#ifndef MATH_BACKEND
#if defined(__APPLE__)
#define MATH_BACKEND MATH_BACKEND_APPLE
#elif defined(__clang__)
#define MATH_BACKEND MATH_BACKEND_SIMD
#elif defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#define MATH_BACKEND MATH_BACKEND_INTEL
#else
#define MATH_BACKEND MATH_BACKEND_GENERIC
#endif
#endif

///----------------------------------------
/// @brief Include exactly the chosen backend; each defines the same @ref Math::detail interface.
///----------------------------------------

#if MATH_BACKEND == MATH_BACKEND_APPLE
#include "MathLib/detail/AppleBackend.h"
#elif MATH_BACKEND == MATH_BACKEND_SIMD
#include "MathLib/detail/SimdBackend.h"
#elif MATH_BACKEND == MATH_BACKEND_INTEL
#include "MathLib/detail/IntelBackend.h"
#else
#include "MathLib/detail/GenericBackend.h"
#endif

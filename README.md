# MathLib

Portable **C++23 math primitives**: vector / matrix / quaternion algebra, color
and spectral types, noise and splines, and a set of closed-form numerical
solvers. The implementation is header-only. The vector-math core lowers to
hardware SIMD on Apple, x86, and ARM, and falls back to portable plain structs
anywhere else, so the same headers build on Clang, GCC, and MSVC.

## One API, four backends

`Math::Vector`, `Math::Matrix`, and `Math::Quaternion` are thin value wrappers.
Each is trivially copyable and carries nothing but its backend datum, so it
lowers to that datum with no overhead. The backend is selected once, in
`include/MathLib/detail/BackendSelect.h`, and every backend implements the same
internal interface, so the type you write against is identical on every
platform.

| Backend | Header | Built on | Auto-selected for |
|---|---|---|---|
| Apple | `detail/AppleBackend.h` | the SDK's `<simd/simd.h>` | Apple platforms |
| Simd | `detail/SimdBackend.h` | Clang's `ext_vector_type` extension, lowering to NEON / SSE / AVX | Clang off Apple |
| Intel | `detail/IntelBackend.h` | baseline-SSE2 `<immintrin.h>` intrinsics | MSVC and GCC on x86 |
| Generic | `detail/GenericBackend.h` | portable plain structs, whose scalar loops the optimizer still vectorizes | any other target |

The auto-detect is a fallback, not a lock-in. Pass
`-DMATH_BACKEND=MATH_BACKEND_APPLE` (or `_SIMD`, `_INTEL`, `_GENERIC`) on the
command line to force any backend on any compiler that supports it — useful for
differential testing or for pinning a build to the portable path.

The strided-array layer follows the same pattern. `Blast.h` and `Tridiagonal.h`
dispatch to Apple's **Accelerate** framework (`vDSP` / `vForce`) on Apple
platforms other than watchOS, and compile a portable scalar fallback everywhere
else; there is no third-party math dependency off Apple.

Continuous integration builds MathLib and runs its self-test suite on macOS
(AppleClang → Apple backend), Linux with GCC (→ Intel backend), Linux with Clang
(→ Simd backend), Windows with MSVC (→ Intel backend), and a Linux leg that
forces `MATH_BACKEND_GENERIC`, so all four backends are exercised on every push.

## Modules

### Linear algebra

```cpp
#include "MathLib/Vector.h"       // Math::Vector2/3/4 (float & double), SIMD-backed
#include "MathLib/Matrix.h"       // Math::Matrix3/4, column-major, column-vector convention
#include "MathLib/Quaternion.h"   // Math::Quaternion, SLERP, axis-angle
#include "MathLib/Transforms.h"   // view / projection / viewport transforms
#include "MathLib/PolarCoordinates.h"
```

- **`Vector`** wraps the backend vector type with named, directly writable
  components (`x`, `y`, `z`, `w`), arithmetic, dot / cross / length, and reaches
  the raw backend type only through a single `native()` escape hatch.
- **`Matrix`** is column-major with the column-vector convention: `a * b`
  applies `b` then `a`, and `m * v` transforms a column vector. **`Quaternion`**
  covers composition, rotation, SLERP, and axis-angle, converting to and from
  matrices directly.
- **`Transforms`** builds the camera and projection matrices — perspective and
  frustum, orthographic, viewport, and an observer-on-a-sphere viewer transform,
  with helpers to retarget a projection's depth range (finite or infinite far
  plane).
- **`PolarCoordinates`** converts between right-handed polar angles and
  `Math::Vector` directions.

### Scalars, angles, and geometry

```cpp
#include "MathLib/Scalar.h"        // power-of-two, approxEqual, interpolation, Gaussian kernel
#include "MathLib/Angle.h"         // sexagesimal decomposition: HM / HMS / DM / DMS
#include "MathLib/Numbers.h"       // math constants and unit conversions
#include "MathLib/Point2d.h"
#include "MathLib/Rect.h"
#include "MathLib/Ray.h"
#include "MathLib/RescalingMap.h"  // map cumulative distance to a normalized [0,1] factor
```

- **`Scalar`** collects small utilities: power-of-two tests and rounding,
  `approxEqual`, the haversine, cosine and bilinear interpolation, and a
  normalized Gaussian-kernel generator.
- **`Angle`** decomposes radians into the sexagesimal forms — hours/minutes
  (`HM`, `HMS`) and degrees/arcminutes/arcseconds (`DM`, `DMS`) — with the sign
  carried separately from the non-negative magnitudes.
- **`Numbers`** supplies the mathematical constants and unit conversions
  (degrees / radians / arcminutes / arcseconds, and friends) in both `float` and
  `double`.
- **`Point2d`**, **`Rect`**, and **`Ray`** are the small geometric value types:
  a 2D point with Chebyshev distance, an axis-aligned rectangle with
  inset / offset / intersection / union, and an origin-plus-direction ray.

### Numerical solvers

```cpp
#include "MathLib/Polynomial.h"      // solveLinear / solveQuadratic / solveCubic
#include "MathLib/CubicPolynomial.h" // a cubic value type over those solvers
#include "MathLib/Tridiagonal.h"     // Thomas algorithm over strided operands
#include "MathLib/Blast.h"           // strided-array ops over vDSP / vForce
```

- **`Polynomial`** returns a `RealRoots<Scalar, Capacity>` — a fixed-capacity
  root set with a known count that is indexable, iterable, and sortable.
  Quadratics use the numerically stable Citardauq form to avoid cancellation;
  cubics take the trigonometric branch for three real roots and Cardano for one.
- **`Tridiagonal`** solves a tridiagonal system by the Thomas algorithm, with
  every band and the result expressed as a `Math::Blast::StridedSpan`, so an
  operand may address one column of a grid or one field of an array-of-structs
  in place, with no copy.
- **`Blast`** is the strided-array primitive layer: a non-owning `StridedSpan`
  view (base pointer, count, element stride) plus elementwise and reduction ops
  that dispatch to Accelerate (`vDSP`, `vForce`) on Apple and a portable scalar
  backend elsewhere.

### Curves and noise

```cpp
#include "MathLib/CatmullRom.h"          // single segment: uniform / centripetal / non-uniform
#include "MathLib/SegmentedCatmullRom.h" // piecewise, arc-length parameterized
#include "MathLib/PerlinNoise.h"         // seedable gradient noise in 1/2/3 dimensions
#include "MathLib/Random.h"              // thread-safe random number utilities
#include "MathLib/OneEuroFilter.h"       // low-latency 1€ smoothing filter for noisy input
```

### Storage and color

```cpp
#include "MathLib/HalfFloat.h"        // IEEE 754 binary16 storage type
#include "MathLib/PackedVector.h"     // compact, SIMD-padding-free layout for on-disk / GPU records
#include "MathLib/Color.h"            // Math::RGBColor / RGBAColor, RGB↔HSV, blend, luminance
#include "MathLib/BlackbodyColor.h"   // Planck radiance and temperature ↔ color-index estimation
#include "MathLib/XYZColor.h"         // CIE 1931 XYZ / xyY, the 2° standard-observer table
#include "MathLib/Spectrum.h"         // sampled spectral power distribution → XYZ
```

- **`HalfFloat`** converts through the compiler's native `_Float16` where
  available — one hardware instruction on arm64 and x86 F16C — and a portable
  software path otherwise. **`PackedVector`** drops the SIMD alignment padding so
  a vector is byte-compatible with a packed record.
- **`Color`** operates on `Math::RGBColor` / `RGBAColor` plus 8-bit integer color
  types. **`BlackbodyColor`**, **`XYZColor`**, and **`Spectrum`** form the
  physical-color chain: a Planck-radiance / color-temperature model, the CIE 1931
  tristimulus and chromaticity types over the shared 2° color-matching table, and
  a sampled spectrum that integrates to XYZ.

## Requirements

- A C++23 compiler and standard library. MathLib builds under Clang (including
  AppleClang), GCC, and MSVC.
- No third-party dependency for the library itself. On Apple platforms other than
  watchOS, `Blast.h` and `Tridiagonal.h` call `vDSP` / `vForce`, so a consumer
  that compiles them links the **Accelerate** framework; the CMake target carries
  that as a transitive `INTERFACE` link on Apple only. Everywhere else those
  headers compile their scalar fallback and need nothing.
- Only the opt-in `MathLib/SelfTest.h` aggregator depends on **CoreLib**: it
  includes `CoreLib/SelfTestRegistry.h` to register each module's inline
  self-test with CoreLib's runner. Compiling it — or the standalone test runner —
  needs CoreLib's `include/` on the header search path. Every other MathLib
  header asserts through the self-contained `MathLib/SelfTestCheck.h` and pulls in
  no CoreLib. CoreLib is at <https://github.com/wobbleworks/CoreLib>; check it out
  as a sibling of MathLib.

## Usage

Add the `include/` directory to your header search path; headers are referenced
as `MathLib/<Module>.h`.

```cpp
#include "MathLib/Vector.h"
#include "MathLib/Polynomial.h"

Math::Double3 axis(0.0, 0.0, 1.0);
auto roots = Math::solveQuadratic(1.0, -3.0, 2.0);   // RealRoots<double, 2>: {1, 2}
```

Each self-tested module ships an inline self-test in its own header, so it
generates code only where referenced. Include the opt-in `MathLib/SelfTest.h` to
register them all with CoreLib's runner, then execute them with
`Core::selftest::runAll()` — a no-op outside a `DEBUG` build.

## Building

MathLib is a CMake library target, available by three routes. All of them
provide the same namespaced target, `MathLib::MathLib`, so nothing downstream
has to know which route was used.

Fetched at configure time — pin a tag, never a branch:

```cmake
include(FetchContent)
FetchContent_Declare(MathLib
	GIT_REPOSITORY https://github.com/wobbleworks/MathLib.git
	GIT_TAG        v1.0.0)
FetchContent_MakeAvailable(MathLib)
target_link_libraries(your_target PRIVATE MathLib::MathLib)
```

From a checkout or submodule you already manage:

```cmake
add_subdirectory(MathLib)
target_link_libraries(your_target PRIVATE MathLib::MathLib)
```

Or from an installed package, after `cmake --install`:

```cmake
find_package(MathLib 1.0 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MathLib::MathLib)
```

The header-only implementation is anchored by one empty source
(`src/MathLib.cpp`) so the target still produces `libMathLib.a`. When MathLib is
the top-level project it also builds `mathlib_selftest`, a CTest runner that
executes every registered module self-test and maps the failure count to its exit
code (honoring the standard `BUILD_TESTING` flag). That runner needs CoreLib; the
`CORELIB_INCLUDE_DIR` cache variable points at it, defaulting to
`../CoreLib/include` — a CoreLib checkout beside MathLib — and can be overridden
to CoreLib's actual location.

```sh
cmake -S MathLib -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Releases and versioning

Releases are git tags of the form `v1.0.0`; [CHANGELOG.md](CHANGELOG.md)
records what each one contains. Source is the only distribution — which for a
header-only library is nearly all there is to ship, since the archive it builds
is an anchor around one empty translation unit and the backend is chosen by the
consumer's own compiler and flags.

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html): the
major component changes when the public API breaks, the minor when it grows
compatibly, the patch for compatible fixes. MathLib is versioned independently
of anything that consumes it, so `find_package(MathLib 1.0)` accepting any 1.x
is a compatibility promise rather than an accident of release timing.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for details.

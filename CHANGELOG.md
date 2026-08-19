# Changelog

All notable changes to MathLib are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Versioning

MathLib follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). In a
version `MAJOR.MINOR.PATCH`:

- **MAJOR** changes when the public API breaks — a symbol removed or renamed, a
  signature or a documented behavior changed, a requirement tightened.
- **MINOR** changes when the API grows in a backward-compatible way.
- **PATCH** changes for backward-compatible fixes.

The version describes this library's API and nothing else. MathLib is versioned
independently of any application that consumes it, so its numbers move when its
own surface moves, not on somebody else's release schedule.

Every breaking change is listed under **Changed** or **Removed** in the entry
for the release that carries it; those are the ones that force a major bump.

The CMake package version file is generated with `COMPATIBILITY
SameMajorVersion`, which is exactly this contract expressed to CMake:
`find_package(MathLib 1.0)` accepts any 1.x — everything in the 1.x line is
compatible with 1.0 by construction — and rejects 2.x.

## [1.0.0] — 2026-08-19

First tagged release. This code has been in production use for some time; 1.0.0
is the point at which it becomes independently versioned and separately
consumable, so the entry below describes the surface rather than a diff against
a predecessor.

### Added

- Public release of the header-only C++23 math layer: vectors, matrices, and
  quaternions over a compile-time-selected backend (`detail/BackendSelect.h`),
  polynomial solving, the `Blast.h` bulk operations, and the supporting
  geometric types.
- Four interchangeable backends behind one API — Apple `<simd/simd.h>` +
  Accelerate, Clang `ext_vector_type`, baseline SSE2 `<immintrin.h>`, and a
  plain-struct generic fallback — selected automatically or forced with
  `-DMATH_BACKEND`.
- CMake package export: `find_package(MathLib CONFIG)` provides
  `MathLib::MathLib`, the same imported target name an in-tree
  `add_subdirectory()` build supplies.
- Continuous integration exercising all four backends across macOS (AppleClang),
  Linux (GCC and Clang 20), and Windows (MSVC).

### Fixed

- The generated package version file read `PROJECT_VERSION`, which this library
  leaves unset whenever it is not the top-level project — including when an
  enclosing project builds it as a deliberate package member by setting the
  `WW_SUPERBUILD` option, which is the case that also turns its install rules
  on. An installed package could therefore advertise the enclosing project's
  version instead of MathLib's. The version is now carried in `MATHLIB_VERSION`,
  set unconditionally ahead of `project()`.

### Notes for packagers

MathLib is distributed as **source only**, and being header-only it is close to
meaningless to ship as a binary: the archive it produces is an anchor around one
empty translation unit. The backend selection is a compile-time decision made in
the consumer's own build, so the headers are the deliverable.

The library itself depends on nothing beyond the standard library and, on Apple,
Accelerate. Only the self-test runner reaches CoreLib, for its self-test
registry.

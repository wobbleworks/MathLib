///----------------------------------------
///       @file PackedVector.h
///    @ingroup MathLib
///      @brief Tightly-packed, naturally-aligned vector storage for serialized/GPU records.
///    @details @ref Math::Vector is SIMD-backed and hardware-aligned (a 3-vector occupies
///             16/32 bytes), which is wrong for packed on-disk or GPU-buffer records that
///             assume the compact 12/24-byte, 4/8-aligned layout. @ref Math::PackedVector3
///             is a plain trivially-copyable POD with exactly that layout. Use it only for
///             storage; convert to/from @ref Math::Vector (implicitly) to compute.
///     @author Created by John Stephen.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <type_traits>
#include "MathLib/Vector.h"

///----------------------------------------
namespace Math {
///----------------------------------------

/// @brief Compact 3-component storage POD (no SIMD padding), byte-compatible with packed records.
/// @tparam TScalar Scalar type (float or double).
template <class TScalar>
struct PackedVector3 {
	using Native = typename Vector<TScalar, 3>::Native;
	
	TScalar x, y, z;
	
	PackedVector3() noexcept = default;
	
	PackedVector3(TScalar x, TScalar y, TScalar z) noexcept : x(x), y(y), z(z) {}
	
	/// @brief Implicitly stores a computed @ref Math::Vector into packed form.
	PackedVector3(const Vector<TScalar, 3> &vector) noexcept : x(vector.x), y(vector.y), z(vector.z) {}
	
	/// @brief Implicitly stores a native SIMD vector into packed form (simd-boundary interop).
	PackedVector3(const Native &native) noexcept : PackedVector3(Vector<TScalar, 3>(native)) {}
	
	/// @brief Implicitly unpacks to a @ref Math::Vector for computation.
	operator Vector<TScalar, 3>() const noexcept { return Vector<TScalar, 3>(x, y, z); }
	
	/// @brief Implicitly unpacks to a native SIMD vector (simd-boundary interop).
	operator Native() const noexcept { return Vector<TScalar, 3>(x, y, z); }
};

using PackedVector3f = PackedVector3<float>;
using PackedVector3d = PackedVector3<double>;

// Layout guarantees: this exact field layout and size is a persistence and GPU wire format —
// on-disk records and GPU buffers are read and written as these bytes, so it must never change.
static_assert(sizeof(PackedVector3f) == 12 && alignof(PackedVector3f) == 4);
static_assert(sizeof(PackedVector3d) == 24 && alignof(PackedVector3d) == 8);
static_assert(std::is_trivially_copyable_v<PackedVector3f> && std::is_standard_layout_v<PackedVector3f>);
static_assert(std::is_trivially_copyable_v<PackedVector3d> && std::is_standard_layout_v<PackedVector3d>);

///----------------------------------------
} // namespace Math
///----------------------------------------

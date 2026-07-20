///----------------------------------------
///       @file Blast.h
///    @ingroup MathLib
///      @brief Backend-accelerated elementwise operations over strided arrays of scalars.
///    @details A small performance layer for the bulk arithmetic and transcendental operations
///             that Apple's Accelerate framework exposes through @c vDSP and @c vForce. Operands are
///             expressed as @ref StridedSpan views — a base pointer, an element count, and an element
///             stride — so a single field of an array of structs can be operated on in place without
///             copying. Every operation dispatches through a backend policy: @ref Blast::Plain is a
///             portable scalar implementation, and @ref Blast::Accelerate maps onto @c vDSP / @c vForce
///             on Apple platforms (watchOS excepted). The default backend is selected at compile time,
///             but each operation may be pointed at a specific backend, which the self-test uses to
///             assert the two agree. Additional backends (NEON, SVE, GPU) can be added as further
///             policy structs without touching call sites.
///     @author Created by John Stephen on 7/7/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
/// @brief When 1, the Accelerate backend is available and selected as the default; when 0, only the
///        portable scalar backend exists. Auto-detected as Apple platforms other than watchOS, whose
///        modular headers cannot be included through the framework umbrella here.
///----------------------------------------

#if !defined(MATH_BLAST_HAS_ACCELERATE)
	#if defined(__APPLE__)
		#include <TargetConditionals.h>
		#if !TARGET_OS_WATCH
			#define MATH_BLAST_HAS_ACCELERATE 1
		#endif
	#endif
#endif

#ifndef MATH_BLAST_HAS_ACCELERATE
	#define MATH_BLAST_HAS_ACCELERATE 0
#endif

#if MATH_BLAST_HAS_ACCELERATE
	#include <Accelerate/Accelerate.h>
#endif

///----------------------------------------
namespace Math::Blast {
///----------------------------------------

///----------------------------------------
/// MARK: Scalar
///----------------------------------------

///----------------------------------------
///   @brief Satisfied by the element types the backends support: the two floating-point widths @c vDSP provides.
///----------------------------------------
template <typename T>
concept Scalar = std::same_as<T, float> || std::same_as<T, double>;

///----------------------------------------
/// MARK: StridedSpan
///----------------------------------------

///----------------------------------------
///   @class StridedSpan
///   @brief A non-owning view over evenly spaced elements: a base pointer, a count, and an element stride.
/// @details Models @c vDSP's operand shape directly. A stride of one is a contiguous run, so a
///          @c std::span, @c std::array, @c std::vector, or C array converts implicitly. A larger
///          stride addresses one field across an array of structs — see @ref field. The view never
///          owns or outlives its storage.
/// @tparam TElement Element type, possibly @c const qualified for read-only operands.
///----------------------------------------

template <typename TElement>
class StridedSpan final {
public:
	using element_type = TElement;
	using value_type = std::remove_cv_t<TElement>;
	
	///----------------------------------------
	/// @brief An empty view.
	///----------------------------------------
	
	constexpr StridedSpan() noexcept = default;
	
	///----------------------------------------
	///   @brief Construct from an explicit base, count, and stride.
	///   @param data Pointer to the first element.
	///   @param count Number of elements addressed.
	///   @param stride Distance between consecutive elements, in units of @c TElement.
	///----------------------------------------
	
	constexpr StridedSpan(TElement* data, std::size_t count, std::size_t stride = 1) noexcept :
		_data(data),
		_count(count),
		_stride(stride) {
	}
	
	///----------------------------------------
	///   @brief Construct a contiguous (stride 1) view from any contiguous range.
	/// @details Accepts @c std::span, @c std::array, @c std::vector, and C arrays whose element
	///          pointer converts to @c TElement*, which also permits binding a mutable range to a
	///          @c const view.
	/// @tparam TRange A contiguous range over a compatible element type.
	///   @param range The range whose storage is viewed.
	///----------------------------------------
	
	template <std::ranges::contiguous_range TRange>
		requires (!std::same_as<std::remove_cvref_t<TRange>, StridedSpan>) &&
			requires (TRange&& range) { { std::ranges::data(range) } -> std::convertible_to<TElement*>; }
	constexpr StridedSpan(TRange&& range) noexcept :
		_data(std::ranges::data(range)),
		_count(std::ranges::size(range)),
		_stride(1) {
	}
	
	///----------------------------------------
	///   @brief Convert a mutable view to a @c const view of the same storage.
	/// @tparam TMutable The non-const element type this view adds @c const to.
	///   @param other The mutable view to borrow.
	///----------------------------------------
	
	template <typename TMutable>
		requires std::same_as<TElement, const TMutable>
	constexpr StridedSpan(StridedSpan<TMutable> other) noexcept :
		_data(other.data()),
		_count(other.size()),
		_stride(other.stride()) {
	}
	
	///----------------------------------------
	/// @brief Element at @p index, honoring the stride.
	///----------------------------------------
	
	[[nodiscard]] constexpr TElement& operator[](std::size_t index) const noexcept { return _data[index * _stride]; }
	
	///----------------------------------------
	/// @brief Pointer to the first element.
	///----------------------------------------
	
	[[nodiscard]] constexpr TElement* data() const noexcept { return _data; }
	
	///----------------------------------------
	/// @brief Number of elements addressed.
	///----------------------------------------
	
	[[nodiscard]] constexpr std::size_t size() const noexcept { return _count; }
	
	///----------------------------------------
	/// @brief Distance between consecutive elements, in units of @c TElement.
	///----------------------------------------
	
	[[nodiscard]] constexpr std::size_t stride() const noexcept { return _stride; }
	
	///----------------------------------------
	/// @brief Whether the view addresses no elements.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool isEmpty() const noexcept { return _count == 0; }
	
	///----------------------------------------
	/// @brief Whether the elements are packed with unit stride.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool isContiguous() const noexcept { return _stride == 1; }
	
private:
	TElement* _data = nullptr;
	std::size_t _count = 0;
	std::size_t _stride = 1;
};

///----------------------------------------
/// MARK: field
///----------------------------------------

///----------------------------------------
///   @brief View a single struct member across a contiguous range of structs as a strided span.
/// @details The member's element stride is @c sizeof(TStruct)/sizeof(TMember), so consecutive
///          elements skip the sibling members. The resulting view is @c const when @p items is.
/// @tparam TRange A contiguous range of @p TStruct.
/// @tparam TStruct The struct type, deduced from the member pointer.
/// @tparam TMember The member type, deduced from the member pointer.
///   @param items The range of structs to view.
///   @param member Pointer to the member to project.
/// @returns A @ref StridedSpan addressing @p member of every element, or an empty view if @p items is empty.
///----------------------------------------

template <std::ranges::contiguous_range TRange, typename TStruct, typename TMember>
	requires std::same_as<std::remove_cv_t<std::remove_reference_t<std::ranges::range_reference_t<TRange>>>, TStruct>
[[nodiscard]] constexpr auto field(TRange&& items, TMember TStruct::* member) noexcept {
	using Element = std::remove_reference_t<std::ranges::range_reference_t<TRange>>;
	using ResultMember = std::conditional_t<std::is_const_v<Element>, const TMember, TMember>;
	static_assert(sizeof(TStruct) % sizeof(TMember) == 0, "member stride must be a whole number of elements");
	
	// An empty range has no valid base element to address
	if (std::ranges::empty(items)) {
		return StridedSpan<ResultMember>{};
	}
	
	// Address the member of the first element and step by the struct's element footprint
	ResultMember* base = std::addressof(std::ranges::data(items)->*member);
	return StridedSpan<ResultMember>{base, std::ranges::size(items), sizeof(TStruct) / sizeof(TMember)};
}

///----------------------------------------
/// MARK: Plain backend
///----------------------------------------

///----------------------------------------
///   @class Plain
///   @brief Portable scalar backend: every operation is a straightforward strided loop.
/// @details Always available, correct on every platform, and the fallback the Accelerate backend
///          defers to for cases @c vForce cannot express. The floating-point results are the
///          reference the self-test compares the accelerated backend against.
///----------------------------------------

struct Plain final {
	static constexpr bool isBlastBackend = true;
	static constexpr std::string_view name = "plain";
	
	template <Scalar T>
	static void multiplyAddScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar, StridedSpan<const T> b) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] * scalar + b[index];
		}
	}
	
	template <Scalar T>
	static void multiplyAdd(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b, StridedSpan<const T> c) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] * b[index] + c[index];
		}
	}
	
	template <Scalar T>
	static void multiplyScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] * scalar;
		}
	}
	
	template <Scalar T>
	static void addScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] + scalar;
		}
	}
	
	template <Scalar T>
	static void add(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] + b[index];
		}
	}
	
	template <Scalar T>
	static void subtract(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] - b[index];
		}
	}
	
	template <Scalar T>
	static void multiply(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = a[index] * b[index];
		}
	}
	
	template <Scalar T>
	[[nodiscard]] static T dot(StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		auto total = T(0);
		const auto count = a.size();
		for (std::size_t index = 0; index < count; ++index) {
			total += a[index] * b[index];
		}
		
		return total;
	}
	
	template <Scalar T>
	[[nodiscard]] static T sum(StridedSpan<const T> a) noexcept {
		auto total = T(0);
		const auto count = a.size();
		for (std::size_t index = 0; index < count; ++index) {
			total += a[index];
		}
		
		return total;
	}
	
	template <Scalar T>
	static void cosine(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = std::cos(src[index]);
		}
	}
	
	template <Scalar T>
	static void sine(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = std::sin(src[index]);
		}
	}
	
	template <Scalar T>
	static void sinCosine(StridedSpan<T> sineDest, StridedSpan<T> cosineDest, StridedSpan<const T> src) noexcept {
		const auto count = src.size();
		for (std::size_t index = 0; index < count; ++index) {
			sineDest[index] = std::sin(src[index]);
			cosineDest[index] = std::cos(src[index]);
		}
	}
	
	template <Scalar T>
	static void squareRoot(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = std::sqrt(src[index]);
		}
	}
	
	template <Scalar T>
	static void exponential(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = std::exp(src[index]);
		}
	}
	
	template <Scalar T>
	static void naturalLog(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		const auto count = dest.size();
		for (std::size_t index = 0; index < count; ++index) {
			dest[index] = std::log(src[index]);
		}
	}
};

#if MATH_BLAST_HAS_ACCELERATE

///----------------------------------------
/// MARK: Accelerate primitives
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

// Overload the double- and single-precision vDSP / vForce primitives behind common names so the
// backend can be written once. The vDSP argument order is preserved except where a primitive's
// signature reverses operands, which the wrapper hides.

inline void vsma(const double* a, vDSP_Stride strideA, const double* scalar, const double* b, vDSP_Stride strideB, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsmaD(a, strideA, scalar, b, strideB, dest, strideDest, count); }
inline void vsma(const float* a, vDSP_Stride strideA, const float* scalar, const float* b, vDSP_Stride strideB, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsma(a, strideA, scalar, b, strideB, dest, strideDest, count); }

inline void vma(const double* a, vDSP_Stride strideA, const double* b, vDSP_Stride strideB, const double* c, vDSP_Stride strideC, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vmaD(a, strideA, b, strideB, c, strideC, dest, strideDest, count); }
inline void vma(const float* a, vDSP_Stride strideA, const float* b, vDSP_Stride strideB, const float* c, vDSP_Stride strideC, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vma(a, strideA, b, strideB, c, strideC, dest, strideDest, count); }

inline void vsmul(const double* a, vDSP_Stride strideA, const double* scalar, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsmulD(a, strideA, scalar, dest, strideDest, count); }
inline void vsmul(const float* a, vDSP_Stride strideA, const float* scalar, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsmul(a, strideA, scalar, dest, strideDest, count); }

inline void vsadd(const double* a, vDSP_Stride strideA, const double* scalar, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsaddD(a, strideA, scalar, dest, strideDest, count); }
inline void vsadd(const float* a, vDSP_Stride strideA, const float* scalar, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsadd(a, strideA, scalar, dest, strideDest, count); }

inline void vadd(const double* a, vDSP_Stride strideA, const double* b, vDSP_Stride strideB, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vaddD(a, strideA, b, strideB, dest, strideDest, count); }
inline void vadd(const float* a, vDSP_Stride strideA, const float* b, vDSP_Stride strideB, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vadd(a, strideA, b, strideB, dest, strideDest, count); }

// vDSP_vsub computes C = A - B but takes the subtrahend B first; swap so the wrapper reads dest = a - b.
inline void vsub(const double* a, vDSP_Stride strideA, const double* b, vDSP_Stride strideB, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsubD(b, strideB, a, strideA, dest, strideDest, count); }
inline void vsub(const float* a, vDSP_Stride strideA, const float* b, vDSP_Stride strideB, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vsub(b, strideB, a, strideA, dest, strideDest, count); }

inline void vmul(const double* a, vDSP_Stride strideA, const double* b, vDSP_Stride strideB, double* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vmulD(a, strideA, b, strideB, dest, strideDest, count); }
inline void vmul(const float* a, vDSP_Stride strideA, const float* b, vDSP_Stride strideB, float* dest, vDSP_Stride strideDest, vDSP_Length count) noexcept { vDSP_vmul(a, strideA, b, strideB, dest, strideDest, count); }

inline void dotpr(const double* a, vDSP_Stride strideA, const double* b, vDSP_Stride strideB, double* dest, vDSP_Length count) noexcept { vDSP_dotprD(a, strideA, b, strideB, dest, count); }
inline void dotpr(const float* a, vDSP_Stride strideA, const float* b, vDSP_Stride strideB, float* dest, vDSP_Length count) noexcept { vDSP_dotpr(a, strideA, b, strideB, dest, count); }

inline void sve(const double* a, vDSP_Stride strideA, double* dest, vDSP_Length count) noexcept { vDSP_sveD(a, strideA, dest, count); }
inline void sve(const float* a, vDSP_Stride strideA, float* dest, vDSP_Length count) noexcept { vDSP_sve(a, strideA, dest, count); }

// vForce operates on contiguous arrays only and takes its count by pointer.
inline void vcos(double* dest, const double* src, const int* count) noexcept { vvcos(dest, src, count); }
inline void vcos(float* dest, const float* src, const int* count) noexcept { vvcosf(dest, src, count); }

inline void vsin(double* dest, const double* src, const int* count) noexcept { vvsin(dest, src, count); }
inline void vsin(float* dest, const float* src, const int* count) noexcept { vvsinf(dest, src, count); }

inline void vsincos(double* sineDest, double* cosineDest, const double* src, const int* count) noexcept { vvsincos(sineDest, cosineDest, src, count); }
inline void vsincos(float* sineDest, float* cosineDest, const float* src, const int* count) noexcept { vvsincosf(sineDest, cosineDest, src, count); }

inline void vsqrt(double* dest, const double* src, const int* count) noexcept { vvsqrt(dest, src, count); }
inline void vsqrt(float* dest, const float* src, const int* count) noexcept { vvsqrtf(dest, src, count); }

inline void vexp(double* dest, const double* src, const int* count) noexcept { vvexp(dest, src, count); }
inline void vexp(float* dest, const float* src, const int* count) noexcept { vvexpf(dest, src, count); }

inline void vlog(double* dest, const double* src, const int* count) noexcept { vvlog(dest, src, count); }
inline void vlog(float* dest, const float* src, const int* count) noexcept { vvlogf(dest, src, count); }

///----------------------------------------
} // namespace detail
///----------------------------------------

///----------------------------------------
/// MARK: Accelerate backend
///----------------------------------------

///----------------------------------------
///   @class Accelerate
///   @brief Backend mapping each operation onto Apple's @c vDSP and @c vForce primitives.
/// @details The arithmetic operations honor arbitrary strides natively. The transcendental
///          operations use @c vForce, which requires contiguous storage, so they take the accelerated
///          path only when all operands are contiguous and otherwise defer to @ref Plain.
///----------------------------------------

struct Accelerate final {
	static constexpr bool isBlastBackend = true;
	static constexpr std::string_view name = "accelerate";
	
	template <Scalar T>
	static void multiplyAddScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar, StridedSpan<const T> b) noexcept {
		const auto scalarValue = scalar;
		detail::vsma(a.data(), vDSP_Stride(a.stride()), &scalarValue, b.data(), vDSP_Stride(b.stride()), dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void multiplyAdd(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b, StridedSpan<const T> c) noexcept {
		detail::vma(a.data(), vDSP_Stride(a.stride()), b.data(), vDSP_Stride(b.stride()), c.data(), vDSP_Stride(c.stride()), dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void multiplyScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar) noexcept {
		const auto scalarValue = scalar;
		detail::vsmul(a.data(), vDSP_Stride(a.stride()), &scalarValue, dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void addScalar(StridedSpan<T> dest, StridedSpan<const T> a, T scalar) noexcept {
		const auto scalarValue = scalar;
		detail::vsadd(a.data(), vDSP_Stride(a.stride()), &scalarValue, dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void add(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		detail::vadd(a.data(), vDSP_Stride(a.stride()), b.data(), vDSP_Stride(b.stride()), dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void subtract(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		detail::vsub(a.data(), vDSP_Stride(a.stride()), b.data(), vDSP_Stride(b.stride()), dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	static void multiply(StridedSpan<T> dest, StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		detail::vmul(a.data(), vDSP_Stride(a.stride()), b.data(), vDSP_Stride(b.stride()), dest.data(), vDSP_Stride(dest.stride()), vDSP_Length(dest.size()));
	}
	
	template <Scalar T>
	[[nodiscard]] static T dot(StridedSpan<const T> a, StridedSpan<const T> b) noexcept {
		auto result = T(0);
		detail::dotpr(a.data(), vDSP_Stride(a.stride()), b.data(), vDSP_Stride(b.stride()), &result, vDSP_Length(a.size()));
		return result;
	}
	
	template <Scalar T>
	[[nodiscard]] static T sum(StridedSpan<const T> a) noexcept {
		auto result = T(0);
		detail::sve(a.data(), vDSP_Stride(a.stride()), &result, vDSP_Length(a.size()));
		return result;
	}
	
	template <Scalar T>
	static void cosine(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		// vForce has no stride arguments, so only contiguous operands take the accelerated path
		if (dest.isContiguous() && src.isContiguous()) {
			const auto count = int(dest.size());
			detail::vcos(dest.data(), src.data(), &count);
		} else {
			Plain::cosine(dest, src);
		}
	}
	
	template <Scalar T>
	static void sine(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		if (dest.isContiguous() && src.isContiguous()) {
			const auto count = int(dest.size());
			detail::vsin(dest.data(), src.data(), &count);
		} else {
			Plain::sine(dest, src);
		}
	}
	
	template <Scalar T>
	static void sinCosine(StridedSpan<T> sineDest, StridedSpan<T> cosineDest, StridedSpan<const T> src) noexcept {
		if (sineDest.isContiguous() && cosineDest.isContiguous() && src.isContiguous()) {
			const auto count = int(src.size());
			detail::vsincos(sineDest.data(), cosineDest.data(), src.data(), &count);
		} else {
			Plain::sinCosine(sineDest, cosineDest, src);
		}
	}
	
	template <Scalar T>
	static void squareRoot(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		if (dest.isContiguous() && src.isContiguous()) {
			const auto count = int(dest.size());
			detail::vsqrt(dest.data(), src.data(), &count);
		} else {
			Plain::squareRoot(dest, src);
		}
	}
	
	template <Scalar T>
	static void exponential(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		if (dest.isContiguous() && src.isContiguous()) {
			const auto count = int(dest.size());
			detail::vexp(dest.data(), src.data(), &count);
		} else {
			Plain::exponential(dest, src);
		}
	}
	
	template <Scalar T>
	static void naturalLog(StridedSpan<T> dest, StridedSpan<const T> src) noexcept {
		if (dest.isContiguous() && src.isContiguous()) {
			const auto count = int(dest.size());
			detail::vlog(dest.data(), src.data(), &count);
		} else {
			Plain::naturalLog(dest, src);
		}
	}
};

#endif // MATH_BLAST_HAS_ACCELERATE

///----------------------------------------
/// MARK: Backend selection
///----------------------------------------

///----------------------------------------
///   @brief Satisfied by a backend policy struct usable for @ref Blast operation dispatch.
///----------------------------------------
template <typename TBackend>
concept Backend = requires {
	{ TBackend::isBlastBackend } -> std::convertible_to<bool>;
};

///----------------------------------------
/// @brief The backend every operation uses unless a call names another: Accelerate where available.
///----------------------------------------

#if MATH_BLAST_HAS_ACCELERATE
using DefaultBackend = Accelerate;
#else
using DefaultBackend = Plain;
#endif

///----------------------------------------
/// MARK: Scaled combine
///----------------------------------------

///----------------------------------------
///   @brief Compute @c dest = a * scalar + b elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a First factor view.
///   @param scalar Scalar multiplier applied to @p a.
///   @param b Addend view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void multiplyAddScalar(StridedSpan<double> dest, StridedSpan<const double> a, double scalar, StridedSpan<const double> b) noexcept {
	TBackend::multiplyAddScalar(dest, a, scalar, b);
}

/// @brief Single-precision overload of @ref multiplyAddScalar.
template <Backend TBackend = DefaultBackend>
inline void multiplyAddScalar(StridedSpan<float> dest, StridedSpan<const float> a, float scalar, StridedSpan<const float> b) noexcept {
	TBackend::multiplyAddScalar(dest, a, scalar, b);
}

///----------------------------------------
///   @brief Compute @c dest = a * b + c elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a First factor view.
///   @param b Second factor view.
///   @param c Addend view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void multiplyAdd(StridedSpan<double> dest, StridedSpan<const double> a, StridedSpan<const double> b, StridedSpan<const double> c) noexcept {
	TBackend::multiplyAdd(dest, a, b, c);
}

/// @brief Single-precision overload of @ref multiplyAdd.
template <Backend TBackend = DefaultBackend>
inline void multiplyAdd(StridedSpan<float> dest, StridedSpan<const float> a, StridedSpan<const float> b, StridedSpan<const float> c) noexcept {
	TBackend::multiplyAdd(dest, a, b, c);
}

///----------------------------------------
///   @brief Compute @c dest = a * scalar elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a Source view.
///   @param scalar Scalar multiplier.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void multiplyScalar(StridedSpan<double> dest, StridedSpan<const double> a, double scalar) noexcept {
	TBackend::multiplyScalar(dest, a, scalar);
}

/// @brief Single-precision overload of @ref multiplyScalar.
template <Backend TBackend = DefaultBackend>
inline void multiplyScalar(StridedSpan<float> dest, StridedSpan<const float> a, float scalar) noexcept {
	TBackend::multiplyScalar(dest, a, scalar);
}

///----------------------------------------
///   @brief Compute @c dest = a + scalar elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a Source view.
///   @param scalar Scalar addend.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void addScalar(StridedSpan<double> dest, StridedSpan<const double> a, double scalar) noexcept {
	TBackend::addScalar(dest, a, scalar);
}

/// @brief Single-precision overload of @ref addScalar.
template <Backend TBackend = DefaultBackend>
inline void addScalar(StridedSpan<float> dest, StridedSpan<const float> a, float scalar) noexcept {
	TBackend::addScalar(dest, a, scalar);
}

///----------------------------------------
/// MARK: Elementwise arithmetic
///----------------------------------------

///----------------------------------------
///   @brief Compute @c dest = a + b elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a First addend view.
///   @param b Second addend view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void add(StridedSpan<double> dest, StridedSpan<const double> a, StridedSpan<const double> b) noexcept {
	TBackend::add(dest, a, b);
}

/// @brief Single-precision overload of @ref add.
template <Backend TBackend = DefaultBackend>
inline void add(StridedSpan<float> dest, StridedSpan<const float> a, StridedSpan<const float> b) noexcept {
	TBackend::add(dest, a, b);
}

///----------------------------------------
///   @brief Compute @c dest = a - b elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a Minuend view.
///   @param b Subtrahend view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void subtract(StridedSpan<double> dest, StridedSpan<const double> a, StridedSpan<const double> b) noexcept {
	TBackend::subtract(dest, a, b);
}

/// @brief Single-precision overload of @ref subtract.
template <Backend TBackend = DefaultBackend>
inline void subtract(StridedSpan<float> dest, StridedSpan<const float> a, StridedSpan<const float> b) noexcept {
	TBackend::subtract(dest, a, b);
}

///----------------------------------------
///   @brief Compute @c dest = a * b elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param a First factor view.
///   @param b Second factor view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void multiply(StridedSpan<double> dest, StridedSpan<const double> a, StridedSpan<const double> b) noexcept {
	TBackend::multiply(dest, a, b);
}

/// @brief Single-precision overload of @ref multiply.
template <Backend TBackend = DefaultBackend>
inline void multiply(StridedSpan<float> dest, StridedSpan<const float> a, StridedSpan<const float> b) noexcept {
	TBackend::multiply(dest, a, b);
}

///----------------------------------------
/// MARK: Reductions
///----------------------------------------

///----------------------------------------
///   @brief Sum of the elementwise products of @p a and @p b.
/// @tparam TBackend Backend to dispatch through.
///   @param a First operand view.
///   @param b Second operand view.
/// @returns The dot product of the two views.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
[[nodiscard]] inline double dot(StridedSpan<const double> a, StridedSpan<const double> b) noexcept {
	return TBackend::dot(a, b);
}

/// @brief Single-precision overload of @ref dot.
template <Backend TBackend = DefaultBackend>
[[nodiscard]] inline float dot(StridedSpan<const float> a, StridedSpan<const float> b) noexcept {
	return TBackend::dot(a, b);
}

///----------------------------------------
///   @brief Sum of the elements of @p a.
/// @tparam TBackend Backend to dispatch through.
///   @param a Operand view.
/// @returns The total of every element.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
[[nodiscard]] inline double sum(StridedSpan<const double> a) noexcept {
	return TBackend::sum(a);
}

/// @brief Single-precision overload of @ref sum.
template <Backend TBackend = DefaultBackend>
[[nodiscard]] inline float sum(StridedSpan<const float> a) noexcept {
	return TBackend::sum(a);
}

///----------------------------------------
/// MARK: Transcendental
///----------------------------------------

///----------------------------------------
///   @brief Compute @c dest = cos(src) elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param src Source view of angles in radians.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void cosine(StridedSpan<double> dest, StridedSpan<const double> src) noexcept {
	TBackend::cosine(dest, src);
}

/// @brief Single-precision overload of @ref cosine.
template <Backend TBackend = DefaultBackend>
inline void cosine(StridedSpan<float> dest, StridedSpan<const float> src) noexcept {
	TBackend::cosine(dest, src);
}

///----------------------------------------
///   @brief Compute @c dest = sin(src) elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param src Source view of angles in radians.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void sine(StridedSpan<double> dest, StridedSpan<const double> src) noexcept {
	TBackend::sine(dest, src);
}

/// @brief Single-precision overload of @ref sine.
template <Backend TBackend = DefaultBackend>
inline void sine(StridedSpan<float> dest, StridedSpan<const float> src) noexcept {
	TBackend::sine(dest, src);
}

///----------------------------------------
///   @brief Compute @c sineDest = sin(src) and @c cosineDest = cos(src) in one pass.
/// @tparam TBackend Backend to dispatch through.
///   @param sineDest Destination view for the sines.
///   @param cosineDest Destination view for the cosines.
///   @param src Source view of angles in radians.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void sinCosine(StridedSpan<double> sineDest, StridedSpan<double> cosineDest, StridedSpan<const double> src) noexcept {
	TBackend::sinCosine(sineDest, cosineDest, src);
}

/// @brief Single-precision overload of @ref sinCosine.
template <Backend TBackend = DefaultBackend>
inline void sinCosine(StridedSpan<float> sineDest, StridedSpan<float> cosineDest, StridedSpan<const float> src) noexcept {
	TBackend::sinCosine(sineDest, cosineDest, src);
}

///----------------------------------------
///   @brief Compute @c dest = sqrt(src) elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param src Source view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void squareRoot(StridedSpan<double> dest, StridedSpan<const double> src) noexcept {
	TBackend::squareRoot(dest, src);
}

/// @brief Single-precision overload of @ref squareRoot.
template <Backend TBackend = DefaultBackend>
inline void squareRoot(StridedSpan<float> dest, StridedSpan<const float> src) noexcept {
	TBackend::squareRoot(dest, src);
}

///----------------------------------------
///   @brief Compute @c dest = exp(src) elementwise.
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param src Source view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void exponential(StridedSpan<double> dest, StridedSpan<const double> src) noexcept {
	TBackend::exponential(dest, src);
}

/// @brief Single-precision overload of @ref exponential.
template <Backend TBackend = DefaultBackend>
inline void exponential(StridedSpan<float> dest, StridedSpan<const float> src) noexcept {
	TBackend::exponential(dest, src);
}

///----------------------------------------
///   @brief Compute @c dest = log(src) elementwise (natural logarithm).
/// @tparam TBackend Backend to dispatch through.
///   @param dest Destination view.
///   @param src Source view.
///----------------------------------------

template <Backend TBackend = DefaultBackend>
inline void naturalLog(StridedSpan<double> dest, StridedSpan<const double> src) noexcept {
	TBackend::naturalLog(dest, src);
}

/// @brief Single-precision overload of @ref naturalLog.
template <Backend TBackend = DefaultBackend>
inline void naturalLog(StridedSpan<float> dest, StridedSpan<const float> src) noexcept {
	TBackend::naturalLog(dest, src);
}

///----------------------------------------
/// MARK: Self-test
///----------------------------------------

///----------------------------------------
///   @class Tests
///   @brief Hosts the @ref Blast self-test.
///----------------------------------------

struct Tests final {
	///----------------------------------------
	/// @brief Validates the operations against closed forms and asserts every backend agrees.
	/// @details Throws @ref Math::selftest::Failure on the first violation.
	///----------------------------------------
	
	static void selfTest();
};

inline void Tests::selfTest() {
	// Scalar multiply-add matches the closed form over a contiguous run
	{
		constexpr std::array<double, 4> a = {1.0, 2.0, 3.0, 4.0};
		constexpr std::array<double, 4> b = {10.0, 20.0, 30.0, 40.0};
		std::array<double, 4> dest = {};
		multiplyAddScalar(std::span(dest), std::span(a), 2.0, std::span(b));
		for (std::size_t index = 0; index < dest.size(); ++index) {
			Math::selftest::check(dest[index] == a[index] * 2.0 + b[index], "multiplyAddScalar");
		}
	}
	
	// A struct member projects to a strided view with the expected stride and values
	{
		struct Sample {
			double x;
			double y;
			double z;
		};
		std::array<Sample, 3> data = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};
		auto ys = field(std::span(data), &Sample::y);
		Math::selftest::check(ys.size() == 3 && ys.stride() == 3, "field shape");
		Math::selftest::check(ys[0] == 2.0 && ys[1] == 5.0 && ys[2] == 8.0, "field values");
		Math::selftest::check(sum(StridedSpan<const double>(ys)) == 15.0, "strided sum");
	}
	
	// The dot product weights one strided field against a contiguous buffer, as VSOP series summation does
	{
		struct Term {
			double amplitude;
			double phase;
			double frequency;
		};
		std::array<Term, 3> terms = {{{2.0, 0.0, 1.0}, {3.0, 0.5, 2.0}, {4.0, 1.0, 3.0}}};
		std::array<double, 3> scratch = {};
		const auto time = 0.25;
		multiplyAddScalar(std::span(scratch), field(std::span(terms), &Term::frequency), time, field(std::span(terms), &Term::phase));
		cosine(std::span(scratch), std::span(scratch));
		const auto total = dot(field(std::span(terms), &Term::amplitude), std::span(scratch));
		
		// Compare against the direct evaluation of the same series
		auto reference = 0.0;
		for (const auto& term : terms) {
			reference += term.amplitude * std::cos(term.phase + term.frequency * time);
		}
		Math::selftest::check(std::abs(total - reference) < 1.0e-12, "vsop series");
	}
	
	// Every backend agrees on the transcendental results over a ramp, in both precisions
	{
		std::array<double, 64> input = {};
		for (std::size_t index = 0; index < input.size(); ++index) {
			input[index] = static_cast<double>(index) * 0.1;
		}
		
		std::array<double, 64> viaPlain = {};
		std::array<double, 64> viaDefault = {};
		cosine<Plain>(std::span(viaPlain), std::span(input));
		cosine<DefaultBackend>(std::span(viaDefault), std::span(input));
		for (std::size_t index = 0; index < input.size(); ++index) {
			Math::selftest::check(std::abs(viaPlain[index] - viaDefault[index]) < 1.0e-12, "backend agreement (double)");
		}
	}
	
	{
		std::array<float, 64> input = {};
		for (std::size_t index = 0; index < input.size(); ++index) {
			input[index] = static_cast<float>(index) * 0.1f;
		}
		
		std::array<float, 64> viaPlain = {};
		std::array<float, 64> viaDefault = {};
		sine<Plain>(std::span(viaPlain), std::span(input));
		sine<DefaultBackend>(std::span(viaDefault), std::span(input));
		for (std::size_t index = 0; index < input.size(); ++index) {
			Math::selftest::check(std::abs(viaPlain[index] - viaDefault[index]) < 1.0e-5f, "backend agreement (float)");
		}
	}
}

///----------------------------------------
} // namespace Math::Blast
///----------------------------------------

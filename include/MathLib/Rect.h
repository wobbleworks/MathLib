///----------------------------------------
/// @file Rect.h
/// @ingroup MathLib
/// @brief An axis-aligned rectangle template with inset, offset, intersection, and union operations.
/// @author Created by John Stephen on 1/1/26.
/// @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///            Licensed under the Apache License, Version 2.0.
///            SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "MathLib/SelfTestCheck.h"

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief An axis-aligned rectangle defined by left, top, right, bottom edges.
/// @tparam TType Arithmetic type for the edge coordinates.
///----------------------------------------

template <class TType>
struct rect {
///----------------------------------------
	TType _left{}, _top{}, _right{}, _bottom{};
	
	constexpr rect() = default;
	constexpr rect(TType left, TType top, TType right, TType bottom) : _left(left), _top(top), _right(right), _bottom(bottom) {}
	constexpr rect(const rect& rect, TType insetLeft, TType insetTop, TType insetRight, TType insetBottom)
		: _left(rect._left + insetLeft), _top(rect._top + insetTop), _right(rect._right - insetRight), _bottom(rect._bottom - insetBottom) {}
	constexpr rect(const rect& rect, TType insetHoriz, TType insetVert)
		: _left(rect._left + insetHoriz), _top(rect._top + insetVert), _right(rect._right - insetHoriz), _bottom(rect._bottom - insetVert) {}
		
	///----------------------------------------
	/// @brief Set all four edges.
	///----------------------------------------
	
	constexpr void set(TType left, TType top, TType right, TType bottom) noexcept { _left = left; _top = top; _right = right; _bottom = bottom; }
	
	/// @brief Width of the rectangle.
	[[nodiscard]] constexpr TType width() const noexcept { return _right - _left; }
	
	/// @brief Height of the rectangle.
	[[nodiscard]] constexpr TType height() const noexcept { return _bottom - _top; }
	
	/// @brief Horizontal center coordinate.
	[[nodiscard]] constexpr TType horizCenter() const noexcept { return (_right + _left) / 2; }
	
	/// @brief Vertical center coordinate.
	[[nodiscard]] constexpr TType vertCenter() const noexcept { return (_bottom + _top) / 2; }
	
	///----------------------------------------
	/// @brief Inset (shrink) the rectangle symmetrically.
	/// @param horiz Horizontal inset applied to both left and right edges.
	/// @param vert Vertical inset applied to both top and bottom edges.
	///----------------------------------------
	
	constexpr void inset(TType horiz, TType vert) noexcept { _left += horiz; _top += vert; _right -= horiz; _bottom -= vert; }
	
	///----------------------------------------
	/// @brief Inset the rectangle with independent edge amounts.
	///----------------------------------------
	
	constexpr void inset(TType left, TType top, TType right, TType bottom) noexcept { _left += left; _top += top; _right -= right; _bottom -= bottom; }
	
	///----------------------------------------
	/// @brief Set this rect to a symmetric inset of another rect.
	///----------------------------------------
	
	constexpr void inset(const rect& rect, TType horiz, TType vert) noexcept { _left = rect._left + horiz; _top = rect._top + vert; _right = rect._right - horiz; _bottom = rect._bottom - vert; }
	
	///----------------------------------------
	/// @brief Set this rect to an independent-edge inset of another rect.
	///----------------------------------------
	
	constexpr void inset(const rect& rect, TType left, TType top, TType right, TType bottom) noexcept { _left = rect._left + left; _top = rect._top + top; _right = rect._right - right; _bottom = rect._bottom - bottom; }
	
	///----------------------------------------
	/// @brief Translate the rectangle uniformly.
	/// @param horiz Horizontal translation.
	/// @param vert Vertical translation.
	///----------------------------------------
	
	constexpr void offset(TType horiz, TType vert) noexcept { _left += horiz; _top += vert; _right += horiz; _bottom += vert; }
	
	///----------------------------------------
	/// @brief Offset each edge independently.
	///----------------------------------------
	
	constexpr void offset(TType left, TType top, TType right, TType bottom) noexcept { _left += left; _top += top; _right += right; _bottom += bottom; }
	
	///----------------------------------------
	/// @brief Set this rect to a uniform translation of another rect.
	///----------------------------------------
	
	constexpr void offset(const rect& rect, TType horiz, TType vert) noexcept { _left = rect._left + horiz; _top = rect._top + vert; _right = rect._right + horiz; _bottom = rect._bottom + vert; }
	
	///----------------------------------------
	/// @brief Set this rect to an independent-edge offset of another rect.
	///----------------------------------------
	
	constexpr void offset(const rect& rect, TType left, TType top, TType right, TType bottom) noexcept { _left = rect._left + left; _top = rect._top + top; _right = rect._right + right; _bottom = rect._bottom + bottom; }
	
	///----------------------------------------
	/// @brief Center this rect within another rect, preserving size.
	/// @param rect The bounding rectangle to center within.
	///----------------------------------------
	
	void centerInRect(const rect& rect);
	
	///----------------------------------------
	/// @brief Scale this rect to fit inside another rect while preserving aspect ratio.
	/// @details If this rect has zero or negative dimensions, it is replaced by the target rect.
	/// @param rect The bounding rectangle to fit within.
	///----------------------------------------
	
	void fitInRect(const rect& rect);
	
	///----------------------------------------
	/// @brief Test whether this rect overlaps another (non-mutating).
	/// @param r The rectangle to test against.
	/// @return True if the rectangles overlap.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool intersects(const rect& r) const noexcept { return _top < r._bottom && _bottom > r._top && _left < r._right && _right > r._left; }
	
	///----------------------------------------
	/// @brief Test whether this rect fully contains another.
	/// @param r The rectangle to test.
	/// @return True if r is entirely inside this rect.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool contains(const rect& r) const noexcept { return r._left >= _left && r._right <= _right && r._top >= _top && r._bottom <= _bottom; }
	
	/// @brief Whether the rectangle has zero or negative area.
	[[nodiscard]] constexpr bool isEmpty() const noexcept { return (_right <= _left) || (_bottom <= _top); }
	
	/// @brief Reset the rectangle to zero.
	constexpr void makeEmpty() noexcept { set(0, 0, 0, 0); }
	
	///----------------------------------------
	/// @brief Intersect this rect with another, storing the result in this rect.
	/// @param rect The rectangle to intersect with.
	/// @return True if the intersection is non-empty.
	///----------------------------------------
	
	[[nodiscard]] bool intersect(const rect& rect);
	
	///----------------------------------------
	/// @brief Set this rect to the intersection of two other rects.
	/// @param rect1 First rectangle.
	/// @param rect2 Second rectangle.
	/// @return True if the intersection is non-empty.
	///----------------------------------------
	
	[[nodiscard]] bool intersect(const rect& rect1, const rect& rect2);
	
	///----------------------------------------
	/// @brief Expand this rect to include another rect (union).
	/// @param rect The rectangle to merge in.
	///----------------------------------------
	
	constexpr void makeUnion(const rect& rect) noexcept {
		// Skip empty source
		if (rect.isEmpty()) {
			return;
		}
		
		// Replace if currently empty
		if (isEmpty()) {
			*this = rect;
			return;
		}
		
		// Expand each edge as needed
		if (rect._left < _left) {
			_left = rect._left;
		}
		if (rect._top < _top) {
			_top = rect._top;
		}
		if (rect._right > _right) {
			_right = rect._right;
		}
		if (rect._bottom > _bottom) {
			_bottom = rect._bottom;
		}
	}
	
	///----------------------------------------
	/// @brief Test whether a point lies inside the rectangle (half-open: inclusive left/top, exclusive right/bottom).
	/// @param x Horizontal coordinate to test.
	/// @param y Vertical coordinate to test.
	/// @return True if the point is inside.
	///----------------------------------------
	
	[[nodiscard]] constexpr bool pointInRect(TType x, TType y) const noexcept { return x >= _left && x < _right && y >= _top && y < _bottom; }
	
	[[nodiscard]] constexpr bool operator==(const rect&) const = default;
	
	[[nodiscard]] constexpr TType& operator[](size_t index) noexcept { assert(index < 4); return (&_left)[index]; }
	[[nodiscard]] constexpr TType operator[](size_t index) const noexcept { assert(index < 4); return (&_left)[index]; }
	
	[[nodiscard]] constexpr rect operator+(const rect& rect) const noexcept { return {_left + rect._left, _top + rect._top, _right + rect._right, _bottom + rect._bottom}; }
	[[nodiscard]] constexpr rect operator-(const rect& rect) const noexcept { return {_left - rect._left, _top - rect._top, _right - rect._right, _bottom - rect._bottom}; }
	[[nodiscard]] constexpr rect operator*(TType factor) const noexcept { return {_left * factor, _top * factor, _right * factor, _bottom * factor}; }
};

using rect_int16 = rect<int16_t>;
using rect_int32 = rect<int32_t>;
using rect_int = rect<int>;
using rect_float = rect<float>;
using rect_double = rect<double>;

///----------------------------------------
/// MARK: Rect out-of-line definitions
///----------------------------------------

template <class TType>
void rect<TType>::centerInRect(const rect& rect) {
	// Compute the centering offset and apply it
	offset((rect.width() - width()) / 2 + rect._left - _left, (rect.height() - height()) / 2 + rect._top - _top);
}

template <class TType>
void rect<TType>::fitInRect(const rect& rect) {
	// Measure the source dimensions
	const auto vWidth = width();
	const auto vHeight = height();
	
	// Guard against zero/negative source dimensions
	if (vWidth <= TType(0) || vHeight <= TType(0)) {
		*this = rect;
		return;
	}
	
	// Measure the destination dimensions
	const auto vDestWidth = rect.width();
	const auto vDestHeight = rect.height();
	
	// Scale to fit, preserving aspect ratio
	if (vWidth * vDestHeight > vHeight * vDestWidth) {
		// Width-limited: scale height to match width ratio
		const auto vNewHeight = vHeight * vDestWidth / vWidth;
		
		_left = rect._left;
		_top = rect._top + (vDestHeight - vNewHeight) / 2;
		_right = rect._right;
		_bottom = _top + vNewHeight;
	} else {
		// Height-limited: scale width to match height ratio
		const auto vNewWidth = vWidth * vDestHeight / vHeight;
		
		_left = rect._left + (vDestWidth - vNewWidth) / 2;
		_top = rect._top;
		_right = _left + vNewWidth;
		_bottom = rect._bottom;
	}
}

template <class TType>
bool rect<TType>::intersect(const rect& rect) {
	// Empty input yields empty result
	if (rect.isEmpty() || isEmpty()) {
		makeEmpty();
		return false;
	}
	
	// Clamp each edge to the intersection
	if (rect._left > _left) {
		_left = rect._left;
	}
	if (rect._top > _top) {
		_top = rect._top;
	}
	if (rect._bottom < _bottom) {
		_bottom = rect._bottom;
	}
	if (rect._right < _right) {
		_right = rect._right;
	}
	
	return !isEmpty();
}

template <class TType>
bool rect<TType>::intersect(const rect& rect1, const rect& rect2) {
	// Compute the intersection edges directly
	_left = std::max(rect1._left, rect2._left);
	_top = std::max(rect1._top, rect2._top);
	_right = std::min(rect1._right, rect2._right);
	_bottom = std::min(rect1._bottom, rect2._bottom);
	
	return !isEmpty();
}

///----------------------------------------
/// @brief Verifies letterbox fitting, intersection, union, and the half-open point test.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void rectSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	
	// A 2:1 rect fitted into the unit square is centered and vertically letterboxed.
	{
		rect_double fitted(0.0, 0.0, 2.0, 1.0);
		fitted.fitInRect(rect_double(0.0, 0.0, 1.0, 1.0));
		check(near(fitted._left, 0.0) && near(fitted._right, 1.0), "letterbox spans full width");
		check(near(fitted._top, 0.25) && near(fitted._bottom, 0.75), "letterbox centered vertically");
	}
	
	// Two overlapping rects intersect in their shared corner region; disjoint rects do not.
	{
		rect_double overlap;
		const auto ok = overlap.intersect(rect_double(0.0, 0.0, 4.0, 4.0), rect_double(2.0, 2.0, 6.0, 6.0));
		check(ok, "overlapping rects report a non-empty intersection");
		check(near(overlap._left, 2.0) && near(overlap._top, 2.0) && near(overlap._right, 4.0) && near(overlap._bottom, 4.0),
			"intersection edges");
			
		rect_double disjoint;
		check(!disjoint.intersect(rect_double(0.0, 0.0, 1.0, 1.0), rect_double(2.0, 2.0, 3.0, 3.0)),
			"disjoint rects report an empty intersection");
	}
	
	// A union grows to the bounding rect; an empty source leaves the target untouched.
	{
		rect_double bounds(0.0, 0.0, 2.0, 2.0);
		bounds.makeUnion(rect_double(1.0, 1.0, 4.0, 4.0));
		check(near(bounds._left, 0.0) && near(bounds._top, 0.0) && near(bounds._right, 4.0) && near(bounds._bottom, 4.0),
			"union spans both rects");
			
		rect_double kept(1.0, 1.0, 3.0, 3.0);
		kept.makeUnion(rect_double(5.0, 5.0, 5.0, 5.0));
		check(kept == rect_double(1.0, 1.0, 3.0, 3.0), "union ignores an empty source");
		
		rect_double empty;
		empty.makeUnion(rect_double(2.0, 2.0, 6.0, 6.0));
		check(empty == rect_double(2.0, 2.0, 6.0, 6.0), "union replaces an empty target");
	}
	
	// pointInRect is half-open: inclusive on the left/top edges, exclusive on the right/bottom.
	{
		const rect_double box(0.0, 0.0, 10.0, 10.0);
		check(box.pointInRect(0.0, 0.0), "top-left corner is inside");
		check(box.pointInRect(5.0, 5.0), "interior point is inside");
		check(!box.pointInRect(10.0, 10.0), "bottom-right corner is outside");
		check(!box.pointInRect(10.0, 5.0), "right edge is outside");
		check(!box.pointInRect(5.0, 10.0), "bottom edge is outside");
	}
}
	
} // namespace Math

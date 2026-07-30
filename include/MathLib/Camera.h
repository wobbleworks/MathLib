///----------------------------------------
///       @file Camera.h
///    @ingroup MathLib
///      @brief A viewpoint: the settings that describe one, the transforms derived from them, and a
///             thread-safe holder that publishes consistent snapshots of both.
///    @details The type is split into three pieces, and the split is the design:
///
///             @ref Math::CameraSettings is a plain aggregate of *inputs* — where the viewpoint is,
///             which way it faces, how wide it sees, what pixels it fills. It holds nothing derived,
///             so no two fields can disagree.
///
///             @ref Math::CameraView is everything *derived* from one settings value: the view,
///             projection and viewport transforms with their inverses, the culling frusta, the
///             resolved per-axis fields of view, and screen/world projection. It is computed once, in
///             its constructor, and is immutable thereafter — so every matrix a caller reads from one
///             describes the same viewpoint. Construct one directly whenever the settings are already
///             in hand; it needs no camera object.
///
///             @ref Math::Camera is a small thread-safe holder around the two: a writer edits the
///             settings, readers take @ref Math::Camera::view snapshots. A snapshot's lifetime is
///             independent of later edits, which is what lets a render thread work from a coherent
///             viewpoint while an animation thread moves the camera.
///
///             Directions are right-handed with **+Z forward**, +X right and +Y up in view space, and
///             normalized device coordinates run @c [-1,1] with +Y up and the near plane at @c -1.
///             Everything in @ref Math::CameraSettings is expressed the way the *screen* is, though —
///             y grows downward — so the viewport, the clip bounds and the viewport offset all read
///             the same way. The flip to NDC happens once, inside @ref Math::CameraView.
///
///             @c position and @c orientation place a *platform* — the vantage point the application
///             drives, which is where the viewer logically is. @c riderTransform then carries the eye
///             away from that platform: a head pose from a headset, or a motion-driven parallax nudge on
///             a handheld. It is identity whenever the eye simply is the platform, which is every
///             ordinary camera, and it composes so that a camera with no rider behaves exactly as though
///             the concept did not exist.
///     @author Created by John Stephen on 7/30/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/FieldOfView.h"
#include "MathLib/Frustum.h"
#include "MathLib/Matrix.h"
#include "MathLib/Numbers.h"
#include "MathLib/Quaternion.h"
#include "MathLib/Ray.h"
#include "MathLib/Rect.h"
#include "MathLib/SelfTestCheck.h"
#include "MathLib/Transforms.h"
#include "MathLib/Vector.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
/// @brief How a camera maps view space onto the screen.
///----------------------------------------

enum class Projection : uint8_t {
	perspective,          ///< A finite frustum; depth maps the near plane to -1 and the far plane to +1.
	infinitePerspective,  ///< The same frustum with the far plane at infinity — nothing is ever too distant to draw.
	orthographic          ///< A parallel projection; size on screen does not fall off with distance.
};

///----------------------------------------
///   @brief Where the near and far planes land in normalized device coordinates.
/// @details Only the depth row of the projection changes; the image is identical under all three.
///
///          @c reversedZeroToOne exists for precision rather than taste. A floating-point depth buffer
///          packs its resolution near zero, while a perspective projection packs its own near the near
///          plane — so the conventional mapping stacks both concentrations at the same end and starves
///          the far distances, and reversing it cancels one against the other almost exactly. It costs
///          a depth buffer cleared to 0 and a @c greater-than depth test instead of @c less-than.
///   @note  Paired with @ref Projection::infinitePerspective the reversed mapping reduces to
///          @c depth = nearDepth / viewDepth, the canonical reversed-Z form, with no far plane to choose
///          and no precision cliff at any distance.
///----------------------------------------

enum class DepthConvention : uint8_t {
	negativeOneToOne,   ///< Near plane at -1, far plane at +1. OpenGL's traditional range.
	zeroToOne,          ///< Near plane at 0, far plane at +1. Metal, Vulkan and Direct3D.
	reversedZeroToOne   ///< Near plane at +1, far plane at 0. Best precision from a float depth buffer.
};

///----------------------------------------
/// @brief The normalized device depth of the near plane under a convention.
///----------------------------------------

[[nodiscard]] constexpr double nearPlaneDepth(DepthConvention convention) noexcept {
	switch (convention) {
		case DepthConvention::negativeOneToOne:  return -1;
		case DepthConvention::zeroToOne:         return 0;
		case DepthConvention::reversedZeroToOne: return 1;
	}
	return -1;
}

///----------------------------------------
///   @brief The normalized device depth of the far plane under a convention.
/// @details Also the value to clear a depth buffer to, and the limit an infinite projection approaches
///          without reaching.
///----------------------------------------

[[nodiscard]] constexpr double farPlaneDepth(DepthConvention convention) noexcept {
	return convention == DepthConvention::reversedZeroToOne ? 0.0 : 1.0;
}

///----------------------------------------
/// @brief Which axis of the view a field of view or angular scale is measured along.
/// @details @c narrow and @c wide follow the shorter and longer edge of the viewport, so they keep
///          their meaning as the device rotates; @c horizontal and @c vertical are fixed to the screen.
///----------------------------------------

enum class CameraAxis : uint8_t {
	narrow,      ///< The shorter viewport edge.
	wide,        ///< The longer viewport edge.
	horizontal,  ///< The screen's x axis.
	vertical     ///< The screen's y axis.
};

///----------------------------------------
/// @brief Inset amounts for the four edges of a rectangle, in the same units as the rectangle.
/// @details Distinct from @ref Math::rect on purpose: these are widths measured inward from each edge,
///          not coordinates, and the two are easy to confuse when both are four doubles.
///----------------------------------------

struct EdgeInsets final {
	double left{};
	double top{};
	double right{};
	double bottom{};
	
	[[nodiscard]] bool operator==(const EdgeInsets &) const = default;
	
	/// @brief Whether every edge is zero.
	[[nodiscard]] bool isEmpty() const noexcept { return left == 0 && top == 0 && right == 0 && bottom == 0; }
};

///----------------------------------------
///   @brief The four tangents that fix a perspective frustum's lateral extent.
/// @details Each is the tangent of the angle from the forward axis out to that plane, positive in the
///          direction its name says — @c left positive means the left plane really is to the left. The
///          four are independent, so an off-centre projection is expressible directly, and a negative
///          value places that plane on the far side of the axis, which a strongly off-centre frustum
///          needs.
///
///          This is the general form the field-of-view framing reduces to. Supplying it directly is how
///          you accept a projection you do not get to choose — a headset reports its optics this way —
///          or how you aim one at a fixed window rather than along the view axis (see @ref window).
///    @note Tangents describe a pyramid through an apex, which @ref Projection::orthographic does not
///          have; a camera cannot use both.
///----------------------------------------

struct FrustumTangents final {
	double left{};
	double right{};
	double top{};
	double bottom{};
	
	[[nodiscard]] bool operator==(const FrustumTangents &) const = default;
	
	/// @brief Whether the tangents enclose a non-empty volume.
	[[nodiscard]] bool isValid() const noexcept {
		return std::isfinite(left) && std::isfinite(right) && std::isfinite(top) && std::isfinite(bottom)
		    && left + right > 0 && top + bottom > 0;
	}
	
	/// @brief The total horizontal field of view, in radians.
	[[nodiscard]] double horizontalFieldOfViewRadians() const noexcept { return std::atan(left) + std::atan(right); }
	
	/// @brief The total vertical field of view, in radians.
	[[nodiscard]] double verticalFieldOfViewRadians() const noexcept { return std::atan(top) + std::atan(bottom); }
	
	///----------------------------------------
	/// @brief A centred frustum spanning the given total fields of view, in radians.
	///----------------------------------------
	
	[[nodiscard]] static FrustumTangents symmetric(double horizontalFieldOfViewRadians, double verticalFieldOfViewRadians) noexcept {
		const double horizontal = std::tan(horizontalFieldOfViewRadians / 2);
		const double vertical = std::tan(verticalFieldOfViewRadians / 2);
		return FrustumTangents{.left = horizontal, .right = horizontal, .top = vertical, .bottom = vertical};
	}
	
	///----------------------------------------
	///   @brief The frustum that keeps a fixed rectangular window filling the view as the eye moves behind it.
	/// @details Off-axis, "fish tank" projection: the window stays put in space and the viewer moves, so
	///          what is drawn at the window's depth does not shift at all, what is drawn at infinity shifts
	///          the most, and everything between shifts in proportion. That gradient is the effect — it is
	///          what reads as depth rather than as the image sliding around.
	///
	///          It only works if @p eyeOffset is *also* applied to the camera's position. These tangents
	///          hold the window still; the eye displacement supplies the parallax. Apply one without the
	///          other and the window drifts, which looks like a soft, swimmy version of the same effect.
	///   @param halfWidth,halfHeight Half the window's extent, in world units, centred on the forward axis.
	///   @param distance Distance from the unoffset viewpoint to the window plane, in the same units.
	///   @param eyeOffset The eye's displacement in view space — @c +x right, @c +y up, @c +z forward.
	///----------------------------------------
	
	[[nodiscard]] static FrustumTangents window(double halfWidth, double halfHeight, double distance, const Double3 &eyeOffset) noexcept {
		const double windowDepth = distance - eyeOffset.z;
		return FrustumTangents{
			.left = (halfWidth + eyeOffset.x) / windowDepth,
			.right = (halfWidth - eyeOffset.x) / windowDepth,
			.top = (halfHeight - eyeOffset.y) / windowDepth,
			.bottom = (halfHeight + eyeOffset.y) / windowDepth};
	}
};

///----------------------------------------
/// @struct CameraSettings
/// @brief The complete input description of a viewpoint — position, orientation, framing and projection.
/// @details A plain aggregate: every field is independent, nothing is cached, and two settings values
///          that compare equal describe the same view. Designated initializers work, and so does
///          copying one, adjusting a field and handing it back to @ref Math::Camera::setSettings.
///----------------------------------------

struct CameraSettings final {
	///----------------------------------------
	/// @name Placement
	///----------------------------------------
	///@{
	
	/// @brief The platform's position, in world coordinates — where the viewer logically is.
	Double3 position{0, 0, 0};
	
	/// @brief The platform's orientation; its local x, y and z axes are the view's right, up and forward.
	Quaternion64 orientation{};
	
	///----------------------------------------
	///   @brief The eye's displacement from the platform: a rigid transform mapping platform space into eye
	///          space. Identity when the eye is the platform, which is the ordinary case.
	/// @details Where a viewer rides a vantage point the application drives rather than being that vantage
	///          point. A headset supplies a tracked head pose here; a handheld can supply a motion-driven
	///          nudge to give the display parallax. @c position and @c orientation move the platform, this
	///          moves the viewer on it, and neither has to know about the other.
	///
	///          Translation is meaningful, not a caveat. Rotating the eye about a pivot some distance ahead
	///          — @c T(0,0,d)·R·T(0,0,-d) — swings it on a small arc, which is what produces parallax
	///          against something at that distance rather than a bare change of aim. @ref CameraView takes
	///          the eye position, the frustum apex, the world axes and the pick-ray origin from the composed
	///          transform, so all of them follow the displacement.
	///    @note This is @c eyeFromPlatform — the *inverse* of the eye's pose on the platform. A pose from a
	///          tracking API is usually the other direction and needs inverting on the way in.
	///    @note Expected to be rigid. A reflection is fine (the inverse is still the transpose); a scale or
	///          shear is not, and falls back to a general inverse with the frustum axes no longer unit
	///          length.
	///----------------------------------------
	
	Double4x4 riderTransform = Double4x4::identity();
	
	///@}
	///----------------------------------------
	/// @name Framing
	///----------------------------------------
	///@{
	
	/// @brief The field of view along the shorter viewport edge, in radians. The longer edge follows from the aspect ratio.
	double narrowAxisFieldOfViewRadians = defaultFieldOfViewRadians;
	
	/// @brief The viewport, in pixels, with y growing downward — @c _top is above @c _bottom.
	rect_double viewport{0, 0, 1, 1};
	
	/// @brief The visible sub-rectangle of the full view, normalized to @c [-1,1] and oriented like the
	///        viewport: @c {-1,-1,1,1} is the whole view, @c _top is the upper edge.
	rect_double clipBounds{-1, -1, 1, 1};
	
	/// @brief Shifts the image within the viewport, in half-viewports: @c +x moves it right, @c +y moves it down.
	Double2 viewportOffset{0, 0};
	
	///----------------------------------------
	///   @brief An explicit frustum, superseding @c narrowAxisFieldOfViewRadians, @c clipBounds and
	///          @c viewportOffset.
	/// @details Set this when the projection is handed to you rather than chosen — headset optics — or when
	///          it is aimed at a fixed window rather than along the view axis. Those three fields are a
	///          parameterization of exactly these four tangents, so setting this does not add a capability
	///          so much as expose the form they already reduce to.
	///    @note The superseded fields are ignored, not reconciled: assigning a field of view while this is
	///          set is a silent no-op. @ref CameraView::fieldOfViewRadians always reports what the resolved
	///          frustum actually spans, so reading it back tells you which one won.
	///          Invalid together with @ref Projection::orthographic.
	///----------------------------------------
	
	std::optional<FrustumTangents> frustumTangents;
	
	/// @brief Edges of the viewport covered by other content, in pixels. Narrows @ref CameraView::unobstructedFrustum only.
	EdgeInsets obstructionMargins{};
	
	/// @brief Mirrors the image left-to-right.
	bool mirrorHorizontally = false;
	
	/// @brief Mirrors the image top-to-bottom.
	bool mirrorVertically = false;
	
	///@}
	///----------------------------------------
	/// @name Projection
	///----------------------------------------
	///@{
	
	/// @brief How view space maps onto the screen.
	Projection projection = Projection::perspective;
	
	/// @brief Where the near and far planes land in normalized device coordinates. Affects depth only.
	DepthConvention depthConvention = DepthConvention::negativeOneToOne;
	
	/// @brief Distance to the near plane. Must be positive under a perspective projection.
	double nearDepth = 1;
	
	/// @brief Distance to the far plane. Ignored under @ref Projection::infinitePerspective.
	double farDepth = 1000;
	
	/// @brief The world-space height the view spans under @ref Projection::orthographic; the width follows
	///        from the aspect ratio. Ignored by the perspective projections, which take their scale from the
	///        field of view instead.
	double orthographicHeight = 1;
	
	///@}
	
	[[nodiscard]] bool operator==(const CameraSettings &) const = default;
};

///----------------------------------------
/// @class CameraView
/// @brief Everything derived from one @ref Math::CameraSettings: transforms, inverses, frusta and projection.
/// @details Computed once in the constructor and immutable thereafter, so every accessor describes the
///          same viewpoint. Construction is not free — it builds and inverts four matrices and two
///          frusta — so build one per settings change, not per query.
///----------------------------------------

class CameraView final {
///----------------------------------------
public:
	///----------------------------------------
	/// @brief Derives the view from @p settings.
	///----------------------------------------
	
	explicit CameraView(const CameraSettings &settings) noexcept;
	
	///----------------------------------------
	/// @brief The settings this view was derived from.
	///----------------------------------------
	
	[[nodiscard]] const CameraSettings &settings() const noexcept { return _settings; }
	
	///----------------------------------------
	///   @brief Whether the view is usable.
	/// @details False when the viewport has no area, or when the projection parameters leave nothing to
	///          see. Every transform is then the identity and both frusta are invalid, so a degenerate
	///          viewport yields nothing rather than NaN propagated through a caller's geometry.
	///----------------------------------------
	
	[[nodiscard]] bool isValid() const noexcept { return _isValid; }
	
	///----------------------------------------
	/// @name Transforms
	///----------------------------------------
	///@{
	
	/// @brief World→view, including @c riderTransform — the transform to render the eye's image with.
	[[nodiscard]] const Double4x4 &viewTransform() const noexcept { return _viewTransform; }
	
	/// @brief View→world.
	[[nodiscard]] const Double4x4 &inverseViewTransform() const noexcept { return _inverseViewTransform; }
	
	/// @brief World→platform, without @c riderTransform — the vantage point the application placed, which
	///        is the one astronomy and other logical queries should use rather than the eye's.
	[[nodiscard]] const Double4x4 &unadjustedViewTransform() const noexcept { return _unadjustedViewTransform; }
	
	/// @brief View→normalized device coordinates, including mirroring and the viewport offset.
	[[nodiscard]] const Double4x4 &projectionTransform() const noexcept { return _projectionTransform; }
	
	/// @brief Normalized device coordinates→view. A true inverse of @ref projectionTransform in every projection mode.
	[[nodiscard]] const Double4x4 &inverseProjectionTransform() const noexcept { return _inverseProjectionTransform; }
	
	/// @brief Normalized device coordinates→viewport pixels. Carries no mirroring or offset; those live in the projection.
	[[nodiscard]] const Double4x4 &viewportTransform() const noexcept { return _viewportTransform; }
	
	/// @brief Viewport pixels→normalized device coordinates.
	[[nodiscard]] const Double4x4 &inverseViewportTransform() const noexcept { return _inverseViewportTransform; }
	
	/// @brief World→normalized device coordinates: the projection composed with the view.
	[[nodiscard]] Double4x4 viewProjectionTransform() const noexcept { return _projectionTransform * _viewTransform; }
	
	/// @brief The normalized device depth the near plane projects to, under the settings' convention.
	[[nodiscard]] double nearPlaneDepth() const noexcept { return Math::nearPlaneDepth(_settings.depthConvention); }
	
	///----------------------------------------
	///   @brief The normalized device depth the far plane projects to, under the settings' convention.
	/// @details The value to clear a depth buffer to. Greater than @ref nearPlaneDepth under every
	///          convention except @ref Math::DepthConvention::reversedZeroToOne, where it is less — which
	///          is the same comparison a depth test has to make, so deriving the test direction from these
	///          two beats hard-coding it.
	///----------------------------------------
	
	[[nodiscard]] double farPlaneDepth() const noexcept { return Math::farPlaneDepth(_settings.depthConvention); }
	
	/// @brief The viewpoint in world coordinates — where the projection actually places the eye.
	[[nodiscard]] const Double3 &eyePosition() const noexcept { return _eyePosition; }
	
	/// @brief The view's forward (+Z) axis, in world coordinates.
	[[nodiscard]] const Double3 &forwardAxis() const noexcept { return _forwardAxis; }
	
	/// @brief The view's right (+X) axis, in world coordinates.
	[[nodiscard]] const Double3 &rightAxis() const noexcept { return _rightAxis; }
	
	/// @brief The view's up (+Y) axis, in world coordinates.
	[[nodiscard]] const Double3 &upAxis() const noexcept { return _upAxis; }
	
	///@}
	///----------------------------------------
	/// @name Culling volumes
	///----------------------------------------
	///@{
	
	///----------------------------------------
	///   @brief The world-space viewing volume.
	/// @details Invalid under @ref Projection::orthographic — a @ref Math::Frustum is a pyramid through an
	///          apex and a parallel projection has none. Use @ref orthographicVolume there, and check
	///          @ref Math::Frustum::isValid before relying on this.
	///----------------------------------------
	
	[[nodiscard]] const Frustum3d &frustum() const noexcept { return _frustum; }
	
	///----------------------------------------
	///   @brief The world-space viewing volume with @c obstructionMargins removed.
	/// @details What the viewer can actually see once overlaid content is accounted for — the volume to
	///          place labels in. Invalid when the margins leave no area, and under an orthographic
	///          projection.
	///----------------------------------------
	
	[[nodiscard]] const Frustum3d &unobstructedFrustum() const noexcept { return _unobstructedFrustum; }
	
	///----------------------------------------
	/// @brief The view-space box an orthographic projection maps to clip space, or @c nullopt in the
	///        perspective modes.
	/// @details @c bounds is in world units, oriented like the viewport (@c _top above @c _bottom), and
	///          applies at every depth in @c [nearDepth, farDepth].
	///----------------------------------------
	
	struct OrthographicVolume final {
		rect_double bounds;  ///< The view-space extent, in world units.
		double nearDepth{};  ///< View-space z of the near plane.
		double farDepth{};   ///< View-space z of the far plane.
	};
	
	[[nodiscard]] const std::optional<OrthographicVolume> &orthographicVolume() const noexcept { return _orthographicVolume; }
	
	///----------------------------------------
	///   @brief The frustum the framing actually resolved to.
	/// @details What was asked for, once the clip rectangle and viewport offset have been folded in — or
	///          simply @c settings().frustumTangents when those were supplied. This is the form to hand to
	///          another projection API, and feeding it back through @c frustumTangents reproduces this view
	///          exactly.
	///   @return The tangents, or an invalid @ref Math::FrustumTangents under @ref Projection::orthographic.
	///----------------------------------------
	
	[[nodiscard]] const FrustumTangents &frustumTangents() const noexcept { return _resolvedTangents; }
	
	///@}
	///----------------------------------------
	/// @name Angular scale
	///----------------------------------------
	///@{
	
	///----------------------------------------
	///   @brief The field of view along an axis, in radians.
	///   @return The angle, or NaN under @ref Projection::orthographic, where a parallel projection has no
	///           field of view to report.
	///----------------------------------------
	
	[[nodiscard]] double fieldOfViewRadians(CameraAxis axis) const noexcept;
	
	///----------------------------------------
	///   @brief How many pixels one radian spans along an axis.
	///   @return The scale, or NaN under @ref Projection::orthographic.
	///----------------------------------------
	
	[[nodiscard]] double pixelsPerRadian(CameraAxis axis) const noexcept;
	
	///----------------------------------------
	///   @brief The finest angular detail the view can resolve — half a pixel on the narrow axis, in radians.
	/// @details The threshold below which two directions land on the same pixel, so there is no point
	///          distinguishing them.
	///   @return The angle, or NaN under @ref Projection::orthographic.
	///----------------------------------------
	
	[[nodiscard]] double angularResolutionRadians() const noexcept;
	
	/// @brief The viewport's width divided by its height.
	[[nodiscard]] double aspectRatio() const noexcept { return _aspectRatio; }
	
	///@}
	///----------------------------------------
	/// @name Screen and world
	///----------------------------------------
	///@{
	
	///----------------------------------------
	///   @brief The world-space ray through a viewport point.
	/// @details Under a perspective projection every ray starts at @ref eyePosition and fans outward;
	///          under an orthographic one they are parallel and start on the near plane. The direction is
	///          unit length.
	///   @param screenPoint The viewport point, in pixels, in the same coordinates as @c settings().viewport.
	///----------------------------------------
	
	[[nodiscard]] Ray3d rayThroughScreenPoint(const Double2 &screenPoint) const noexcept;
	
	///----------------------------------------
	///   @brief The unit world-space direction through a viewport point.
	/// @details Constant across the viewport under an orthographic projection, where every ray is parallel.
	///----------------------------------------
	
	[[nodiscard]] Double3 worldDirectionThroughScreenPoint(const Double2 &screenPoint) const noexcept;
	
	///----------------------------------------
	/// @brief The unit view-space direction through a viewport point — the same ray, before the view
	///        transform, for a caller aligning against the device rather than the world.
	///----------------------------------------
	
	[[nodiscard]] Double3 viewDirectionThroughScreenPoint(const Double2 &screenPoint) const noexcept;
	
	///----------------------------------------
	///   @brief Where a world-space direction lands on the viewport.
	/// @details A direction has no position, so it is projected as a point on the unit sphere about the eye.
	///   @return The viewport point in pixels, or @c nullopt when the direction is at or behind the eye
	///           plane, or under an orthographic projection, where a direction alone does not determine a
	///           screen position.
	///----------------------------------------
	
	[[nodiscard]] std::optional<Double2> screenPointForWorldDirection(const Double3 &direction) const noexcept;
	
	///----------------------------------------
	///   @brief Where a world-space position lands on the viewport.
	///   @return The viewport point in pixels, or @c nullopt when a perspective projection puts the position
	///           at or behind the eye plane, where it has no projection. An orthographic projection always
	///           has one, at any depth.
	/// @note    The result is not clamped to the viewport; a visible-on-screen test is the caller's, and
	///          @ref unobstructedFrustum is usually the better one.
	///----------------------------------------
	
	[[nodiscard]] std::optional<Double2> screenPointForWorldPosition(const Double3 &position) const noexcept;
	
	///@}
	
private:
	[[nodiscard]] static Double3 divideByW(const Double4 &vector) noexcept;
	[[nodiscard]] Double3 viewPointForScreenPoint(const Double2 &screenPoint, double normalizedDepth) const noexcept;
	[[nodiscard]] std::optional<Double2> screenPointForViewPoint(const Double3 &viewPoint) const noexcept;
	
	CameraSettings _settings;
	bool _isValid = false;
	
	Double4x4 _unadjustedViewTransform = Double4x4::identity();
	Double4x4 _viewTransform = Double4x4::identity();
	Double4x4 _inverseViewTransform = Double4x4::identity();
	Double4x4 _projectionTransform = Double4x4::identity();
	Double4x4 _inverseProjectionTransform = Double4x4::identity();
	Double4x4 _viewportTransform = Double4x4::identity();
	Double4x4 _inverseViewportTransform = Double4x4::identity();
	
	Double3 _eyePosition{0, 0, 0};
	Double3 _rightAxis{1, 0, 0};
	Double3 _upAxis{0, 1, 0};
	Double3 _forwardAxis{0, 0, 1};
	
	Frustum3d _frustum;
	Frustum3d _unobstructedFrustum;
	std::optional<OrthographicVolume> _orthographicVolume;
	FrustumTangents _resolvedTangents;
	
	double _aspectRatio = 1;
	double _horizontalFieldOfViewRadians = 0;
	double _verticalFieldOfViewRadians = 0;
	bool _viewportIsWide = true;
};

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief The inverse of a view transform: the cheap rigid inverse when it is one, the general inverse otherwise.
/// @details A camera's world→view transform is rigid unless @c riderTransform makes it something else,
///          so the fast path covers every ordinary case without making the odd one wrong.
///----------------------------------------

[[nodiscard]] inline Double4x4 inverseViewTransformOf(const Double4x4 &transform) noexcept {
	return isOrtho(transform) ? orthoInverse(transform) : inverse(transform);
}

///----------------------------------------
///   @brief An orthographic projection mapping a view-space box onto the @c [-1,1] clip cube.
/// @details Built here rather than through @ref Math::orthographicMatrix because this one follows the
///          camera's depth convention — @p nearDepth to -1 and @p farDepth to +1, matching
///          @ref Math::frustumMatrix — so the two projection modes agree about which end of clip space is
///          close, and so the matrix inverts consistently for picking.
///   @param left,top,right,bottom The view-space extent, in NDC orientation: @p top above @p bottom, so @p top > @p bottom.
///----------------------------------------

///----------------------------------------
///   @brief Remaps clip-space depth from the @c [-1,1] convention the projection builders produce onto
///          another one.
/// @details Applied on the clip-space side, where the mapping is linear in @c z and @c w and so is just
///          another matrix — the projection builders never learn about the convention, and the inverse
///          the camera already takes stays a true inverse. The 0.5 factors are exact in binary floating
///          point, so composing rather than building each variant directly costs nothing: under
///          @ref Math::DepthConvention::reversedZeroToOne the infinite projection collapses to exactly
///          @c z_clip = nearDepth, @c w_clip = z_view.
///----------------------------------------

[[nodiscard]] inline Double4x4 depthConventionMatrix(DepthConvention convention) noexcept {
	switch (convention) {
		case DepthConvention::negativeOneToOne:
			return Double4x4::identity();
			
		// z' = (z + w) / 2
		case DepthConvention::zeroToOne:
			return Double4x4(Double4(1, 0, 0, 0), Double4(0, 1, 0, 0), Double4(0, 0, 0.5, 0), Double4(0, 0, 0.5, 1));
			
		// z' = (w - z) / 2
		case DepthConvention::reversedZeroToOne:
			return Double4x4(Double4(1, 0, 0, 0), Double4(0, 1, 0, 0), Double4(0, 0, -0.5, 0), Double4(0, 0, 0.5, 1));
	}
	return Double4x4::identity();
}

///----------------------------------------

[[nodiscard]] inline Double4x4 cameraOrthographicMatrix(double left, double top, double right, double bottom,
                                                        double nearDepth, double farDepth) noexcept {
	const double rangeX = right - left;
	const double rangeY = top - bottom;
	const double rangeZ = farDepth - nearDepth;
	return Double4x4(
		Double4(2 / rangeX, 0, 0, 0),
		Double4(0, 2 / rangeY, 0, 0),
		Double4(0, 0, 2 / rangeZ, 0),
		Double4(-(left + right) / rangeX, -(top + bottom) / rangeY, -(nearDepth + farDepth) / rangeZ, 1));
}
	
} // namespace detail

///----------------------------------------
/// @brief Derives every transform, frustum and angular scale from @p settings.
///----------------------------------------

inline CameraView::CameraView(const CameraSettings &settings) noexcept : _settings(settings) {
	// The view transform is well defined regardless of the framing, so build it before deciding whether
	// anything is visible — a caller with a collapsed viewport still gets a usable orientation.
	Double3 orientationRight, orientationUp, orientationForward;
	_settings.orientation.getLocalAxes(orientationRight, orientationUp, orientationForward);
	_unadjustedViewTransform = localizationMatrix(orientationRight, orientationUp, orientationForward, _settings.position);
	_viewTransform = _settings.riderTransform * _unadjustedViewTransform;
	_inverseViewTransform = detail::inverseViewTransformOf(_viewTransform);
	
	// The eye sits wherever the rider carries it, which is only the platform's position when there is no
	// rider. Taking it from the composed transform rather than from settings.position is what keeps the
	// frustum apex, the pick rays and the world axes on the eye instead of on the platform.
	_eyePosition = translation(_inverseViewTransform);
	
	// Likewise the world-space axes are the columns of the composed view→world transform rather than the
	// platform orientation's own.
	const auto axisFromColumn = [](const Double4x4 &matrix, int index) {
		const Double4 column = matrix.column(index);
		return normalize(Double3(column.x, column.y, column.z));
	};
	_rightAxis = axisFromColumn(_inverseViewTransform, 0);
	_upAxis = axisFromColumn(_inverseViewTransform, 1);
	_forwardAxis = axisFromColumn(_inverseViewTransform, 2);
	
	const double viewportWidth = _settings.viewport.width();
	const double viewportHeight = _settings.viewport.height();
	const bool perspective = _settings.projection != Projection::orthographic;
	const bool hasExplicitTangents = _settings.frustumTangents.has_value();
	
	// Everything below divides by one of these. A collapsed viewport, an inverted depth range or a
	// non-positive near plane leaves nothing to see, and propagating the resulting infinities into the
	// transforms would hand the caller geometry that silently poisons whatever it touches.
	if (!(viewportWidth > 0) || !(viewportHeight > 0)
	 || !(_settings.farDepth > _settings.nearDepth)
	 || (perspective && !(_settings.nearDepth > 0))
	 || (!perspective && !(_settings.orthographicHeight > 0))) {
		return;
	}
	
	// Tangents are angles from an apex, which a parallel projection has not got. Rather than quietly
	// reinterpret them as something else, refuse the combination.
	if (hasExplicitTangents && (!perspective || !_settings.frustumTangents->isValid())) {
		return;
	}
	
	_aspectRatio = viewportWidth / viewportHeight;
	_viewportIsWide = viewportWidth >= viewportHeight;
	
	// The view-space extent the framing selects, measured on the near plane under a perspective projection
	// and on the projection plane under an orthographic one — NDC-oriented, so top is above bottom. Both
	// are linear in the same way, which is what lets the projection, frustum and margin arithmetic below be
	// shared between every framing and projection mode.
	double extentLeft = 0, extentRight = 0, extentTop = 0, extentBottom = 0;
	double ndcOffsetX = 0, ndcOffsetY = 0;
	
	if (hasExplicitTangents) {
		// An explicit frustum already says exactly what is visible, so there is nothing left for the clip
		// rectangle or the viewport offset to contribute — they are another way of writing the same four
		// numbers, and applying them on top would shift a frustum the caller had already placed.
		const FrustumTangents &tangents = *_settings.frustumTangents;
		extentLeft = -tangents.left * _settings.nearDepth;
		extentRight = tangents.right * _settings.nearDepth;
		extentTop = tangents.top * _settings.nearDepth;
		extentBottom = -tangents.bottom * _settings.nearDepth;
	} else {
		const double clipWidth = _settings.clipBounds.width();
		const double clipHeight = _settings.clipBounds.height();
		if (!(clipWidth > 0) || !(clipHeight > 0)) {
			return;
		}
		
		// The clip rectangle and the viewport offset are given in the viewport's orientation, where y grows
		// downward; NDC has y growing upward. Flip both once, here.
		const double clipLeft = _settings.clipBounds._left;
		const double clipRight = _settings.clipBounds._right;
		const double clipTop = -_settings.clipBounds._top;
		const double clipBottom = -_settings.clipBounds._bottom;
		ndcOffsetX = _settings.viewportOffset.x;
		ndcOffsetY = -_settings.viewportOffset.y;
		
		// Resolve the narrow-axis field of view onto the screen axes. Tracking the narrow edge rather than a
		// fixed one is what keeps the framing stable as the device rotates.
		const double narrowToWideRatio = _viewportIsWide ? _aspectRatio : 1 / _aspectRatio;
		const double narrowHalfAngleTangent = std::tan(_settings.narrowAxisFieldOfViewRadians / 2);
		const double wideFieldOfViewRadians = 2 * std::atan(narrowHalfAngleTangent * narrowToWideRatio);
		const double nominalVerticalFieldOfView = _viewportIsWide ? _settings.narrowAxisFieldOfViewRadians : wideFieldOfViewRadians;
		const double verticalHalfAngleTangent = std::tan(nominalVerticalFieldOfView / 2);
		
		const double horizontalUnitScale = perspective ? verticalHalfAngleTangent * _settings.nearDepth * _aspectRatio
		                                               : _settings.orthographicHeight * _aspectRatio / 2;
		const double verticalUnitScale = perspective ? verticalHalfAngleTangent * _settings.nearDepth
		                                             : _settings.orthographicHeight / 2;
		extentLeft = clipLeft * horizontalUnitScale;
		extentRight = clipRight * horizontalUnitScale;
		extentTop = clipTop * verticalUnitScale;
		extentBottom = clipBottom * verticalUnitScale;
	}
	
	Double4x4 projectionCore = Double4x4::identity();
	switch (_settings.projection) {
		case Projection::perspective:
			projectionCore = frustumMatrix(extentLeft, extentTop, extentRight, extentBottom, _settings.nearDepth, _settings.farDepth);
			break;
			
		case Projection::infinitePerspective:
			// Retarget the finite frustum's depth rather than building a symmetric infinite projection and
			// bolting the clip rectangle on afterwards: this keeps the off-axis terms in place, so the
			// matrix and its inverse describe the same volume.
			projectionCore = withInfinitePerspectiveNearZ(
				frustumMatrix(extentLeft, extentTop, extentRight, extentBottom, _settings.nearDepth, _settings.farDepth),
				_settings.nearDepth);
			break;
			
		case Projection::orthographic:
			projectionCore = detail::cameraOrthographicMatrix(extentLeft, extentTop, extentRight, extentBottom,
				_settings.nearDepth, _settings.farDepth);
			_orthographicVolume = OrthographicVolume{
				.bounds = rect_double(extentLeft, -extentTop, extentRight, -extentBottom),
				.nearDepth = _settings.nearDepth,
				.farDepth = _settings.farDepth};
			break;
	}
	
	// Mirroring and the viewport offset are screen-space adjustments, so they belong on the NDC side of the
	// projection and nowhere else. Applying the offset here and again in the viewport transform — which is
	// the obvious-looking thing to do — shifts picked points twice as far as drawn ones.
	// The depth convention touches only z and w, the mirror and offset only x and y, so the three commute
	// and the order below is for reading rather than correctness.
	const double mirrorX = _settings.mirrorHorizontally ? -1.0 : 1.0;
	const double mirrorY = _settings.mirrorVertically ? -1.0 : 1.0;
	_projectionTransform = detail::depthConventionMatrix(_settings.depthConvention)
	                     * translationMatrix(Double3(ndcOffsetX, ndcOffsetY, 0))
	                     * scaleMatrix(mirrorX, mirrorY, 1.0)
	                     * projectionCore;
	_inverseProjectionTransform = inverse(_projectionTransform);
	
	_viewportTransform = viewportMatrix(_settings.viewport._left, _settings.viewport._top,
	                                    _settings.viewport._right, _settings.viewport._bottom);
	_inverseViewportTransform = inverse(_viewportTransform);
	
	_isValid = true;
	
	// A parallel projection has no apex, so there is no frustum to build; orthographicVolume carries the
	// culling volume instead.
	if (!perspective) {
		return;
	}
	
	// The visible near-plane rectangle, after the offset moves the image within the viewport. Mirroring
	// reverses which side of the axis the offset opens up, so fold it in rather than applying it twice.
	const double effectiveOffsetX = mirrorX * ndcOffsetX;
	const double effectiveOffsetY = mirrorY * ndcOffsetY;
	const double centerX = (extentLeft + extentRight) / 2;
	const double centerY = (extentTop + extentBottom) / 2;
	const double halfSpanX = (extentRight - extentLeft) / 2;
	const double halfSpanY = (extentTop - extentBottom) / 2;
	double visibleLeft = centerX - (1 + effectiveOffsetX) * halfSpanX;
	double visibleRight = centerX + (1 - effectiveOffsetX) * halfSpanX;
	double visibleBottom = centerY - (1 + effectiveOffsetY) * halfSpanY;
	double visibleTop = centerY + (1 - effectiveOffsetY) * halfSpanY;
	
	// Report the field of view the resolved frustum actually spans, not the one that was asked for. The two
	// differ whenever a clip rectangle, a viewport offset or an explicit frustum is in play, and it is the
	// resolved one that label sizing and level-of-detail thresholds want.
	_resolvedTangents = FrustumTangents{
		.left = -visibleLeft / _settings.nearDepth,
		.right = visibleRight / _settings.nearDepth,
		.top = visibleTop / _settings.nearDepth,
		.bottom = -visibleBottom / _settings.nearDepth};
	_horizontalFieldOfViewRadians = _resolvedTangents.horizontalFieldOfViewRadians();
	_verticalFieldOfViewRadians = _resolvedTangents.verticalFieldOfViewRadians();
	
	// Frustum::perspective reads the view axes from the first three columns of an orientation matrix.
	const Double4x4 frustumOrientation(
		Double4(_rightAxis.x, _rightAxis.y, _rightAxis.z, 0),
		Double4(_upAxis.x, _upAxis.y, _upAxis.z, 0),
		Double4(_forwardAxis.x, _forwardAxis.y, _forwardAxis.z, 0),
		Double4(0, 0, 0, 1));
		
	// Half-angles are signed, measured from the forward axis outward. An off-centre clip rectangle can put
	// both edges of the image on the same side of the axis, and a negative half-angle expresses that.
	const auto perspectiveFrustum = [&](double left, double top, double right, double bottom) {
		return Frustum3d::perspective(_eyePosition, frustumOrientation,
			std::atan2(-left, _settings.nearDepth), std::atan2(right, _settings.nearDepth),
			std::atan2(top, _settings.nearDepth), std::atan2(-bottom, _settings.nearDepth));
	};
	
	_frustum = perspectiveFrustum(visibleLeft, visibleTop, visibleRight, visibleBottom);
	
	// The margins are given in pixels; the visible rectangle is in near-plane units. One scale factor
	// converts, and it stays correct however far off centre the rectangle sits.
	const EdgeInsets &margins = _settings.obstructionMargins;
	if (margins.isEmpty()) {
		_unobstructedFrustum = _frustum;
		return;
	}
	
	const double nearPlaneUnitsPerPixelX = (visibleRight - visibleLeft) / viewportWidth;
	const double nearPlaneUnitsPerPixelY = (visibleTop - visibleBottom) / viewportHeight;
	visibleLeft += margins.left * nearPlaneUnitsPerPixelX;
	visibleRight -= margins.right * nearPlaneUnitsPerPixelX;
	visibleTop -= margins.top * nearPlaneUnitsPerPixelY;
	visibleBottom += margins.bottom * nearPlaneUnitsPerPixelY;
	
	// Margins that meet leave nothing visible. An invalid frustum says so; a zero-angle one would still
	// report the forward axis as inside.
	if (visibleRight > visibleLeft && visibleTop > visibleBottom) {
		_unobstructedFrustum = perspectiveFrustum(visibleLeft, visibleTop, visibleRight, visibleBottom);
	}
}

///----------------------------------------

inline double CameraView::fieldOfViewRadians(CameraAxis axis) const noexcept {
	if (!_isValid || _settings.projection == Projection::orthographic) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	switch (axis) {
		case CameraAxis::narrow:     return _viewportIsWide ? _verticalFieldOfViewRadians : _horizontalFieldOfViewRadians;
		case CameraAxis::wide:       return _viewportIsWide ? _horizontalFieldOfViewRadians : _verticalFieldOfViewRadians;
		case CameraAxis::horizontal: return _horizontalFieldOfViewRadians;
		case CameraAxis::vertical:   return _verticalFieldOfViewRadians;
	}
	return std::numeric_limits<double>::quiet_NaN();
}

///----------------------------------------

inline double CameraView::pixelsPerRadian(CameraAxis axis) const noexcept {
	const double fieldOfView = fieldOfViewRadians(axis);
	if (!std::isfinite(fieldOfView) || fieldOfView <= 0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const double viewportWidth = _settings.viewport.width();
	const double viewportHeight = _settings.viewport.height();
	switch (axis) {
		case CameraAxis::narrow:     return std::min(viewportWidth, viewportHeight) / fieldOfView;
		case CameraAxis::wide:       return std::max(viewportWidth, viewportHeight) / fieldOfView;
		case CameraAxis::horizontal: return viewportWidth / fieldOfView;
		case CameraAxis::vertical:   return viewportHeight / fieldOfView;
	}
	return std::numeric_limits<double>::quiet_NaN();
}

///----------------------------------------

inline double CameraView::angularResolutionRadians() const noexcept {
	const double scale = pixelsPerRadian(CameraAxis::narrow);
	return std::isfinite(scale) ? 0.5 / scale : std::numeric_limits<double>::quiet_NaN();
}

///----------------------------------------

inline Double3 CameraView::divideByW(const Double4 &vector) noexcept {
	if (vector.w == 0 || !std::isfinite(vector.w)) {
		return Double3(vector.x, vector.y, vector.z);
	}
	const double inverseW = 1 / vector.w;
	return Double3(vector.x * inverseW, vector.y * inverseW, vector.z * inverseW);
}

///----------------------------------------
/// @brief The view-space point a viewport point unprojects to at the given NDC depth (-1 near, +1 far).
///----------------------------------------

inline Double3 CameraView::viewPointForScreenPoint(const Double2 &screenPoint, double normalizedDepth) const noexcept {
	const Double3 normalizedPoint = transformPoint(_inverseViewportTransform, Double3(screenPoint.x, screenPoint.y, normalizedDepth));
	return divideByW(_inverseProjectionTransform * Double4(normalizedPoint.x, normalizedPoint.y, normalizedPoint.z, 1));
}

///----------------------------------------

inline std::optional<Double2> CameraView::screenPointForViewPoint(const Double3 &viewPoint) const noexcept {
	if (!_isValid) {
		return std::nullopt;
	}
	// A perspective projection has no image of anything at or behind the eye plane; the divide below would
	// produce a plausible-looking point mirrored through the origin instead of reporting that.
	if (_settings.projection != Projection::orthographic && !(viewPoint.z > 0)) {
		return std::nullopt;
	}
	const Double3 normalizedPoint = divideByW(_projectionTransform * Double4(viewPoint.x, viewPoint.y, viewPoint.z, 1));
	const Double3 screenPoint = transformPoint(_viewportTransform, normalizedPoint);
	return Double2(screenPoint.x, screenPoint.y);
}

///----------------------------------------

inline Ray3d CameraView::rayThroughScreenPoint(const Double2 &screenPoint) const noexcept {
	if (!_isValid) {
		return Ray3d(_eyePosition, _forwardAxis);
	}
	const Double3 nearPoint = viewPointForScreenPoint(screenPoint, nearPlaneDepth());
	if (_settings.projection == Projection::orthographic) {
		// Parallel rays: the screen point chooses where the ray starts, not which way it points.
		return Ray3d(transformPoint(_inverseViewTransform, nearPoint), _forwardAxis);
	}
	// The projection places the eye at the view-space origin, so the near-plane point is the direction.
	return Ray3d(_eyePosition, normalize(transformDirection(_inverseViewTransform, nearPoint)));
}

///----------------------------------------

inline Double3 CameraView::worldDirectionThroughScreenPoint(const Double2 &screenPoint) const noexcept {
	return rayThroughScreenPoint(screenPoint).direction;
}

///----------------------------------------

inline Double3 CameraView::viewDirectionThroughScreenPoint(const Double2 &screenPoint) const noexcept {
	if (!_isValid || _settings.projection == Projection::orthographic) {
		return Double3(0, 0, 1);
	}
	return normalize(viewPointForScreenPoint(screenPoint, nearPlaneDepth()));
}

///----------------------------------------

inline std::optional<Double2> CameraView::screenPointForWorldDirection(const Double3 &direction) const noexcept {
	// Under a parallel projection a direction carries no screen position — every point along it projects
	// somewhere different, and there is no eye to anchor it to.
	if (_settings.projection == Projection::orthographic) {
		return std::nullopt;
	}
	return screenPointForViewPoint(transformDirection(_viewTransform, direction));
}

///----------------------------------------

inline std::optional<Double2> CameraView::screenPointForWorldPosition(const Double3 &position) const noexcept {
	return screenPointForViewPoint(transformPoint(_viewTransform, position));
}

///----------------------------------------
/// @class Camera
/// @brief A thread-safe holder for a @ref Math::CameraSettings and the @ref Math::CameraView derived from it.
/// @details Readers call @ref view for a snapshot whose lifetime is independent of any later edit, so a
///          render pass keeps a coherent viewpoint while another thread moves the camera. The derived view
///          is computed on the first @ref view after a change and shared by every reader until the next one.
/// @note    Edits are expected to come from one thread at a time. Concurrent edits are memory-safe but the
///          later publish wins outright, so an interleaved read-modify-write can lose the other's change.
///----------------------------------------

class Camera final {
///----------------------------------------
public:
	Camera() : Camera(CameraSettings{}) {}
	
	explicit Camera(const CameraSettings &settings)
		: _settings(std::make_shared<const CameraSettings>(settings)) {}
		
	Camera(const Camera &other) {
		const auto state = other.state();
		_settings = state.first;
		_view = state.second;
	}
	
	Camera &operator=(const Camera &other) {
		if (this != &other) {
			const auto state = other.state();
			const std::lock_guard lock(_mutex);
			_settings = state.first;
			_view = state.second;
		}
		return *this;
	}
	
	///----------------------------------------
	/// @brief The current settings.
	///----------------------------------------
	
	[[nodiscard]] CameraSettings settings() const {
		const std::lock_guard lock(_mutex);
		return *_settings;
	}
	
	///----------------------------------------
	///   @brief Replaces the settings.
	/// @details Publishing identical settings is a no-op, so the cached view — and its identity, which
	///          callers use to detect change — survives a redundant write.
	///----------------------------------------
	
	void setSettings(const CameraSettings &settings) {
		const std::lock_guard lock(_mutex);
		if (*_settings == settings) {
			return;
		}
		_settings = std::make_shared<const CameraSettings>(settings);
		_view.reset();
	}
	
	///----------------------------------------
	///   @brief Applies @p edit to a copy of the settings and publishes the result.
	/// @details The point of batching: however many fields the edit touches, it costs one publish and one
	///          derivation, and no reader ever observes a half-applied change.
	///   @param edit A callable taking @c CameraSettings& — for example
	///          @code camera.edit([](Math::CameraSettings &settings) { settings.position = origin; settings.nearDepth = 0.1; }); @endcode
	/// @note    @p edit runs without the camera's lock held, so it may call back into the camera. It sees a
	///          snapshot taken when it started.
	///----------------------------------------
	
	template <std::invocable<CameraSettings &> TEdit>
	void edit(TEdit &&editSettings) {
		CameraSettings settings = this->settings();
		std::forward<TEdit>(editSettings)(settings);
		setSettings(settings);
	}
	
	///----------------------------------------
	///   @brief A snapshot of everything derived from the current settings.
	/// @details Never null. The returned view is immutable and outlives any later edit, so a caller may
	///          hold it for as long as it needs one consistent viewpoint. Two calls with no edit between
	///          them return the same object, which makes pointer identity a change test.
	///----------------------------------------
	
	[[nodiscard]] std::shared_ptr<const CameraView> view() const {
		const std::lock_guard lock(_mutex);
		if (!_view) {
			_view = std::make_shared<const CameraView>(*_settings);
		}
		return _view;
	}
	
private:
	[[nodiscard]] std::pair<std::shared_ptr<const CameraSettings>, std::shared_ptr<const CameraView>> state() const {
		const std::lock_guard lock(_mutex);
		return {_settings, _view};
	}
	
	mutable std::mutex _mutex;
	std::shared_ptr<const CameraSettings> _settings;
	mutable std::shared_ptr<const CameraView> _view;
};

///----------------------------------------
/// @brief Verifies the camera's conventions, invertibility, culling volumes and projection edge cases.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------

inline void cameraSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs, double tolerance = 1.0e-9) noexcept { return std::abs(lhs - rhs) < tolerance; };
	
	// The clip-space depth of a view-space point, with the perspective divide the transforms deliberately
	// leave to the caller.
	const auto normalizedDepthOf = [](const Double4x4 &projection, double viewDepth) {
		const Double4 clip = projection * Double4(0, 0, viewDepth, 1);
		return clip.z / clip.w;
	};
	
	const auto landscapeSettings = [] {
		CameraSettings settings;
		settings.viewport = rect_double(0, 0, 800, 400);
		settings.narrowAxisFieldOfViewRadians = 60 * deg2rad;
		settings.nearDepth = 1;
		settings.farDepth = 1000;
		return settings;
	};
	
	// The view transform carries the viewpoint to the view-space origin and aims the forward axis down +Z.
	{
		CameraSettings settings = landscapeSettings();
		settings.position = Double3(10, -4, 7);
		const CameraView view(settings);
		check(view.isValid(), "a well-formed camera is valid");
		const Double3 origin = transformPoint(view.viewTransform(), settings.position);
		check(near(origin.x, 0) && near(origin.y, 0) && near(origin.z, 0), "the viewpoint maps to the view-space origin");
		check(near(view.eyePosition().x, 10) && near(view.eyePosition().y, -4) && near(view.eyePosition().z, 7), "the eye is at the viewpoint");
		check(near(view.forwardAxis().z, 1), "the default forward axis is +Z");
		check(approxEqual(view.inverseViewTransform() * view.viewTransform(), Double4x4::identity(), 1.0e-9), "the view transform inverts");
	}
	
	// The rider carries the eye away from the platform. A rotation only changes the aim; a rotation about a
	// pivot ahead swings the eye onto an arc, which is what produces parallax rather than a bare change of
	// aim — and the platform must not move either way, because that is what astronomy queries read.
	{
		constexpr double pivotDistance = 40;
		CameraSettings settings = landscapeSettings();
		settings.position = Double3(5, -2, 9);
		const CameraView unridden(settings);
		
		// A pure rotation leaves the eye on the platform and turns the view.
		settings.riderTransform = yawPitchRollMatrix<double>(0.2, 0.0, 0.0);
		const CameraView turned(settings);
		check(approxEqual(turned.eyePosition(), settings.position, 1.0e-9), "a rotating rider leaves the eye on the platform");
		check(!approxEqual(turned.forwardAxis(), unridden.forwardAxis(), 1.0e-6), "but it does turn the view");
		
		// A pivot orbit moves the eye onto an arc of the pivot's radius.
		const Double3 pivot(5, -2, 9 + pivotDistance);
		settings.riderTransform = translationMatrix(Double3(0, 0, pivotDistance))
		                        * yawPitchRollMatrix<double>(0.05, 0.0, 0.0)
		                        * translationMatrix(Double3(0, 0, -pivotDistance));
		const CameraView orbited(settings);
		check(!approxEqual(orbited.eyePosition(), settings.position, 1.0e-6), "a pivoting rider moves the eye off the platform");
		check(near(length(orbited.eyePosition() - pivot), pivotDistance, 1.0e-9), "and keeps it on the pivot's arc");
		
		// The platform is untouched by any of it — that is the whole point of keeping the two apart.
		check(approxEqual(orbited.unadjustedViewTransform(), unridden.unadjustedViewTransform(), 1.0e-12), "the rider never moves the platform");
		check(approxEqual(turned.unadjustedViewTransform(), unridden.unadjustedViewTransform(), 1.0e-12), "however it turns");
		
		// The frustum apex and the pick rays follow the eye, not the platform. Taking either from
		// settings.position would leave culling and picking behind wherever the rider went.
		check(approxEqual(orbited.frustum().apex(), orbited.eyePosition(), 1.0e-12), "the frustum apex is at the eye");
		check(approxEqual(orbited.rayThroughScreenPoint(Double2(400, 200)).origin, orbited.eyePosition(), 1.0e-12), "and so is the pick ray origin");
		
		// The parallax that pivoting buys: what sits at the pivot stays put on screen, what lies beyond it
		// does not. A rider that only turned the view would move both together.
		const auto atPivotBefore = unridden.screenPointForWorldPosition(pivot);
		const auto atPivotAfter = orbited.screenPointForWorldPosition(pivot);
		check(atPivotBefore.has_value() && atPivotAfter.has_value(), "the pivot projects either way");
		check(near(atPivotAfter->x, atPivotBefore->x, 1.0e-6) && near(atPivotAfter->y, atPivotBefore->y, 1.0e-6), "content at the pivot does not move");
		
		const Double3 beyond(5, -2, 9 + 100 * pivotDistance);
		const auto beyondBefore = unridden.screenPointForWorldPosition(beyond);
		const auto beyondAfter = orbited.screenPointForWorldPosition(beyond);
		check(beyondBefore.has_value() && beyondAfter.has_value(), "the distant point projects either way");
		check(std::abs(beyondAfter->x - beyondBefore->x) > 1.0, "content beyond the pivot does");
	}
	
	// The narrow axis follows the shorter viewport edge, so the framing holds as the device rotates. A
	// fixed vertical field of view would make the same scene wider in landscape than in portrait.
	{
		CameraSettings settings = landscapeSettings();
		const CameraView landscape(settings);
		check(near(landscape.fieldOfViewRadians(CameraAxis::narrow), 60 * deg2rad), "landscape narrow axis is the requested angle");
		check(near(landscape.fieldOfViewRadians(CameraAxis::vertical), 60 * deg2rad), "landscape narrow axis is the vertical one");
		check(landscape.fieldOfViewRadians(CameraAxis::horizontal) > landscape.fieldOfViewRadians(CameraAxis::vertical), "landscape is wider than tall");
		
		settings.viewport = rect_double(0, 0, 400, 800);
		const CameraView portrait(settings);
		check(near(portrait.fieldOfViewRadians(CameraAxis::narrow), 60 * deg2rad), "portrait narrow axis is the requested angle");
		check(near(portrait.fieldOfViewRadians(CameraAxis::horizontal), 60 * deg2rad), "portrait narrow axis is the horizontal one");
		check(near(landscape.fieldOfViewRadians(CameraAxis::wide), portrait.fieldOfViewRadians(CameraAxis::wide)), "rotating the device preserves both angles");
	}
	
	// Picking round-trips. Every projection mode has to invert, and the infinite one is the trap: pairing
	// an infinite forward matrix with a finite frustum's inverse looks right until a picked point is
	// compared against a drawn one.
	for (const Projection projection : {Projection::perspective, Projection::infinitePerspective, Projection::orthographic}) {
		CameraSettings settings = landscapeSettings();
		settings.projection = projection;
		settings.position = Double3(3, 2, -5);
		settings.orientation.setRotationRad(0.4, 0.2, 0.1);
		settings.orthographicHeight = 6;
		const CameraView view(settings);
		check(view.isValid(), "every projection mode produces a valid view");
		check(approxEqual(view.inverseProjectionTransform() * view.projectionTransform(), Double4x4::identity(), 1.0e-9), "the projection inverts");
		check(approxEqual(view.inverseViewportTransform() * view.viewportTransform(), Double4x4::identity(), 1.0e-9), "the viewport transform inverts");
		
		for (const Double2 screenPoint : {Double2(400, 200), Double2(0, 0), Double2(800, 400), Double2(123, 345)}) {
			const Ray3d ray = view.rayThroughScreenPoint(screenPoint);
			check(near(length(ray.direction), 1), "the pick ray direction is unit length");
			const auto recovered = view.screenPointForWorldPosition(ray.pointAt(50));
			check(recovered.has_value(), "a point along the pick ray projects");
			check(near(recovered->x, screenPoint.x, 1.0e-6) && near(recovered->y, screenPoint.y, 1.0e-6), "screen point round-trips through the pick ray");
		}
	}
	
	// The viewport centre looks along the forward axis, and the projection agrees with the frustum about
	// what is in front of the camera.
	{
		const CameraView view(landscapeSettings());
		const Double3 forward = view.worldDirectionThroughScreenPoint(Double2(400, 200));
		check(near(forward.x, 0, 1.0e-9) && near(forward.y, 0, 1.0e-9) && near(forward.z, 1, 1.0e-9), "the viewport centre looks forward");
		check(view.frustum().containsPoint(Double3(0, 0, 10)), "a point straight ahead is inside the frustum");
		check(!view.frustum().containsPoint(Double3(0, 0, -10)), "a point behind the camera is not");
		check(!view.screenPointForWorldDirection(Double3(0, 0, -1)).has_value(), "a direction behind the camera has no screen point");
		check(!view.screenPointForWorldPosition(Double3(0, 0, -10)).has_value(), "a position behind the camera has no screen point");
	}
	
	// The frustum and the projection have to agree about the edges of the image, or culling drops things
	// that would have been drawn.
	{
		const CameraView view(landscapeSettings());
		for (const Double2 corner : {Double2(1, 1), Double2(799, 1), Double2(1, 399), Double2(799, 399)}) {
			check(view.frustum().containsPoint(view.rayThroughScreenPoint(corner).pointAt(100)), "a point just inside a viewport corner is inside the frustum");
		}
		for (const Double2 corner : {Double2(-4, -4), Double2(804, -4), Double2(-4, 404), Double2(804, 404)}) {
			check(!view.frustum().containsPoint(view.rayThroughScreenPoint(corner).pointAt(100)), "a point just outside a viewport corner is outside the frustum");
		}
	}
	
	// Obstruction margins narrow the unobstructed frustum without touching the projection, and the result
	// nests inside the full one.
	{
		CameraSettings settings = landscapeSettings();
		const CameraView unmargined(settings);
		settings.obstructionMargins = EdgeInsets{.left = 0, .top = 100, .right = 0, .bottom = 0};
		const CameraView view(settings);
		check(approxEqual(view.projectionTransform(), unmargined.projectionTransform(), 1.0e-12), "margins do not change the projection");
		check(view.unobstructedFrustum().isValid(), "a partial margin leaves a usable frustum");
		
		// A point just below the covered strip stays visible; one just inside it does not.
		const Double3 belowMargin = view.rayThroughScreenPoint(Double2(400, 110)).pointAt(100);
		const Double3 withinMargin = view.rayThroughScreenPoint(Double2(400, 90)).pointAt(100);
		check(view.frustum().containsPoint(belowMargin) && view.frustum().containsPoint(withinMargin), "the full frustum holds both points");
		check(view.unobstructedFrustum().containsPoint(belowMargin), "a point below the margin is unobstructed");
		check(!view.unobstructedFrustum().containsPoint(withinMargin), "a point behind the margin is obstructed");
		
		settings.obstructionMargins = EdgeInsets{.left = 500, .top = 0, .right = 500, .bottom = 0};
		check(!CameraView(settings).unobstructedFrustum().isValid(), "margins that meet leave no unobstructed frustum");
	}
	
	// A parallel projection does not fall off with distance, and reports no frustum or field of view
	// rather than a plausible-looking wrong one.
	{
		CameraSettings settings = landscapeSettings();
		settings.projection = Projection::orthographic;
		settings.orthographicHeight = 10;
		const CameraView view(settings);
		const auto nearPoint = view.screenPointForWorldPosition(Double3(2, 1, 5));
		const auto farPoint = view.screenPointForWorldPosition(Double3(2, 1, 500));
		check(nearPoint.has_value() && farPoint.has_value(), "an orthographic projection reaches any depth");
		check(near(nearPoint->x, farPoint->x) && near(nearPoint->y, farPoint->y), "depth does not change orthographic screen position");
		check(std::isnan(view.fieldOfViewRadians(CameraAxis::narrow)), "an orthographic view reports no field of view");
		check(!view.frustum().isValid(), "an orthographic view has no frustum");
		check(view.orthographicVolume().has_value(), "an orthographic view reports its volume");
		check(near(view.orthographicVolume()->bounds.height(), 10), "the orthographic volume spans the requested height");
		check(near(view.orthographicVolume()->bounds.width(), 20), "the orthographic width follows the aspect ratio");
		
		// The depth mapping has to match the perspective modes', or a scene that switches projection
		// reverses its depth sort. A self-consistent but inverted mapping still round-trips through
		// picking, so this has to be asserted directly.
		check(near(normalizedDepthOf(view.projectionTransform(), settings.nearDepth), view.nearPlaneDepth()), "the orthographic near plane maps to the convention's near depth");
		check(near(normalizedDepthOf(view.projectionTransform(), settings.farDepth), view.farPlaneDepth()), "the orthographic far plane maps to the convention's far depth");
		
		// Parallel rays: two screen points give the same direction and different origins.
		const Ray3d leftRay = view.rayThroughScreenPoint(Double2(100, 200));
		const Ray3d rightRay = view.rayThroughScreenPoint(Double2(700, 200));
		check(near(leftRay.direction.x, rightRay.direction.x) && near(leftRay.direction.z, rightRay.direction.z), "orthographic rays are parallel");
		check(!near(leftRay.origin.x, rightRay.origin.x), "orthographic rays start at different points");
		check(!view.screenPointForWorldDirection(Double3(0, 0, 1)).has_value(), "a direction has no orthographic screen point");
	}
	
	// The near and far planes land on the ends of the depth range, and the infinite projection keeps the
	// near plane while never running out of far.
	{
		CameraSettings settings = landscapeSettings();
		const CameraView finite(settings);
		check(near(normalizedDepthOf(finite.projectionTransform(), settings.nearDepth), -1), "the near plane maps to NDC -1");
		check(near(normalizedDepthOf(finite.projectionTransform(), settings.farDepth), 1), "the far plane maps to NDC +1");
		
		settings.projection = Projection::infinitePerspective;
		const CameraView infinite(settings);
		const double veryFarDepth = normalizedDepthOf(infinite.projectionTransform(), 1.0e12);
		check(near(normalizedDepthOf(infinite.projectionTransform(), settings.nearDepth), -1), "the infinite projection keeps the near plane at NDC -1");
		check(veryFarDepth < 1 && veryFarDepth > 0.999999, "the infinite projection approaches NDC +1 without reaching it");
	}
	
	// Every depth convention puts the planes where it says, in every projection mode, and changes nothing
	// but depth. Picking has to follow the convention too — it unprojects from the near plane, and a
	// hard-coded -1 there silently aims the pick ray from the wrong end of a reversed range.
	for (const DepthConvention convention : {DepthConvention::negativeOneToOne, DepthConvention::zeroToOne, DepthConvention::reversedZeroToOne}) {
		for (const Projection projection : {Projection::perspective, Projection::infinitePerspective, Projection::orthographic}) {
			CameraSettings settings = landscapeSettings();
			settings.projection = projection;
			settings.orthographicHeight = 6;
			settings.depthConvention = convention;
			const CameraView view(settings);
			
			check(near(view.nearPlaneDepth(), nearPlaneDepth(convention)), "the view reports its convention's near depth");
			check(near(normalizedDepthOf(view.projectionTransform(), settings.nearDepth), nearPlaneDepth(convention)), "the near plane lands where the convention says");
			if (projection == Projection::infinitePerspective) {
				// No far plane to land on — the far depth is a limit approached but never reached.
				const double veryFar = normalizedDepthOf(view.projectionTransform(), 1.0e12);
				check(std::abs(veryFar - farPlaneDepth(convention)) < 1.0e-6, "an infinite projection approaches the convention's far depth");
				check(veryFar != farPlaneDepth(convention), "an infinite projection never reaches its far depth");
			} else {
				check(near(normalizedDepthOf(view.projectionTransform(), settings.farDepth), farPlaneDepth(convention)), "the far plane lands where the convention says");
			}
			
			// Monotonic in the direction the convention implies, so a depth test comparing toward
			// farPlaneDepth sorts correctly.
			const double closer = normalizedDepthOf(view.projectionTransform(), 10);
			const double further = normalizedDepthOf(view.projectionTransform(), 100);
			check(convention == DepthConvention::reversedZeroToOne ? closer > further : closer < further,
				"depth grows away from the viewer, or shrinks when reversed");
				
			// The image is untouched: only the depth row differs from the baseline convention.
			settings.depthConvention = DepthConvention::negativeOneToOne;
			const CameraView baseline(settings);
			const auto projected = view.screenPointForWorldPosition(Double3(2, 1, 20));
			const auto baselineProjected = baseline.screenPointForWorldPosition(Double3(2, 1, 20));
			check(projected.has_value() && baselineProjected.has_value(), "the target projects under both conventions");
			check(near(projected->x, baselineProjected->x) && near(projected->y, baselineProjected->y), "the depth convention leaves the image alone");
			
			// Picking still round-trips, which is what pins the near-plane constant used to unproject.
			for (const Double2 screenPoint : {Double2(400, 200), Double2(60, 350)}) {
				const auto recovered = view.screenPointForWorldPosition(view.rayThroughScreenPoint(screenPoint).pointAt(40));
				check(recovered.has_value(), "a picked point projects back");
				check(near(recovered->x, screenPoint.x, 1.0e-6) && near(recovered->y, screenPoint.y, 1.0e-6), "picking round-trips under every depth convention");
			}
		}
	}
	
	// Reversed depth over an infinite far plane is the configuration worth having: the composed matrix
	// reduces to z_clip = nearDepth, w_clip = z_view, so the depth is exactly nearDepth/viewDepth — no
	// cancellation anywhere, no far plane to choose, and no precision cliff at any distance.
	{
		CameraSettings settings = landscapeSettings();
		settings.projection = Projection::infinitePerspective;
		settings.depthConvention = DepthConvention::reversedZeroToOne;
		settings.nearDepth = 0.25;
		const CameraView view(settings);
		for (const double viewDepth : {0.25, 1.0, 37.5, 1.0e6, 1.0e15}) {
			check(near(normalizedDepthOf(view.projectionTransform(), viewDepth), settings.nearDepth / viewDepth, 1.0e-15),
				"reversed infinite depth is exactly nearDepth / viewDepth");
		}
	}
	
	// Mirroring reflects the image about the viewport centre and leaves the other axis alone.
	{
		CameraSettings settings = landscapeSettings();
		const Double3 target(2, 1, 10);
		const auto plain = CameraView(settings).screenPointForWorldPosition(target);
		settings.mirrorHorizontally = true;
		const auto mirrored = CameraView(settings).screenPointForWorldPosition(target);
		check(plain.has_value() && mirrored.has_value(), "the target projects both ways");
		check(near(plain->x - 400, 400 - mirrored->x), "mirroring reflects x about the viewport centre");
		check(near(plain->y, mirrored->y), "mirroring x leaves y alone");
	}
	
	// The viewport offset has to move the drawn image and the picked point by the same amount. Applying it
	// to the projection and to the viewport transform both — the arrangement that looks symmetric — shifts
	// picks twice as far as draws, and the two disagree only when the offset is non-zero.
	{
		CameraSettings settings = landscapeSettings();
		settings.viewportOffset = Double2(0.25, -0.5);
		const CameraView view(settings);
		for (const Double2 screenPoint : {Double2(400, 200), Double2(120, 300)}) {
			const auto recovered = view.screenPointForWorldPosition(view.rayThroughScreenPoint(screenPoint).pointAt(40));
			check(recovered.has_value(), "an offset view still projects");
			check(near(recovered->x, screenPoint.x, 1.0e-6) && near(recovered->y, screenPoint.y, 1.0e-6), "an offset view picks where it draws");
		}
		// The offset moves the image, so the centre of the viewport no longer looks along the forward axis,
		// and the frustum has to follow it.
		check(view.frustum().containsPoint(view.rayThroughScreenPoint(Double2(700, 380)).pointAt(60)), "the offset frustum follows the image");
		check(!view.frustum().containsPoint(view.rayThroughScreenPoint(Double2(-40, 200)).pointAt(60)), "the offset frustum still excludes what is off screen");
	}
	
	// Clip bounds select a sub-rectangle of the view without changing the field of view the settings ask
	// for, and the frustum narrows with them.
	{
		CameraSettings settings = landscapeSettings();
		settings.clipBounds = rect_double(-1, -1, 0, 0);
		const CameraView view(settings);
		check(view.isValid(), "a clipped view is valid");
		const Double3 upperLeft = view.rayThroughScreenPoint(Double2(400, 200)).pointAt(50);
		check(view.frustum().containsPoint(upperLeft), "the clipped frustum holds what the clipped view draws");
		const auto recovered = view.screenPointForWorldPosition(upperLeft);
		check(recovered.has_value() && near(recovered->x, 400, 1.0e-6) && near(recovered->y, 200, 1.0e-6), "a clipped view picks where it draws");
		// The full view's forward axis sits at the clipped view's lower-right corner.
		const auto forwardPoint = view.screenPointForWorldPosition(Double3(0, 0, 50));
		check(forwardPoint.has_value() && near(forwardPoint->x, 800, 1.0e-6) && near(forwardPoint->y, 400, 1.0e-6), "clipping to a quadrant puts the axis at its corner");
	}
	
	// An explicit frustum is the general form the field-of-view framing reduces to, so reading the resolved
	// tangents back out and feeding them in must reproduce the view exactly — including the clip rectangle
	// and the viewport offset, which are two more ways of writing the same four numbers. This is also what
	// pins the claim that folding the offset into the extents equals applying it as an NDC translation.
	{
		CameraSettings settings = landscapeSettings();
		settings.clipBounds = rect_double(-0.8, -0.6, 0.9, 1.0);
		settings.viewportOffset = Double2(0.15, -0.2);
		const CameraView derived(settings);
		
		CameraSettings explicitSettings = settings;
		explicitSettings.frustumTangents = derived.frustumTangents();
		const CameraView explicitView(explicitSettings);
		
		check(explicitView.isValid(), "an explicit frustum produces a valid view");
		check(approxEqual(explicitView.projectionTransform(), derived.projectionTransform(), 1.0e-12), "explicit tangents reproduce the derived projection");
		check(near(explicitView.fieldOfViewRadians(CameraAxis::horizontal), derived.fieldOfViewRadians(CameraAxis::horizontal)), "and its horizontal field of view");
		check(near(explicitView.fieldOfViewRadians(CameraAxis::vertical), derived.fieldOfViewRadians(CameraAxis::vertical)), "and its vertical field of view");
		for (const Double2 screenPoint : {Double2(400, 200), Double2(90, 310)}) {
			const Ray3d a = derived.rayThroughScreenPoint(screenPoint);
			const Ray3d b = explicitView.rayThroughScreenPoint(screenPoint);
			check(approxEqual(a.direction, b.direction, 1.0e-12), "and the rays through the viewport");
		}
	}
	
	// An asymmetric frustum of the kind a headset reports. Nothing downstream may assume the axis runs
	// through the middle of the image.
	{
		CameraSettings settings = landscapeSettings();
		settings.frustumTangents = FrustumTangents{.left = 1.0, .right = 0.7, .top = 0.8, .bottom = 0.95};
		settings.narrowAxisFieldOfViewRadians = 10 * deg2rad;
		const CameraView view(settings);
		
		check(view.isValid(), "an asymmetric frustum is valid");
		check(near(view.fieldOfViewRadians(CameraAxis::horizontal), std::atan(1.0) + std::atan(0.7)), "the horizontal field of view follows the tangents");
		check(near(view.fieldOfViewRadians(CameraAxis::vertical), std::atan(0.8) + std::atan(0.95)), "the vertical field of view follows the tangents");
		check(view.fieldOfViewRadians(CameraAxis::horizontal) > 10 * deg2rad, "an explicit frustum supersedes the requested field of view");
		
		// The forward axis is off centre, in the direction the wider side opens.
		const auto axisPoint = view.screenPointForWorldPosition(Double3(0, 0, 50));
		check(axisPoint.has_value() && axisPoint->x > 400, "the forward axis sits off centre in an asymmetric frustum");
		
		for (const Double2 screenPoint : {Double2(400, 200), Double2(5, 5), Double2(795, 395)}) {
			const auto recovered = view.screenPointForWorldPosition(view.rayThroughScreenPoint(screenPoint).pointAt(60));
			check(recovered.has_value(), "an asymmetric frustum still projects");
			check(near(recovered->x, screenPoint.x, 1.0e-6) && near(recovered->y, screenPoint.y, 1.0e-6), "picking round-trips through an asymmetric frustum");
			check(view.frustum().containsPoint(view.rayThroughScreenPoint(screenPoint).pointAt(60)), "and the frustum follows it");
		}
	}
	
	// FrustumTangents::window is the off-axis projection that holds a fixed window still while the viewer
	// moves behind it. The defining property is the depth gradient: nothing at the window's own depth moves,
	// everything beyond it moves in proportion, and the far limit is a uniform shift.
	{
		constexpr double halfWidth = 4, halfHeight = 2, windowDistance = 10;
		CameraSettings base = landscapeSettings();
		base.farDepth = 1.0e9;
		
		// The tangents hold the window still; the eye displacement supplies the parallax. Both, or neither.
		const auto viewFromEye = [&](const Double3 &eyeOffset, bool moveTheEye) {
			CameraSettings settings = base;
			settings.frustumTangents = FrustumTangents::window(halfWidth, halfHeight, windowDistance, eyeOffset);
			if (moveTheEye) {
				settings.position = eyeOffset;
			}
			return CameraView(settings);
		};
		
		const Double3 eyeOffset(1.5, -0.5, 0);
		const CameraView centred = viewFromEye(Double3(0, 0, 0), true);
		const CameraView shifted = viewFromEye(eyeOffset, true);
		
		for (const Double3 onWindow : {Double3(0, 0, windowDistance), Double3(3, 1, windowDistance), Double3(-2, -1.5, windowDistance)}) {
			const auto a = centred.screenPointForWorldPosition(onWindow);
			const auto b = shifted.screenPointForWorldPosition(onWindow);
			check(a.has_value() && b.has_value(), "a point on the window projects from both eye positions");
			check(near(a->x, b->x, 1.0e-9) && near(a->y, b->y, 1.0e-9), "content at the window depth does not move when the eye does");
		}
		
		// Beyond the window the shift grows with depth and converges on the eye offset measured in window
		// half-widths. A depth-independent shift would be a plain image slide, not parallax.
		double previousShift = 0;
		for (const double depth : {2 * windowDistance, 10 * windowDistance, 1000 * windowDistance}) {
			const auto a = centred.screenPointForWorldPosition(Double3(0, 0, depth));
			const auto b = shifted.screenPointForWorldPosition(Double3(0, 0, depth));
			check(a.has_value() && b.has_value(), "a distant point projects from both eye positions");
			const double shift = b->x - a->x;
			check(shift > previousShift, "the shift grows with depth");
			previousShift = shift;
		}
		const double farLimit = eyeOffset.x / halfWidth * 400;
		check(previousShift < farLimit && previousShift > farLimit * 0.999, "the shift converges on the eye offset in window half-widths");
		
		// Applying the tangents without moving the eye is the failure this helper exists to prevent: the
		// window drifts instead of staying put, which reads as a soft, swimmy version of the same effect.
		const auto drifted = viewFromEye(eyeOffset, false).screenPointForWorldPosition(Double3(0, 0, windowDistance));
		const auto anchored = centred.screenPointForWorldPosition(Double3(0, 0, windowDistance));
		check(drifted.has_value() && anchored.has_value(), "both project");
		check(!near(drifted->x, anchored->x, 1.0), "tangents without the matching eye offset let the window drift");
	}
	
	// Combinations that have no meaning are refused rather than reinterpreted.
	{
		CameraSettings settings = landscapeSettings();
		settings.frustumTangents = FrustumTangents::symmetric(60 * deg2rad, 40 * deg2rad);
		check(CameraView(settings).isValid(), "a symmetric explicit frustum is valid");
		
		settings.projection = Projection::orthographic;
		check(!CameraView(settings).isValid(), "tangents and a parallel projection are refused");
		
		settings.projection = Projection::perspective;
		settings.frustumTangents = FrustumTangents{.left = -1.0, .right = 0.5, .top = 0.5, .bottom = 0.5};
		check(!CameraView(settings).isValid(), "a frustum with no horizontal extent is refused");
		
		settings.frustumTangents = FrustumTangents{.left = 0.5, .right = 0.5, .top = 0.5, .bottom = -0.6};
		check(!CameraView(settings).isValid(), "a frustum with no vertical extent is refused");
	}
	
	// Angular scale: pixels per radian follows the axis it is asked about, and the resolution is half a
	// pixel on the narrow one.
	{
		const CameraView view(landscapeSettings());
		check(near(view.pixelsPerRadian(CameraAxis::vertical), 400 / (60 * deg2rad)), "vertical pixels per radian");
		check(near(view.pixelsPerRadian(CameraAxis::narrow), view.pixelsPerRadian(CameraAxis::vertical)), "the narrow axis is the vertical one in landscape");
		check(near(view.angularResolutionRadians(), 0.5 / view.pixelsPerRadian(CameraAxis::narrow)), "angular resolution is half a narrow-axis pixel");
		check(near(view.aspectRatio(), 2), "the aspect ratio is the viewport's");
	}
	
	// Degenerate settings produce an unusable view rather than NaN threaded through every matrix.
	{
		CameraSettings settings = landscapeSettings();
		settings.viewport = rect_double(0, 0, 800, 0);
		check(!CameraView(settings).isValid(), "a collapsed viewport is invalid");
		
		settings = landscapeSettings();
		settings.nearDepth = 0;
		check(!CameraView(settings).isValid(), "a zero near plane is invalid under perspective");
		
		settings = landscapeSettings();
		settings.farDepth = settings.nearDepth;
		check(!CameraView(settings).isValid(), "an empty depth range is invalid");
		
		settings = landscapeSettings();
		settings.projection = Projection::orthographic;
		settings.orthographicHeight = 0;
		check(!CameraView(settings).isValid(), "a zero orthographic height is invalid");
	}
	
	// The holder publishes whole edits and shares one derived view until the next change, so pointer
	// identity is a change test and a snapshot outlives the edit that replaces it.
	{
		Camera camera(landscapeSettings());
		const auto first = camera.view();
		check(camera.view() == first, "an unchanged camera returns the same view");
		
		camera.setSettings(camera.settings());
		check(camera.view() == first, "publishing identical settings keeps the view");
		
		camera.edit([](CameraSettings &settings) {
			settings.position = Double3(1, 2, 3);
			settings.narrowAxisFieldOfViewRadians = 30 * deg2rad;
		});
		const auto second = camera.view();
		check(second != first, "an edit publishes a new view");
		check(near(second->settings().position.x, 1) && near(second->settings().narrowAxisFieldOfViewRadians, 30 * deg2rad), "the edit applied every field");
		check(near(first->settings().position.x, 0), "the earlier snapshot is unchanged by the edit");
		check(near(first->fieldOfViewRadians(CameraAxis::narrow), 60 * deg2rad), "the earlier snapshot keeps its own derived state");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

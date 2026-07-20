///----------------------------------------
///       @file Quaternion.h
///    @ingroup MathLib
///      @brief Zero-overhead quaternion over the backend quaternion type.
///    @details A thin, header-only abstraction whose only data member is the backend quaternion,
///             so it is trivially copyable and lowers to that type with no overhead. The backend is
///             chosen by the @ref detail::native_quaternion trait and reached only through
///             @ref Quaternion::native, the single escape hatch, so the implementation can be
///             retargeted to another platform. The Euler conventions and yaw/pitch/roll
///             decomposition are defined on the members below; storage, composition, rotation,
///             SLERP and matrix conversion delegate to the backend.
///     @author Created by John Stephen on 6/23/26.
///  @copyright Copyright © 2026 John Stephen (wobbleworks.com)
///             Licensed under the Apache License, Version 2.0.
///             SPDX-License-Identifier: Apache-2.0
///----------------------------------------

#pragma once

#include "MathLib/detail/BackendSelect.h"

#include "MathLib/Numbers.h"
#include "MathLib/Vector.h"
#include "MathLib/Matrix.h"
#include "MathLib/SelfTestCheck.h"

#include <algorithm>
#include <cmath>
#include <numbers>

///----------------------------------------
namespace Math {
///----------------------------------------

///----------------------------------------
namespace detail {
///----------------------------------------

///----------------------------------------
/// @brief Alias to the backend's native quaternion type for a scalar type.
/// @details The @ref native_quaternion trait is declared and specialized by the selected backend
///          (@ref detail/AppleBackend.h or @ref detail/GenericBackend.h).
///----------------------------------------

template <class TScalar>
using native_quaternion_t = typename native_quaternion<TScalar>::type;
	
} // namespace detail

///----------------------------------------
/// @class Quaternion
/// @brief A unit rotation quaternion backed by the native backend type.
/// @tparam TFloat Scalar type (@c float or @c double).
/// @details Stores the rotation as the backend quaternion. The component order is @c (x,y,z,w)
///          where @c (x,y,z) is the imaginary (axis) part and @c w is the real (scalar) part.
///----------------------------------------

template <class TFloat>
class Quaternion {
public:
	using Scalar = TFloat;
	using Native = detail::native_quaternion_t<TFloat>;
	using Vector3 = Vector<TFloat, 3>;
	using Matrix3 = Matrix<TFloat, 3, 3>;
	using Matrix4 = Matrix<TFloat, 4, 4>;
	using NativeVector3 = detail::native_vector_t<TFloat, 3>;
	using NativeMatrix3 = detail::native_matrix_t<TFloat, 3, 3>;
	
	///----------------------------------------
	/// @brief Constructs the identity rotation.
	///----------------------------------------
	
	Quaternion() noexcept : _native(detail::Backend::makeQuaternion(TFloat(0), TFloat(0), TFloat(0), TFloat(1))) {}
	
	///----------------------------------------
	/// @brief Wraps a native backend quaternion.
	///----------------------------------------
	
	[[nodiscard]] static Quaternion fromNative(const Native& native) noexcept {
		Quaternion result;
		result._native = native;
		return result;
	}
	
	///----------------------------------------
	/// @brief The native backend quaternion (read-only escape hatch).
	///----------------------------------------
	
	[[nodiscard]] const Native& native() const noexcept { return _native; }
	
	///----------------------------------------
	/// @brief The native backend quaternion (mutable escape hatch).
	///----------------------------------------
	
	[[nodiscard]] Native& native() noexcept { return _native; }
	
	/// @brief The imaginary @c x component.
	[[nodiscard]] TFloat x() const noexcept { return _native.vector.x; }
	
	/// @brief The imaginary @c y component.
	[[nodiscard]] TFloat y() const noexcept { return _native.vector.y; }
	
	/// @brief The imaginary @c z component.
	[[nodiscard]] TFloat z() const noexcept { return _native.vector.z; }
	
	/// @brief The real (scalar) @c w component.
	[[nodiscard]] TFloat w() const noexcept { return _native.vector.w; }
	
	/// @brief True when all four components are finite (no NaN or infinity).
	[[nodiscard]] bool isFinite() const noexcept {
		auto v = _native.vector;
		return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
	}
	
	///----------------------------------------
	/// @brief Converts from another precision, renormalizing on narrowing.
	///----------------------------------------
	
	template <class TOther>
	explicit Quaternion(const Quaternion<TOther>& other) noexcept {
		auto v = other.native().vector;
		_native = detail::Backend::makeQuaternion(TFloat(v.x), TFloat(v.y), TFloat(v.z), TFloat(v.w));
		if constexpr (sizeof(TOther) != sizeof(TFloat)) {
			normalize();
		}
	}
	
	///----------------------------------------
	/// @brief Constructs a rotation of @p degrees about the axis @c (axisX,axisY,axisZ).
	/// @details The fourth argument is in DEGREES.
	///----------------------------------------
	
	Quaternion(TFloat axisX, TFloat axisY, TFloat axisZ, TFloat degrees) noexcept {
		setRotationRad(axisX, axisY, axisZ, degrees * TFloat(deg2rad));
	}
	
	///----------------------------------------
	/// @brief Constructs a rotation of @p degrees about @p axis.
	/// @details The angle is in DEGREES.
	///----------------------------------------
	
	Quaternion(const Vector3& axis, TFloat degrees) noexcept {
		setRotationRad(axis, degrees * TFloat(deg2rad));
	}
	
	///----------------------------------------
	/// @brief Constructs a rotation from yaw, pitch and roll in RADIANS.
	///----------------------------------------
	
	Quaternion(TFloat yawRad, TFloat pitchRad, TFloat rollRad) noexcept {
		setRotationRad(yawRad, pitchRad, rollRad);
	}
	
	///----------------------------------------
	/// @brief Sets a rotation of @p radians about the axis @c (axisX,axisY,axisZ).
	///----------------------------------------
	
	void setRotationRad(TFloat axisX, TFloat axisY, TFloat axisZ, TFloat radians) noexcept {
		TFloat half = radians * TFloat(0.5);
		TFloat sinHalf = std::sin(half);
		
		_native = detail::Backend::makeQuaternion(sinHalf * axisX, sinHalf * axisY, sinHalf * axisZ, std::cos(half));
		normalize();
	}
	
	///----------------------------------------
	/// @brief Sets a rotation of @p radians about @p axis.
	///----------------------------------------
	
	void setRotationRad(const Vector3& axis, TFloat radians) noexcept {
		setRotationRad(axis.x, axis.y, axis.z, radians);
	}
	
	///----------------------------------------
	/// @brief Sets a rotation from yaw, pitch and roll in RADIANS.
	/// @details The composition is RollQ · PitchQ · YawQ. Yaw is about +Y,
	///          pitch about +X and roll about +Z.
	///----------------------------------------
	
	void setRotationRad(TFloat yaw, TFloat pitch, TFloat roll) noexcept {
		TFloat halfYaw = yaw * TFloat(0.5);
		TFloat halfPitch = pitch * TFloat(0.5);
		TFloat halfRoll = roll * TFloat(0.5);
		
		TFloat sinYaw = std::sin(halfYaw);
		TFloat cosYaw = std::cos(halfYaw);
		TFloat sinPitch = std::sin(halfPitch);
		TFloat cosPitch = std::cos(halfPitch);
		TFloat sinRoll = std::sin(halfRoll);
		TFloat cosRoll = std::cos(halfRoll);
		
		TFloat cosYawSinPitch = cosYaw * sinPitch;
		TFloat sinYawCosPitch = sinYaw * cosPitch;
		
		TFloat imaginaryX = cosYawSinPitch * cosRoll - sinYawCosPitch * sinRoll;
		TFloat imaginaryY = cosYawSinPitch * sinRoll + sinYawCosPitch * cosRoll;
		TFloat imaginaryZ = cosYaw * cosPitch * sinRoll + sinYaw * sinPitch * cosRoll;
		TFloat real = cosYaw * cosPitch * cosRoll - sinYaw * sinPitch * sinRoll;
		
		_native = detail::Backend::makeQuaternion(imaginaryX, imaginaryY, imaginaryZ, real);
		normalize();
	}
	
	///----------------------------------------
	/// @brief Sets the rotation from a 3x3 rotation matrix.
	///----------------------------------------
	
	void setRotation(const Matrix3& matrix) noexcept {
		_native = detail::Backend::makeQuaternion(matrix.native());
		normalize();
	}
	
	///----------------------------------------
	/// @brief Sets the rotation from a native 3x3 rotation matrix.
	///----------------------------------------
	
	void setRotation(const NativeMatrix3& matrix) noexcept {
		_native = detail::Backend::makeQuaternion(matrix);
		normalize();
	}
	
	///----------------------------------------
	/// @brief Returns the rotation as a 3x3 matrix.
	///----------------------------------------
	
	[[nodiscard]] Matrix3 getRotationMatrix() const noexcept {
		return Matrix3::fromNative(detail::Backend::rotationMatrix3x3(_native));
	}
	
	///----------------------------------------
	/// @brief Returns the rotation as a 4x4 matrix.
	///----------------------------------------
	
	[[nodiscard]] Matrix4 getRotationMatrix4() const noexcept {
		return Matrix4::fromNative(detail::Backend::rotationMatrix4x4(_native));
	}
	
	///----------------------------------------
	/// @brief Writes the rotation into a 3x3 matrix.
	///----------------------------------------
	
	void getRotation(Matrix3& matrix) const noexcept {
		matrix = getRotationMatrix();
	}
	
	///----------------------------------------
	/// @brief Returns the axis and angle (RADIANS) of the rotation.
	///----------------------------------------
	
	void getRotationRad(Vector3& axis, TFloat& radians) const noexcept {
		auto v = _native.vector;
		TFloat w = std::min(std::max(v.w, TFloat(-1)), TFloat(1));
		radians = std::acos(w);
		
		if (radians == TFloat(0)) {
			axis = Vector3(TFloat(0), TFloat(1), TFloat(0));
		} else {
			TFloat scale = TFloat(1) / std::sin(radians);
			axis = Vector3(v.x * scale, v.y * scale, v.z * scale);
		}
		
		radians *= TFloat(2);
	}
	
	///----------------------------------------
	/// @brief Returns the local x, y and z basis axes (columns of the rotation matrix).
	///----------------------------------------
	
	void getLocalAxes(Vector3& xAxis, Vector3& yAxis, Vector3& zAxis) const noexcept {
		auto v = _native.vector;
		TFloat twoXX = TFloat(2) * v.x * v.x;
		TFloat twoYY = TFloat(2) * v.y * v.y;
		TFloat twoZZ = TFloat(2) * v.z * v.z;
		TFloat twoXY = TFloat(2) * v.x * v.y;
		TFloat twoXZ = TFloat(2) * v.x * v.z;
		TFloat twoXW = TFloat(2) * v.x * v.w;
		TFloat twoYZ = TFloat(2) * v.y * v.z;
		TFloat twoYW = TFloat(2) * v.y * v.w;
		TFloat twoZW = TFloat(2) * v.z * v.w;
		
		xAxis = Vector3(TFloat(1) - twoYY - twoZZ, twoXY - twoZW, twoXZ + twoYW);
		yAxis = Vector3(twoXY + twoZW, TFloat(1) - twoXX - twoZZ, twoYZ - twoXW);
		zAxis = Vector3(twoXZ - twoYW, twoYZ + twoXW, TFloat(1) - twoXX - twoYY);
	}
	
	///----------------------------------------
	/// @brief Decomposes the rotation into yaw, pitch and roll in RADIANS.
	/// @details Yaw is wrapped to @c [0, 2π).
	///----------------------------------------
	
	void getRotation(TFloat& yaw, TFloat& pitch, TFloat& roll) const noexcept {
		auto v = _native.vector;
		TFloat sinPitch = TFloat(2) * (v.y * v.z + v.x * v.w);
		
		if (sinPitch > TFloat(0.999999)) {
			// Gimbal pole: only yaw+roll is defined, reported entirely as yaw. The quaternion here is
			// (w, z) = (cos, sin) of HALF the combined angle, so the atan2 must be doubled.
			pitch = TFloat(half_pi);
			roll = TFloat(0);
			yaw = TFloat(2) * std::atan2(v.z, v.w);
		} else if (sinPitch < TFloat(-0.999999)) {
			// Gimbal pole: only yaw-roll is defined, reported entirely as yaw
			pitch = -TFloat(half_pi);
			roll = TFloat(0);
			yaw = TFloat(-2) * std::atan2(v.z, v.w);
		} else {
			TFloat xSquared = v.x * v.x;
			yaw = std::atan2(TFloat(2) * (v.y * v.w - v.x * v.z), TFloat(1) - TFloat(2) * (xSquared + v.y * v.y));
			pitch = std::asin(sinPitch);
			roll = std::atan2(TFloat(2) * (v.z * v.w - v.x * v.y), TFloat(1) - TFloat(2) * (xSquared + v.z * v.z));
		}
		
		if (yaw < TFloat(0)) {
			yaw += TFloat(two_pi);
		} else if (yaw >= TFloat(two_pi)) {
			yaw -= TFloat(two_pi);
		}
	}
	
	///----------------------------------------
	/// @brief Rotates @p vector by this quaternion.
	///----------------------------------------
	
	[[nodiscard]] Vector3 act(const Vector3& vector) const noexcept {
		return Vector3::fromNative(detail::Backend::rotate(_native, vector.native()));
	}
	
	///----------------------------------------
	/// @brief Rotates a native 3-vector by this quaternion.
	///----------------------------------------
	
	[[nodiscard]] NativeVector3 act(const NativeVector3& vector) const noexcept {
		return detail::Backend::rotate(_native, vector);
	}
	
	///----------------------------------------
	/// @brief The dot product with @p quaternion.
	///----------------------------------------
	
	[[nodiscard]] TFloat dot(const Quaternion& quaternion) const noexcept {
		return detail::Backend::dot(_native, quaternion._native);
	}
	
	///----------------------------------------
	/// @brief The dot product with @p quaternion, accumulated in double precision.
	///----------------------------------------
	
	[[nodiscard]] double dotd(const Quaternion& quaternion) const noexcept {
		auto a = _native.vector;
		auto b = quaternion._native.vector;
		return double(a.x) * double(b.x) + double(a.y) * double(b.y) + double(a.z) * double(b.z) + double(a.w) * double(b.w);
	}
	
	///----------------------------------------
	/// @brief Negates all four components (the antipodal, equivalent rotation).
	///----------------------------------------
	
	void negate() noexcept {
		_native = detail::Backend::makeQuaternion(-_native.vector);
	}
	
	///----------------------------------------
	/// @brief The conjugate — the inverse rotation for a unit quaternion.
	/// @details Distinct from @ref negate: the conjugate @c (-x,-y,-z,w) reverses the rotation, whereas
	///          negation @c (-x,-y,-z,-w) is the antipodal quaternion representing the same rotation.
	///----------------------------------------
	
	[[nodiscard]] Quaternion conjugate() const noexcept {
		return fromNative(detail::Backend::conjugate(_native));
	}
	
	///----------------------------------------
	/// @brief True when the rotation is the identity.
	///----------------------------------------
	
	[[nodiscard]] bool isZero() const noexcept {
		return _native.vector.w == TFloat(1);
	}
	
	///----------------------------------------
	/// @brief A coarse Euclidean distance between the two component vectors, used to gauge how
	///        far apart two orientations are before interpolating.
	///----------------------------------------
	
	[[nodiscard]] TFloat calcDeltaFactor(const Quaternion& quaternion) const noexcept {
		auto delta = quaternion._native.vector - _native.vector;
		return std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + delta.w * delta.w);
	}
	
	///----------------------------------------
	/// @brief Sets this to the shortest-arc SLERP from @p quaternion1 to @p quaternion2 at @p factor.
	///----------------------------------------
	
	void interpolate(const Quaternion& quaternion1, const Quaternion& quaternion2, TFloat factor) noexcept {
		_native = detail::Backend::slerp(quaternion1._native, quaternion2._native, factor);
	}
	
	///----------------------------------------
	/// @brief Component-wise addition followed by renormalization.
	///----------------------------------------
	
	void operator +=(const Quaternion& quaternion) noexcept {
		_native = detail::Backend::makeQuaternion(_native.vector + quaternion._native.vector);
		normalize();
	}
	
	///----------------------------------------
	/// @brief Composes @p quaternion into this rotation (apply @p quaternion, then this).
	///----------------------------------------
	
	void operator *=(const Quaternion& quaternion) noexcept {
		_native = detail::Backend::multiply(_native, quaternion._native);
		normalize();
	}
	
	///----------------------------------------
	/// @brief The composition of this rotation with @p quaternion (apply @p quaternion, then this).
	///----------------------------------------
	
	[[nodiscard]] Quaternion operator *(const Quaternion& quaternion) const noexcept {
		Quaternion result = fromNative(detail::Backend::multiply(_native, quaternion._native));
		result.normalize();
		return result;
	}
	
	///----------------------------------------
	/// @brief Exact component-wise equality.
	///----------------------------------------
	
	[[nodiscard]] bool operator ==(const Quaternion& quaternion) const noexcept {
		auto a = _native.vector;
		auto b = quaternion._native.vector;
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}
	
	///----------------------------------------
	/// @brief Normalizes to a unit quaternion and clamps the real part to @c [-1,1].
	///----------------------------------------
	
	void normalize() noexcept {
		auto v = _native.vector;
		TFloat reciprocalMagnitude = TFloat(1) / std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
		v *= reciprocalMagnitude;
		v.w = std::min(std::max(v.w, TFloat(-1)), TFloat(1));
		_native = detail::Backend::makeQuaternion(v);
	}
	
private:
	Native _native;
};

static_assert(sizeof(Quaternion<double>) == sizeof(detail::native_quaternion_t<double>), "Quaternion must be a zero-overhead wrapper");
static_assert(sizeof(Quaternion<float>) == sizeof(detail::native_quaternion_t<float>), "Quaternion must be a zero-overhead wrapper");

using Quaternion32 = Quaternion<float>;
using Quaternion64 = Quaternion<double>;

///----------------------------------------
/// @brief Verifies the axis-angle, Euler and matrix conventions against known rotations.
/// @details Throws @ref Math::selftest::Failure on the first violation.
///----------------------------------------
inline void quaternionSelfTest() {
	using selftest::check;
	const auto near = [](double lhs, double rhs) noexcept { return std::abs(lhs - rhs) < 1.0e-9; };
	const double sqrtHalf = std::sqrt(0.5);
	
	// The axis-angle constructor takes DEGREES: a 90° turn about +Z is a 45° half-angle, so both
	// z and w are √½, and the rotation sends +X onto +Y (right-handed active rotation).
	{
		const Quaternion64 aboutZ(0.0, 0.0, 1.0, 90.0);
		check(near(aboutZ.x(), 0.0) && near(aboutZ.y(), 0.0), "axis-angle leaves x and y zero");
		check(near(aboutZ.z(), sqrtHalf) && near(aboutZ.w(), sqrtHalf), "90 degrees is a 45 degree half-angle");
		const Double3 rotatedX = aboutZ.act(Double3(1.0, 0.0, 0.0));
		check(near(rotatedX.x, 0.0) && near(rotatedX.y, 1.0) && near(rotatedX.z, 0.0), "+Z rotation sends +X to +Y");
	}
	
	// Axis-angle round-trip: the recovered angle is in RADIANS and the axis is the original +Z.
	{
		const Quaternion64 aboutZ(0.0, 0.0, 1.0, 90.0);
		Double3 recoveredAxis;
		double recoveredRadians = 0.0;
		aboutZ.getRotationRad(recoveredAxis, recoveredRadians);
		check(near(recoveredRadians, half_pi), "recovered angle is a quarter turn in radians");
		check(near(recoveredAxis.x, 0.0) && near(recoveredAxis.y, 0.0) && near(recoveredAxis.z, 1.0), "recovered axis is +Z");
	}
	
	// Euler round-trip: small yaw/pitch/roll in RADIANS, clear of gimbal lock and the [0,2π) yaw wrap.
	{
		const double yaw = 0.3;
		const double pitch = 0.2;
		const double roll = 0.1;
		Quaternion64 rotation;
		rotation.setRotationRad(yaw, pitch, roll);
		double recoveredYaw = 0.0;
		double recoveredPitch = 0.0;
		double recoveredRoll = 0.0;
		rotation.getRotation(recoveredYaw, recoveredPitch, recoveredRoll);
		check(near(recoveredYaw, yaw) && near(recoveredPitch, pitch) && near(recoveredRoll, roll), "yaw, pitch and roll round-trip");
	}

	// Euler round-trip at the gimbal poles: pitch ±90° collapses yaw and roll into one degree of
	// freedom, reported entirely as yaw. The pole branch extracts the combined angle from (w, z),
	// which hold its HALF-angle — an undoubled extraction rebuilds a rotation off by half the
	// combined angle (seen as a mid-launch 180° camera flip when aiming at the nadir).
	{
		const auto sameRotation = [&near](const Quaternion64 &a, const Quaternion64 &b) {
			Double3 ax, ay, az, bx, by, bz;
			a.getLocalAxes(ax, ay, az);
			b.getLocalAxes(bx, by, bz);
			return near(ax.x, bx.x) && near(ax.y, bx.y) && near(ax.z, bx.z)
			    && near(ay.x, by.x) && near(ay.y, by.y) && near(ay.z, by.z)
			    && near(az.x, bz.x) && near(az.y, bz.y) && near(az.z, bz.z);
		};
		for (double pole : {half_pi, -half_pi}) {
			for (double yaw : {0.0, 1.0, 4.826900}) {
				for (double roll : {0.0, -1.455170, 3.0}) {
					Quaternion64 original;
					original.setRotationRad(yaw, pole, roll);
					double recoveredYaw = 0.0;
					double recoveredPitch = 0.0;
					double recoveredRoll = 0.0;
					original.getRotation(recoveredYaw, recoveredPitch, recoveredRoll);
					check(recoveredYaw >= 0.0 && recoveredYaw < two_pi, "pole yaw is wrapped to [0, 2pi)");
					Quaternion64 rebuilt;
					rebuilt.setRotationRad(recoveredYaw, recoveredPitch, recoveredRoll);
					check(sameRotation(original, rebuilt), "yaw, pitch and roll round-trip at the gimbal poles");
				}
			}
		}
	}

	// getLocalAxes returns the rows of the rotation matrix (the columns of its transpose).
	{
		Quaternion64 rotation;
		rotation.setRotationRad(0.3, 0.2, 0.1);
		Double3 xAxis;
		Double3 yAxis;
		Double3 zAxis;
		rotation.getLocalAxes(xAxis, yAxis, zAxis);
		const Double3x3 matrix = rotation.getRotationMatrix();
		check(near(matrix(0, 0), xAxis.x) && near(matrix(0, 1), xAxis.y) && near(matrix(0, 2), xAxis.z), "local x axis is matrix row 0");
		check(near(matrix(1, 0), yAxis.x) && near(matrix(1, 1), yAxis.y) && near(matrix(1, 2), yAxis.z), "local y axis is matrix row 1");
		check(near(matrix(2, 0), zAxis.x) && near(matrix(2, 1), zAxis.y) && near(matrix(2, 2), zAxis.z), "local z axis is matrix row 2");
	}
	
	// conjugate reverses the axis but keeps w; negate flips all four — the two are distinct.
	{
		const Quaternion64 rotation(0.0, 0.0, 1.0, 90.0);
		const Quaternion64 reversed = rotation.conjugate();
		Quaternion64 negated = rotation;
		negated.negate();
		check(near(reversed.z(), -rotation.z()) && near(reversed.w(), rotation.w()), "conjugate flips the axis and keeps w");
		check(near(negated.z(), -rotation.z()) && near(negated.w(), -rotation.w()), "negate flips every component");
		check(!(reversed == negated), "conjugate and negate are distinct");
	}
	
	// The default quaternion is the identity rotation; a quarter turn is not.
	{
		check(Quaternion64().isZero(), "identity quaternion reports isZero");
		check(!Quaternion64(0.0, 0.0, 1.0, 90.0).isZero(), "a quarter turn is not the identity");
	}
}

///----------------------------------------
} // namespace Math
///----------------------------------------

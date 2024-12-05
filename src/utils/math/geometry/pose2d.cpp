#pragma once

// These are required for Eigen to compile
// https://www.vexforum.com/t/eigen-integration-issue/61474/5
#undef __ARM_NEON__
#undef __ARM_NEON
#include <Eigen/Dense>

#include "../core/include/utils/math/geometry/rotation2d.h"
#include "../core/include/utils/math/geometry/transform2d.h"
#include "../core/include/utils/math/geometry/translation2d.h"
#include "../core/include/utils/math/geometry/twist2d.h"
#include "../core/include/utils/math/geometry/pose2d.h"

/**
 * Constructs a pose with given translation and rotation components.
 *
 * @param translation translational component.
 * @param rotation rotational component.
 */
Pose2d::Pose2d(const Translation2d &translation, const Rotation2d &rotation) : m_translation{translation}, m_rotation{rotation} {}

/**
 * Constructs a pose with given translation and rotation components.
 *
 * @param x x component.
 * @param y y component.
 * @param rotation rotational component.
 */
Pose2d::Pose2d(const double &x, const double &y, const Rotation2d &rotation) : m_translation{x, y}, m_rotation{rotation}

/**
 * Constructs a pose with given translation and rotation components.
 *
 * @param x x component.
 * @param y y component.
 * @param radians rotational component in radians.
 */
Pose2d::Pose2d(const double &x, const double &y, const double &radians);

/**
 * Constructs a pose with given translation and rotation components.
 *
 * @param translation translational component.
 * @param radians rotational component in radians.
 */
Pose2d::Pose2d(const Translation2d &translation, const double &radians);

/**
 * Constructs a pose with given translation and rotation components.
 *
 * @param pose_vector vector of the form [x, y, theta].
 */
Pose2d::Pose2d(const Eigen::Vector3d &pose_vector);

/**
 * Returns the translational component.
 *
 * @return the translational component.
 */
Translation2d Pose2d::translation() const;

/**
 * Returns the x value of the translational component.
 *
 * @return the x value of the translational component.
 */
double Pose2d::x() const;

/**
 * Returns the y value of the translational component.
 *
 * @return the y value of the translational component.
 */
double Pose2d::y() const;

/**
 * Returns the rotational component.
 *
 * @return the rotational component.
 */
Rotation2d Pose2d::rotation() const;

/**
 * Compares this to another pose.
 *
 * @param other the other pose to compare to.
 *
 * @return true if each of the components are within 1e-9 of each other.
 */
bool Pose2d::operator==(const Pose2d other) const;

/**
 * Multiplies this pose by a scalar.
 * Simply multiplies each component.
 *
 * @param scalar the scalar value to multiply by.
 */
Pose2d Pose2d::operator*(const double &scalar) const;

/**
 * Divides this pose by a scalar.
 * Simply divides each component.
 *
 * @param scalar the scalar value to divide by.
 */
Pose2d Pose2d::operator/(const double &scalar) const;

/**
 * Adds a transform to this pose.
 * Simply adds each component.
 *
 * @param transform the change in pose.
 */
Pose2d Pose2d::operator+(const Transform2d &transform) const;

/**
 * Subtracts one pose from another to find the transform between them.
 *
 * @param other the pose to subtract.
 */
Transform2d Pose2d::operator-(const Pose2d &other) const;

/**
 * Finds the pose equivalent to this pose relative to another arbitrary pose rather than the origin.
 *
 * @param other the pose representing the new origin.
 *
 * @return this pose relative to another pose.
 */
Pose2d Pose2d::relative_to(const Pose2d &other) const;

/**
 * Adds a transform to this pose.
 * Simply adds each component.
 *
 * @param transform the change in pose.
 *
 * @return the pose after being transformed.
 */
Pose2d Pose2d::transform_by(const Transform2d &transform) const;

/**
 * Applies a twist (pose delta) to a pose by including first order dynamics of heading.
 *
 * When applying a twist, imagine a constant angular velocity, the translational components must
 * be rotated into the global frame at every point along the twist, simply adding the deltas does not do this,
 * and using euler integration results in some error. This is the analytic solution that that problem.
 *
 * Can also be thought of more simply as applying a twist as following an arc rather than a straight line.
 *
 * See this document for more information on the pose exponential and its derivation.
 * https://file.tavsys.net/control/controls-engineering-in-frc.pdf#section.10.2
 *
 * @param old_pose  The pose to which the twist will be applied.
 * @param twist     The twist, represents a pose delta.
 *
 * @return new pose that has been moved forward according to the twist.
 */
Pose2d Pose2d::exp(const Twist2d &twist) const;

/**
 * The inverse of the pose exponential.
 *
 * Determines the twist required to go from this pose to the given end pose.
 * suppose you have Pose2d a, Twist2d twist
 * if a.exp(twist) = b then a.log(b) = twist
 *
 * @param end_pose the end pose to find the mapping to.
 *
 * @return the twist required to go from this pose to the given end
 */
Twist2d Pose2d::log(const Pose2d &end_pose) const;

/**
 * Calculates the mean of a list of poses.
 *
 * @param list std::vector containing a list of poses.
 *
 * @return the single pose mean of the list of poses.
 */
Pose2d wrapped_mean(const std::vector<Pose2d> &list);

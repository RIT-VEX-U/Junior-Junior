// These are required for Eigen to compile
// https://www.vexforum.com/t/eigen-integration-issue/61474/5
#undef __ARM_NEON__
#undef __ARM_NEON
#include <Eigen/Dense>

#include "../core/include/utils/math/geometry/rotation2d.h"
#include "../core/include/utils/math/geometry/translation2d.h"
#include "../core/include/utils/math/geometry/transform2d.h"

/**
 * Constructs a transform given translation and rotation components.
 *
 * @param translation the translational component of the transform.
 * @param rotation the rotational component of the transform.
 */
Transform2d::Transform2d(const Translation2d &translation, const Rotation2d &rotation) : 
  m_translation{std::move(translation)}, 
  m_rotation{std::move(rotation)} {}

/**
 * Constructs a transform given translation and rotation components.
 *
 * @param x the x component of the transform.
 * @param y the y component of the transform.
 * @param rotation the rotational component of the transform.
 */
Transform2d::Transform2d(const double &x, const double &y, const Rotation2d &rotation) : m_translation{x, y}, m_rotation{std::move(rotation)} {}

/**
 * Constructs a transform given translation and rotation components.
 *
 * @param x the x component of the transform.
 * @param y the y component of the transform.
 * @param radians the rotational component of the transform in radians.
 */
Transform2d::Transform2d(const double &x, const double &y, const double &radians) : m_translation{x, y}, m_rotation{radians} {}

/**
 * Constructs a transform given translation and rotation components.
 *
 * @param translation the translational component of the transform.
 * @param radians the rotational component of the transform in radians.
 */
Transform2d::Transform2d(const Translation2d &translation, const double &radians) : m_translation{std::move(translation)}, m_rotation{radians} {}

/**
 * Constructs a transform given translation and rotation components given as a vector.
 *
 * @param transform_vector vector of the form [x, y, theta]
 */
Transform2d::Transform2d(const Eigen::Vector3d &transform_vector) : m_translation{transform_vector(0), transform_vector(1)}, m_rotation{transform_vector(2)} {}

/**
 * Constructs a transform given translation and rotation components.
 *
 * @param translation the translational component of the transform.
 * @param rotation the rotational component of the transform.
 */
// Transform2d(const Pose2d &start, const Pose2d &end);

/**
 * Returns the translational component of the transform.
 *
 * @return the translational component of the transform.
 */
Translation2d Transform2d::translation() const {
  return m_translation;
}

/**
 * Returns the x component of the transform.
 *
 * @return the x component of the transform.
 */
double Transform2d::x() const {
  return m_translation.x();
}

/**
 * Returns the y component of the transform.
 *
 * @return the y component of the transform.
 */
double Transform2d::y() const {
  return m_translation.y();
}

/**
 * Returns the rotational component of the transform.
 *
 * @return the rotational component of the transform.
 */
Rotation2d Transform2d::rotation() const {
  return m_rotation;
}

/**
 * Inverts the transform.
 *
 * @return the inverted transform.
 */
Transform2d Transform2d::inverse() const {
  return Transform2d((-translation()).rotate_by(-rotation()), -rotation());
}

/**
 * Multiplies this transform by a scalar.
 *
 * @param scalar the scalar to multiply this transform by.
 */
Transform2d Transform2d::operator*(const double &scalar) const {
  return Transform2d(translation() * scalar, rotation() * scalar);
}

/**
 * Divides this transform by a scalar.
 *
 * @param scalar the scalar to divide this transform by.
 */
Transform2d Transform2d::operator/(const double &scalar) const {
  return *this * (1. / scalar);
}

/**
 * Compares this to another transform.
 *
 * @param other the other transform to compare to.
 *
 * @return true if the components are within 1e-9 of each other.
 */
bool Transform2d::operator==(const Transform2d &other) const {
  return (translation() == other.translation()) && (rotation() == other.rotation()); 
}

#pragma once

#include <tuple>
#include <../vendor/eigen/Eigen/Dense>
#include <functional>


#include "../core/include/utils/math/numerical/numerical_integration.h"

/**
 * A Kalman filter combines predictions from a model and measurements to
 * estimate the state of a syste. The motivation for this is the fact that
 * sensors always provide measurements with some amount of noise. Also, there
 * are often parts of the state of a system that cannot be directly observed
 * by a sensor, but are related to measurements in some way.
 * 
 * The predict function moves the state forward in time and adds
 * noise to the state covariance increasing uncertainty. The correct function
 * adjusts the state according to a measurement and reduces uncertainty.
 * 
 * The Unscented Kalman Filter is a nonlinear Kalman filter. Instead of using
 * matrices representing the linear system, a function f(x, u) is used to 
 * propagate the state forward in time. Similarly, a function h(x, u) is used
 * to predict a measurement given a state. Neither of these functions need to
 * be linear. 
 * 
 * For more information on the math behind the Kalman filter and the standard
 * Unscented Kalman Filter read:
 * https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
 * 
 * This particular implementation uses the square-root or factored form of the
 * Unscented Kalman Filter. The main reason for this is to ensure that the
 * covariance matrix remains positive definite. For more information on the
 * SR-UKF, and to read the equations referenced in this implementation, read:
 * https://www.researchgate.net/publication/3908304
 * 
 * @tparam STATES  Number of states to estimate, also L or N, size of x.
 * @tparam INPUTS  Number of inputs used in the process model, size of u.
 * @tparam OUTPUTS Number of outputs from sensors, size of y.
 */

template <int STATES>
class MerweScaledSigmaPoints;

template <int COV_DIM, int STATES>
std::tuple<Eigen::Vector<double, COV_DIM>,
           Eigen::Matrix<double, COV_DIM, COV_DIM>>
square_root_ut(const Eigen::Matrix<double, COV_DIM, 2 * STATES + 1> &sigmas,
               const Eigen::Vector<double, 2 * STATES + 1> &Wm,
               const Eigen::Vector<double, 2 * STATES + 1> &Wc,
               std::function<Eigen::Vector<double, COV_DIM>(
                   const Eigen::Matrix<double, COV_DIM, 2 * STATES + 1> &,
                   const Eigen::Vector<double, 2 * STATES + 1> &)>
                   mean_func,
               std::function<Eigen::Vector<double, COV_DIM>(
                   const Eigen::Vector<double, COV_DIM> &,
                   const Eigen::Vector<double, COV_DIM> &)>
                   residual_func,
               const Eigen::Matrix<double, COV_DIM, COV_DIM> &square_root_R);

template <int STATES, int INPUTS, int OUTPUTS>
class SquareRootUnscentedKalmanFilter {
 public:
  using StateVector = Eigen::Vector<double, STATES>;
  using InputVector = Eigen::Vector<double, INPUTS>;
  using OutputVector = Eigen::Vector<double, OUTPUTS>;

  using StateMatrix = Eigen::Matrix<double, STATES, STATES>;

  /**
   * Constructs an Unscented Kalman Filter using default mean, addition, and
   * residual functions.
   * 
   * @param f A vector-valued function of x and u, which returns the derivative
   * of a state with respect to time.
   * @param h A vector-valued function of x and u that returns the expected
   * measurement given a state and input.
   * @param state_std_devs Vector containing the standard deviations of the
   * states, lower stddev means more trust in the model.
   * @param measurement_std_devs Vector containing the standard deviations of
   * of the measurements, lower stddev means more trust in the measurements.
   */
  SquareRootUnscentedKalmanFilter(std::function<StateVector(const StateVector &, const InputVector &)> f,
        std::function<OutputVector(const StateVector &, const InputVector &)> h,
        const StateVector &state_std_devs,
        const OutputVector &measurement_std_devs)
      : m_f(std::move(f)), m_h(std::move(h)) {
    m_sqrt_Q = state_std_devs.asDiagonal();
    m_sqrt_R = measurement_std_devs.asDiagonal();
    m_mean_func_X =
        [](const Eigen::Matrix<double, STATES, 2 * STATES + 1> &sigmas,
           const Eigen::Vector<double, 2 * STATES + 1> &Wm) -> StateVector {
      return sigmas * Wm;
    };

    m_mean_func_Y =
        [](const Eigen::Matrix<double, OUTPUTS, 2 * STATES + 1> &sigmas,
           const Eigen::Vector<double, 2 * STATES + 1> &Wc) -> OutputVector {
      return sigmas * Wc;
    };
    m_residual_func_X = [](const StateVector &a,
                           const StateVector &b) -> StateVector {
      return a - b;
    };
    m_residual_func_Y = [](const OutputVector &a,
                           const OutputVector &b) -> OutputVector {
      return a - b;
    };
    m_add_func_X = [](const StateVector &a,
                      const StateVector &b) -> StateVector { return a + b; };

    reset();
  }

  /**
   * Constructs an Unscented Kalman Filter using custom mean, addition, and
   * residual functions.
   * 
   * @param f A vector-valued function of x and u, which returns the derivative
   * of a state with respect to time.
   * @param h A vector-valued function of x and u that returns the expected
   * measurement given a state and input.
   * @param state_std_devs Vector containing the standard deviations of the
   * states, lower stddev means more trust in the model.
   * @param measurement_std_devs Vector containing the standard deviations of
   * of the measurements, lower stddev means more trust in the measurements.
   * @param mean_func_X A function that computes the mean of a set of
   * 2 * STATES + 1 state sigma points given a weight for each.
   * @param mean_func_Y A function that computes the mean of a set of
   * 2 * STATES + 1 measurement sigma points given a weight for each.
   * @param residual_func_X A function that computes the residual of
   * two state vectors, usually this is a simple subtraction.
   * @param residual_func_Y A function that computes the residual of
   * two measurement vectors, usually this is a simple subtraction.
   * @param add_func_X A function that adds two state vectors.
   */
  SquareRootUnscentedKalmanFilter(std::function<StateVector(const StateVector &, const InputVector &)> f,
        std::function<OutputVector(const StateVector &, const InputVector &)> h,
        const StateVector &state_std_devs,
        const OutputVector &measurement_std_devs,
        std::function<
            StateVector(const Eigen::Matrix<double, STATES, 2 * STATES + 1> &,
                        const Eigen::Vector<double, 2 * STATES + 1> &)>
            mean_func_X,
        std::function<
            OutputVector(const Eigen::Matrix<double, OUTPUTS, 2 * STATES + 1> &,
                         const Eigen::Vector<double, 2 * STATES + 1> &)>
            mean_func_Y,
        std::function<StateVector(const StateVector &, const StateVector &)>
            residual_func_X,
        std::function<OutputVector(const OutputVector &, const OutputVector &)>
            residual_func_Y,
        std::function<StateVector(const StateVector &, const StateVector &)>
            add_func_X)
      : m_f(std::move(f)),
        m_h(std::move(h)),
        m_mean_func_X(std::move(mean_func_X)),
        m_mean_func_Y(std::move(mean_func_Y)),
        m_residual_func_X(std::move(residual_func_X)),
        m_residual_func_Y(std::move(residual_func_Y)),
        m_add_func_X(std::move(add_func_X)) {
    m_sqrt_Q = state_std_devs.asDiagonal();
    m_sqrt_R = measurement_std_devs.asDiagonal();

    reset();
  }

  /**
   * Returns the lower triangular square-root covariance matrix S.
   * 
   * @return the lower triangular square-root covariance matrix S.
   */
  const StateMatrix &S() const { return m_S; }

  /**
   * Returns an element of the square-root covariance matrix S.
   * 
   * @param i Row of S.
   * @param j Column of S.
   * 
   * @return The element of the square-root covariance matrix S.
   */
  double S(int i, int j) const { return m_S(i, j); }

  /**
   * Sets the current lower triangular square-root covariance matrix S.
   * 
   * @param S The square-root covariance matrix S.
   */
  void set_S(const StateMatrix &S) { m_S = S; }

  /**
   * Returns the reconstructed covariance matrix P.
   * 
   * @return The reconstructed covariance matrix P.
   */
  StateMatrix P() const { return m_S.transpose() * m_S; }

  /**
   * Set the current square-root covariance S by taking the lower triangular
   * square-root of P.
   * 
   * @param P The covariance matrix P.
   */
  void set_P(const StateMatrix &P) { m_S = P.llt().matrixL(); }

  /**
   * Returns the current state estimate x̂.
   * 
   * @return The current state estimate x̂.
   */
  const StateVector &xhat() const { return m_xhat; }

  /**
   * Returns an element of the current state estimate x̂.
   * 
   * @param i Row of x̂.
   * 
   * @return The element of the current state estimate x̂.
   */
  double xhat(int i) const { return m_xhat(i); }

  /**
   * Sets the complete state estimate x̂.
   * 
   * @param xhat The new state estimate x̂.
   */
  void set_xhat(const StateVector &xhat) { m_xhat = xhat; }

  /**
   * Sets one value of the state estimate x̂.
   * 
   * @param i Row of x̂.
   * @param value The value to set.
   */
  void set_xhat(int i, double value) { m_xhat(i) = value; }

  /**
   * Resets the observer.
   * This is dangerous as it sets the square-root covariance to all zeroes.
   * The covariance must be set manually before the filter is used again!
   */
  void reset() {
    m_xhat.setZero();
    m_S.setZero();
    m_sigmas_F.setZero();
  }

  /**
   * Project the state into the future by dt, with control input u.
   * 
   * @param u The control input.
   * @param dt The timestep in seconds.
   */
  void predict(const InputVector &u, double dt, std::function<) {
    // Generate sigma points around the state mean
    //
    // Equation (17)
    Eigen::Matrix<double, STATES, 2 * STATES + 1> sigmas =
        m_pts.square_root_sigma_points(m_xhat, m_S);

    // Project each sigma point forward in time according to the
    // dynamics f(x, u)
    //
    //   sigmas  = 𝒳ₖ₋₁
    //   sigmasF = 𝒳ₖ,ₖ₋₁ or just 𝒳 for readability
    //
    // Equation (18)
    for (int i = 0; i < m_pts.num_sigmas(); ++i) {
      StateVector x = sigmas.template block<STATES, 1>(0, i);
      m_sigmas_F.template block<STATES, 1>(0, i) = euler(m_f, x, u, dt);
    }

    // Pass the predicted sigmas (𝒳) through the Unscented Transform
    // to compute the prior state mean and covariance
    //
    // Equations (18) (19) and (20)
    auto [xhat, S] = square_root_ut<STATES, STATES>(
        m_sigmas_F, m_pts.Wm(), m_pts.Wc(), m_mean_func_X, m_residual_func_X,
        discQ.template triangularView<Eigen::Lower>());

    m_xhat = xhat;
    m_S = S;
  }

  /**
   * Correct the state estimate and covariance using the measurements in y.
   * 
   * @param u The control input for this epoch, same as in the predict step.
   * @param y The vector containing the measurement.
   */
  void correct(const InputVector &u, const OutputVector &y) {
    correct<OUTPUTS>(u, y, m_h, m_sqrt_R, m_mean_func_Y, m_residual_func_Y,
                     m_residual_func_X, m_add_func_X);
  }

  /**
   * Correct the state estimate and covariance using the measurements in y,
   * and a custom measurement noise matrix R.
   * 
   * @param u The control input for this epoch, same as in the predict step.
   * @param y The vector containing the measurement.
   * @param R The square-root measurement noise matrix to use for this step.
   */
  void correct(const InputVector &u, const OutputVector &y,
               const Eigen::Matrix<double, OUTPUTS, OUTPUTS> &R) {
    correct<OUTPUTS>(u, y, m_h, R, m_mean_func_Y, m_residual_func_Y);
  }

  /**
   * Correct the state estimate and covariance using the measurement in y,
   * using a custom measurement function h and measurement noise matrix R.
   * 
   * @param u The control input for this epoch, same as in the predict step.
   * @param y The vector containing the measurement.
   * @param h The vector-valued function of x and u that returns the expected
   * measurement given a state and input to use for this step.
   * @param R The square-root measurement noise matrix to use for this step.
   */
  template <int ROWS>
  void correct(const InputVector &u, const Eigen::Vector<double, ROWS> &y,
               std::function<Eigen::Vector<double, ROWS>(const StateVector &,
                                                         const InputVector &)>
                   h,
               const Eigen::Matrix<double, ROWS, ROWS> &R) {
    correct<ROWS>(u, y, std::move(h), R, std::move(m_mean_func_Y),
                  std::move(m_residual_func_Y));
  }

  /**
  * Correct the state estimate and covariance using the measurement in y,
  * using a custom measurement function h and measurement noise matrix R.
  * 
  * @param u The control input for this epoch, same as in the predict step.
  * @param y The vector containing the measurement.
  * @param h A vector-valued function of x and u that returns the expected
  * measurement given a state and input.
  * @param R The square-root measurement noise matrix to use for this step.
  * @param mean_func_Y A function that computes the mean of a set of
  * 2 * STATES + 1 measurement sigma points given a weight for each.
  * @param residual_func_Y A function that computes the residual of
  * two measurement vectors, usually this is a simple subtraction.
  */
  template <int ROWS>
  void correct(
      const InputVector &u, const Eigen::Vector<double, ROWS> &y,
      std::function<Eigen::Vector<double, ROWS>(const StateVector &,
                                                const InputVector &)>
          h,
      const Eigen::Matrix<double, ROWS, ROWS> &R,
      std::function<Eigen::Vector<double, ROWS>(
          const Eigen::Matrix<double, ROWS, 2 * STATES + 1> &,
          const Eigen::Vector<double, 2 * STATES + 1> &)>
          mean_func_Y,
      std::function<
          Eigen::Vector<double, ROWS>(const Eigen::Vector<double, ROWS> &,
                                      const Eigen::Vector<double, ROWS> &)>
          residual_func_Y) {
    // Generate new sigma points from the prior mean and covariance
    // and transform them into measurement space using h(x, u)
    //
    //   sigmas  = 𝒳
    //   sigmasH = 𝒴
    //
    // This differs from equation (22) which uses
    // the prior sigma points, regenerating them allows
    // multiple measurement updates per time update
    Eigen::Matrix<double, ROWS, 2 * STATES + 1> sigmas_H;
    Eigen::Matrix<double, STATES, 2 * STATES + 1> sigmas = m_pts.square_root_sigma_points(m_xhat, m_S);
    for (int i = 0; i < m_pts.num_sigmas(); ++i) {
      sigmas_H.template block<ROWS, 1>(0, i) =
          h(m_sigmas_F.template block<STATES, 1>(0, i), u);
    }

    // Pass the predicted measurement sigmas through the Unscented Transform
    // to compute the mean predicted measurement and square-root innovation
    // covariance.
    //
    // Equations (23) (24) and (25)
    auto [yhat, Sy] = square_root_ut<ROWS, STATES>(
        sigmas_H, m_pts.Wm(), m_pts.Wc(), mean_func_Y, residual_func_Y,
        discR.template triangularView<Eigen::Lower>());

    // Compute cross covariance of the predicted state and measurement sigma
    // points given as:
    //
    //           2n
    //   P_{xy} = Σ Wᵢ⁽ᶜ⁾[𝒳ᵢ - x̂][𝒴ᵢ - ŷ⁻]ᵀ
    //           i=0
    //
    // Equation (26)
    Eigen::Matrix<double, STATES, ROWS> Pxy;
    Pxy.setZero();
    for (int i = 0; i < m_pts.num_sigmas(); ++i) {
      Pxy += m_pts.Wc(i) *
             (m_residual_func_X(m_sigmas_F.template block<STATES, 1>(0, i),
                              m_xhat)) *
             (residual_func_Y(sigmas_H.template block<ROWS, 1>(0, i), yhat))
                 .transpose();
    }

    // Compute the Kalman gain. We use Eigen's QR decomposition to solve. This
    // is equivalent to MATLAB's \ operator, so we need to rearrange to use
    // that.
    //
    //   K = (P_{xy} / S_{y}ᵀ) / S_{y}
    //   K = (S_{y} \ P_{xy})ᵀ / S_{y}
    //   K = (S_{y}ᵀ \ (S_{y} \ P_{xy}ᵀ))ᵀ
    //
    // Equation (27)
    Eigen::Matrix<double, STATES, ROWS> K =
        (Sy.transpose().fullPivHouseholderQr().solve(
             Sy.fullPivHouseholderQr().solve(Pxy.transpose())))
            .transpose();

    // Compute the posterior state mean
    //
    //   x̂ = x̂⁻ + K(y − ŷ⁻)
    //
    // Second part of equation (27)
    m_xhat = add_func_X(m_xhat, K * residual_func_Y(y, yhat));

    // Compute the intermediate matrix U for downdating
    // the square-root covariance
    //
    // Equation (28)
    Eigen::Matrix<double, STATES, ROWS> U = K * Sy;

    // Downdate the posterior square-root state covariance
    //
    // Equation (29)
    for (int i = 0; i < ROWS; i++) {
      Eigen::internal::llt_inplace<double, Eigen::Lower>::rankUpdate(
          m_S, U.template block<STATES, 1>(0, i), -1);
    }
  }

 private:
  std::function<StateVector(const StateVector &, const InputVector &)> m_f;
  std::function<OutputVector(const StateVector &, const InputVector &)> m_h;
  std::function<StateVector(
      const Eigen::Matrix<double, STATES, 2 * STATES + 1> &,
      const Eigen::Vector<double, 2 * STATES + 1> &)>
      m_mean_func_X;
  std::function<OutputVector(
      const Eigen::Matrix<double, OUTPUTS, 2 * STATES + 1> &,
      const Eigen::Vector<double, 2 * STATES + 1> &)>
      m_mean_func_Y;
  std::function<StateVector(const StateVector &, const StateVector &)>
      m_residual_func_X;
  std::function<OutputVector(const OutputVector &, const OutputVector &)>
      m_residual_func_Y;
  std::function<StateVector(const StateVector &, const StateVector &)>
      m_add_func_X;
  StateVector m_xhat;
  StateMatrix m_S;
  StateMatrix m_sqrt_Q;
  Eigen::Matrix<double, OUTPUTS, OUTPUTS> m_sqrt_R;
  Eigen::Matrix<double, STATES, 2 * STATES + 1> m_sigmas_F;

  MerweScaledSigmaPoints<STATES> m_pts;
};

/**
 * Computes the Unscented transform of a set of sigma points and weights.
 * Returns the mean and square-root covariance of the sigma points as a tuple.
 * 
 * @tparam COV_DIM The dimension of the covariance of the transformed sigma
 * points.
 * @tparam STATES The number of states.
 * 
 * @param sigmas Matrix with each column being one sigma point.
 * @param Wm Weight of each sigma point for the mean.
 * @param Wc Weight of each sigma point for the covariance.
 * @param mean_func A function that computes the mean of 2 * STATES + 1 state
 * vectors (the sigma points) using the vector of weights.
 * @param residual_func A function that computes the residual of two state
 * vectors, usually this is a simple subtraction.
 * @param square_root_R square-root of the noise covariance of the sigma points.
 * 
 * @return Tuple of x, the mean of the sigma points; S, the square-root
 * covariance of the sigma points.
 */
template <int COV_DIM, int STATES>
std::tuple<Eigen::Vector<double, COV_DIM>,
           Eigen::Matrix<double, COV_DIM, COV_DIM>>
square_root_ut(const Eigen::Matrix<double, COV_DIM, 2 * STATES + 1> &sigmas,
               const Eigen::Vector<double, 2 * STATES + 1> &Wm,
               const Eigen::Vector<double, 2 * STATES + 1> &Wc,
               std::function<Eigen::Vector<double, COV_DIM>(
                   const Eigen::Matrix<double, COV_DIM, 2 * STATES + 1> &,
                   const Eigen::Vector<double, 2 * STATES + 1> &)>
                   mean_func,
               std::function<Eigen::Vector<double, COV_DIM>(
                   const Eigen::Vector<double, COV_DIM> &,
                   const Eigen::Vector<double, COV_DIM> &)>
                   residual_func,
               const Eigen::Matrix<double, COV_DIM, COV_DIM> &square_root_R) {
  // New mean is usually just the sum of the sigmas * weights:
  //
  //      2n
  //   x̂ = Σ Wᵢ⁽ᵐ⁾𝒳ᵢ
  //      i=0
  //
  // Equations (19) and (23) in the paper show this,
  // but we allow a custom function, usually for angle wrapping
  Eigen::Vector<double, COV_DIM> x = mean_func(sigmas, Wm);

  // Form an intermediate matrix S⁻ as:
  //
  //   [√{W₁⁽ᶜ⁾}(𝒳_{1:2L} - x̂) √{Rᵛ}]
  //
  // The part of equations (20) and (24) within the "qr{}"
  Eigen::Matrix<double, COV_DIM, STATES * 2 + COV_DIM> S_bar;
  for (int i = 0; i < STATES * 2; i++) {
    S_bar.template block<COV_DIM, 1>(0, i) =
        std::sqrt(Wc[1]) *
        residual_func(sigmas.template block<COV_DIM, 1>(0, i + 1), x);
  }
  S_bar.template block<COV_DIM, COV_DIM>(0, STATES * 2) = square_root_R;

  // Compute the square-root covariance of the sigma points.
  //
  // We transpose S⁻ first because we formed it by horizontally
  // concatenating each part; it should be vertical so we can take
  // the QR decomposition as defined in the "QR Decomposition" passage
  // of section 3. "EFFICIENT SQUARE-ROOT IMPLEMENTATION"
  //
  // The resulting matrix R is the square-root covariance S, but it
  // is upper triangular, so we need to transpose it.
  //
  // Equations (20) and (24)
  Eigen::Matrix<double, COV_DIM, COV_DIM> S =
      S_bar.transpose()
          .householderQr()
          .matrixQR()
          .template block<COV_DIM, COV_DIM>(0, 0)
          .template triangularView<Eigen::Upper>().transpose();

  // Update or downdate the square-root covariance with (𝒳₀-x̂)
  // depending on whether its weight (W₀⁽ᶜ⁾) is positive or negative.
  //
  // Equations (21) and (25)
  Eigen::internal::llt_inplace<double, Eigen::Lower>::rankUpdate(
      S, residual_func(sigmas.template block<COV_DIM, 1>(0, 0), x), Wc[0]);

  return std::make_tuple(x, S);
}

/**
 * Generates the sigma points used for the Unscented Kalman Filter,
 * specifically uses the square-root covariance, so this implementation must
 * be used with the SR-UKF.
 * 
 * @tparam STATES The number of states being estimated.
 */
template <int STATES>
class MerweScaledSigmaPoints {
public:
  /**
   * Constructs a generator for sigma points according to the formulation in 
   * Wan and Merwe's paper linked at the top of this header. 
   * 
   * @param alpha Determines the spread of the sigma points around the mean,
   * usually a small value, defaults to 0.001
   * @param beta Incorporates prior knowledge of the distribution of the mean,
   * defaults to 2 as that is optimal for Gaussian distributions.
   * @param kappa Secondary scaling parameter, defaults to 3 - STATES,
   * also frequently 0 in literature. 
   */
  explicit MerweScaledSigmaPoints(double alpha = 0.001, double beta = 2,
                                int kappa = 3 - STATES) {
    m_alpha = alpha;
    m_kappa = kappa;

    compute_weights(beta);
  }

  /**
   * Returns the number of sigma points generated.
   * 
   * @return The number of sigma points generated.
   */
  int num_sigmas() { return 2 * STATES + 1; }

  /**
   * Computes the sigma points given the mean (x) and the lower triangular
   * square-root covariance (S) from the UKF.
   * 
   * @param x The current state mean.
   * @param S The current square-root covariance.
   * 
   * @return Matrix containing all the sigma points, one per column.
   */
  Eigen::Matrix<double, STATES, 2 * STATES + 1> square_root_sigma_points(
      const Eigen::Vector<double, STATES> &x,
      const Eigen::Matrix<double, STATES, STATES> &S) {
    double lambda = (m_alpha * m_alpha) * (STATES + m_kappa) - STATES;
    double eta = std::sqrt(lambda + STATES);
    Eigen::Matrix<double, STATES, STATES> U = eta * S;

    // Generates the sigmas and constructs the matrix containing them
    //
    // [x̂  x̂ + ηS  x̂ - ηS]
    //
    // Equation (17)
    Eigen::Matrix<double, STATES, 2 * STATES + 1> sigmas;
    sigmas.template block<STATES, 1>(0, 0) = x;
    for (int k = 0; k < STATES; ++k) {
      sigmas.template block<STATES, 1>(0, k + 1) =
          x + U.template block<STATES, 1>(0, k);
      sigmas.template block<STATES, 1>(0, STATES + k + 1) =
          x - U.template block<STATES, 1>(0, k);
    }

    return sigmas;
  }

  /**
   * Returns the vector containing each sigma point's weight for the mean.
   * 
   * @return the vector containing each sigma point's weight for the mean.
   */
  const Eigen::Vector<double, 2 * STATES + 1> &Wm() const { return m_Wm; }

  /**
   * Returns the vector containing each sigma point's weight for the covariance.
   * 
   * @return the vector containing each sigma point's weight for the covariance.
   */
  const Eigen::Vector<double, 2 * STATES + 1> &Wc() const { return m_Wc; }

  /**
   * Returns the weight of a single sigma point for the mean.
   * 
   * @param i The index of the sigma point.
   * 
   * @return The weight of the sigma point for the mean.
   */
  double Wm(int i) const { return m_Wm(i); }

  /**
   * Returns the weight of a single sigma point for the covariance.
   * 
   * @param i The index of the sigma point.
   * 
   * @return The weight of the sigma point for the mean.
   */
  double Wc(int i) const { return m_Wc(i); }

 private:
  Eigen::Vector<double, 2 * STATES + 1> m_Wm;
  Eigen::Vector<double, 2 * STATES + 1> m_Wc;
  double m_alpha;
  int m_kappa;

  /**
   * Computes the weights for the sigma points.
   * 
   * @param beta The parameter incorporating prior knowledge of the distribution.
   */
  void compute_weights(double beta) {
    double lambda = (m_alpha * m_alpha) * (STATES + m_kappa) - STATES;

    double c = 0.5 / (STATES + lambda);
    m_Wm = Eigen::Vector<double, 2 * STATES + 1>::Constant(c);
    m_Wc = Eigen::Vector<double, 2 * STATES + 1>::Constant(c);

    m_Wm(0) = lambda / (STATES + lambda);
    m_Wc(0) = lambda / (STATES + lambda) + (1 - (m_alpha * m_alpha) + beta);
  }
};

// Allow both SRUKF and SquareRootUnscentedKalmanFilter to be usable names.
template <int STATES, int INPUTS, int OUTPUTS>
using SRUKF = SquareRootUnscentedKalmanFilter<STATES, INPUTS, OUTPUTS>;

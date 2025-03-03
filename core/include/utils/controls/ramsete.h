#pragma once

#include <Eigen/Dense>
#include "../core/include/utils/controls/feedback_base.h"
#include "../core/include/utils/math/geometry/pose2d.h"
#include "vex.h"

/**
 * Adapted from Purdue Sigbots
 */

class Ramsete {
    public:
        typedef struct {
            double lin_vel;
            double ang_vel;
        } output_t;

        Ramsete(double ib, double iz);

        void set_target(Pose2d t, double vel, double ang_vel);

        void set_target(double x, double y, double vel, double ang_vel);

        void set_gains(double ib, double iz);

        output_t step(double x, double y, double theta);

        output_t step(Pose2d pose);

    private:
        double b;
        double z;
        double target_x;
        double target_y;
        double target_vel;
        double target_ang_vel;

};
#pragma once
#include "../core/include/subsystems/odometry/odometry_serial.h"
#include "TempSubSystems/TempSubSystems.h"
#include "core.h"
#include "vex.h"

#define WALLSTAKE_POT_OFFSET

extern vex::brain brain;
extern vex::controller con;

bool blue_alliance();
// ================ INPUTS ================
// Digital sensors

// Analog sensors
extern CustomEncoder Left_enc;
extern CustomEncoder right_enc;
extern CustomEncoder front_enc;

// ================ OUTPUTS ================
// Motors
extern vex::motor left_top;
extern vex::motor left_front;
extern vex::motor left_middle;
extern vex::motor left_rear;

extern vex::motor right_top;
extern vex::motor right_front;
extern vex::motor right_middle;
extern vex::motor right_rear;

extern vex::motor conveyor;
extern vex::motor intake_motor;

extern vex::motor wallstake_left;
extern vex::motor wallstake_right;
extern vex::motor_group wallstake_motors;

extern Rotation2d initial;
extern Rotation2d tolerance;
extern double pot_offset;
extern vex::pot wall_pot;
// extern WallStakeMech wallstakemech_sys;

extern vex::motor_group left_drive_motors;
extern vex::motor_group right_drive_motors;

// Pneumatics
extern vex::digital_out goal_grabber_sol;
extern vex::inertial imu;

extern vex::distance goal_sensor;

extern vex::pot wall_pot;

// ================ SUBSYSTEMS ================
extern ClamperSys clamper_sys;
extern IntakeSys intake_sys;
extern PID drive_pid;
extern PID turn_pid;
extern MotionController::m_profile_cfg_t drive_motioncontroller_cfg;
extern MotionController drive_motioncontroller;

// extern AsymmetricMotionController::a_m_profile_cfg_t drive_motioncontroller_slow_decel_cfg;
// extern AsymmetricMotionController drive_motioncontroller_slow_decel;

extern vex::digital_out mcglight_board;
// extern vex::pwm_out mcglight_board;

extern vex::optical color_sensor;

extern PID::pid_config_t correction_pid_cfg;
extern OdometrySerial odom;
extern OdometryTank tankodom;

extern robot_specs_t robot_cfg;
extern TankDrive drive_sys;

// ================ UTILS ================

void robot_init();

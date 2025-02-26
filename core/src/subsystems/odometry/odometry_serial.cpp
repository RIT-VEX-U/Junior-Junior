// These are required for Eigen to compile
// https://www.vexforum.com/t/eigen-integration-issue/61474/5
#undef __ARM_NEON__
#undef __ARM_NEON
#include <Eigen/Dense>

#include "../core/include/subsystems/odometry/odometry_serial.h"

#include "../core/include/subsystems/custom_encoder.h"
#include "../core/include/subsystems/odometry/odometry_base.h"
#include "../core/include/utils/math_util.h"

#include "../core/include/utils/math/geometry/pose2d.h"

/**
 * OdometrySerial
 *
 * This class handles the code for an odometry setup where calculations are done on an external coprocessor.
 * Data is sent to the brain via smart port, using a generic serial (UART) connection.
 *
 *
 *
 * This is a "set and forget" class, meaning once the object is created, the robot will immediately begin
 * tracking it's movement in the background.
 *
 * https://rit.enterprise.slack.com/files/U04112Y5RB6/F080M01KPA5/predictperpindiculars2.pdf
 * 2024-2025 Notebook: Entries/Software Entries/Localization/N-Pod Odometry
 *
 * @author Jack Cammarata
 * @date Jan 16 2025
 */

/**
 * Construct a new Odometry Serial Object
 */
OdometrySerial::OdometrySerial(
  bool is_async, int32_t port,
  int32_t baudrate
)
    : OdometryBase(is_async), _port(port) {
    vexGenericSerialEnable(_port, 0);
    vexGenericSerialBaudrate(_port, baudrate);
    
}

// 0b00000001 is pose (48 bit packet)
// 0b00000010 is vel (48 bit packet)
// 0b00000100 is acc (48 bit packet)
// 0b00000011 is pos/vel (96 bit packet)
// 0b00000101 is pos/acc (96 bit packet)
// 0b00000110 is vel/acc (96 bit packet)
// 0b00000111 is pos/vel/acc (144 bit packet)
// 0b00001000 is set config mode (wait for another 48 bit packet with the pos to set) respond 0xFF when finished
// 0b00010000 is calibrate imu, (wait for another 8 bit packet with one byte for 0-255 samples) respond 0xFF when finished
// 0b00100000 is reset tracking respond 0xFF when finished
// 0b01000000 is set linear scalar, (wait for another 16 bit packet float) respond 0xFF when finished
// 0b10000000 is set angular scalar, (wait for another 16 bit packet float) respond 0xFF when finished
// 0b10001000 is set offset (wait for another 48 bit packet with the offset to set) respond 0xFF when finished

void OdometrySerial::calibrate_imu(uint8_t &samples) {
    uint8_t ctl_byte = 0b00010000;
    
    send_packet(_port, &ctl_byte, 1);
    send_packet(_port, &samples, 1);

    receive_packet(_port, &ctl_byte, 1); // block until response received
    printf("imu calibrated!\n");
}

void OdometrySerial::reset_tracking() {
    uint8_t ctl_byte = 0b00100000;

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, &ctl_byte, 1);
    printf("tracking reset!\n");
}
void OdometrySerial::set_linear_scalar(const float &linear_scalar) {
    uint8_t ctl_byte = 0b00010000;
    uint8_t data[sizeof(linear_scalar)];
    memcpy(&data, &linear_scalar, sizeof(linear_scalar));
    
    send_packet(_port, &ctl_byte, 1);
    send_packet(_port, data, 1);

    receive_packet(_port, &ctl_byte, 1); // block until response received
}
void OdometrySerial::set_angular_scalar(const float &angular_scalar) {
    uint8_t ctl_byte = 0b00010000;
    uint8_t data[sizeof(angular_scalar)];
    memcpy(data, &angular_scalar, sizeof(angular_scalar));
    
    send_packet(_port, &ctl_byte, 1);
    send_packet(_port, data, 1);

    receive_packet(_port, &ctl_byte, 1); // block until response received
}
void OdometrySerial::set_offset(Pose2d &sensor_offset) {
    while (!pausable) {

    }
    handle->suspend();
    printf("setting offset\n");
    uint8_t ctl_byte = 0b10001000;
    uint8_t raw[6];
    pose_to_regs(raw, sensor_offset, METER_TO_INT16, RAD_TO_INT16);
    
    send_packet(_port, &ctl_byte, 1);
    send_packet(_port, raw, sizeof(raw));

    receive_packet(_port, &ctl_byte, 1);
    handle->resume();
    printf("offset set!\n");
}
void OdometrySerial::set_position(Pose2d &pose) {
    while (!pausable) {

    }
    handle->suspend();
    uint8_t ctl_byte = 0b00001000;
    uint8_t raw[6];
    pose_to_regs(raw, pose, METER_TO_INT16, RAD_TO_INT16);
    
    send_packet(_port, &ctl_byte, 1);
    send_packet(_port, raw, sizeof(raw));

    receive_packet(_port, &ctl_byte, 1);
    handle->resume();
    printf("position set!\n");
}

void OdometrySerial::request_pos(Pose2d &pos) {
    uint8_t ctl_byte = 0b00000001;
    uint8_t raw[6];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    pos = regs_to_pose(raw, INT16_TO_METER, INT16_TO_RAD);
}
void OdometrySerial::request_vel(Pose2d &vel) {
    uint8_t ctl_byte = 0b00000010;
    uint8_t raw[6];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    vel = regs_to_pose(raw, INT16_TO_MPS, INT16_TO_RPS);
}
void OdometrySerial::request_acc(Pose2d &acc) {
    uint8_t ctl_byte = 0b00000100;
    uint8_t raw[6];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    acc = regs_to_pose(raw, INT16_TO_MPSS, INT16_TO_RPSS);
}


void OdometrySerial::request_pos_vel(Pose2d &pos, Pose2d &vel) {
    uint8_t ctl_byte = 0b00000011;
    uint8_t raw[12];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    pos = regs_to_pose(raw, INT16_TO_METER, INT16_TO_RAD);
    vel = regs_to_pose(raw + 6, INT16_TO_MPS, INT16_TO_RPS);
}
void OdometrySerial::request_pos_acc(Pose2d &pos, Pose2d &acc) {
    uint8_t ctl_byte = 0b00000101;
    uint8_t raw[12];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    pos = regs_to_pose(raw, INT16_TO_METER, INT16_TO_RAD);
    acc = regs_to_pose(raw + 6, INT16_TO_MPSS, INT16_TO_RPSS);
}
void OdometrySerial::request_vel_acc(Pose2d &vel, Pose2d &acc) {
    uint8_t ctl_byte = 0b00000110;
    uint8_t raw[12];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    vel = regs_to_pose(raw, INT16_TO_MPS, INT16_TO_RPS);
    acc = regs_to_pose(raw + 6, INT16_TO_MPSS, INT16_TO_RPSS);
}

void OdometrySerial::request_pos_vel_acc(Pose2d &pos, Pose2d &vel, Pose2d &acc) {
    uint8_t ctl_byte = 0b00000111;
    uint8_t raw[18];

    send_packet(_port, &ctl_byte, 1);

    receive_packet(_port, raw, sizeof(raw));
    pos = regs_to_pose(raw, INT16_TO_METER, INT16_TO_RAD);
    vel = regs_to_pose(raw + 6, INT16_TO_MPS, INT16_TO_RPS);
    acc = regs_to_pose(raw + 12, INT16_TO_MPSS, INT16_TO_RPSS);
}

int OdometrySerial::send_packet(uint32_t port, uint8_t *data, size_t data_size) {
    uint8_t buffer[data_size + 2];
    cobs_encode(data, data_size, buffer);

    for (int i = 0; i < data_size + 2; i++) {
        printf("%x ", buffer[i]);
    }

    printf("\n");
    return vexGenericSerialTransmit(port, buffer, data_size + 1);
}

int OdometrySerial::receive_packet(uint32_t port, uint8_t *buffer, size_t buffer_size) {
    uint8_t cobs_encoded[buffer_size + 1];
    receive_cobs_packet(port, cobs_encoded, buffer_size + 1);
    return cobs_decode(buffer, buffer_size + 1, cobs_encoded);
}

/**
 * Attempts to recieve an entire packet encoded with COBS, stops at delimiter or there's a buffer overflow
 * 
 * @param port the port number the serial is plugged into, counts from 0 instead of 1
 * @param buffer pointer to a uint8_t[] where we put the data
 * @param buffer_size length in bytes of the buffer
 * @return 0 success
 */
int OdometrySerial::receive_cobs_packet(uint32_t port, uint8_t *buffer, size_t buffer_size) {
    size_t index = 0;

    while (true) {
        // wait for a byte (we read byte by byte into our own buffer rather than grabbing the whole packet all at once)
        if (vexGenericSerialReceiveAvail(port) > 0) {
            printf("avail\n");
            uint8_t character = vexGenericSerialReadChar(port);

            // if delimiter
            if (character == 0x00) {
                return index; // return packet length
            }

            // store character in buffer
            if (index < buffer_size) {
                buffer[index++] = character;
            } else {
                // buffer overflow
                printf("bufferoverflow\n");
                return -1;
            }
        }
        vex::this_thread::yield();
    }
}

/**
 * Update the current position of the robot once by reading a single packet from the serial port, then updating all over values, velocity, accel
 *
 * @return the robot's updated position
 */
pose_t OdometrySerial::update() {
    printf("update\n");
    this->pausable = false;
    request_pos_vel_acc(pos, vel, acc);
    this->pausable = true;
    this->speed = vel.translation().norm();
    this->accel = acc.translation().norm();
    this->ang_speed_deg = vel.rotation().degrees();
    this->ang_accel_deg = acc.rotation().degrees();


    return {pos.x(), pos.y(), pos.rotation().wrapped_degrees_360()};
}

/**
 * Gets the current position and rotation
 * 
 * @return the position that the odometry believes the robot is at
 */
pose_t OdometrySerial::get_position(void) {
    Pose2d pose = get_pose2d();
    return pose_t{pose.x(), pose.y(), pose.rotation().wrapped_degrees_360()};
}

/**
 * Gets the current position and rotation
 * 
 * @return the position that the odometry believes the robot is at
 */
Pose2d OdometrySerial::get_pose2d(void) {
    return pos;
}

/** COBS encode data to buffer
 * 
 * @param data Pointer to input data to encode
 * @param length Number of bytes to encode
 * @param buffer Pointer to encoded output buffer
 * 
 * @return Encoded buffer length in bytes
 * @note Does not output delimiter byte
*/
size_t OdometrySerial::cobs_encode(const void *data, size_t length, uint8_t *buffer) {
    assert(data && buffer);

    uint8_t *encode = buffer;  // Encoded byte pointer
    uint8_t *codep = encode++; // Output code pointer
    uint8_t code = 1;          // Code value

    for (const uint8_t *byte = (const uint8_t *)data; length--; ++byte) {
        if (*byte) // Byte not zero, write it
            *encode++ = *byte, ++code;

        if (!*byte || code == 0xff) // Input is zero or block completed, restart
        {
            *codep = code, code = 1, codep = encode;
            if (!*byte || length)
                ++encode;
        }
    }
    *codep = code; // Write final code value

    *encode++ = 0x00; // Append 0x00 delimiter

    return (size_t)(encode - buffer);
}

/** COBS decode data from buffer
 * @param buffer Pointer to encoded input bytes
 * @param length Number of bytes to decode
 * @param data Pointer to decoded output data
 * 
 * @return Number of bytes successfully decoded
 * @note Stops decoding if delimiter byte is found
*/
size_t OdometrySerial::cobs_decode(const uint8_t *buffer, size_t length, void *data) {
    assert(buffer && data);

    const uint8_t *byte = buffer;      // Encoded input byte pointer
    uint8_t *decode = (uint8_t *)data; // Decoded output byte pointer

    for (uint8_t code = 0xff, block = 0; byte < buffer + length; --block) {
        if (block) // Decode block byte
            *decode++ = *byte++;
        else {
            block = *byte++;             // Fetch the next block length
            if (block && (code != 0xff)) // Encoded zero, write it unless it's delimiter.
                *decode++ = 0;
            code = block;
            if (!code) // Delimiter code found
                break;
        }
    }

    return (size_t)(decode - (uint8_t *)data);
}

Pose2d OdometrySerial::regs_to_pose(uint8_t *raw, float raw_to_xy, float raw_to_h) {
    int16_t raw_x = (raw[1] << 8) | raw[0];
    int16_t raw_y = (raw[3] << 8) | raw[2];
    int16_t raw_h = (raw[5] << 8) | raw[4];

    double x = raw_x * raw_to_xy * METER_TO_INCH;
    double y = raw_y * raw_to_xy * METER_TO_INCH;
    double h = raw_h * raw_to_h;

    return Pose2d(x, y, h);
}

void OdometrySerial::pose_to_regs(uint8_t *raw, Pose2d &pose, float xy_to_raw, float h_to_raw) {
    int16_t rawx = (float)(pose.x()) * xy_to_raw / METER_TO_INCH;
    int16_t rawy = (float)(pose.y()) * xy_to_raw / METER_TO_INCH;
    int16_t rawh = (float)(pose.rotation().wrapped_radians_360()) * h_to_raw / RADIAN_TO_DEGREE;

    raw[0] = rawx & 0xFF;
    raw[1] = (rawx >> 8) & 0xFF;
    raw[2] = rawy & 0xFF;
    raw[3] = (rawy >> 8) & 0xFF;
    raw[4] = rawh & 0xFF;
    raw[5] = (rawh >> 8) & 0xFF;
}

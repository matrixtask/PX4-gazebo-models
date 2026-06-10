#!/usr/bin/env python3

import argparse
import math
import time

from pymavlink import mavutil


def wait_message(master, msg_type, timeout=5.0):
    return master.recv_match(type=msg_type, blocking=True, timeout=timeout)


def send_setpoint(master, x_m, y_m, z_m, yaw_rad, vx_m_s=0.0, vy_m_s=0.0, vz_m_s=0.0,
                  ax_m_s2=0.0, ay_m_s2=0.0, az_m_s2=0.0, use_feedforward=False):
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    if not use_feedforward:
        type_mask |= (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x_m,
        y_m,
        z_m,
        vx_m_s,
        vy_m_s,
        vz_m_s,
        ax_m_s2,
        ay_m_s2,
        az_m_s2,
        yaw_rad,
        0.0,
    )


def send_command(master, command, *params):
    full_params = list(params) + [0.0] * (7 - len(params))
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command,
        0,
        *full_params[:7],
    )


def set_mode(master, mode_name):
    mapping = master.mode_mapping()
    if not mapping or mode_name not in mapping:
        raise RuntimeError(f"mode {mode_name} not available; got {mapping}")
    master.set_mode_px4(mode_name, 0, 0)


def set_param(master, name, value):
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("ascii"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def run_segment(master, duration_s, target_fn, rate_hz=20.0):
    dt = 1.0 / rate_hz
    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        send_setpoint(master, *target_fn(time.monotonic()))
        time.sleep(dt)


def trapezoid_profile(distance_m, speed_m_s, accel_m_s2):
    speed_m_s = max(speed_m_s, 0.1)
    accel_m_s2 = max(accel_m_s2, 0.1)
    t_accel = speed_m_s / accel_m_s2
    d_accel = 0.5 * accel_m_s2 * t_accel * t_accel

    if 2.0 * d_accel > distance_m:
        t_accel = math.sqrt(distance_m / accel_m_s2)
        speed_m_s = accel_m_s2 * t_accel
        t_cruise = 0.0
        d_accel = 0.5 * accel_m_s2 * t_accel * t_accel
    else:
        t_cruise = (distance_m - 2.0 * d_accel) / speed_m_s

    total = 2.0 * t_accel + t_cruise

    def sample(elapsed_s):
        t = min(max(elapsed_s, 0.0), total)

        if t < t_accel:
            acc = accel_m_s2
            vel = acc * t
            pos = 0.5 * acc * t * t
        elif t < t_accel + t_cruise:
            acc = 0.0
            vel = speed_m_s
            pos = d_accel + speed_m_s * (t - t_accel)
        else:
            td = t - t_accel - t_cruise
            acc = -accel_m_s2
            vel = max(speed_m_s - accel_m_s2 * td, 0.0)
            pos = d_accel + speed_m_s * t_cruise + speed_m_s * td - 0.5 * accel_m_s2 * td * td

        if t >= total:
            return distance_m, 0.0, 0.0

        return min(pos, distance_m), vel, acc

    return total, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="udpin:127.0.0.1:14540")
    parser.add_argument("--alt", type=float, default=3.0)
    parser.add_argument("--forward", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--accel", type=float, default=0.6)
    parser.add_argument("--profile", choices=["smoothstep", "trapezoid"], default="smoothstep")
    parser.add_argument("--initial-hold", type=float, default=25.0)
    parser.add_argument("--final-hold", type=float, default=25.0)
    parser.add_argument("--rate", type=float, default=80.0)
    parser.add_argument("--param", action="append", default=[], help="NAME=VALUE")
    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.connect, autoreconnect=True)
    master.wait_heartbeat(timeout=30)
    print(f"heartbeat system={master.target_system} component={master.target_component}")

    for assignment in args.param:
        name, value = assignment.split("=", 1)
        print(f"set {name}={value}")
        set_param(master, name, float(value))
        time.sleep(0.15)

    # PX4 requires a short stream of setpoints before OFFBOARD can be entered.
    for _ in range(60):
        send_setpoint(master, 0.0, 0.0, -args.alt, 0.0)
        time.sleep(0.05)

    set_mode(master, "OFFBOARD")
    time.sleep(0.5)
    send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)

    print("hold at origin")
    run_segment(master, args.initial_hold, lambda _: (0.0, 0.0, -args.alt, 0.0), args.rate)

    print(f"move forward {args.forward} m")
    move_start = time.monotonic()
    if args.profile == "trapezoid":
        move_duration, sample_profile = trapezoid_profile(args.forward, args.speed, args.accel)

        def ramp_target(now):
            x, vx, ax = sample_profile(now - move_start)
            return (x, 0.0, -args.alt, 0.0, vx, 0.0, 0.0, ax, 0.0, 0.0, True)
    else:
        move_duration = max(args.forward / max(args.speed, 0.1), 1.0)

        def ramp_target(now):
            progress = min(max((now - move_start) / move_duration, 0.0), 1.0)
            # Smoothstep to avoid a hard step in position demand.
            s = progress * progress * (3.0 - 2.0 * progress)
            return (args.forward * s, 0.0, -args.alt, 0.0)

    run_segment(master, move_duration, ramp_target, args.rate)

    print("hold at forward point")
    run_segment(master, args.final_hold, lambda _: (args.forward, 0.0, -args.alt, 0.0), args.rate)

    print("land")
    set_mode(master, "LAND")
    time.sleep(20.0)
    send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0)


if __name__ == "__main__":
    main()

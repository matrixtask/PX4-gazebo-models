#!/usr/bin/env python3

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

from pymavlink import mavutil


R_EARTH_M = 6378137.0


@dataclass
class PositionMetric:
    name: str
    target_x_m: float
    target_y_m: float
    distance_m: float
    peak_speed_m_s: float
    time_to_speed_1_m_s: float | None
    time_to_speed_5_m_s: float | None
    time_to_speed_10_m_s: float | None
    overshoot_along_m: float
    cross_track_max_m: float
    target_crossings: int
    first_enter_time_s: float | None
    radius_exits_after_first_enter: int
    max_error_after_first_enter_m: float | None
    max_speed_after_first_enter_m_s: float | None
    final_error_m: float
    final_speed_m_s: float
    settle_time_s: float | None
    duration_s: float


@dataclass
class YawMetric:
    name: str
    target_yaw_deg: float
    commanded_delta_deg: float
    peak_yaw_rate_deg_s: float
    overshoot_deg: float
    target_crossings: int
    first_enter_time_s: float | None
    error_exits_after_first_enter: int
    max_error_after_first_enter_deg: float | None
    final_error_deg: float
    settle_time_s: float | None
    duration_s: float


def wrap_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_deg(value_deg, target_deg):
    return math.degrees(wrap_pi(math.radians(value_deg - target_deg)))


def heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        0,
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


INTEGER_PARAM_TYPES = {
    mavutil.mavlink.MAV_PARAM_TYPE_UINT8,
    mavutil.mavlink.MAV_PARAM_TYPE_INT8,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT16,
    mavutil.mavlink.MAV_PARAM_TYPE_INT16,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT32,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32,
}


def request_param_message(master, name, timeout_s=4.0):
    master.mav.param_request_read_send(master.target_system, master.target_component, name.encode("ascii"), -1)
    end_time = time.monotonic() + timeout_s

    while time.monotonic() < end_time:
        heartbeat(master)
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)

        if msg and msg.param_id.strip("\x00") == name:
            return msg

    return None


def set_param(master, name, value):
    current = request_param_message(master, name, timeout_s=1.5)
    param_type = (
        current.param_type
        if current is not None
        else mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    param_value = int(round(value)) if param_type in INTEGER_PARAM_TYPES else float(value)

    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("ascii"),
        param_value,
        param_type,
    )


def request_param(master, name, timeout_s=4.0):
    msg = request_param_message(master, name, timeout_s)
    return float(msg.param_value) if msg else None


def set_mode(master, mode_name):
    mapping = master.mode_mapping()

    if not mapping or mode_name not in mapping:
        raise RuntimeError(f"mode {mode_name} not available; got {mapping}")

    print(f"set mode {mode_name}", flush=True)
    master.set_mode_px4(mode_name, 0, 0)
    time.sleep(0.5)


def wait_message(master, msg_type, timeout_s=5.0):
    end_time = time.monotonic() + timeout_s

    while time.monotonic() < end_time:
        heartbeat(master)
        msg = master.recv_match(type=msg_type, blocking=True, timeout=0.5)

        if msg:
            return msg

    raise RuntimeError(f"timeout waiting for {msg_type}")


def global_offset(lat_deg, lon_deg, north_m, east_m):
    lat_rad = math.radians(lat_deg)
    return (
        lat_deg + math.degrees(north_m / R_EARTH_M),
        lon_deg + math.degrees(east_m / (R_EARTH_M * max(math.cos(lat_rad), 1e-6))),
    )


def latest_local_position(master, timeout_s=5.0):
    return wait_message(master, "LOCAL_POSITION_NED", timeout_s)


def latest_attitude(master, timeout_s=5.0):
    return wait_message(master, "ATTITUDE", timeout_s)


def takeoff(master, min_alt_m):
    set_mode(master, "TAKEOFF")
    send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
    print("armed/takeoff", flush=True)

    start = time.monotonic()
    last_print = -999.0
    last_position = None

    while time.monotonic() - start < 50.0:
        heartbeat(master)
        pos = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=0.5)

        if not pos:
            continue

        last_position = pos
        elapsed = time.monotonic() - start

        if elapsed - last_print >= 2.0:
            print(
                f"takeoff t={elapsed:.1f} x={pos.x:.1f} y={pos.y:.1f} z={pos.z:.1f} "
                f"vx={pos.vx:.2f} vy={pos.vy:.2f}",
                flush=True,
            )
            last_print = elapsed

        if pos.z < -min_alt_m:
            return

    last_z = None if last_position is None else last_position.z
    raise RuntimeError(f"takeoff did not reach altitude; z={last_z}")


def command_reposition(master, origin_lat, origin_lon, alt_msl_m, north_m, east_m, speed_m_s, yaw_rad=None):
    target_lat, target_lon = global_offset(origin_lat, origin_lon, north_m, east_m)
    yaw_param = float("nan") if yaw_rad is None else float(yaw_rad)
    send_command(
        master,
        mavutil.mavlink.MAV_CMD_DO_REPOSITION,
        float(speed_m_s),
        1.0,
        0.0,
        yaw_param,
        target_lat,
        target_lon,
        alt_msl_m,
    )


def run_position_segment(master, name, target_x, target_y, origin_lat, origin_lon, alt_msl_m,
                         origin_x, origin_y, speed_m_s, yaw_rad, max_time_s, settle_radius_m,
                         settle_speed_m_s, stable_time_s, print_period_s):
    start_pos = latest_local_position(master)
    start_x = float(start_pos.x)
    start_y = float(start_pos.y)
    path_dx = target_x - start_x
    path_dy = target_y - start_y
    path_len = math.hypot(path_dx, path_dy)
    ux = path_dx / path_len if path_len > 0.01 else 1.0
    uy = path_dy / path_len if path_len > 0.01 else 0.0

    command_reposition(
        master,
        origin_lat,
        origin_lon,
        alt_msl_m,
        target_x - origin_x,
        target_y - origin_y,
        speed_m_s,
        yaw_rad,
    )
    print(f"segment {name}: target=({target_x:.1f}, {target_y:.1f}) speed={speed_m_s}", flush=True)

    start = time.monotonic()
    last_print = -999.0
    peak_speed = 0.0
    time_to_speed_1 = None
    time_to_speed_5 = None
    time_to_speed_10 = None
    overshoot_along = -1e9
    cross_track_max = 0.0
    target_crossings = 0
    previous_along_sign = -1
    first_enter_time = None
    was_inside_radius = False
    radius_exits_after_first_enter = 0
    max_error_after_first_enter = None
    max_speed_after_first_enter = None
    settle_start = None
    settle_time = None
    final_dist = float("nan")
    final_speed = float("nan")

    while time.monotonic() - start < max_time_s:
        heartbeat(master)
        pos = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=0.5)

        if not pos:
            continue

        elapsed = time.monotonic() - start
        err_x = float(pos.x) - target_x
        err_y = float(pos.y) - target_y
        dist = math.hypot(err_x, err_y)
        speed = math.hypot(float(pos.vx), float(pos.vy))
        from_target_x = float(pos.x) - target_x
        from_target_y = float(pos.y) - target_y
        along_over = from_target_x * ux + from_target_y * uy
        cross = abs(from_target_x * (-uy) + from_target_y * ux)

        peak_speed = max(peak_speed, speed)

        if time_to_speed_1 is None and speed >= 1.0:
            time_to_speed_1 = elapsed

        if time_to_speed_5 is None and speed >= 5.0:
            time_to_speed_5 = elapsed

        if time_to_speed_10 is None and speed >= 10.0:
            time_to_speed_10 = elapsed

        overshoot_along = max(overshoot_along, along_over)
        cross_track_max = max(cross_track_max, cross)
        final_dist = dist
        final_speed = speed

        inside_radius = dist <= settle_radius_m

        if first_enter_time is not None:
            max_error_after_first_enter = max(max_error_after_first_enter or 0.0, dist)
            max_speed_after_first_enter = max(max_speed_after_first_enter or 0.0, speed)

        if first_enter_time is None and inside_radius:
            first_enter_time = elapsed
            max_error_after_first_enter = dist
            max_speed_after_first_enter = speed

        elif first_enter_time is not None and was_inside_radius and not inside_radius:
            radius_exits_after_first_enter += 1

        was_inside_radius = inside_radius

        if along_over > 0.25:
            along_sign = 1

        elif along_over < -0.25:
            along_sign = -1

        else:
            along_sign = previous_along_sign

        if along_sign != previous_along_sign:
            target_crossings += 1
            previous_along_sign = along_sign

        is_stable = inside_radius and speed <= settle_speed_m_s

        if is_stable:
            if settle_start is None:
                settle_start = elapsed

            if settle_time is None and elapsed - settle_start >= stable_time_s:
                settle_time = settle_start
                break

        else:
            settle_start = None

        if elapsed - last_print >= print_period_s:
            print(
                f"{name} t={elapsed:.1f} x={pos.x:.1f} y={pos.y:.1f} dist={dist:.2f} "
                f"vxy={speed:.2f} along_over={along_over:.2f}",
                flush=True,
            )
            last_print = elapsed

    duration = time.monotonic() - start
    metric = PositionMetric(
        name=name,
        target_x_m=float(target_x),
        target_y_m=float(target_y),
        distance_m=float(path_len),
        peak_speed_m_s=float(peak_speed),
        time_to_speed_1_m_s=None if time_to_speed_1 is None else float(time_to_speed_1),
        time_to_speed_5_m_s=None if time_to_speed_5 is None else float(time_to_speed_5),
        time_to_speed_10_m_s=None if time_to_speed_10 is None else float(time_to_speed_10),
        overshoot_along_m=float(max(0.0, overshoot_along)),
        cross_track_max_m=float(cross_track_max),
        target_crossings=int(target_crossings),
        first_enter_time_s=None if first_enter_time is None else float(first_enter_time),
        radius_exits_after_first_enter=int(radius_exits_after_first_enter),
        max_error_after_first_enter_m=None if max_error_after_first_enter is None else float(max_error_after_first_enter),
        max_speed_after_first_enter_m_s=None if max_speed_after_first_enter is None else float(max_speed_after_first_enter),
        final_error_m=float(final_dist),
        final_speed_m_s=float(final_speed),
        settle_time_s=None if settle_time is None else float(settle_time),
        duration_s=float(duration),
    )
    print(f"metric {json.dumps(asdict(metric), sort_keys=True)}", flush=True)
    return metric


def command_yaw(master, origin_lat, origin_lon, alt_msl_m, yaw_deg, speed_m_s, method):
    if method in ("reposition", "both"):
        gp = wait_message(master, "GLOBAL_POSITION_INT", 5.0)
        yaw_rad = wrap_pi(math.radians(yaw_deg))
        command_reposition(
            master,
            gp.lat / 1e7,
            gp.lon / 1e7,
            alt_msl_m,
            0.0,
            0.0,
            speed_m_s,
            yaw_rad,
        )

    if method in ("condition", "both"):
        send_command(
            master,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            yaw_deg,
            45.0,
            0.0,
            0.0,
        )


def run_yaw_segment(master, name, target_yaw_deg, origin_lat, origin_lon, alt_msl_m, speed_m_s,
                    method, max_time_s, settle_error_deg, settle_rate_deg_s,
                    stable_time_s, print_period_s):
    att0 = latest_attitude(master)
    yaw0_deg = math.degrees(att0.yaw)
    delta_deg = angle_diff_deg(target_yaw_deg, yaw0_deg)
    direction = 1.0 if delta_deg >= 0.0 else -1.0

    command_yaw(master, origin_lat, origin_lon, alt_msl_m, target_yaw_deg, speed_m_s, method)
    print(f"segment {name}: yaw_target={target_yaw_deg:.1f} deg method={method}", flush=True)

    start = time.monotonic()
    last_print = -999.0
    peak_rate = 0.0
    overshoot = -1e9
    target_crossings = 0
    initial_error_deg = angle_diff_deg(yaw0_deg, target_yaw_deg)

    if initial_error_deg > settle_error_deg:
        previous_error_sign = 1

    elif initial_error_deg < -settle_error_deg:
        previous_error_sign = -1

    else:
        previous_error_sign = 0
    first_enter_time = None
    was_inside_error = False
    error_exits_after_first_enter = 0
    max_error_after_first_enter = None
    settle_start = None
    settle_time = None
    final_error = float("nan")

    while time.monotonic() - start < max_time_s:
        heartbeat(master)
        att = master.recv_match(type="ATTITUDE", blocking=True, timeout=0.5)

        if not att:
            continue

        elapsed = time.monotonic() - start
        yaw_deg = math.degrees(att.yaw)
        rate_deg_s = math.degrees(att.yawspeed)
        error_deg = angle_diff_deg(yaw_deg, target_yaw_deg)
        progress_deg = direction * angle_diff_deg(yaw_deg, yaw0_deg)
        target_progress_deg = abs(delta_deg)
        along_over = progress_deg - target_progress_deg

        peak_rate = max(peak_rate, abs(rate_deg_s))
        overshoot = max(overshoot, along_over)
        final_error = error_deg

        abs_error = abs(error_deg)
        inside_error = abs_error <= settle_error_deg

        if first_enter_time is not None:
            max_error_after_first_enter = max(max_error_after_first_enter or 0.0, abs_error)

        if first_enter_time is None and inside_error:
            first_enter_time = elapsed
            max_error_after_first_enter = abs_error

        elif first_enter_time is not None and was_inside_error and not inside_error:
            error_exits_after_first_enter += 1

        was_inside_error = inside_error

        if error_deg > settle_error_deg:
            error_sign = 1

        elif error_deg < -settle_error_deg:
            error_sign = -1

        else:
            error_sign = previous_error_sign

        if error_sign != previous_error_sign:
            target_crossings += 1
            previous_error_sign = error_sign

        is_stable = inside_error and abs(rate_deg_s) <= settle_rate_deg_s

        if is_stable:
            if settle_start is None:
                settle_start = elapsed

            if settle_time is None and elapsed - settle_start >= stable_time_s:
                settle_time = settle_start
                break

        else:
            settle_start = None

        if elapsed - last_print >= print_period_s:
            print(
                f"{name} t={elapsed:.1f} yaw={yaw_deg:.1f} err={error_deg:.2f} "
                f"yrate={rate_deg_s:.2f} over={along_over:.2f}",
                flush=True,
            )
            last_print = elapsed

    duration = time.monotonic() - start
    metric = YawMetric(
        name=name,
        target_yaw_deg=float(target_yaw_deg),
        commanded_delta_deg=float(delta_deg),
        peak_yaw_rate_deg_s=float(peak_rate),
        overshoot_deg=float(max(0.0, overshoot)),
        target_crossings=int(target_crossings),
        first_enter_time_s=None if first_enter_time is None else float(first_enter_time),
        error_exits_after_first_enter=int(error_exits_after_first_enter),
        max_error_after_first_enter_deg=None if max_error_after_first_enter is None else float(max_error_after_first_enter),
        final_error_deg=float(final_error),
        settle_time_s=None if settle_time is None else float(settle_time),
        duration_s=float(duration),
    )
    print(f"metric {json.dumps(asdict(metric), sort_keys=True)}", flush=True)
    return metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="udpin:0.0.0.0:14550")
    parser.add_argument("--distance", type=float, default=30.0)
    parser.add_argument("--alt", type=float, default=7.0)
    parser.add_argument("--speed", type=float, default=-1.0)
    parser.add_argument("--max-time", type=float, default=55.0)
    parser.add_argument("--yaw-max-time", type=float, default=35.0)
    parser.add_argument("--settle-radius", type=float, default=1.0)
    parser.add_argument("--settle-speed", type=float, default=0.20)
    parser.add_argument("--settle-yaw-error", type=float, default=2.0)
    parser.add_argument("--settle-yaw-rate", type=float, default=3.0)
    parser.add_argument("--stable-time", type=float, default=2.0)
    parser.add_argument("--initial-hold", type=float, default=4.0)
    parser.add_argument("--between-hold", type=float, default=1.0)
    parser.add_argument("--print-period", type=float, default=3.0)
    parser.add_argument("--yaw-step-deg", type=float, default=90.0)
    parser.add_argument("--yaw-method", choices=["reposition", "condition", "both"], default="reposition")
    parser.add_argument("--metrics-out")
    parser.add_argument("--skip-position", action="store_true")
    parser.add_argument("--skip-yaw", action="store_true")
    parser.add_argument("--position-frame", choices=["local", "body"], default="local")
    parser.add_argument("--position-segments", help="Comma-separated position segment names to run")
    parser.add_argument("--yaw-segments", help="Comma-separated yaw segment names to run")
    parser.add_argument("--param", action="append", default=[], help="NAME=VALUE runtime override")
    parser.add_argument("--param-read", action="append", default=[
        "MPC_XY_P",
        "MPC_XY_CRUISE",
        "MPC_XY_VEL_MAX",
        "MPC_XY_VEL_P_ACC",
        "MPC_XY_VEL_I_ACC",
        "MPC_XY_VEL_D_ACC",
        "MPC_ACC_HOR",
        "MPC_JERK_AUTO",
        "MPC_XY_ERR_MAX",
        "MPC_XY_TRAJ_P",
        "MPC_YAWRAUTO_MAX",
        "MPC_YAWRAUTO_ACC",
        "MC_YAW_P",
        "MC_YAWRATE_P",
        "MC_YAWRATE_I",
        "MC_YAWRATE_D",
        "VT_FWD_THRUST_EN",
        "VT_FWD_THRUST_SC",
        "VT_PITCH_MIN",
        "VT_TETRA_FWD_EN",
        "VT_TETRA_FWD_SC",
        "VT_TETRA_FWD_MX",
        "VT_TETRA_FWD_DB",
        "VT_TETRA_FWD_SL",
        "CA_R12_SLEW",
    ])
    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.connect, autoreconnect=True)
    master.wait_heartbeat(timeout=30)
    print(f"heartbeat system={master.target_system} component={master.target_component}", flush=True)

    for assignment in args.param:
        name, value = assignment.split("=", 1)
        print(f"set {name}={value}", flush=True)
        set_param(master, name, float(value))
        time.sleep(0.12)

    params = {}
    for name in args.param_read:
        params[name] = request_param(master, name)
        print(f"param {name} {params[name]}", flush=True)

    takeoff(master, args.alt * 0.9)
    set_mode(master, "LOITER")
    time.sleep(args.initial_hold)

    gp = wait_message(master, "GLOBAL_POSITION_INT", 10.0)
    pos0 = latest_local_position(master)
    att0 = latest_attitude(master)
    origin_lat = gp.lat / 1e7
    origin_lon = gp.lon / 1e7
    alt_msl_m = gp.alt / 1000.0
    origin_x = float(pos0.x)
    origin_y = float(pos0.y)
    initial_yaw_deg = math.degrees(att0.yaw)

    print(
        f"origin local=({origin_x:.2f}, {origin_y:.2f}) global=({origin_lat:.7f}, {origin_lon:.7f}) "
        f"alt={alt_msl_m:.1f} yaw={initial_yaw_deg:.1f}",
        flush=True,
    )

    metrics = {"params": params, "position_frame": args.position_frame, "position": [], "yaw": []}
    if not args.skip_position:
        d = args.distance

        if args.position_frame == "body":
            yaw_rad = att0.yaw
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)
            right_x = -math.sin(yaw_rad)
            right_y = math.cos(yaw_rad)

        else:
            forward_x = 1.0
            forward_y = 0.0
            right_x = 0.0
            right_y = 1.0

        position_targets = [
            ("forward", origin_x + d * forward_x, origin_y + d * forward_y),
            ("return_after_forward", origin_x, origin_y),
            ("back", origin_x - d * forward_x, origin_y - d * forward_y),
            ("return_after_back", origin_x, origin_y),
            ("right", origin_x + d * right_x, origin_y + d * right_y),
            ("return_after_right", origin_x, origin_y),
            ("left", origin_x - d * right_x, origin_y - d * right_y),
            ("return_after_left", origin_x, origin_y),
        ]

        if args.position_segments:
            selected = {item.strip() for item in args.position_segments.split(",") if item.strip()}
            position_targets = [item for item in position_targets if item[0] in selected]

        for name, x, y in position_targets:
            metric = run_position_segment(
                master,
                name,
                x,
                y,
                origin_lat,
                origin_lon,
                alt_msl_m,
                origin_x,
                origin_y,
                args.speed,
                None,
                args.max_time,
                args.settle_radius,
                args.settle_speed,
                args.stable_time,
                args.print_period,
            )
            metrics["position"].append(asdict(metric))
            time.sleep(args.between_hold)

    if not args.skip_yaw:
        yaw_targets = [
            ("yaw_positive", initial_yaw_deg + args.yaw_step_deg),
            ("yaw_return_positive", initial_yaw_deg),
            ("yaw_negative", initial_yaw_deg - args.yaw_step_deg),
            ("yaw_return_negative", initial_yaw_deg),
        ]

        if args.yaw_segments:
            selected = {item.strip() for item in args.yaw_segments.split(",") if item.strip()}
            yaw_targets = [item for item in yaw_targets if item[0] in selected]

        for name, yaw in yaw_targets:
            metric = run_yaw_segment(
                master,
                name,
                yaw,
                origin_lat,
                origin_lon,
                alt_msl_m,
                args.speed,
                args.yaw_method,
                args.yaw_max_time,
                args.settle_yaw_error,
                args.settle_yaw_rate,
                args.stable_time,
                args.print_period,
            )
            metrics["yaw"].append(asdict(metric))
            time.sleep(args.between_hold)

    print("land", flush=True)
    set_mode(master, "LAND")
    time.sleep(12.0)
    send_command(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0)

    print("SUMMARY_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

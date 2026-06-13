#!/usr/bin/env python3

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from pymavlink import mavutil


@dataclass
class TrajectoryPlan:
    distance_m: float
    duration_s: float
    peak_speed_m_s: float
    peak_accel_m_s2: float
    peak_jerk_m_s3: float
    peak_snap_m_s4: float
    accel_energy_proxy: float
    drag_energy_proxy: float
    snap_energy_proxy: float
    objective: float


@dataclass
class SegmentMetric:
    name: str
    target_x_m: float
    target_y_m: float
    duration_s: float
    planned_duration_s: float
    peak_speed_m_s: float
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


def wrap_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


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

    while time.monotonic() - start < 55.0:
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


def c4_smoothstep(u):
    u = np.asarray(u, dtype=float)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    u6 = u5 * u
    u7 = u6 * u
    u8 = u7 * u
    u9 = u8 * u
    pos = 126.0 * u5 - 420.0 * u6 + 540.0 * u7 - 315.0 * u8 + 70.0 * u9
    vel = 630.0 * u4 - 2520.0 * u5 + 3780.0 * u6 - 2520.0 * u7 + 630.0 * u8
    accel = 2520.0 * u3 - 12600.0 * u4 + 22680.0 * u5 - 17640.0 * u6 + 5040.0 * u7
    jerk = 7560.0 * u2 - 50400.0 * u3 + 113400.0 * u4 - 105840.0 * u5 + 35280.0 * u6
    snap = 15120.0 * u - 151200.0 * u2 + 453600.0 * u3 - 529200.0 * u4 + 211680.0 * u5
    return pos, vel, accel, jerk, snap


def build_trajectory_plan(distance_m, max_speed_m_s, max_accel_m_s2, max_jerk_m_s3,
                          max_snap_m_s4, duration_min_s, duration_max_s,
                          time_weight, accel_energy_weight, drag_energy_weight,
                          snap_weight, duration_scale):
    samples = np.linspace(0.0, 1.0, 1201)
    _, vel_n, accel_n, jerk_n, snap_n = c4_smoothstep(samples)
    peak_v_n = float(np.nanmax(np.abs(vel_n)))
    peak_a_n = float(np.nanmax(np.abs(accel_n)))
    peak_j_n = float(np.nanmax(np.abs(jerk_n)))
    peak_s_n = float(np.nanmax(np.abs(snap_n)))
    accel_n_sq_int = float(np.trapezoid(accel_n * accel_n, samples))
    drag_n_int = float(np.trapezoid(np.abs(vel_n) ** 3, samples))
    snap_n_sq_int = float(np.trapezoid(snap_n * snap_n, samples))

    min_t = max(0.1, duration_min_s)

    if max_speed_m_s > 0.0:
        min_t = max(min_t, distance_m * peak_v_n / max_speed_m_s)

    if max_accel_m_s2 > 0.0:
        min_t = max(min_t, math.sqrt(distance_m * peak_a_n / max_accel_m_s2))

    if max_jerk_m_s3 > 0.0:
        min_t = max(min_t, (distance_m * peak_j_n / max_jerk_m_s3) ** (1.0 / 3.0))

    if max_snap_m_s4 > 0.0:
        min_t = max(min_t, (distance_m * peak_s_n / max_snap_m_s4) ** 0.25)

    max_t = max(duration_max_s, min_t)
    durations = np.linspace(min_t, max_t, 240)
    def make_plan(duration_s):
        peak_speed = distance_m * peak_v_n / duration_s
        peak_accel = distance_m * peak_a_n / (duration_s ** 2)
        peak_jerk = distance_m * peak_j_n / (duration_s ** 3)
        peak_snap = distance_m * peak_s_n / (duration_s ** 4)
        accel_energy = (distance_m ** 2) * accel_n_sq_int / (duration_s ** 3)
        drag_energy = (distance_m ** 3) * drag_n_int / (duration_s ** 2)
        snap_energy = (distance_m ** 2) * snap_n_sq_int / (duration_s ** 7)
        objective = (
            time_weight * duration_s
            + accel_energy_weight * accel_energy
            + drag_energy_weight * drag_energy
            + snap_weight * snap_energy
        )
        return TrajectoryPlan(
            distance_m=float(distance_m),
            duration_s=float(duration_s),
            peak_speed_m_s=float(peak_speed),
            peak_accel_m_s2=float(peak_accel),
            peak_jerk_m_s3=float(peak_jerk),
            peak_snap_m_s4=float(peak_snap),
            accel_energy_proxy=float(accel_energy),
            drag_energy_proxy=float(drag_energy),
            snap_energy_proxy=float(snap_energy),
            objective=float(objective),
        )

    best = None

    for duration_s in durations:
        plan = make_plan(duration_s)

        if best is None or plan.objective < best.objective:
            best = plan

    if duration_scale > 1.0:
        best = make_plan(best.duration_s * duration_scale)

    return best


def position_target_type_mask(feedforward):
    type_mask = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

    if feedforward == "position":
        type_mask |= (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )

    elif feedforward == "velocity":
        type_mask |= (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        )

    return type_mask


def send_position_target(master, boot_time, x, y, z, vx, vy, vz, ax, ay, az, yaw, feedforward):
    type_mask = position_target_type_mask(feedforward)
    time_boot_ms = int((time.monotonic() - boot_time) * 1000.0) & 0xFFFFFFFF
    master.mav.set_position_target_local_ned_send(
        time_boot_ms,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        float(x),
        float(y),
        float(z),
        float(vx),
        float(vy),
        float(vz),
        float(ax),
        float(ay),
        float(az),
        float(yaw),
        0.0,
    )


def stream_hold(master, boot_time, x, y, z, yaw, duration_s, rate_hz, feedforward):
    period_s = 1.0 / rate_hz
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        heartbeat(master)
        send_position_target(master, boot_time, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw, feedforward)
        time.sleep(period_s)


def run_smooth_segment(master, boot_time, name, target_x, target_y, target_z, yaw,
                       plan, rate_hz, min_hold_s, max_hold_s, settle_radius_m,
                       settle_speed_m_s, stable_time_s, print_period_s, feedforward):
    start_pos = latest_local_position(master)
    start_x = float(start_pos.x)
    start_y = float(start_pos.y)
    start_z = float(start_pos.z)
    path_dx = target_x - start_x
    path_dy = target_y - start_y
    path_dz = target_z - start_z
    path_len = math.sqrt(path_dx * path_dx + path_dy * path_dy + path_dz * path_dz)
    ux = path_dx / path_len if path_len > 0.01 else 1.0
    uy = path_dy / path_len if path_len > 0.01 else 0.0
    uz = path_dz / path_len if path_len > 0.01 else 0.0
    period_s = 1.0 / rate_hz
    duration_s = max(plan.duration_s, period_s)

    print(
        f"segment {name}: target=({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) "
        f"T={duration_s:.2f}s peak_v={plan.peak_speed_m_s:.2f}",
        flush=True,
    )

    start = time.monotonic()
    last_print = -999.0
    peak_speed = 0.0
    overshoot_along = -1e9
    cross_track_max = 0.0
    target_crossings = 0
    previous_along_sign = -1
    first_enter_time = None
    was_inside_radius = False
    radius_exits_after_first_enter = 0
    max_error_after_first_enter = None
    max_speed_after_first_enter = None
    final_dist = float("nan")
    final_speed = float("nan")
    settle_start = None
    settle_time = None

    while True:
        elapsed = time.monotonic() - start
        u = min(1.0, elapsed / duration_s)
        s, ds, dds, _, _ = c4_smoothstep(np.array([u]))
        distance = path_len
        pos_scale = distance * float(s[0])
        vel_scale = distance * float(ds[0]) / duration_s
        acc_scale = distance * float(dds[0]) / (duration_s * duration_s)
        sp_x = start_x + ux * pos_scale
        sp_y = start_y + uy * pos_scale
        sp_z = start_z + uz * pos_scale
        sp_vx = ux * vel_scale
        sp_vy = uy * vel_scale
        sp_vz = uz * vel_scale
        sp_ax = ux * acc_scale
        sp_ay = uy * acc_scale
        sp_az = uz * acc_scale

        heartbeat(master)
        send_position_target(master, boot_time, sp_x, sp_y, sp_z, sp_vx, sp_vy, sp_vz, sp_ax, sp_ay, sp_az, yaw, feedforward)

        while True:
            pos = master.recv_match(type="LOCAL_POSITION_NED", blocking=False)

            if not pos:
                break

            err_x = float(pos.x) - target_x
            err_y = float(pos.y) - target_y
            err_z = float(pos.z) - target_z
            dist_xy = math.hypot(err_x, err_y)
            dist = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
            speed = math.hypot(float(pos.vx), float(pos.vy))
            from_target_x = float(pos.x) - target_x
            from_target_y = float(pos.y) - target_y
            along_over = from_target_x * ux + from_target_y * uy
            cross = abs(from_target_x * (-uy) + from_target_y * ux)

            peak_speed = max(peak_speed, speed)
            overshoot_along = max(overshoot_along, along_over)
            cross_track_max = max(cross_track_max, cross)
            final_dist = dist
            final_speed = speed

            inside_radius = dist_xy <= settle_radius_m

            if first_enter_time is not None:
                max_error_after_first_enter = max(max_error_after_first_enter or 0.0, dist_xy)
                max_speed_after_first_enter = max(max_speed_after_first_enter or 0.0, speed)

            if first_enter_time is None and inside_radius:
                first_enter_time = elapsed
                max_error_after_first_enter = dist_xy
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

            if elapsed - last_print >= print_period_s:
                print(
                    f"{name} t={elapsed:.1f} x={pos.x:.1f} y={pos.y:.1f} "
                    f"dist={dist_xy:.2f} vxy={speed:.2f} over={along_over:.2f}",
                    flush=True,
                )
                last_print = elapsed

            if elapsed >= duration_s:
                is_stable = inside_radius and speed <= settle_speed_m_s

                if is_stable:
                    if settle_start is None:
                        settle_start = elapsed

                    if settle_time is None and elapsed - settle_start >= stable_time_s:
                        settle_time = settle_start

                else:
                    settle_start = None

        if elapsed >= duration_s + min_hold_s and settle_time is not None:
            break

        if elapsed >= duration_s + max_hold_s:
            break

        time.sleep(period_s)

    metric = SegmentMetric(
        name=name,
        target_x_m=float(target_x),
        target_y_m=float(target_y),
        duration_s=float(time.monotonic() - start),
        planned_duration_s=float(duration_s),
        peak_speed_m_s=float(peak_speed),
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
    )
    print(f"metric {json.dumps(asdict(metric), sort_keys=True)}", flush=True)
    return metric


def body_segment_offsets(segment_names, distance_m, yaw_rad):
    forward = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (-math.sin(yaw_rad), math.cos(yaw_rad))
    offsets = []

    for name in segment_names:
        if name == "forward":
            offsets.append((name, forward[0] * distance_m, forward[1] * distance_m))
        elif name == "back":
            offsets.append((name, -forward[0] * distance_m, -forward[1] * distance_m))
        elif name == "right":
            offsets.append((name, right[0] * distance_m, right[1] * distance_m))
        elif name == "left":
            offsets.append((name, -right[0] * distance_m, -right[1] * distance_m))
        elif name.startswith("return"):
            offsets.append((name, 0.0, 0.0))
        else:
            raise ValueError(f"unknown segment {name}")

    return offsets


def main():
    parser = argparse.ArgumentParser(
        description="Fly a C4 snap-continuous Offboard position mission for the Mk-7 SITL model."
    )
    parser.add_argument("--connect", default="udpin:0.0.0.0:14550")
    parser.add_argument("--distance", type=float, default=200.0)
    parser.add_argument("--alt", type=float, default=7.0)
    parser.add_argument("--segments", default="forward,return_after_forward")
    parser.add_argument("--max-speed", type=float, default=20.0)
    parser.add_argument("--max-accel", type=float, default=5.05)
    parser.add_argument("--max-jerk", type=float, default=1.38)
    parser.add_argument("--max-snap", type=float, default=0.0)
    parser.add_argument("--duration-min", type=float, default=0.1)
    parser.add_argument("--duration-max", type=float, default=90.0)
    parser.add_argument("--duration-scale", type=float, default=1.25)
    parser.add_argument("--time-weight", type=float, default=1.0)
    parser.add_argument("--accel-energy-weight", type=float, default=0.01)
    parser.add_argument("--drag-energy-weight", type=float, default=0.0002)
    parser.add_argument("--snap-weight", type=float, default=0.000001)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--feedforward", choices=["position", "velocity", "full"], default="position")
    parser.add_argument("--prestream", type=float, default=2.5)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--max-hold", type=float, default=12.0)
    parser.add_argument("--settle-radius", type=float, default=1.0)
    parser.add_argument("--settle-speed", type=float, default=0.20)
    parser.add_argument("--stable-time", type=float, default=1.5)
    parser.add_argument("--print-period", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    plan = build_trajectory_plan(
        distance_m=args.distance,
        max_speed_m_s=args.max_speed,
        max_accel_m_s2=args.max_accel,
        max_jerk_m_s3=args.max_jerk,
        max_snap_m_s4=args.max_snap,
        duration_min_s=args.duration_min,
        duration_max_s=args.duration_max,
        time_weight=args.time_weight,
        accel_energy_weight=args.accel_energy_weight,
        drag_energy_weight=args.drag_energy_weight,
        snap_weight=args.snap_weight,
        duration_scale=args.duration_scale,
    )
    segment_names = [item.strip() for item in args.segments.split(",") if item.strip()]
    result = {
        "plan": asdict(plan),
        "feedforward": args.feedforward,
        "profile": "c4_smoothstep_zero_v_a_j_snap_at_endpoints",
        "segments": [],
    }

    if args.dry_run:
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")

        return

    master = mavutil.mavlink_connection(args.connect)
    master.wait_heartbeat(timeout=10)
    print(f"heartbeat system={master.target_system} component={master.target_component}", flush=True)

    boot_time = time.monotonic()
    takeoff(master, args.alt)
    set_mode(master, "LOITER")
    time.sleep(2.0)

    origin = latest_local_position(master)
    att = latest_attitude(master)
    origin_x = float(origin.x)
    origin_y = float(origin.y)
    target_z = -float(args.alt)
    yaw = float(att.yaw)
    print(
        f"origin local=({origin_x:.2f}, {origin_y:.2f}, {origin.z:.2f}) yaw={math.degrees(yaw):.1f}",
        flush=True,
    )

    stream_hold(master, boot_time, origin_x, origin_y, target_z, yaw, args.prestream, args.rate, args.feedforward)
    set_mode(master, "OFFBOARD")

    offsets = body_segment_offsets(segment_names, args.distance, yaw)

    for name, dx, dy in offsets:
        target_x = origin_x + dx
        target_y = origin_y + dy
        metric = run_smooth_segment(
            master,
            boot_time,
            name,
            target_x,
            target_y,
            target_z,
            yaw,
            plan,
            args.rate,
            args.hold,
            args.max_hold,
            args.settle_radius,
            args.settle_speed,
            args.stable_time,
            args.print_period,
            args.feedforward,
        )
        result["segments"].append(asdict(metric))

    print("land", flush=True)
    set_mode(master, "LAND")
    text = json.dumps(result, sort_keys=True)
    print(f"SUMMARY_JSON={text}", flush=True)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

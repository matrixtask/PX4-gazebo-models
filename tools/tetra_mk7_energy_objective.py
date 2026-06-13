#!/usr/bin/env python3

import argparse
import json

import numpy as np
from pyulog import ULog


def dataset(ulog, name, multi_id=0):
    for item in ulog.data_list:
        if item.name == name and item.multi_id == multi_id:
            return item.data
    return None


def time_s(data, start_us):
    return (data["timestamp"].astype(float) - float(start_us)) * 1e-6


def finite_diff(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)

    dt = np.diff(times)
    dv = np.diff(values)
    out = np.zeros_like(values)
    valid = dt > 1e-6
    diff = np.zeros_like(dt)
    diff[valid] = dv[valid] / dt[valid]
    out[1:] = diff
    out[0] = diff[0] if diff.size else 0.0
    return out


def integrate(times, values):
    if len(times) < 2:
        return 0.0
    return float(np.trapezoid(values, times))


def interp_safe(source_t, source_v, target_t):
    mask = np.isfinite(source_t) & np.isfinite(source_v)
    if np.count_nonzero(mask) < 2:
        return np.full_like(target_t, np.nan, dtype=float)
    return np.interp(target_t, source_t[mask], source_v[mask])


def load_segment_penalties(metrics_path):
    if not metrics_path:
        return {
            "segment_count": 0,
            "segment_overshoot_m": 0.0,
            "segment_cross_track_m": 0.0,
            "segment_final_error_m": 0.0,
            "segment_post_enter_speed_m_s": 0.0,
            "segment_time_to_speed_1_m_s": 0.0,
            "segment_time_to_speed_5_m_s": 0.0,
            "segment_time_to_speed_10_m_s": 0.0,
            "segment_target_crossings": 0,
            "segment_radius_exits": 0,
        }

    with open(metrics_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    segments = data.get("position", data.get("segments", []))

    if not segments:
        return {
            "segment_count": 0,
            "segment_overshoot_m": 0.0,
            "segment_cross_track_m": 0.0,
            "segment_final_error_m": 0.0,
            "segment_post_enter_speed_m_s": 0.0,
            "segment_time_to_speed_1_m_s": 0.0,
            "segment_time_to_speed_5_m_s": 0.0,
            "segment_time_to_speed_10_m_s": 0.0,
            "segment_target_crossings": 0,
            "segment_radius_exits": 0,
        }

    def max_float(key):
        values = [float(item.get(key) or 0.0) for item in segments]
        return max(values) if values else 0.0

    return {
        "segment_count": len(segments),
        "segment_overshoot_m": max_float("overshoot_along_m"),
        "segment_cross_track_m": max_float("cross_track_max_m"),
        "segment_final_error_m": max_float("final_error_m"),
        "segment_post_enter_speed_m_s": max_float("max_speed_after_first_enter_m_s"),
        "segment_time_to_speed_1_m_s": max_float("time_to_speed_1_m_s"),
        "segment_time_to_speed_5_m_s": max_float("time_to_speed_5_m_s"),
        "segment_time_to_speed_10_m_s": max_float("time_to_speed_10_m_s"),
        "segment_target_crossings": sum(int(item.get("target_crossings") or 0) for item in segments),
        "segment_radius_exits": sum(int(item.get("radius_exits_after_first_enter") or 0) for item in segments),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Mk-7 mission energy and smoothness objective from a ULog."
    )
    parser.add_argument("ulg")
    parser.add_argument("--time-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--snap-weight", type=float, default=0.01)
    parser.add_argument("--snap-rate-weight", type=float, default=0.0002)
    parser.add_argument("--snap-jump-weight", type=float, default=0.5)
    parser.add_argument("--motor-slew-weight", type=float, default=5.0)
    parser.add_argument("--terminal-weight", type=float, default=10.0)
    parser.add_argument("--overshoot-weight", type=float, default=20.0)
    parser.add_argument("--metrics-json")
    parser.add_argument("--segment-overshoot-weight", type=float, default=100.0)
    parser.add_argument("--segment-crossing-weight", type=float, default=200.0)
    parser.add_argument("--segment-exit-weight", type=float, default=100.0)
    parser.add_argument("--segment-cross-track-weight", type=float, default=2.0)
    parser.add_argument("--segment-final-weight", type=float, default=10.0)
    parser.add_argument("--segment-post-enter-speed-weight", type=float, default=20.0)
    parser.add_argument("--segment-speed-5-time-weight", type=float, default=0.2)
    parser.add_argument("--segment-speed-10-time-weight", type=float, default=1.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    ulog = ULog(args.ulg)
    local_pos = dataset(ulog, "vehicle_local_position")
    motors = dataset(ulog, "actuator_motors")
    traj = dataset(ulog, "trajectory_setpoint")

    if local_pos is None:
        raise SystemExit("vehicle_local_position is required")

    if motors is None:
        raise SystemExit("actuator_motors is required")

    start_us = ulog.start_timestamp
    t_pos = time_s(local_pos, start_us)
    z = local_pos["z"].astype(float)
    vxy = np.hypot(local_pos["vx"].astype(float), local_pos["vy"].astype(float))
    airborne = z < -1.0

    if np.count_nonzero(airborne) > 2:
        t0 = float(t_pos[airborne][0])
        t1 = float(t_pos[airborne][-1])
    else:
        t0 = float(t_pos[0])
        t1 = float(t_pos[-1])

    mission_mask_pos = (t_pos >= t0) & (t_pos <= t1)
    duration_s = max(0.0, t1 - t0)

    t_mot = time_s(motors, start_us)
    mission_mask_mot = (t_mot >= t0) & (t_mot <= t1)
    t_energy = t_mot[mission_mask_mot]

    controls = []
    for i in range(16):
        key = f"control[{i}]"
        if key in motors:
            values = motors[key].astype(float)[mission_mask_mot]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            controls.append(np.clip(values, 0.0, 1.0))

    if controls:
        control_matrix = np.vstack(controls)
        # In Gazebo this output is closer to rotor speed command than thrust.
        # Rotor power is approximately proportional to speed^3, while u^2 is a
        # useful allocator effort proxy. Report both and use cubic energy in J.
        effort_l2 = np.sum(control_matrix * control_matrix, axis=0)
        power_proxy = np.sum(control_matrix ** 3, axis=0)
        energy_l2 = integrate(t_energy, effort_l2)
        energy_cubic = integrate(t_energy, power_proxy)
        lift_energy = integrate(t_energy, np.sum(control_matrix[:12] ** 3, axis=0)) if control_matrix.shape[0] >= 12 else 0.0
        pusher_energy = integrate(t_energy, control_matrix[12] ** 3) if control_matrix.shape[0] >= 13 else 0.0
        motor_delta = np.linalg.norm(np.diff(control_matrix, axis=1), axis=0) if control_matrix.shape[1] > 1 else np.array([])
        motor_slew_proxy = float(np.nanmax(motor_delta)) if motor_delta.size else 0.0
    else:
        energy_l2 = 0.0
        energy_cubic = 0.0
        lift_energy = 0.0
        pusher_energy = 0.0
        motor_slew_proxy = 0.0

    snap_integral = 0.0
    snap_peak = 0.0
    snap_rate_integral = 0.0
    snap_rate_peak = 0.0
    snap_jump_peak = 0.0

    if traj is not None and "jerk[0]" in traj:
        t_traj = time_s(traj, start_us)
        mission_mask_traj = (t_traj >= t0) & (t_traj <= t1)
        tt = t_traj[mission_mask_traj]

        if tt.size > 2:
            snap_sq = np.zeros_like(tt)
            snap_rate_sq = np.zeros_like(tt)

            for axis in range(3):
                jerk = traj[f"jerk[{axis}]"].astype(float)[mission_mask_traj]
                jerk = np.nan_to_num(jerk, nan=0.0, posinf=0.0, neginf=0.0)
                snap = finite_diff(jerk, tt)
                snap_rate = finite_diff(snap, tt)
                snap_sq += snap * snap
                snap_rate_sq += snap_rate * snap_rate
                snap_peak = max(snap_peak, float(np.nanmax(np.abs(snap))))
                snap_rate_peak = max(snap_rate_peak, float(np.nanmax(np.abs(snap_rate))))

                if snap.size > 1:
                    snap_jump_peak = max(snap_jump_peak, float(np.nanmax(np.abs(np.diff(snap)))))

            snap_integral = integrate(tt, snap_sq)
            snap_rate_integral = integrate(tt, snap_rate_sq)

    terminal_error_m = 0.0
    overshoot_m = 0.0

    if traj is not None and "position[0]" in traj:
        t_traj = time_s(traj, start_us)
        target_x = traj["position[0]"].astype(float)
        target_y = traj["position[1]"].astype(float)
        finite_target = np.isfinite(target_x) & np.isfinite(target_y)

        if np.count_nonzero(finite_target) > 2:
            tx = interp_safe(t_traj[finite_target], target_x[finite_target], t_pos)
            ty = interp_safe(t_traj[finite_target], target_y[finite_target], t_pos)
            err = np.hypot(local_pos["x"].astype(float) - tx, local_pos["y"].astype(float) - ty)
            err = err[mission_mask_pos & np.isfinite(err)]

            if err.size:
                terminal_error_m = float(err[-1])
                overshoot_m = float(max(0.0, np.nanmax(err[-max(10, min(500, err.size // 5)):]) - err[-1]))

    segment_penalties = load_segment_penalties(args.metrics_json)
    segment_penalty = (
        args.segment_overshoot_weight * segment_penalties["segment_overshoot_m"]
        + args.segment_crossing_weight * segment_penalties["segment_target_crossings"]
        + args.segment_exit_weight * segment_penalties["segment_radius_exits"]
        + args.segment_cross_track_weight * segment_penalties["segment_cross_track_m"]
        + args.segment_final_weight * segment_penalties["segment_final_error_m"]
        + args.segment_post_enter_speed_weight * segment_penalties["segment_post_enter_speed_m_s"]
        + args.segment_speed_5_time_weight * segment_penalties["segment_time_to_speed_5_m_s"]
        + args.segment_speed_10_time_weight * segment_penalties["segment_time_to_speed_10_m_s"]
    )

    objective = (
        args.energy_weight * energy_cubic
        + args.snap_weight * snap_integral
        + args.snap_rate_weight * snap_rate_integral
        + args.snap_jump_weight * snap_jump_peak
        + args.motor_slew_weight * motor_slew_proxy
        + args.time_weight * duration_s
        + args.terminal_weight * terminal_error_m
        + args.overshoot_weight * overshoot_m
        + segment_penalty
    )

    result = {
        "ulg": args.ulg,
        "time_s": [t0, t1],
        "duration_s": duration_s,
        "energy_proxy_u2_s": energy_l2,
        "energy_proxy_u3_s": energy_cubic,
        "lift_energy_proxy_u3_s": lift_energy,
        "pusher_energy_proxy_u3_s": pusher_energy,
        "snap_proxy_integral": snap_integral,
        "snap_proxy_peak": snap_peak,
        "snap_rate_proxy_integral": snap_rate_integral,
        "snap_rate_proxy_peak": snap_rate_peak,
        "snap_jump_proxy_peak": snap_jump_peak,
        "motor_slew_proxy_max_delta": motor_slew_proxy,
        "terminal_error_m": terminal_error_m,
        "terminal_overshoot_proxy_m": overshoot_m,
        "segment_penalty": segment_penalty,
        "segment_metrics": segment_penalties,
        "vxy_peak_m_s": float(np.nanmax(vxy[mission_mask_pos])) if np.count_nonzero(mission_mask_pos) else 0.0,
        "objective": objective,
        "weights": {
            "energy": args.energy_weight,
            "motor_slew": args.motor_slew_weight,
            "snap": args.snap_weight,
            "snap_rate": args.snap_rate_weight,
            "snap_jump": args.snap_jump_weight,
            "time": args.time_weight,
            "terminal": args.terminal_weight,
            "overshoot": args.overshoot_weight,
            "segment_overshoot": args.segment_overshoot_weight,
            "segment_crossing": args.segment_crossing_weight,
            "segment_exit": args.segment_exit_weight,
            "segment_cross_track": args.segment_cross_track_weight,
            "segment_final": args.segment_final_weight,
            "segment_post_enter_speed": args.segment_post_enter_speed_weight,
            "segment_speed_5_time": args.segment_speed_5_time_weight,
            "segment_speed_10_time": args.segment_speed_10_time_weight,
        },
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()

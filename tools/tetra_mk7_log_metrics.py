#!/usr/bin/env python3

import argparse
import json
import math

import numpy as np
from pyulog import ULog


def dataset(ulog, name):
    for item in ulog.data_list:
        if item.name == name and item.multi_id == 0:
            return item.data
    return None


def interp(source_t, source_v, target_t):
    return np.interp(target_t, source_t, source_v)


def rms(values):
    values = np.asarray(values)
    if values.size == 0:
        return None
    return float(np.sqrt(np.nanmean(values * values)))


def max_abs(values):
    values = np.asarray(values)
    if values.size == 0:
        return None
    return float(np.nanmax(np.abs(values)))


def quat_to_euler_deg(q0, q1, q2, q3):
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def segment_metrics(name, mask, pos, sp, att_interp=None):
    out = {"samples": int(np.count_nonzero(mask))}
    if out["samples"] == 0:
        return out

    x = pos["x"][mask]
    y = pos["y"][mask]
    z = pos["z"][mask]
    vx = pos["vx"][mask]
    vy = pos["vy"][mask]
    vz = pos["vz"][mask]
    xsp = sp["x"][mask]
    ysp = sp["y"][mask]
    zsp = sp["z"][mask]

    ex = x - xsp
    ey = y - ysp
    ez = z - zsp
    xy = np.sqrt(ex * ex + ey * ey)
    vxy = np.sqrt(vx * vx + vy * vy)

    out.update(
        {
            "time_s": [float(pos["t"][mask][0]), float(pos["t"][mask][-1])],
            "xy_error_rms_m": rms(xy),
            "xy_error_max_m": float(np.nanmax(xy)),
            "z_error_rms_m": rms(ez),
            "z_error_max_m": max_abs(ez),
            "xy_velocity_rms_m_s": rms(vxy),
            "xy_velocity_max_m_s": float(np.nanmax(vxy)),
            "z_velocity_rms_m_s": rms(vz),
            "x_final_m": float(x[-1]),
            "y_final_m": float(y[-1]),
            "z_final_m": float(z[-1]),
            "x_sp_final_m": float(xsp[-1]),
            "y_sp_final_m": float(ysp[-1]),
            "z_sp_final_m": float(zsp[-1]),
        }
    )

    if name == "forward_hold":
        out["x_overshoot_m"] = float(np.nanmax(x - xsp[-1]))
        out["x_undershoot_m"] = float(np.nanmin(x - xsp[-1]))

    if att_interp is not None:
        roll, pitch, yaw = att_interp
        out.update(
            {
                "roll_rms_deg": rms(roll[mask]),
                "roll_max_deg": max_abs(roll[mask]),
                "pitch_rms_deg": rms(pitch[mask]),
                "pitch_max_deg": max_abs(pitch[mask]),
                "yaw_span_deg": float(np.nanmax(yaw[mask]) - np.nanmin(yaw[mask])),
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ulg")
    args = parser.parse_args()

    ulog = ULog(args.ulg)
    local_pos = dataset(ulog, "vehicle_local_position")
    traj = dataset(ulog, "vehicle_local_position_setpoint")
    attitude = dataset(ulog, "vehicle_attitude")
    status = dataset(ulog, "vehicle_status")

    if local_pos is None or traj is None:
        raise SystemExit("vehicle_local_position and vehicle_local_position_setpoint are required")

    t = (local_pos["timestamp"].astype(float) - local_pos["timestamp"][0]) * 1e-6
    pos = {
        "t": t,
        "x": local_pos["x"].astype(float),
        "y": local_pos["y"].astype(float),
        "z": local_pos["z"].astype(float),
        "vx": local_pos["vx"].astype(float),
        "vy": local_pos["vy"].astype(float),
        "vz": local_pos["vz"].astype(float),
    }

    ts = (traj["timestamp"].astype(float) - local_pos["timestamp"][0]) * 1e-6
    sp = {
        "x": interp(ts, traj["x"].astype(float), t),
        "y": interp(ts, traj["y"].astype(float), t),
        "z": interp(ts, traj["z"].astype(float), t),
    }

    offboard = np.ones_like(t, dtype=bool)
    if status is not None:
        ts_status = (status["timestamp"].astype(float) - local_pos["timestamp"][0]) * 1e-6
        status_index = np.searchsorted(ts_status, t, side="right") - 1
        status_index = np.clip(status_index, 0, len(ts_status) - 1)
        offboard = status["nav_state"][status_index] == 14

    finite_sp = np.isfinite(sp["x"]) & np.isfinite(sp["y"]) & np.isfinite(sp["z"]) & (sp["z"] < -1.0)
    airborne = finite_sp & offboard & (pos["z"] < -0.8)
    if np.count_nonzero(airborne) == 0:
        raise SystemExit("no airborne offboard segment found")

    t_air = t[airborne]
    start = float(t_air[0])
    end = float(t_air[-1])
    duration = end - start

    hover_mask = airborne & (t > start + 14.0) & (t < start + min(30.0, duration * 0.35)) & (np.abs(sp["x"]) < 0.2)
    forward_level = airborne & (sp["x"] > 25.0) & (sp["z"] < -2.5)
    if np.count_nonzero(forward_level) > 0:
        forward_level_end = float(t[forward_level][-1])
    else:
        forward_level_end = end
    forward_hold_mask = forward_level & (t > forward_level_end - 18.0)

    att_interp = None
    if attitude is not None:
        ta = (attitude["timestamp"].astype(float) - local_pos["timestamp"][0]) * 1e-6
        q0 = interp(ta, attitude["q[0]"].astype(float), t)
        q1 = interp(ta, attitude["q[1]"].astype(float), t)
        q2 = interp(ta, attitude["q[2]"].astype(float), t)
        q3 = interp(ta, attitude["q[3]"].astype(float), t)
        att_interp = quat_to_euler_deg(q0, q1, q2, q3)

    moving_mask = airborne & (sp["x"] > 1.0) & (sp["x"] < 29.0) & (sp["z"] < -2.5)

    result = {
        "log": args.ulg,
        "airborne_time_s": [start, end],
        "hover": segment_metrics("hover", hover_mask, pos, sp, att_interp),
        "forward_move": segment_metrics("forward_move", moving_mask, pos, sp, att_interp),
        "forward_hold": segment_metrics("forward_hold", forward_hold_mask, pos, sp, att_interp),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

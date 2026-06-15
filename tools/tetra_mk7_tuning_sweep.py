#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
AXIS_TEST = REPO_ROOT / "Tools/simulation/gz/tools/tetra_mk7_axis_response_test.py"
ENERGY_OBJECTIVE = REPO_ROOT / "Tools/simulation/gz/tools/tetra_mk7_energy_objective.py"
STOP_SITL = REPO_ROOT / "Tools/simulation/gz/tools/stop_tetra_mk7_sitl.sh"

COMMON_SAFE_PARAMS = {
    "VT_FWD_THRUST_EN": 0,
    "VT_FWD_THRUST_SC": 0,
    "VT_TETRA_FWD_EN": 0,
    "VT_TETRA_FWD_SC": 0,
    "VT_TETRA_FWD_MX": 0,
}

BUILTIN_CANDIDATES = [
    {
        "name": "E_adopted",
        "params": {
            "MPC_XY_CRUISE": 16.3,
            "MPC_JERK_AUTO": 1.38,
            "MPC_XY_VEL_D_ACC": 1.66,
            "lift_slew": 0.055,
            "pusher_slew": 0.11,
        },
    },
    {
        "name": "E_more_damping",
        "params": {
            "MPC_XY_CRUISE": 16.3,
            "MPC_JERK_AUTO": 1.38,
            "MPC_XY_VEL_D_ACC": 1.72,
            "lift_slew": 0.055,
            "pusher_slew": 0.11,
        },
    },
    {
        "name": "E_faster_mid",
        "params": {
            "MPC_XY_CRUISE": 16.6,
            "MPC_JERK_AUTO": 1.40,
            "MPC_XY_VEL_D_ACC": 1.70,
            "lift_slew": 0.052,
            "pusher_slew": 0.105,
        },
    },
]


def run(cmd, cwd=REPO_ROOT, stdout=None, check=True):
    print("+ " + " ".join(shlex.quote(str(item)) for item in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=stdout, stderr=subprocess.STDOUT, check=check)


def stop_sitl(session):
    subprocess.run([str(STOP_SITL)], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_session = ""

    if os.environ.get("TMUX"):
        current = subprocess.run(["tmux", "display-message", "-p", "#{session_name}"], cwd=REPO_ROOT,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        current_session = current.stdout.strip()

    if session != current_session and not session.endswith("_runner"):
        subprocess.run(["tmux", "kill-session", "-t", session], cwd=REPO_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    listed = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], cwd=REPO_ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)

    for active_session in listed.stdout.splitlines():
        if active_session == current_session:
            continue

        if active_session.endswith("_runner"):
            continue

        if active_session.startswith(session):
            subprocess.run(["tmux", "kill-session", "-t", active_session], cwd=REPO_ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for extra in ("tetra_axis_test", "tetra_tune", "tetra_smooth", "tetra_smoke"):
        subprocess.run(["tmux", "kill-session", "-t", extra], cwd=REPO_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for name in ("parameters.bson", "parameters_backup.bson"):
        path = REPO_ROOT / "build/px4_sitl_default/rootfs" / name

        try:
            path.unlink()
        except FileNotFoundError:
            pass


def tmux_pane(session, lines=80):
    result = subprocess.run(["tmux", "capture-pane", "-pt", session, "-S", f"-{lines}"], cwd=REPO_ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout


def wait_sitl_ready(session, startup_wait_s):
    deadline = time.monotonic() + startup_wait_s
    last_pane = ""

    while time.monotonic() < deadline:
        has_session = subprocess.run(["tmux", "has-session", "-t", session], cwd=REPO_ROOT,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        if has_session.returncode != 0:
            raise RuntimeError(f"SITL session {session} exited before ready:\n{last_pane}")

        last_pane = tmux_pane(session)

        if "Ready for takeoff" in last_pane or "Startup script returned successfully" in last_pane:
            return

        time.sleep(1.0)

    raise RuntimeError(f"SITL session {session} did not become ready within {startup_wait_s}s:\n{last_pane}")


def start_sitl(session, startup_wait_s):
    stop_sitl(session)
    run([
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        f"cd {shlex.quote(str(REPO_ROOT))} && HEADLESS=1 make px4_sitl gz_teTra_mk-7_EM2",
    ])
    wait_sitl_ready(session, startup_wait_s)


def latest_ulog():
    log_root = REPO_ROOT / "build/px4_sitl_default/rootfs/log"
    logs = sorted(log_root.glob("**/*.ulg"), key=lambda item: item.stat().st_mtime, reverse=True)

    if not logs:
        raise RuntimeError(f"no ULog found under {log_root}")

    return logs[0]


def expand_params(candidate):
    params = dict(COMMON_SAFE_PARAMS)
    params.update(candidate.get("params", {}))
    lift_slew = params.pop("lift_slew", None)
    pusher_slew = params.pop("pusher_slew", None)

    if lift_slew is not None:
        for index in range(12):
            params[f"CA_R{index}_SLEW"] = lift_slew

    if pusher_slew is not None:
        params["CA_R12_SLEW"] = pusher_slew

    return params


def load_candidates(path):
    if not path:
        return BUILTIN_CANDIDATES

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        return data["candidates"]

    return data


def filter_candidates(candidates, only, limit):
    if only:
        selected = {item.strip() for item in only.split(",") if item.strip()}
        candidates = [item for item in candidates if item["name"] in selected]

    if limit > 0:
        candidates = candidates[:limit]

    if not candidates:
        raise SystemExit("no candidates selected")

    return candidates


def build_axis_command(args, metrics_path, candidate):
    cmd = [
        str(AXIS_TEST),
        "--connect",
        args.connect,
        "--distance",
        str(args.distance),
        "--speed",
        str(args.speed),
        "--alt",
        str(args.alt),
        "--position-frame",
        args.position_frame,
        "--position-segments",
        args.segments,
        "--skip-yaw",
        "--max-time",
        str(args.max_time),
        "--settle-radius",
        str(args.settle_radius),
        "--settle-speed",
        str(args.settle_speed),
        "--stable-time",
        str(args.stable_time),
        "--between-hold",
        str(args.between_hold),
        "--initial-hold",
        str(args.initial_hold),
        "--print-period",
        str(args.print_period),
        "--metrics-out",
        str(metrics_path),
    ]

    for name, value in sorted(expand_params(candidate).items()):
        cmd.extend(["--param", f"{name}={value}"])

    return cmd


def run_candidate(args, candidate, out_dir):
    name = candidate["name"]
    session = f"{args.session_prefix}_{name[:20].replace('-', '_')}"
    metrics_path = out_dir / f"{name}_metrics.json"
    energy_path = out_dir / f"{name}_energy.json"
    stdout_path = out_dir / f"{name}.out"

    if args.resume and metrics_path.exists() and energy_path.exists():
        print(f"skip existing {name}", flush=True)
        return json.loads(energy_path.read_text(encoding="utf-8"))

    try:
        if args.start_sitl and not args.dry_run:
            start_sitl(session, args.startup_wait)

        axis_cmd = build_axis_command(args, metrics_path, candidate)

        if args.dry_run:
            print("AXIS:", " ".join(shlex.quote(str(item)) for item in axis_cmd))
            print("ENERGY:", str(ENERGY_OBJECTIVE), "<latest.ulg>", "--metrics-json", metrics_path)
            return {
                "candidate": name,
                "dry_run": True,
                "params": expand_params(candidate),
            }

        with open(stdout_path, "w", encoding="utf-8") as stdout:
            run(axis_cmd, stdout=stdout)

        ulg = latest_ulog()
        energy_cmd = [str(ENERGY_OBJECTIVE), str(ulg), "--metrics-json", str(metrics_path), "--json-out", str(energy_path)]
        run(energy_cmd)
        result = json.loads(energy_path.read_text(encoding="utf-8"))
        result["candidate"] = name
        result["params_override"] = expand_params(candidate)
        result["metrics_json"] = str(metrics_path)
        result["stdout"] = str(stdout_path)
        energy_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    except Exception as exc:
        failure = {
            "candidate": name,
            "failed": True,
            "error": str(exc),
            "params_override": expand_params(candidate),
            "metrics_json": str(metrics_path),
            "stdout": str(stdout_path),
        }
        energy_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"candidate {name} failed: {exc}", flush=True)
        return failure

    finally:
        if args.start_sitl and not args.dry_run:
            stop_sitl(session)


def write_summary(out_dir, results):
    ranked = sorted(
        [item for item in results if not item.get("dry_run") and not item.get("failed")],
        key=lambda item: (
            item.get("objective", float("inf")),
            item.get("segment_metrics", {}).get("segment_target_crossings", 0),
            item.get("segment_metrics", {}).get("segment_radius_exits", 0),
        ),
    )
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_json.write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "rank",
        "candidate",
        "objective",
        "duration_s",
        "energy_proxy_u3_s",
        "snap_proxy_integral",
        "snap_proxy_peak",
        "snap_rate_proxy_integral",
        "snap_rate_proxy_peak",
        "snap_jump_proxy_peak",
        "motor_slew_proxy_max_delta",
        "segment_penalty",
        "segment_overshoot_m",
        "segment_target_crossings",
        "segment_radius_exits",
        "segment_cross_track_m",
        "segment_final_error_m",
        "segment_post_enter_speed_m_s",
        "segment_time_to_speed_1_m_s",
        "segment_time_to_speed_5_m_s",
        "segment_time_to_speed_10_m_s",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for rank, item in enumerate(ranked, start=1):
            segment = item.get("segment_metrics", {})
            writer.writerow({
                "rank": rank,
                "candidate": item.get("candidate"),
                "objective": item.get("objective"),
                "duration_s": item.get("duration_s"),
                "energy_proxy_u3_s": item.get("energy_proxy_u3_s"),
                "snap_proxy_integral": item.get("snap_proxy_integral"),
                "snap_proxy_peak": item.get("snap_proxy_peak"),
                "snap_rate_proxy_integral": item.get("snap_rate_proxy_integral"),
                "snap_rate_proxy_peak": item.get("snap_rate_proxy_peak"),
                "snap_jump_proxy_peak": item.get("snap_jump_proxy_peak"),
                "motor_slew_proxy_max_delta": item.get("motor_slew_proxy_max_delta"),
                "segment_penalty": item.get("segment_penalty"),
                "segment_overshoot_m": segment.get("segment_overshoot_m"),
                "segment_target_crossings": segment.get("segment_target_crossings"),
                "segment_radius_exits": segment.get("segment_radius_exits"),
                "segment_cross_track_m": segment.get("segment_cross_track_m"),
                "segment_final_error_m": segment.get("segment_final_error_m"),
                "segment_post_enter_speed_m_s": segment.get("segment_post_enter_speed_m_s"),
                "segment_time_to_speed_1_m_s": segment.get("segment_time_to_speed_1_m_s"),
                "segment_time_to_speed_5_m_s": segment.get("segment_time_to_speed_5_m_s"),
                "segment_time_to_speed_10_m_s": segment.get("segment_time_to_speed_10_m_s"),
            })

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Run Mk-7 SITL tuning candidates and rank them by the energy objective.")
    parser.add_argument("--candidates-json")
    parser.add_argument("--only", help="Comma-separated candidate names to run")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="/tmp/tetra_mk7_tuning_sweep")
    parser.add_argument("--connect", default="udpin:0.0.0.0:14550")
    parser.add_argument("--distance", type=float, default=200.0)
    parser.add_argument("--segments", default="forward,return_after_forward")
    parser.add_argument("--position-frame", choices=["local", "body"], default="body")
    parser.add_argument("--speed", type=float, default=-1.0)
    parser.add_argument("--alt", type=float, default=7.0)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--settle-radius", type=float, default=1.0)
    parser.add_argument("--settle-speed", type=float, default=0.20)
    parser.add_argument("--stable-time", type=float, default=1.5)
    parser.add_argument("--between-hold", type=float, default=1.0)
    parser.add_argument("--initial-hold", type=float, default=4.0)
    parser.add_argument("--print-period", type=float, default=10.0)
    parser.add_argument("--startup-wait", type=float, default=25.0)
    parser.add_argument("--session-prefix", default="tetra_sweep")
    parser.add_argument("--start-sitl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = filter_candidates(load_candidates(args.candidates_json), args.only, args.limit)
    results = []

    ranked = []

    try:
        for candidate in candidates:
            print(f"\n=== candidate {candidate['name']} ===", flush=True)
            results.append(run_candidate(args, candidate, out_dir))

    finally:
        if not args.dry_run and results:
            ranked = write_summary(out_dir, results)

        if args.start_sitl and not args.dry_run:
            stop_sitl(args.session_prefix)

    if args.dry_run:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    print("\nRANKING", flush=True)

    for rank, item in enumerate(ranked, start=1):
        segment = item.get("segment_metrics", {})
        print(
            f"{rank}. {item['candidate']}: objective={item['objective']:.3f} "
            f"energy={item['energy_proxy_u3_s']:.3f} duration={item['duration_s']:.3f} "
            f"seg_penalty={item['segment_penalty']:.3f} "
            f"crossings={segment.get('segment_target_crossings')} exits={segment.get('segment_radius_exits')}",
            flush=True,
        )


if __name__ == "__main__":
    main()

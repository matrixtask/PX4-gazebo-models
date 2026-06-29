#!/usr/bin/env python3
"""Full NEW-PROFILE mission for the teTra Mk-7:
  AUTO.TAKEOFF -> FW transition -> straight settle -> coordinated TURN (loiter) ->
  [P1] FW CLIMB +300 m at max climb rate / constant airspeed (no zoom, MC off) ->
  [P2] LEVEL flight ->
  [P3] FW DESCEND to gate alt (152 m) ->
  [P4] DECEL-DESCENT: glide slope 1/6 avg -> 1/4 terminal, flaps deployed, FW->MC back-transition ->
  [P5] HOVER at 10 m (VRS-safe, vertical descent rate kept small) ->
  [P6] gentle XY-hold vertical LANDING.
Phase machine driven by groundtruth-like MAVLink altitude/airspeed/time. Aero fixed; control/guidance
+ nav sequence only. Param overrides via --params NAME=VAL,... Flap deploy via /tmp/mk7_flap_deploy."""
import argparse, math, threading, time, os
import numpy as np
from pymavlink import mavutil

# ---- 0.3G min-distance "green" descent reference (gate 38 m/s/152 m -> 10 m hover, Vz capped 0.45*vh=7) ----
REF_NPZ = "/home/tasukunakai/dash/mk7_descent_ref.npz"
_R = np.load(REF_NPZ)
REF_VX = np.asarray(_R["Vx"], float)      # 38 -> 0 (monotone decreasing)
REF_Z  = np.asarray(_R["Z"], float)       # 152 -> 10
REF_VZ = np.asarray(_R["Vz"], float)      # 0 -> 7 -> 0
# Z as a function of forward speed Vx, for indexing the back-trans decel-descent on CURRENT airspeed.
_diag = REF_VX > 0.3
_vx_asc = REF_VX[_diag][::-1]             # ascending Vx for np.interp
_z_asc  = REF_Z[_diag][::-1]
def green_Z_of_Vx(vx):
    """Altitude the green trajectory is at when forward speed is vx (diagonal branch)."""
    vx = max(float(_vx_asc[0]), min(float(_vx_asc[-1]), vx))
    return float(np.interp(vx, _vx_asc, _z_asc))
GREEN_Z_AT_VX0 = float(REF_Z[_diag][-1])  # alt where the diagonal ends, vertical tail begins (~57 m)
GREEN_SINK = 3.0                          # m/s vertical-tail sink. 0.25-CONSERVATIVE (user-adopted):
                                          # Vz/vh=0.19 @vh15.6 / 0.23 @vh13 -> below the 0.25 floor at
                                          # BOTH vh values (greenC used 5.0 = the 0.45-cap version).

INT_PARAMS = {"VT_FW_DIFTHR_EN", "VT_TYPE", "VT_FW_QC_P", "VT_FW_QC_R",
              "COM_RCL_EXCEPT", "NAV_RCL_ACT", "COM_OBL_RC_ACT", "VT_FW_LK_MC", "VT_FWD_THRUST_EN"}

# ---- profile constants (tunable without rebuild) ----
# RP-1 (Mk-7 EM2 Reference Profile) geometry: match the analytical DESIGN shape (mk7_fullprofile.py) ---
# takeoff -> climb to 300 m ABSOLUTE -> cruise 72 m/s -> 180 deg turn -> cruise-back -> descend to the
# 152 m gate -> hover-slam landing (an OUT-AND-BACK, horizontal ~5 km), NOT the old one-way 17 km profile.
RP1            = True     # True = RP-1 out-and-back shape. False = legacy one-way (settle 15km/turn/climb+300).
VCLIMB         = False    # False = FW FORWARD climb at the best-climb speed (design 1/4 gradient; the climb is
                         # TECS-schedule-limited NOT power-limited, so a low-speed full-thrust climb gets ~1/4).
                         # True = vertical MC climb to 300 at the origin (workaround).
TEST_LEVELDECEL = False   # TEST-only path (unused).
TEST_DECEL_SWEEP = False  # [MOD-3c] ISOLATED rotor-support-vs-decel-angle sweep. From cruise (72/300 m level),
                         # cut the pusher (drag-decelerate) and hold a fixed lift-rotor support = VT_FW_MC_THR
                         # (read from /tmp/mk7_decel_mcthr, swept 0..0.55). The resulting flight-path angle
                         # (vertical = wing-lift + rotor-up - W) EMERGES: high support -> zoom(+), low -> descend(-).
                         # No turn/land; the run ends after the decel segment. Output via the ulog (GT).
ROTOR_CLIMB    = True     # [MOD-3] ROTOR-BORNE 1/4 CLIMB: pusher-borne FW climb caps at ~1/6 (W*sin14=3711 N
                         # > pusher Tmax 2991 N -> the gravity term alone exceeds the pusher; vertical lift
                         # rotors cannot take the along-path W*sin(g), only trim drag -> rotor-assist IN FW
                         # does NOT steepen it, verified). A 1/4 climb is intrinsically a LOW-SPEED rotor-borne
                         # climb: command an OFFBOARD MC velocity (forward RC_VX + up RC_GRAD*RC_VX) so the
                         # ROTORS carry the climb. Then transition to FW for the 72 m/s cruise.
RC_VX          = 15.0    # m/s forward velocity during the rotor-borne climb. [MOD-3b 0.5-OVERLAP] at this
                         # WING-BORNE speed (>stall 36) the 1/4 climb needs the body NOSE-UP -> the wing flies
                         # at +AoA and carries ~HALF the weight, the lift rotors carry the other half = a
                         # 0.5 rotor/FW overlap hybrid (vs RC_VX=15 where the wing is dead and rotors carry all,
                         # 484 kW). Tunes the rotor-vs-wing split via the dynamic pressure. Raise=more wing.
RC_GRAD        = 0.25    # commanded climb gradient (1/4 = design); climb rate = RC_GRAD*RC_VX
RC_TKO         = 30.0    # m: low takeoff altitude before the slanted rotor-borne climb begins
HYBRID_CLIMB   = False    # [MOD-3b] TRANSITION-HOLD 0.5-OVERLAP climb (takes precedence over ROTOR_CLIMB). Hold
                         # the vehicle in TRANSITION_TO_FW (VT_ARSP_TRANS set ABOVE the capped climb airspeed ->
                         # the front-transition never completes -> the lift rotors stay blended ON at a partial
                         # mc_weight while the FW attitude controller pitches NOSE-UP so the WING flies). Pusher
                         # at full throttle (forward), wing + rotors share the lift -> a true pusher+wing+rotor
                         # hybrid. The OFFBOARD-MC climb cannot do this (it pitches NOSE-DOWN to go forward ->
                         # wing AoA negative -> wing dead -> rotors carry all). mc_weight ~ (TRANS-V)/(TRANS-BLEND).
HC_AS          = 46.0    # m/s held climb airspeed (in the 45-62 lift+cruise stable band, wing-borne)
HC_BLEND       = 30.0    # VT_ARSP_BLEND during the hold
HC_TRANS       = 62.0    # VT_ARSP_TRANS during the hold; at HC_AS=46: mc_weight=(62-46)/(62-30)=0.50 overlap
PRE_LVL_DECEL  = True     # [RP-1f ① user-arch] BEFORE the back-trans, decelerate IN FW at LEVEL 300 m (NO descent ->
LVL_SPOIL      = -0.55    # no glide overshoot) with a MILD spoiler (-0.40, ~10deg = proven non-divergent) lift-dumping
BT_LO_AS       = 44.0     # [NEW-SPEC] bleed 72 -> BT_LO_AS at altitude, THEN back-trans. Lower (44 vs 55) = the lift
ROLLOUT_ATT    = True     # [attitude-cmd roll-out] at the turn exit, OFFBOARD-ATTITUDE command roll_sp = P9(bank->0) and
ROLLOUT_T      = 6.0      # let the FW attitude controller TRACK it (same path as the clean roll-IN), instead of letting
ROLLOUT_THR    = 0.40     # nav drop the bank open-loop (roll fell BELOW sp -> -28deg overshoot + 0.17Hz ring). T=ramp s.
PRETRIM_EN     = True     # [phase-sched I] fallback: re-trim (restore high I) a few s before the back-trans. Usually OFF
                          # because the cruise-trimmed integrator value is RETAINED across the decel gain drop 0.3->0.1.
                          # Separates decel(level, in place) from descent(MC 1/4). Then MC 1/4 descent (compact) to 10 m.
LEVEL_BT       = True     # SPOILER LIFT-DUMP level back-trans: after the cruise-BACK over home, decel at
                         # CONSTANT 300 m with a DYNAMIC flaperon SPOILER (negative lift) that cancels the
                         # nose-up lift surge -> NO zoom -> then a VERTICAL hover-slam at home (no FW
                         # descend-to-gate circling -> a taller, cleaner U). The fixed -0.7 spoiler (rpf1)
                         # was too weak from 72 m/s; this schedules it dynamically on the vertical-rate error.
LEVEL_BT_SPOIL0 = -0.30  # baseline flaperon-up (lift dump) at vz=0 during the level decel
LEVEL_BT_KVZ    = 0.12   # rad per (m/s) of climb: zoom (vz<0) -> more spoiler (toward -0.785=45deg max dump)
BT_ANTICIP_V    = 45.0   # m/s: below this (near stall) pre-command a small CLIMB so rotors lead the wing fade
BT_ANTICIP_VZ   = 2.5    # m/s anticipatory climb (kills the below-stall ~30 m drop)
# [TASK-1 FIX] turn-exit/level-decel roll-spike cure. DIAGNOSED (rpf82 ULog 14_15_01): the -50 deg roll PEAK is
# NOT the turn roll-out (that tracks roll_sp 45->0 within a few deg) and is NOT a position command (no pos/vel
# setpoint is active at the spike; offboard=attitude, roll_sp=0). It is the cruise-back->lvldecel handoff SLAMMING
# the lift-rotor support (VT_FW_MC_THR 0->0.6 = all 12 rotors), the spoiler, AND a roll-I drop (FW_RR_I 0.3->0.1)
# ON as a simultaneous STEP at 82 m/s. The 12 rotors spinning up out-of-regime at high q inject a left-roll
# disturbance; the spoiler cuts Cl_p and the I-drop cuts authority at the same instant -> ailerons saturate
# (servo +-0.98), roll-rate sp saturates (+1.22 rad/s) and roll diverges to -50.8 deg. FIX = bleed speed first
# (aero only, rotors OFF, full roll I) to ROTSUP_BLEED_AS, THEN P9-ramp the rotor support + spoiler in-regime,
# dropping roll I only after the ramp completes. (Same cure the Task-2 phase-8 "Cruise Brake" V<->C decel needs.)
ROTSUP_BLEED_AS = 58.0   # m/s: idle-bleed the cruise-back speed to here (near VT_ARSP_BLEND) BEFORE engaging rotors
ROTSUP_MAX      = 0.60   # target lift-rotor collective (VT_FW_MC_THR) during the level decel (was a 0->0.6 step)
ROTSUP_RAMP_T   = 10.0   # s: [C-smooth2 REVERTED to 10] (the brake lift-gap was not the alt driver). P9 ramp time for VT_FW_MC_THR 0->ROTSUP_MAX AND spoiler 0->LVL_SPOIL (snap-smooth, SLOW so
                         # the 12 rotors engage gently; rpf84b at 6 s/69 m/s still left a -23 deg dip -> slow + lower V)
ROTSUP_STEP     = 0.10   # min VT_FW_MC_THR increment between set_param calls (finer -> smoother rotor spin-up)
LVLBLEED_T      = 150.0  # [TASK-1b] s timeout for the STAGE-1 idle aero-bleed. rpf84b: idle-only shed 79->69 in 60 s,
                         # so the old 60 s cap fired the rotor ramp early at 69 m/s (q still high -> -23 deg dip). 150 s
                         # lets the bleed reach ROTSUP_BLEED_AS=58 (q halved vs 82) so the slow rotor ramp engages truly
                         # in-regime. (No bleed spoiler: idle holds altitude fine and keeps the wing fully loaded/roll-stable.)
CRUISE_ALT_ABS = 300.0   # m: RP-1 absolute cruise/apex altitude (design 300; old +300-rel gave 398).
CLIMB_AS       = 48.0    # m/s: BEST-CLIMB airspeed (min-drag ~50 -> max excess thrust -> steepest climb ~1/4).
CLIMB_AS_CAP   = 51.0    # m/s: hard FW_AIRSPD_MAX cap so it holds the best-climb speed (not creep to 72).
CRUISE_AS      = 72.0    # m/s: RP-1 cruise airspeed at 300 m (canonical Mk-7 cruise; reached on the level legs).
P8_BRAKE_AS    = 38.0    # m/s: [P8 Cruise Brake] target = P9 descent entry speed (level brake 72->38 @300m West).
P8_SPOIL       = -0.65   # set_flap deploy for the brake spoiler: servo=0.61-0.90=-0.29 rad (~17deg UP) -> FlaperonDrag (gentle; rotors hold alt).
P8_RAMP_T      = 3.0     # s: P9 ramp-IN of the spoiler (jerk-continuous decel ENTRY); fade-OUT keyed to (as-38).
P8_PUSH_RAMP_T = 5.0     # s: [C-smooth2] P9 ramp-DOWN of the pusher 0.58->0.03 at brake entry (was instant cut -> wing-lift bleed faster than rotor-support ramp -> -15 m drop). Bridges the lift gap.
P8_BRAKE_HDG   = 270.0   # deg: WEST return heading held through the brake (heading-hold bank).
P9_DESC_T      = 95.0    # s: [P9 Descend X] P9 altitude profile time 300->152 (was 70; SLOWER -> less gravity PE rate -> the spoiler can pin true 38 at completion, was settling ~45).
P9_SPOIL       = -0.95   # set_flap base spoiler for the P9 descent (dissipate the gravity PE -> hold 38; +speed feedback).
CRUISE_DIST    = 1400.0  # [RP-1f/2] LENGTHENED 800->1400: longer downrange straight leg -> taller U + more room for
                         # the decel-descent. Lateral (turn R=1134) unchanged. = mk7_fullprofile_
                         #     3d_v2 "Cruise out 72" 800 m (RP-1 definition). The earlier 3500 m "widen" was discarded.
DESC_HOME_DIST = 3300.0  # m: on the way back, fly level at 300 until this close to home, THEN descend to the gate.
GATE_DESC_DIST = 1800.0  # m: descend-to-gate leg (300 -> 152 m) heading back toward the start.
RP1_TURN_RAD   = 528.0   # m: RP-1 180 deg turn radius = V_CR^2/(g tan25) at 72 m/s (mk7_fullprofile_3d_v2).
TURN_BANK      = 45.0    # deg: FW roll limit during the turn = RP-1 design bank 25 deg (R=1134 m @ 72 m/s).
# Home approach mode (6-DOF disturbance study). The FW roll-setpoint THRASH (+-15 deg, ~0.3 Hz, the
# trajectory disturbance) is caused by the FW nav CHASING the near home WAYPOINT on the cruise-back/descent
# (near-waypoint bearing is hyper-sensitive). NOT fixable by rate/attitude gains (rate-damping tune = no
# effect) nor nav period (broke the gate entry -> divergence). FROZEN_APPROACH=True aims FAR along the
# (once-)frozen home BEARING -> a STRAIGHT path -> the thrash is ELIMINATED (roll-SP std 10->0.1 deg,
# reversals 12->0) AND it stays reliable, BUT the straight descent OVERSHOOTS home (~5 km) because the FW
# cannot drop 148 m to the gate in a straight line at low speed (aim-at-home circles down over home instead).
# False = aim AT home (reliable, RETURNS to ~1 km of home, but the roll thrash remains). The thrash is the
# intrinsic price of a low-speed return-to-the-152 m-gate descent; it is barely visible on the km-scale path.
FROZEN_APPROACH = True   # [TEST] straight frozen-bearing cruise-back -> WINGS-LEVEL at the back-trans (no sustained
                         # course bank) -> tests whether the BT spike is a banked-approach roll-out overshoot.
                         # False = aim-at-home -> COMPACT out-and-back that RETURNS to home = the v2 RP-1
                         # top-down TOPOLOGY (priority). True = thrash-free straight approach but OVERSHOOTS
                         # home ~7 km (stretches the loop, breaks the compact topology). Trade-off is
                         # intrinsic: returning to the 152 m gate at low speed needs FW circling = roll
                         # thrash; a straight no-thrash approach overshoots. Topology match wins here.
CLIMB_DELTA   = 300.0    # m, climb above cruise (legacy one-way profile only)
LEVEL_HOLD    = 15.0     # s level flight at top
GATE_ALT      = 152.0    # m rel, start of decel-descent
HOVER_ALT     = 10.0     # m rel, final hover
P11_HDG       = float(os.environ.get("P11_HDG", "90.0"))   # [WIND] P11 set heading: calm=East 90; in wind, the UPWIND heading (face the flow)
SLOPE_AVG     = 1.0/6.0  # avg glide slope (rise/run) in decel-descent
SLOPE_TERM    = 1.0/4.0  # terminal (near-ground) steeper slope
TURN_HOLD     = 26.0     # s of turn (loiter) before starting the climb
BACKTRANS_AS  = 38.0     # airspeed [m/s] at which to command FW->MC back-transition
SPOIL_DESC      = False  # [MOD-4] STRAIGHT spoiler descent: on the cruise-back, deploy a PARTIAL flaperon spoiler
                         # (FlaperonDrag plugin adds real drag now) and aim STRAIGHT at home -> a steep straight
                         # glide that dissipates energy (no speed runaway) and descends toward home WITHOUT the FW
                         # circling -> collapses the lateral spread -> the design TALL U. Low-speed end: back-trans
                         # to MC (rotor support) so it never stalls. Energy is spent (no regen; fixed-pitch rotors).
SPOIL_DESC_FLAP = -1.05  # set_flap value for the descent spoiler. servo = 0.61 + this = -0.44 rad (~25 deg UP).
                         # Used in MC (rotors carry weight & control roll -> NO unloaded-wing roll divergence, which
                         # killed the FW spoiler glide). The FlaperonDrag plugin turns this into real drag.
SDESC_MC        = True   # [MOD-4] straight MC spoiler-descent: high-speed back-trans (LK_MC) over the cruise-back,
                         # then descend STRAIGHT toward home in MC (rotors control attitude) with the spoiler adding
                         # drag to dissipate KE (no zoom: the new drag sheds the energy the rotors used to bounce up).
SPOIL_TRANS_FLAP = 0.0   # small spoiler during the TRANSITION (vs!=3): servo -0.19 (~11 deg UP) -> sheds some KE
                         # to cut the back-trans zoom, but small enough not to roll-diverge the still-loaded wing.
SDESC_VZ        = 6.0    # m/s descent rate commanded during the MC spoiler-descent
SDESC_TARGET    = 12.0   # [RP-1f] descend to 12 m (was 28) then hand to the P9 hover-slam at 10 m
GRAD_DESC       = True   # [RP-1f/3] hold a 1/4 GLIDE-GRADIENT decel descent: sink rate vd = vcmd * GRAD_RATIO so the
GRAD_RATIO      = 0.25   # path stays 4-horizontal:1-vertical while the forward speed bleeds (spoiler drag + rotor
GRAD_VZ_CAP     = 16.0   # support). vd ~ vx/4 is intrinsically VRS-safe (vz small when vx small). cap for SITL safety.
# [MOD-4b] SPEED-SCHEDULED ZOOM-CANCEL SPOILER during the high-speed back-trans: dump the excess lift (pitch-up
# surge + KE->PE) so the net vertical ~= weight -> HOLD altitude (zoom~0) while decelerating, THEN descend.
SPOIL_HI        = -0.90  # set_flap at high speed (servo -0.29 ~ 17 deg UP, MILD lift dump -> wing stays mostly
                         # loaded -> keeps roll authority during the vs==2 transition). Tamer than the 40deg that
                         # over-decelerated past stall (rpsd6 -> R87 authority spike).
SPOIL_V0        = 46.0   # CAS at/below which the scheduled spoiler -> 0: FADE OUT before stall so the drag stops
                         # decelerating and the wing reloads (keeps speed > 36 stall until vs==3).
SPOIL_VZ_K      = 0.05   # vz feedback [rad per (m/s)]: cur_vz<0 (zooming up) -> more spoiler to push it back down
SPOIL_RAMP      = 0.7    # rad/s max slew of the spoiler -> GRADUAL deploy
SDESC_HOLD_V    = 46.0   # hold altitude (zoom-cancel) until CAS drops below this, then start the straight descent
SPOIL_ROLL_D    = 0.12   # MC_ROLLRATE_D during the spoiler back-trans (up from 0.025) to add roll DAMPING (the
                         # spoiler cuts Cl_p ~-18%; restore damping). Restored to 0.025 is the default (not done).
GATE_BT_AS_MAX = 45.0    # [MOD-3 FIX] only back-trans at the gate if airspeed <= this. If the gate altitude
                         # is reached while still FAST (compact topology -> short descent doesn't bleed; the
                         # 76 m/s back-trans dove to alt -900), LEVEL-OFF at the gate alt + idle + low airspeed
                         # target and bleed speed FIRST, then back-trans. Prevents the high-speed back-trans dive.
FLAP_DEPLOY   = 0.6      # symmetric flap (0..1) during decel-descent (read by gz flap loop)
REV_FLAP      = 0.0      # SMOOTH: NO spoiler at the low-speed (~34 m/s) gate entry. The spoiler was to
                         # kill the back-trans zoom from 75 m/s; at 34 m/s there is little zoom, and the
                         # spoiler dumped wing lift abruptly -> a 19 m/s SINK SPIKE just past the gate.
                         # REV_FLAP=0 keeps lift continuous -> no sink step (the corner is removed).
DECEL_SINK    = 0.0      # HOLD altitude during the FW->MC back-trans (no descent command while the wing
                         # is unloading). Commanding a descent here let the vehicle FREE-FALL to ~19 m/s
                         # sink before the rotors caught it (the gate sink-spike). Hold alt -> rotors
                         # arrest the transition at the gate -> then the smooth vtail descent begins.
GLIDE_VFWD    = 12.0     # MC forward speed during the glide descent [m/s] (slope = sink/Vfwd)
GLIDE_VFWD_T  = 5.0      # terminal forward speed (slower into arrest -> less to bleed -> smaller roll spike)
CLIMB_THR_MAX = 0.95     # FW throttle max during climb (max climb rate)
CLIMB_THR_MIN = 0.95     # [MOD-2] PIN the FW throttle FLOOR near max during the climb. The climb was TECS-
                         # LIMITED not power-limited (pusher only 0.67/0.77 of 0.95 -> surplus thrust unused,
                         # airspeed drooped to 39). Pinning thr_min=0.90 FORCES near-max thrust and SPDWEIGHT=2
                         # makes pitch hold the best-climb speed -> ALL the surplus energy goes to CLIMB (= the
                         # classic Vy max-rate climb). Functionally identical to the requested TECS source mod,
                         # zero rebuild risk. Restored to 0 at cruise so the later idle-throttle descent works.
CLIMB_CLMB_MAX= 18.0     # FW_T_CLMB_MAX during climb [m/s] (1/4 @ ~50 m/s = ~12 m/s; 18 = headroom)
CLIMB_MC_THR  = 0.0      # [MOD-3 ROTOR-ASSIST] lift-rotor collective (VT_FW_MC_THR, src mod) during the FW
                         # climb. Pusher ALONE cannot do 1/4: W*sin(14)=3711 N > pusher Tmax 2991 N (gravity
                         # term alone), and lift rotors (vertical) cannot take the along-path W*sin(g) -> a
                         # 1/4 climb is intrinsically a LOW-SPEED ROTOR-BORNE climb. This collective makes the
                         # rotors carry weight/add climb force so vz=(Tp-D)V/(W-Tr) steepens to 1/4. 0=off.
DESC_THR_MAX  = 0.10     # FW throttle max during descent (idle, let it sink)
# SMOOTH-LANDING: control the descent airspeed so the gate is crossed SLOW (~35), not 75. Pitch holds
# airspeed (FW_T_SPDWEIGHT=2) + low airspeed target + idle throttle -> descends at the commanded speed,
# so the rotor-assist back-trans starts from ~35 (minimal zoom) and reaches the 33 gate-entry target.
DESC_AS_TRIM  = 35.0     # FW airspeed target during the descent-to-gate
DESC_AS_MIN   = 28.0     # lower the FW min so the speed controller can hold ~35 (and reach 33)
GATE_ENTRY_AS = 33.0     # target airspeed crossing the 152 m gate
# CONTINUOUS-ARC vertical descent: the sink is an ALTITUDE-scheduled smooth bump (NOT ease-in +
# constant-hold + flare, which left a straight middle segment with corners at each junction). It is
# ~FLARE_SINK at the top (no corner entering the descent), rises to DESC_SINK near mid-altitude, and
# eases back to FLARE_SINK at the ground -> one gentle arc, continuous curvature, no straight segment.
# HOVER-SLAM (SpaceX-booster-style) descent + flare. Descend briskly at DESC_SINK, then fire a DECISIVE
# min-snap (P9) braking burn that drops the sink DESC_SINK->FLARE_SINK over the altitude band
# [BRAKE_END_ALT, FLARE_ALT], then a short soft settle to the pad. The brake is ALTITUDE-SCHEDULED (not
# time-triggered) so it is frame-consistent and ROBUST to the ~16 m EKF altitude over-estimate (measured):
# the burn COMPLETES at cur_alt=BRAKE_END_ALT which is set safely ABOVE that bias, so the vehicle is at
# the soft FLARE_SINK well before the TRUE ground. P9 (1st..4th derivs zero at both band ends) keeps
# velocity/accel/jerk/snap continuous at brake ENTRY and at burn-out -> no corner, no slam.
DESC_SINK     = 4.0      # m/s brisk steady descent (gz has no VRS; <7 analytic cap at low V -> safe).
TOP_EASE_T    = 4.0      # s: P9 ease the sink 0->DESC_SINK leaving the hover (snap-cont, no top corner).
FLARE_ALT     = 58.0     # m (cur_alt/EKF frame): hover-slam brake ENTRY altitude (start the burn here).
BRAKE_END_ALT = 20.0     # m (cur_alt/EKF frame): burn COMPLETE (sink=FLARE_SINK). ~16.5 m EKF bias +
                         # margin -> true AGL ~3.5 m, so the vehicle is soft before the real pad.
FLARE_SINK    = 0.40     # m/s settle / touchdown sink (vz approx 0; soft for a 1560 kg vehicle)
ARC_RAMP      = 0.5      # (legacy) unused by the hover-slam profile
SINK_RAMP_T   = 8.0      # (legacy) unused
FLARE_ALT_ARC = 12.0     # (legacy) unused
# ROUNDED-ARC ground track: a MODEST forward GLIDE that decays to 0 by GLIDE_END_ALT (well above the
# ground), so the upper descent curves (diagonal) but the last stretch is pure-vertical + flare -> the
# touchdown stays soft (a strong glide all the way down left forward momentum -> hard 2.5 m/s touchdown).
GLIDE_V0      = 0.0      # m/s forward glide. 0 = PURE VERTICAL (a fwd glide left residual forward
                         # momentum that disrupted the flare -> 2.5 m/s touchdown). Rounding is done by the
                         # vertical profile (big FLARE_ALT + long SINK_RAMP_T) which keeps the soft touchdown.
GLIDE_END_ALT = 40.0     # m: forward glide has fully decayed to 0 at/below this altitude
# (A) ZERO-ALT-CHANGE rotor decel: at the gate, OFFBOARD velocity-FEEDFORWARD holds altitude (vd=0,
# pos-z=gate) while the commanded forward speed ramps 34->0 (advancing the XY target so the MC matches
# attitude smoothly instead of slamming to hover = the tumble cause). Eliminates the transition free-fall.
DECEL_FF_DEC  = 1.2      # [MOD-5/RP-1c] m/s^2 forward decel via velocity-FF. With MASK_ALTVEL (DECEL_ALTVEL) the
                         # osc driver (forward pos-vs-vel fight) is gone, so 1.2 is the optimum: 1.0 spread topology
                         # AND raised pitch osc (15 vs 10), 1.5 too brisk. Inner MC-gain increases made roll WORSE
                         # (outer-loop osc, not inner). RP-1c = MASK_ALTVEL + DECEL 1.2 + MPC detune.
                         # (slow lingers at high speed, accumulates sink), 1.5->39 m (SWEET SPOT), 3.0->47 m
                         # (too-sharp transient). NOT monotonic; 1.5 is optimal. ~39 m is the floor.
MASK_POSVEL   = 0b0000110111000000   # use pos(lat,lon,alt)+vel(vn,ve,vd); ignore accel/yaw
MASK_VEL      = 0b0000110111000111   # velocity-only (vn,ve,vd); ignore position+accel+yaw (MASK_POSVEL|0b111)
MASK_ALTVEL   = 0b0000110111000011   # hold ALT(z)+velocity; ignore lat/lon(forward pos)+accel+yaw
MASK_POS_YAW  = 0b0000100111111000   # [STAGE-2] hold pos(lat,lon,alt) + command YAW; ignore vel+accel+yawrate (hover yaw-set)
DECEL_VEL_ONLY = False   # [MOD-6] MASK_VEL hurt (no alt ref -> R70/P44 wander, 130s decel). DECEL_ALTVEL is the fix.
DECEL_ALTVEL   = True    # [MOD-6b] decel commands ALT(z)+velocity (drop the FORWARD lat/lon position) -> removes
                         # the forward pos-vs-vel fight (the 0.12 Hz osc) while KEEPING the altitude reference.
OVERLAP_DESC   = True    # [P10 RP-1v2 ROOT-CAUSE FOUND] the prior "MC cant descend fwd" was WRONG: the tetra airframe set
                         # MPC_XY_VEL_MAX=1.2 / MPC_Z_VEL_MAX_DN=1.0 (~1 m/s) -> mc_pos_control CLAMPED the MC velocity, so
                         # the commanded vz=8/vx diagonal was clipped to ~1. FIX = raise the MPC limits at the back-trans
                         # (below) -> the MC then DOES the 1:4 diagonal descent (sink=vx/4). NOT a physics/VRS/tilt limit.
                         # wing (rotor thrust 0.050->0.058 only) so the rotors stay at ~5% -> no roll authority gained.
                         # Reverted. (Was: MC-velocity-control OVERLAP: start the MC descent DURING the vs==2 back-trans
                         # hold altitude) -> the wing unloads, the LIFT ROTORS take up the descent & spin up ->
                         # they gain roll DIFFERENTIAL authority through the gap, the descent (not a climb) absorbs
                         # the thrust. Tests the user's "overlap MC to supply roll authority" hypothesis.
MASK_POSVZ    = 0b0000110111011000   # use pos(lat,lon,alt)+vz ONLY (XY pos, vertical velocity-FF); ignore vx,vy,accel,yaw
MASK_POSVZ_YAW= 0b0000100111011000   # [P12] like MASK_POSVZ but HOLD yaw (bit10 cleared) -> the land keeps the East-90 heading
MASK_XYPOS_VZ = 0b0000110111011100   # [P1] HOLD XY pos + vz velocity-FF (ignore ALT-pos too) -> vertical climb on a P9 vUp
MASK_VEL_YAW  = 0b0000100111000111   # [P1] velocity (vN,vE,vz) + HOLD yaw; ignore pos+accel+yawrate (no yaw drift on takeoff)
MASK_POSVEL_YAW = 0b0000100111000000 # [P1] FULL position (lat,lon,alt) + velocity-FF + HOLD yaw -> no XY drift, no yaw drift
USE_VEL_FF    = True     # decel method: True=OFFBOARD velocity-FF (smooth but vstate->2=FW, so the
                         # VT_B_RAMP_MIN floor does NOT apply); False=AUTO loiter hold-alt (vstate=3
                         # TRANSITION_TO_MC, so VT_B_RAMP_MIN's immediate rotor thrust DOES apply -> no dip).
# Anticipate the FW->MC mode-switch gap (rotors spin up over ~3 s): command a brief CLIMB (vd<0) for the
# first FF_CLIMB_T s so the controller maxes rotor thrust against the wing-unload drop -> no free-fall dip.
FF_CLIMB      = 0.0      # m/s anticipatory climb. 0 = best (smoothB1): with VT_B_TRANS_RAMP=0.5 the dip
                         # is 39 m / net dAlt -1 m, no bounce. Climb-FF>0 re-introduced a bounce to 159 m
                         # without shrinking the dip, so keep 0.
FF_CLIMB_T    = 3.0      # s duration of the anticipatory climb


def smootherstep(u):
    """C2-continuous 0->1 ramp (6u^5-15u^4+10u^3); zero 1st & 2nd derivative at both ends -> snap-cont."""
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
def P9(u):
    """9th-order min-snap smoothstep 0->1 (70u^9-315u^8+540u^7-420u^6+126u^5). The 1st..4th derivatives
    are ZERO at both ends -> velocity, accel, jerk AND snap are all continuous (rest-to-rest, C4). Same
    family as the DLC / turn-entry / transition shapers. Used here for the hover-slam braking burn."""
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u**5 * (126.0 + u * (-420.0 + u * (540.0 + u * (-315.0 + u * 70.0))))
def P9int(u):
    """Integral of P9 from 0 to u (= 7u^10-35u^9+67.5u^8-60u^7+21u^6); I(1)=0.5. Used to keep a position
    target consistent with a P9-shaped velocity ramp (snap-continuous command, no pos-vs-vel mismatch)."""
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u**6 * (21.0 + u * (-60.0 + u * (67.5 + u * (-35.0 + u * 7.0))))
# === SNAP-SMOOTH COMMAND POLICY: every commanded velocity/rate change ramps via P9 (C4, snap-continuous) ===
# Step/linear command changes excite the finite-bandwidth controller -> commanded banks/overshoot. P9 ramps
# (1st..4th derivatives zero at both ends) keep accel/jerk/snap continuous. See [[px4-13rotor-tuning]] policy.
RC_RAMP_T     = 6.0      # s: P9 ramp of the rotor-climb forward+up velocity command 0->full (was a STEP -> -30 deg
                         # bank). (RP-1e tried 9 = no help: the climb-start bank is course-capture, ramp-rate insensitive.)
VZ_RAMP_T     = 4.0      # s: P9 ramp of the MC-descent sink-rate command 0->SDESC_VZ. (RP-1e tried 6 = worse descent osc.)
# [RP-1e Q2] FT (MC->FW) OVERLAP: hold the LIFT ROTORS through the forward transition via the airspeed-scheduled
# lift-handoff floor, set DYNAMICALLY = ON only for the FT (low speed, the wing genuinely cannot carry the weight ->
# rotors fill the deficit -> no altitude dip) and OFF before the BT (a global airspeed floor wrecks the high-speed
# back-trans: at V<56 it adds excess lift -> maxRoll 88). So mission set_param VT_LIFT_HND_V=FT_LIFT_HND at the
# climb, =0 at the LEVEL_BT entry. The BT gap stays a pure authority limit (rotor thrust=lift, see memory).
FT_OVERLAP    = True     # [P5 HANDOFF] RE-ENABLED for the NEW FW-mode absolute lift-rotor support (VT_LIFT_HND_THR):
                         # the old "dip unchanged" test below was the TRANSITION-floor (mc_weight floor in vstate=1, which
                         # cuts the instant FW entry completes); the new VT_LIFT_HND_THR commands the rotors IN FW mode
                         # past the handoff (CAS 54->72), a different lever. Sets V=72+THR at climb, V=0 at BT.
                         # [RP-1e] TESTED both 60 & 72 -> FT dip UNCHANGED (~19-21 m): holding the rotors up (even to
                         # cruise speed) does NOT prevent the dip; it is an intrinsic MC->FW handoff transient (the wing
                         # settles into lift-bearing flight with a transient loss, independent of the rotor rampdown).
                         # The dynamic FT-only schedule does work (BT unharmed) but yields no benefit. Disabled.
FT_LIFT_HND   = 72.0
FT_LIFT_HND_THR = float(os.environ.get("MK7_THR", "1.00"))   # [P5 HANDOFF] ADOPTED 1.00 (user choice: valley drop 0.0 m /
                         # sink 0.33; cruise & pitch unaffected, P9 -0.4 m/s drift accepted for a fully flat handoff).
                         # lift-rotor hover-collective peak (VT_LIFT_HND_THR) bridging the MC->FW handoff
                         # valley; faded by 1-(CAS/FT_LIFT_HND)^2, gated CAS<72 & <25 s of FW entry. Tune for valley vs drag.
HOVER_HOLD    = 8.0      # s hover before landing
LAND_RATE     = 1.0      # m/s gentle vertical landing (VRS-safe, <=1.5)
DECEL_RATE    = 1.2      # m/s commanded vertical in final MC approach (VRS-safe)

# VRS-safe vertical descent rate vs forward speed. vh~13 m/s (12 rotors R=0.99, T/rotor=1275 N).
# VRS danger ~ Vx/vh<1 & 0.25<Vz/vh<1.5. Keep Vz/vh<0.25 (Vz<3.25) whenever Vx<~15 (Vx/vh<1.15);
# allow a faster descent only while genuinely in forward flight (Vx>15). Tighten toward hover.
VH = 13.0
def vrs_vz(vx):
    # Vz/vh must stay < 0.25 (the VRS-region floor) whenever Vx/vh < 1 (Vx < ~13). At vh=13 that is
    # Vz < 3.25 m/s, so a CONSTANT 2.5 m/s descent (Vz/vh=0.19) is VRS-safe at ANY forward speed below
    # the VRS box; allow faster only in genuine forward flight (Vx>15). 2.5 lands in budget (1.5 timed out).
    if vx > 15.0:
        return 4.0
    return 2.5

def cmd(m, c, *p):
    pp = list(p) + [0.0]*(7-len(p))
    m.mav.command_long_send(m.target_system, m.target_component, c, 0, *pp[:7])

def set_param(m, name, val):
    nb = name.encode(); is_int = name in INT_PARAMS
    ptype = mavutil.mavlink.MAV_PARAM_TYPE_INT32 if is_int else mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    for _ in range(6):
        m.mav.param_set_send(m.target_system, m.target_component, nb, float(val), ptype)
        pv = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.5); t0 = time.time()
        while pv and pv.param_id != name and time.time()-t0 < 1.5:
            pv = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
        if pv and pv.param_id == name and abs(pv.param_value - float(val)) < 1e-3:
            print(f"  param {name} = {pv.param_value:.4g} OK", flush=True); return True
        time.sleep(0.3)
    print(f"  param {name} SET FAILED (want {val})", flush=True); return False

def ahead(lat, lon, hdg_deg, dist_m):
    hd = math.radians(hdg_deg)
    return (lat + (dist_m/111320.0)*math.cos(hd),
            lon + (dist_m/(111320.0*math.cos(math.radians(lat))))*math.sin(hd))

def hdist(lat1, lon1, lat2, lon2):
    """horizontal great-circle distance [m] (equirectangular, fine at these ranges)."""
    x = math.radians(lon2-lon1)*math.cos(math.radians((lat1+lat2)/2.0)); y = math.radians(lat2-lat1)
    return math.hypot(x, y)*6371000.0

def bearing(lat1, lon1, lat2, lon2):
    """initial heading [deg] from point 1 to point 2."""
    dlon = math.radians(lon2-lon1)
    y = math.sin(dlon)*math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1))*math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(dlon))
    return math.degrees(math.atan2(y, x))

def repo(m, lat, lon, alt, radius=-1.0):
    m.mav.command_int_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_DO_REPOSITION,
        0, 0, -1.0, 0.0, radius, float("nan"), int(lat*1e7), int(lon*1e7), float(alt))

def set_flap(v):
    try:
        with open("/tmp/mk7_flap_deploy", "w") as f: f.write("%.3f" % v)
    except Exception: pass

def quat_from_euler(roll, pitch, yaw):   # rad -> [w,x,y,z]; for the OFFBOARD attitude roll-out command
    cr, sr = math.cos(roll/2), math.sin(roll/2); cp, sp = math.cos(pitch/2), math.sin(pitch/2); cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="udpin:0.0.0.0:14550")
    ap.add_argument("--alt", type=float, default=90.0)
    ap.add_argument("--timeout", type=float, default=560.0)
    ap.add_argument("--params", default="")
    a = ap.parse_args()
    m = mavutil.mavlink_connection(a.connect, autoreconnect=True)
    m.wait_heartbeat(timeout=30); print("heartbeat", m.target_system, flush=True)
    threading.Thread(target=lambda: [(m.mav.heartbeat_send(6,8,0,0,4), time.sleep(1.0)) for _ in iter(int,1)], daemon=True).start()
    set_flap(0.0)
    if a.params:
        print("=== setting params ===", flush=True)
        for kv in a.params.split(","):
            kv = kv.strip()
            if kv:
                name, val = kv.split("="); set_param(m, name.strip(), val.strip())
        time.sleep(1.0)
    CRUISE = a.alt
    t0 = time.time()
    while time.time()-t0 < 30:
        g = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if g and g.lat != 0: break
    print("got global pos", flush=True); time.sleep(2)
    # RP-1 FIX (broken-part = the climb): gain altitude as a VERTICAL MC climb at the ORIGIN (NAV_TAKEOFF
    # straight to 300 m) instead of climbing while flying forward (the Mk-7 climbs slowly in FW -> a 7 km
    # out-leg that stretched the top-down topology to 17 km vs the design's ~5 km). Vertical climb -> ~0
    # downrange used for altitude -> compact out-and-back matching mk7_fullprofile_3d_v2.
    if RP1:
        set_param(m, "MPC_Z_VEL_MAX_UP", 6.0); set_param(m, "MPC_TKO_SPEED", 3.0)
    TKO_ALT = RC_TKO if (RP1 and (ROTOR_CLIMB or HYBRID_CLIMB)) else (CRUISE_ALT_ABS if (RP1 and VCLIMB) else a.alt)
    cmd(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0); time.sleep(1.0)
    # [OFFBOARD velocity TAKEOFF] for the rotor-borne climb, SKIP the AUTO NAV_TAKEOFF (it climbs to RC_TKO then
    # DECELERATES to ~0 near the target -> a vz step at the OFFBOARD handoff, plus a +-11deg position-hold limit-cycle).
    # Instead lift off DIRECTLY on the OFFBOARD velocity stream below (vUp P9-ramped 0->design rate) = one continuous
    # velocity profile from the ground, no handoff step, no AUTO position-hold.
    _offb_tko = bool(RP1 and ROTOR_CLIMB and not HYBRID_CLIMB)
    if not _offb_tko:
        cmd(m, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,0,0, float("nan"), float("nan"), float("nan"), TKO_ALT)
        cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 2)
        print(f"takeoff -> vertical climb to {TKO_ALT:.0f} m at origin", flush=True)
        t0 = time.time(); reached = False
        while time.time()-t0 < 200:
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            if v and v.alt > TKO_ALT-8: reached = True; break
        print(f"reached alt={reached}", flush=True); time.sleep(5)
    else:
        print("=> OFFBOARD velocity TAKEOFF (no AUTO): lift off on the vUp P9 ramp from the ground", flush=True); time.sleep(2.0)

    if RP1 and HYBRID_CLIMB:
        # [MOD-3b] TRANSITION-HOLD 0.5-OVERLAP HYBRID CLIMB. Set VT_ARSP_TRANS ABOVE the capped climb airspeed so
        # the front-transition NEVER completes -> rotors stay blended ON (mc_weight ~0.5) while the FW controller
        # pitches nose-up (wing flies). Pusher full throttle. wing + rotors share lift; measure the split + power.
        set_param(m, "VT_ARSP_BLEND", HC_BLEND); set_param(m, "VT_ARSP_TRANS", HC_TRANS)
        set_param(m, "FW_AIRSPD_TRIM", HC_AS); set_param(m, "FW_AIRSPD_MAX", HC_AS+2.0); set_param(m, "FW_AIRSPD_MIN", 38.0)
        set_param(m, "FW_THR_MAX", CLIMB_THR_MAX); set_param(m, "FW_THR_MIN", CLIMB_THR_MIN)
        set_param(m, "FW_T_SPDWEIGHT", 2.0); set_param(m, "FW_T_CLMB_MAX", CLIMB_CLMB_MAX); set_param(m, "FW_P_LIM_MAX", 28.0)
        cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)  # AUTO.LOITER (for repo)
        cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 4)
        tlat, tlon = ahead(g.lat/1e7, g.lon/1e7, 0.0, 1500.0)
        repo(m, tlat, tlon, CRUISE_ALT_ABS)
        print(f"=> HYBRID transition-hold climb: hold {HC_AS:.0f} m/s, mc_weight~{(HC_TRANS-HC_AS)/(HC_TRANS-HC_BLEND):.2f} -> {CRUISE_ALT_ABS:.0f} m", flush=True)
        tc0 = time.time(); hc_alt = TKO_ALT
        while time.time()-tc0 < 220:
            v = m.recv_match(type="VFR_HUD", blocking=False)
            if v: hc_alt = v.alt
            if hc_alt >= CRUISE_ALT_ABS - 8: break
            time.sleep(0.3)
        print(f"=> hybrid climb done alt={hc_alt:.0f}", flush=True)
        # complete the transition for cruise: drop VT_ARSP_TRANS below the current speed + uncap airspeed
        set_param(m, "VT_ARSP_TRANS", 44.0); set_param(m, "FW_AIRSPD_MAX", 78.0); set_param(m, "FW_THR_MIN", 0.0)
        cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 4); time.sleep(3)

    if RP1 and ROTOR_CLIMB and not HYBRID_CLIMB:
        # [MOD-3] ROTOR-BORNE 1/4 CLIMB: OFFBOARD MC velocity (forward RC_VX North + up RC_GRAD*RC_VX) so the
        # LIFT ROTORS carry the climb to CRUISE_ALT, then transition to FW. The rotors easily provide the
        # vertical climb force a forward-thrust-limited pusher cannot -> a true 1/4 gradient (vs FW's ~1/6).
        vclb = RC_GRAD*RC_VX
        set_param(m, "MPC_XY_VEL_MAX", RC_VX+8.0); set_param(m, "MPC_Z_VEL_MAX_UP", vclb+3.0)
        set_param(m, "MPC_TILTMAX_AIR", 35.0)          # allow the nose-up/forward tilt for the wing-borne RC_VX
        set_param(m, "VT_FWD_THRUST_EN", 0)            # [NEW-SPEC P1] PUSHER (C) OFF during the vertical takeoff -- VProp
        set_param(m, "VT_FWD_THRUST_SC", 0.0)          # (lift rotors) ONLY. Re-enabled (EN=1) AFTER P3, before the climb.
        # [P1/P2] MC yaw gains LEFT AT BASE -- rate P higher(0.35->4.6)/lower(0.10->5.0) AND attitude MC_YAW_P=5 (->osc 0.52,
        #   drift 7.97) ALL made the soft rotor-borne yaw WORSE. The base is the floor; instead P2 holds the P1-END heading.
        if FT_OVERLAP: set_param(m, "VT_LIFT_HND_V", FT_LIFT_HND)   # [RP-1e] FT overlap: hold rotors through MC->FW
        if FT_OVERLAP: set_param(m, "VT_LIFT_HND_THR", FT_LIFT_HND_THR)  # [P5 HANDOFF] absolute lift-rotor support to bridge valley
                                                       # pusher carries FORWARD (the user's hybrid). Cleared below.
        # [BODY-FORWARD climb] capture the takeoff heading and command the climb velocity ALONG it (not world-North).
        # The old vN=15 (world North) vs the ~East takeoff heading forced a -29 deg course-capture bank; aligning the
        # velocity with the nose -> no bank, the vehicle just pitches forward along its heading.
        _hdg0 = None
        for _ in range(20):
            _vh = m.recv_match(type="VFR_HUD", blocking=True, timeout=0.5)
            if _vh is not None and getattr(_vh, "heading", None) is not None: _hdg0 = float(_vh.heading)
            if _hdg0 is not None: break
        if _hdg0 is None: _hdg0 = 0.0
        # ===== [STAGE-2 PHASE 1-3] vertical 10 m takeoff -> Hover1 (keep takeoff heading) -> Hover&set (yaw to NORTH) =====
        _tko_hdg = _hdg0                                  # P0 spawns nose EAST(~90); P2 Hover1 keeps this takeoff heading
        gp = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        _hov_lat = gp.lat/1e7; _hov_lon = gp.lon/1e7      # hold this position through the hover + yaw-set
        # P1 Take off: OFFBOARD pure-vertical climb to HOVER_ALT (10 m) on a P9 vUp ramp (no forward) -> a clean lift-off
        tv0 = time.time(); va = 0.0
        cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
        while time.time()-tv0 < 30:
            wv = P9(min(1.0, (time.time()-tv0)/RC_RAMP_T))
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_VEL_YAW,    # [P1] P9 vUp ramp + HOLD yaw
                0, 0, 0.0, 0.0, 0.0, float(-1.5*wv), 0, 0, 0, math.radians(_tko_hdg), 0)
            if time.time()-tv0 < 1.5: cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            v = m.recv_match(type="VFR_HUD", blocking=False)
            if v: va = v.alt
            if va >= HOVER_ALT-0.5: break
            time.sleep(0.1)
        print(f"=> [P1 Take off] vertical lift-off to {va:.1f} m (target {HOVER_ALT:.0f})", flush=True)
        # P2 Hover1: hold 10 m + the takeoff heading ~8 s. HOLD THE CURRENT position (captured here), NOT the takeoff point
        # (the vehicle drifted during the velocity-only P1 climb; commanding the takeoff point would pull it back = a transient).
        _g2 = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        _p2_lat = (_g2.lat/1e7) if _g2 else _hov_lat; _p2_lon = (_g2.lon/1e7) if _g2 else _hov_lon
        _p2_hdg = (_g2.hdg/100.0) if (_g2 and _g2.hdg != 65535) else _tko_hdg   # HOLD the P1-END heading (soft rotor yaw drifted
        #   ~4 deg off the spawn during the climb; commanding the spawn heading would force a slow 4 deg convergence = the "drift").
        # [D-fix TESTED -> REVERTED] settle + median re-capture of _p2_hdg did NOT reduce the drift (1.00 vs 0.98): the
        # "drift" is the climb-yaw SETTLING TAIL caught by the fixed compliance window [alt>9.3, +8 s], not a bad capture.
        # The soft-rotor yaw floor during the velocity-only climb is ~1.0 deg; gains make it worse (see memory). Left as-is.
        for _ in range(80):
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POS_YAW,
                int(_p2_lat*1e7), int(_p2_lon*1e7), float(HOVER_ALT), 0, 0, 0, 0, 0, 0, math.radians(_p2_hdg), 0)
            time.sleep(0.1)
        print(f"=> [P2 Hover1] held P1-end heading {_p2_hdg:.0f} (takeoff ~{_tko_hdg:.0f}) at {HOVER_ALT:.0f} m (8 s)", flush=True)
        # [NEW-SPEC P3] Hover and set: P9-snap yaw from the ACTUAL P2 heading -> EAST (90 deg) at constant alt + held position.
        # Ramp from _p2_hdg (the real current heading), NOT the spawn -> no initial step; hold the P2 XY position.
        SET_HDG = 90.0                                    # new RP-1: out-leg points EAST (was North 0)
        _yaw_err = (SET_HDG - _p2_hdg + 540.0) % 360.0 - 180.0   # signed shortest turn from the P2 heading to East
        ty0 = time.time(); YAW_T = 6.0; _p3_hdg = _p2_hdg
        while True:                                       # P9 ramp, then HOLD East until the soft rotor yaw CONVERGES (<1.2 deg)
            el = time.time()-ty0
            yawc = math.radians(_p2_hdg + _yaw_err*P9(min(1.0, el/YAW_T)))
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POS_YAW,
                int(_p2_lat*1e7), int(_p2_lon*1e7), float(HOVER_ALT), 0, 0, 0, 0, 0, 0, yawc, 0)
            g3 = m.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if g3 and g3.hdg != 65535: _p3_hdg = g3.hdg/100.0
            if el > YAW_T and abs((_p3_hdg - SET_HDG + 540.0) % 360.0 - 180.0) < 1.2: break   # converged to East
            if el > YAW_T + 14.0: break                   # timeout guard (soft rotor yaw)
            time.sleep(0.1)
        print(f"=> [P3 Hover&set] yawed {_p2_hdg:.0f} -> EAST (settled hdg={_p3_hdg:.0f}, target {SET_HDG:.0f}, {time.time()-ty0:.0f}s)", flush=True)
        _hdg0 = SET_HDG                                   # P4+ climb/cruise run along EAST; P7 turn CW 90->270 (West)
        _chdg = math.cos(math.radians(_hdg0)); _shdg = math.sin(math.radians(_hdg0))
        set_param(m, "VT_FWD_THRUST_EN", 1)              # [NEW-SPEC] re-enable the PUSHER for the climb (P4/5 use V+C); OFF only during P1
        set_param(m, "VT_FWD_THRUST_SC", 1.0)
        print(f"=> ROTOR-BORNE climb: OFFBOARD MC body-forward {RC_VX:.0f} m/s along hdg={_hdg0:.0f} vUp={vclb:.1f} (grad 1/{1/RC_GRAD:.0f}) -> {CRUISE_ALT_ABS:.0f} m", flush=True)
        tc0 = time.time(); rc_alt = (0.0 if _offb_tko else TKO_ALT)   # OFFBOARD takeoff starts from the ground
        _p4_done = False; P5_VX = 48.0                     # [P4/P5 split] mark @152 m; P5 ramps fwd RC_VX->P5_VX (V->C, best-rate)
        while time.time()-tc0 < 200:
            for _ in range(5):                          # stream OFFBOARD velocity setpoints at ~10 Hz
                # [SNAP] P9-ramp the forward+up velocity command 0->full over RC_RAMP_T instead of stepping to
                # RC_VX from hover. The step commanded an instantaneous accel the finite-BW MC could not track
                # cleanly -> a -30 deg course-capture bank. P9 (snap-continuous) eases it in -> small bank.
                w = P9((time.time()-tc0) / RC_RAMP_T)
                # [pitch-osc NOTE 2026-06-25] the +-11deg/0.14Hz takeoff/hover pitch osc is NOT a position-velocity
                # competition: the OFFBOARD climb is already velocity-only (MASK_VEL) and a vertical velocity-only climb
                # [vN gated on alt, tested rpf60] STILL rang +-11deg. It is the ROTOR-BORNE HOVER pitch limit-cycle,
                # intrinsic to the heavy airframe MC control (the slanted climb is +-6deg only because the forward speed
                # adds wing lift that stabilises pitch). MPC detune [XY_P and XY_VEL] both made it WORSE. Left as-is.
                # [OFFBOARD takeoff] lift off PURELY VERTICAL (vUp ramp, no forward) until airborne, then ramp the
                # body-forward velocity -> a clean vertical liftoff (no ground-slide) merging into the slanted 1/4 climb.
                _vfwd = w if (not _offb_tko or rc_alt > 8.0) else 0.0   # [P4] rotor-borne 1:4 climb at RC_VX; P5 = the FW transition below
                m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_VEL,
                    0, 0, 0.0, float(RC_VX*_vfwd*_chdg), float(RC_VX*_vfwd*_shdg), float(-vclb*w), 0, 0, 0, 0, 0)
                if time.time()-tc0 < 1.5:               # ensure a setpoint stream exists BEFORE switching mode
                    cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
                time.sleep(0.1)
            v = m.recv_match(type="VFR_HUD", blocking=False)
            if v: rc_alt = v.alt
            if rc_alt >= GATE_ALT:                          # [P4 done @152 m] -> END the rotor-borne 1:4 climb; P5 = FW transition
                print(f"  => [P4 Climb1/4 done @{rc_alt:.0f} m] 1:4 grad -> P5 Climb X: FW transition (V->C) + best-climb to {CRUISE_ALT_ABS:.0f} m", flush=True)
                break
        print(f"=> rotor-borne (P4) climb done alt={rc_alt:.0f}", flush=True)
        set_param(m, "VT_FWD_THRUST_EN", 0)            # [MOD-3] stop the hover pusher-forward assist (cruise is FW)
        # hand off to AUTO, then transition to FW for the cruise
        cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3); time.sleep(2)

    cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3); time.sleep(3)
    cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 4)
    print("=> DO_VTOL_TRANSITION FW", flush=True)

    t0 = time.time(); last = 0; maxR = maxP = vmax = 0; vs = 0; seen = set()
    fw_done_t = None; cur_lat = cur_lon = None; cur_hdg = 0.0; cur_alt = 0.0; cur_as = 0.0; cur_vz = 0.0; cur_roll = 0.0; cur_pitch = 0.0
    cur_vN = 0.0; cur_vE = 0.0    # ground velocity N/E -> course (velocity direction) for COORDINATED (zero-sideslip) yaw cmd
    cur_vN = 0.0; cur_vE = 0.0   # [T1-FIX] ground velocity N/E (m/s) for the cruise-back lateral-loop closure
    _rollout_t0 = 0.0; _rollout_roll0 = 0.0; _rollout_yaw0 = 0.0; _rollout_pitch0 = 0.0   # [attitude-cmd roll-out] state
    _hc_t0 = 0.0; _hc_psi0 = 0.0; _hc_dpsi = 0.0; _hc_T = 1.0; _hc_pitch0 = 0.0           # [attitude-cmd home-capture turn] state
    phase = "fw"; phase_t = time.time(); back_done = False
    _desc0 = None; _ldesc = 0.0
    _g_lat = _g_lon = None; _g_hdg = 0.0; _gd_alt = None; _gd_t = 0.0
    _mg_lat = _mg_lon = None; _mg_hdg = 0.0; _mg_alt = None; _mg_t = 0.0
    _decel_as0 = 0.0; _decel_t0 = 0.0; _decel_alt0 = 0.0
    _spoil_cur = 0.0; _sdesc_t0 = None; _sdesc_alt0 = 0.0   # [MOD-4b] speed-sched spoiler state + descent-start latch
    _pretrim_done = False   # [RP-1f] one-shot: raise FW_RR_I just before the back-trans to re-trim the roll
    _lvl_rs = 0.40; _grad_alt = 0.0   # [RP-1f 3-term] lift-rotor support level during the level decel (ramped on speed)
    _rotsup_t0 = None; _rotsup_cur = 0.0; _rotsup_done = False   # [TASK-1] level-decel lift-rotor-support P9 ramp state
    _push_cut_t0 = None                                          # [C-smooth2] pusher P9 ramp-down start (brake-entry lift-gap bridge)
    _ff_lat = _ff_lon = None; _ff_t = 0.0          # advancing velocity-FF position point (zero-alt decel)
    _vt_alt0 = 0.0; _vt_lat = _vt_lon = None       # vtail descent start state
    _brake_t = None; _brake_v0 = 0.0               # hover-slam P9 brake entry time + entry sink
    _p11_done = False; _hl_lat = _hl_lon = None; _hy_t0 = 0.0; _hy_yaw0 = 0.0; _hy_dpsi = 0.0; _hy_settled = False  # [STAGE-2 P11] pre-land yaw-to-East hover at 10 m
    _rp_lat0 = _rp_lon0 = None; _turn_hdg0 = 0.0   # RP-1 out-leg start + 180-turn entry heading
    _cb_lat0 = _cb_lon0 = None                     # RP-1 cruise-back leg start
    _cb_offb = False; _ret_hdg = 0.0               # [SP-SMOOTH] hold the straight back-leg under OFFBOARD wings-level velocity
    _cb_course0 = None                             # [T1-FIX] ground course captured at the cruise-back handoff (continuous -> no capture bank)
    _turn_offb = False; _turn_psi0 = 0.0           # [SP-SMOOTH P7] attitude-cmd CW coordinated turn (replaces the nav loiter)
    _last_clmb = CLIMB_CLMB_MAX; _cruise_t0 = 0.0; _cruise_as0 = 54.0; _cruise_last = 0.0  # [P5 flare + P6 P9 airspeed ramp]
    _cruise_offb = False; _cruise_pitch0 = 0.0; _cruise_hdg0 = 90.0   # [P6 SP-SMOOTH] attitude-cmd East cruise (level-off + accel)
    _p8_offb = False; _p8_t0 = 0.0; _p8_as0 = 72.0                    # [P8 Cruise Brake] level spoiler brake 72->38 @300m West
    _p8_reached = False; _p8_reach_t = 0.0                            # [P8] 38 m/s reached latch + hold-window timer
    _p8_rs_last = 0.0                                                 # [P8] rotor-support (VT_FW_MC_THR) last-commanded value (P9 ramp, no slam)
    _p9_offb = False; _p9_t0 = 0.0; _p9_rs_last = 0.0; _p9_alt0 = 300.0  # [P9 Descend X] 38-hold descent 300->152 state
    _hdg_i = 0.0                                                          # [Q1] heading-hold INTEGRAL accumulator (null the -1.5 deg proportional steady error)
    _p10bleed_offb = False; _p10bleed_t0 = 0.0                            # [P10a FW pre-bleed] 38->20 @152 before the back-trans (lower the handover speed)
    _home_lat = _home_lon = None                   # takeoff point (for the return-to-home back legs)
    _home_brg = 0.0                                # FROZEN heading toward home (straight approach, no wp thrash)
    _last_reaim = 0.0                              # last time the home-bearing was re-frozen (periodic, gentle)
    _ar_lat = _ar_lon = None; _ar_alt = None
    _fw_retries = 0; _last_fw_retry = time.time()   # FW-transition auto-retry guard (SITL trans non-determinism)
    _last_bleed_print = 0.0                          # [MOD-3 FIX] throttle the gate-bleed log line
    try:
        turn_rad = float(open("/tmp/mk7_turnrad").read().strip()) if os.path.exists("/tmp/mk7_turnrad") else 0.0
    except Exception: turn_rad = 0.0
    def gophase(p):
        nonlocal phase, phase_t; phase = p; phase_t = time.time()
        print(f"  ===> PHASE {p} (t={time.time()-t0:.0f} alt={cur_alt:.0f} as={cur_as:.1f})", flush=True)
    while time.time()-t0 < a.timeout:
        msg = m.recv_match(type=["VFR_HUD","EXTENDED_SYS_STATE","GLOBAL_POSITION_INT","ATTITUDE"], blocking=True, timeout=2)
        if not msg: continue
        ty = msg.get_type(); now = time.time()
        # FW-transition auto-retry: the single DO_VTOL_TRANSITION sometimes does not engage (SITL
        # non-determinism: greenB hovered in MC at as=0.8 for 50 s, never reached vtol_state 4). Re-send.
        if fw_done_t is None and now-t0 > 20 and now-_last_fw_retry > 18 and _fw_retries < 4:
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 4)
            _last_fw_retry = now; _fw_retries += 1
            print(f"  >>> FW transition RETRY #{_fw_retries} (vs={vs} as={cur_as:.1f} alt={cur_alt:.0f} t={now-t0:.0f})", flush=True)
        if ty == "GLOBAL_POSITION_INT":
            cur_lat = msg.lat/1e7; cur_lon = msg.lon/1e7; cur_hdg = msg.hdg/100.0; cur_alt = msg.relative_alt/1000.0
            cur_vz = msg.vz/100.0   # GLOBAL_POSITION_INT vz cm/s, +down (sink) -> for the lift-dump spoiler feedback
            cur_vN = msg.vx/100.0; cur_vE = msg.vy/100.0   # [T1-FIX] ground vel N/E (cm/s->m/s) for lateral-loop closure
            if _home_lat is None: _home_lat, _home_lon = cur_lat, cur_lon   # takeoff point = return target
            # write the vehicle gz-ENU position (x=East, y=North, z=Up) for the chase-cam follow loop
            try:
                _E = (cur_lon - 8.546163739800146) * 111320.0 * math.cos(math.radians(47.397971057728974))
                _N = (cur_lat - 47.397971057728974) * 111320.0
                with open("/tmp/mk7_vehpos", "w") as _vf:
                    _vf.write("%.2f %.2f %.2f %.1f" % (_E, _N, cur_alt, cur_hdg))
            except Exception:
                pass
        elif ty == "ATTITUDE":
            cur_roll = math.degrees(msg.roll); cur_pitch = math.degrees(msg.pitch)
            maxR = max(maxR, abs(math.degrees(msg.roll))); maxP = max(maxP, abs(math.degrees(msg.pitch)))
        elif ty == "VFR_HUD":
            cur_as = msg.airspeed; vmax = max(vmax, msg.airspeed)
            try:
                with open("/tmp/mk7_as.txt", "w") as _f: _f.write("%.2f" % msg.airspeed)
            except Exception: pass
            if now-last > 3:
                last = now
                print(f"  t={now-t0:4.0f} ph={phase} as={cur_as:5.1f} alt={cur_alt:5.0f} vstate={vs} R={maxR:.0f} P={maxP:.0f}", flush=True)
        elif ty == "EXTENDED_SYS_STATE":
            vs = msg.vtol_state
            if vs not in seen: seen.add(vs); print(f"  *** vtol_state -> {vs} t={now-t0:.0f}", flush=True)
            if vs == 4 and fw_done_t is None:
                fw_done_t = now-t0
                print(f"  *** FW TRANSITION COMPLETE t={fw_done_t:.0f} ***", flush=True)
                if cur_lat is not None and RP1:
                    # RP-1: climb to 300 m ABSOLUTE at CLIMB_AS (pitch holds airspeed -> no zoom). Cruise
                    # airspeed (72) is set once at the top so the level legs are at 72 m/s.
                    set_param(m, "FW_THR_MAX", CLIMB_THR_MAX); set_param(m, "FW_T_CLMB_MAX", CLIMB_CLMB_MAX)
                    # [MOD-2] Vy MAX-RATE CLIMB: PIN the throttle floor near max (FW_THR_MIN=0.90) so TECS cannot
                    # leave surplus thrust unused, and SPDWEIGHT=2 -> pitch HOLDS the best-climb airspeed so the
                    # forced surplus energy all becomes CLIMB. (SPDWEIGHT=1 + free throttle was TECS-limited:
                    # thr 0.67, climb 1/26, airspeed drooped to 39.) Raise pitch limit for the steep gradient.
                    set_param(m, "FW_THR_MIN", CLIMB_THR_MIN)
                    set_param(m, "VT_FW_MC_THR", CLIMB_MC_THR)    # [MOD-3] rotor-assist: lift rotors carry weight
                    set_param(m, "FW_T_SPDWEIGHT", 2.0); set_param(m, "FW_AIRSPD_TRIM", CLIMB_AS)
                    set_param(m, "FW_AIRSPD_MAX", CLIMB_AS_CAP); set_param(m, "FW_AIRSPD_MIN", 44.0)
                    set_param(m, "FW_P_LIM_MAX", 28.0)            # allow a steep pitch-up for the 1/4 climb
                    # [#2] FW_PR_D=0.08 made the climb pitch-osc WORSE (0.75->0.93, the rate-D amplified noise) -> reverted.
                    _rp_lat0, _rp_lon0 = cur_lat, cur_lon
                    # CLOSE climb waypoint -> a STEEP demanded gradient so it climbs IMMEDIATELY.
                    tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 1500.0)
                    repo(m, tlat, tlon, CRUISE_ALT_ABS)
                    print(f"  => RP-1 CLIMB to {CRUISE_ALT_ABS:.0f} m @ {CLIMB_AS:.0f} m/s (cap {CLIMB_AS_CAP:.0f}, wp 3km)", flush=True)
                    gophase("climbout")
                else:
                    if cur_lat is not None:
                        tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 15000.0)
                        repo(m, tlat, tlon, CRUISE); print("  => straight settle 15km", flush=True)
                    gophase("settle")
        if cur_lat is None or fw_done_t is None: continue

        # [RP-1f 3-term (5) TESTED -> NO EFFECT] ramping VT_FW_MC_THR (rotor collective) up with speed did NOT hold
        # altitude (still sagged 92 m): in FW the TECS controls height via PITCH and does not coordinate with the
        # open-loop rotor collective -> the added rotor lift is offset by the TECS pitching down. A true level decel
        # needs the MC velocity controller (vz=0) to OWN the vertical axis, not the FW TECS. Reverted.

        # [SP-SMOOTH] CRUISE-BACK streaming hook (runs every loop iteration while the straight leg is held under OFFBOARD):
        # command a wings-level VELOCITY along the reverse heading (MASK_ALTVEL = hold cruise alt + vN/vE, NO lat/lon
        # position ref) -> the FW flies dead-straight and roll_sp stays ~0. No nav waypoint => no cross-track wiggle in
        # the setpoint => no ring. This replaces handing the leg to NPFG (whose bank command oscillated).
        if _cb_offb and cur_lat is not None and phase in ("cruiseback", "lvlbleed", "lvldecel"):
            # [T1-FIX 2026-06-26] CLOSE THE LATERAL LOOP with a CONSISTENT velocity setpoint (was: pure wings-level
            # ATTITUDE, _ret_hdg fixed yaw). DIAGNOSIS (rpf82 ulog, turn-exit dump): the old pure-attitude command had
            # roll_sp==0 yet the REAL roll departed to -50.8 deg ~4 s into cruise-back. Cause: an OPEN lateral loop +
            # an INCONSISTENT fixed-yaw command -- in FW you cannot hold a yaw setpoint wings-level, so the heading
            # drifts off _ret_hdg (drifted 25 deg), sideslip builds, and the lateral mode departs (the EXACT failure the
            # rollout comment fixed by TRACKING yaw, but left unfixed here). roll_sp was never the demand -> it is not a
            # commanded bank, it is an open-loop departure. FIX (continuous & consistent, no capture bank): command a
            # VELOCITY along the course CAPTURED AT THE HANDOFF (_cb_course0 == the actual ground course at entry, so the
            # setpoint direction == the current velocity -> ZERO course-capture bank), MAGNITUDE == the current measured
            # ground speed (so the velocity loop does NOT fight the spoiler/idle decel: along-track error stays ~0, only
            # the cross-track drift is nulled -> roll_sp ~0 AND real roll ~0). Altitude held by the z(alt) ref. This is
            # CLOSED-loop -> it cannot drift off and depart, killing the -50 deg spike while keeping the leg dead-straight.
            # [SP-SMOOTH v3] command roll=0 DIRECTLY with yaw=COURSE (velocity dir) -> zero sideslip AND zero commanded
            # bank. (The velocity-along-captured-course killed the spike too but held a ~15 deg bank chasing a stale
            # captured course; roll=0/yaw=course is straighter and demands no bank.) Altitude via the alt-hold pitch;
            # throttle = cruise on the leg, idle in the decel (the spoiler + lift-rotors do the braking).
            alt_err = CRUISE_ALT_ABS - cur_alt
            pitch_c = math.radians(max(-3.0, min(9.0, 2.0 + 0.30*alt_err)))
            thr = 0.58 if phase == "cruiseback" else 0.03        # [C-smooth2 REVERTED] pusher P9 ramp-down WORSENED alt (range 6.4->11.2 m: it disturbs brake energy mgmt); instant cut kept. P8 alt is set by the TURN APEX, not the cut.
            course = math.degrees(math.atan2(cur_vE, cur_vN))
            qd = quat_from_euler(0.0, pitch_c, math.radians(course))
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [TASK-1 FIX] LEVEL-DECEL rotor-support + spoiler P9 ramp (snap-smooth, in-regime). Spin VT_FW_MC_THR 0->0.6
        # and deploy the spoiler 0->LVL_SPOIL gradually so the 12 lift rotors engage gently (vs slammed on at high q
        # -> the -50 deg roll disturbance). roll I is dropped only AFTER the ramp finishes (full authority during it).
        if phase == "lvldecel" and _rotsup_t0 is not None and not _rotsup_done:
            w = min(1.0, (now - _rotsup_t0) / ROTSUP_RAMP_T); f = P9(w)
            set_flap(LVL_SPOIL * f)                                   # spoiler ramps with the same P9 (flap loop reads /tmp @3 Hz)
            target_mc = ROTSUP_MAX * f
            if target_mc - _rotsup_cur >= ROTSUP_STEP or (w >= 1.0 and _rotsup_cur < ROTSUP_MAX - 1e-3):
                _rotsup_cur = target_mc; set_param(m, "VT_FW_MC_THR", round(_rotsup_cur, 3))
            if w >= 1.0:
                _rotsup_done = True
                set_param(m, "FW_RR_I", 0.1); set_param(m, "FW_RR_D", 0.01)   # drop roll I / add D ONLY after rotors up
                print(f"  => [TASK-1] rotor-support ramp complete (VT_FW_MC_THR={ROTSUP_MAX:.2f}, spoiler={LVL_SPOIL}); FW_RR_I->0.1", flush=True)

        # [SP-SMOOTH P7] TURN streaming hook: attitude-cmd CW coordinated turn. roll = TRAPEZOID keyed to the heading
        # swept (P9 ramp 0->45, HOLD 45 = best turn rate, P9 ramp 45->0), CW => +roll. yaw_sp = cur_hdg (the bank does
        # the turning; no rudder fight). pitch = turn-AoA (scales with bank) + alt-hold. roll_sp is smooth -> no nav wiggle.
        if phase == "turn180" and _turn_offb and cur_lat is not None:
            elapsed = now - phase_t
            dh = (cur_hdg - _turn_psi0 + 360.0) % 360.0
            if dh > 270.0: dh = 0.0                                     # entry wrap guard (heading dipped below psi0)
            T_ENTRY = 8.0   # [C-smooth1c] gentler bank roll-in (6->8 s) -> slower lift-vertical drop at bank onset -> smaller entry sink -> smaller recovery over-climb (lower apex)
            f = P9(elapsed / T_ENTRY) if elapsed < T_ENTRY else 1.0    # TIME-based P9 ease-in 0->45, then HOLD 45 = best
            #   rate. The roll-OUT is handed to the dedicated rollout phase below (proven clean P9 45->0) at dh~155, NOT
            #   a dh-based ramp here (that + the homecap handoff produced a -48 deg exit spike).
            bank = math.radians(TURN_BANK) * f                         # + = CW (right bank)
            alt_err = CRUISE_ALT_ABS - cur_alt
            pitch_c = math.radians(max(-7.0, min(13.0, 2.0*f + 0.85*alt_err + 0.55*cur_vz)))   # [C-smooth1b] turn-AoA FF 4.5->2.0 (was 3x the real bank load-factor need -> excess-lift +14 m climb), STIFFER alt-hold 0.55->0.85 + more vz-damp -> hold 300 in the bank
            qd = quat_from_euler(bank, pitch_c, math.radians(cur_hdg))
            thr_turn = max(0.30, min(0.90, 0.58 + 0.10*(CRUISE_AS - cur_as)))   # [C-smooth1 REVERTED] speed-tighten did NOT reduce the turn climb (excess speed refuted) and perturbed the rollout; back to base
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr_turn)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [P6 SP-SMOOTH CRUISE-OUT hook] OFFBOARD ATTITUDE each loop: roll=0, yaw=EAST(90), pitch = P9 level-off [entry climb
        # pitch -> alt-hold] so the climb vz momentum is bled smoothly (no TECS bounce); throttle FOLLOWS a P9 airspeed
        # target cur_as->72 (jerk-continuous accel via the pusher). No nav (no heading drift), no TECS (no alt bounce).
        if phase == "cruiseout" and _cruise_offb and cur_lat is not None:
            w = now - _cruise_t0
            wl = P9(min(1.0, w / 4.0))                                  # blend entry-attitude -> the cruise controller over 4 s
            # PITCH = alt-hold(300) + vz-DAMP (the vz-damp arrests the climb momentum -> no +40 m level-off overshoot)
            pitch_ctrl = max(-6.0, min(7.0, 0.5 + 0.50*(CRUISE_ALT_ABS - cur_alt) + 0.80*cur_vz))   # alt-hold(stiff) + vz-DAMP -> above/climbing pushes the nose DOWN hard (extra accel-thrust -> SPEED not altitude)
            pitch_c = _cruise_pitch0*(1.0 - wl) + pitch_ctrl*wl
            # ROLL = heading-HOLD: bank proportional to the heading error -> actively steer the heading to EAST(90). roll=0
            # cannot hold a heading in FW (it just drifts to the trim ~77); a proportional bank converges it to 90.
            herr = (90.0 - cur_hdg + 540.0) % 360.0 - 180.0
            roll_c = max(-12.0, min(12.0, 0.5*herr)) * wl
            as_tgt = _cruise_as0 + (CRUISE_AS - _cruise_as0) * P9(min(1.0, w / 15.0))   # P9 airspeed target cur_as->72 (TAS, calibrated)
            # [#3] HOLD true 72: baseline 0.65 balanced at ~77 (overshoot). Lower to 0.55 (the ~72 cruise thrust) + a stronger
            # gain so the throttle backs off HARD as cur_as approaches/exceeds 72 -> no +5 over-speed into the turn.
            thr = max(0.25, min(0.95, 0.48 + 0.10*(as_tgt - cur_as)))   # [C-smooth1 REVERTED to base] (speed not the climb source)
            qd = quat_from_euler(math.radians(roll_c), math.radians(pitch_c), math.radians(cur_hdg))   # yaw=cur_hdg -> coordinated, no sideslip
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [P8 Cruise Brake hook] (A) LEVEL spoiler brake 72->38 at 300 m, WEST(270) held. OFFBOARD attitude: roll=heading-hold
        # (West), pitch=alt-hold(300)+vz-damp (compensate the spoiler lift-dump -> hold 300), yaw=cur_hdg (coordinated). The
        # FLAPERON SPOILER (set_flap lift-dump drag, 45deg=30x) ramps IN on a P9 then FADES (P9) as as->38 -> the DECEL is
        # jerk-continuous (P9-snap). THROTTLE = airspeed-hold targeting 38 (idle while fast -> decel; cruise at 38 -> hold).
        # Decel and descent are SEPARATED here (memory rule: FW cannot decel in a descent; brake level, then descend at 38).
        if phase == "p8brake" and _p8_offb and cur_lat is not None:
            w = now - _p8_t0
            # ROTOR SUPPORT (KEY): P9 ramp VT_FW_MC_THR 0->0.55 over 8 s -> the 12 lift rotors hold altitude + STABILIZE the
            # attitude as the wing unloads. Without this the spoiler lift-dump + the dropping speed stalled the wing -> tumble
            # (rpf117: roll 88, pitch 70). The P9 ramp engages the rotors GENTLY (slamming them on at high q gave a roll spike).
            # rotor support = P9 ramp-in (0->0.50) + ALT-ERROR modulation (closes the alt loop via the rotors: alt dip -> more
            # support -> climb back). This kills the 26 m mid-brake dip [rpf118: the fixed 0.55 could not hold alt as the wing
            # unloaded]. The P9 ramp engages the rotors GENTLY (no high-q roll spike).
            rs_base = 0.50 * P9(min(1.0, w / 9.0))
            rs_tgt = max(0.10, min(0.85, rs_base + 0.035*(CRUISE_ALT_ABS - cur_alt)))   # [#5] stiffer alt-hold via rotors
            if abs(rs_tgt - _p8_rs_last) >= 0.03:
                _p8_rs_last = rs_tgt; set_param(m, "VT_FW_MC_THR", round(rs_tgt, 3))
            # SPOILER: gentle P9 ramp-in, FADE (P9) as as->38 -> jerk-continuous decel; the rotors (not high wing AoA) hold alt
            ramp_in = P9(min(1.0, w / 8.0)); fade = P9(max(0.0, min(1.0, (cur_as - P8_BRAKE_AS) / 10.0)))
            set_flap(P8_SPOIL * ramp_in * fade)
            # attitude: GENTLE heading-hold BANK -> WEST(270) [coordinated, now rotor-stabilized so it cannot tumble]; yaw=COURSE
            # (track the velocity, zero sideslip, coordinate with the bank); MILD alt-hold pitch (rotors hold alt -> wing AoA LOW,
            # no stall). throttle idle while fast (brake), cruise at 38 (hold).
            alt_err = CRUISE_ALT_ABS - cur_alt
            pitch_c = math.radians(max(-3.0, min(7.0, 2.0 + 0.40*alt_err)))   # [#5] stiffer pitch alt-hold
            course = math.degrees(math.atan2(cur_vE, cur_vN))
            herr = (P8_BRAKE_HDG - (cur_hdg % 360.0) + 540.0) % 360.0 - 180.0
            roll_c = max(-12.0, min(12.0, 0.5*herr))                      # bank back to WEST(270) (rotor-stabilized, coordinated)
            thr = max(0.03, min(0.60, 0.50 + 0.05*(P8_BRAKE_AS - cur_as)))
            qd = quat_from_euler(math.radians(roll_c), pitch_c, math.radians(course))
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [P9 Descend X hook] descend 300->152 on a P9 altitude profile while HOLDING 38 m/s. Inherits P8's rotor-supported
        # regime. ROLE SPLIT: rotor support = DESCENT tracking (alt-error vs the descending profile -> less lift -> sink);
        # SPOILER = SPEED hold (dissipate the gravity PE, more drag if as>38 -> hold 38); pitch = mild; throttle = idle
        # (MINIMAL drive energy, gravity drives the descent). West(270) held. Finish at 152 m at exactly 38 m/s.
        if phase == "p9descend" and _p9_offb and cur_lat is not None:
            w = now - _p9_t0
            frac = min(1.0, w / P9_DESC_T)
            # [P9 sink, user hypothesis] FLAT trapezoidal sink (snap ramp-in + constant wmax + snap ramp-out), NOT the
            # PEAKED snap-velocity 630u^4(1-u)^4 (which spiked to 4.28 m/s mid-descent -> a gravity-power spike mg*w ->
            # V rose 38->41). A flat low-peak w keeps mg*w ~constant -> V flat. wmax set so the total drop = H.
            _RAMP = 0.20; _H = _p9_alt0 - GATE_ALT; _wmax = _H / (P9_DESC_T * (1.0 - _RAMP))
            if frac < _RAMP:           _shp = P9(frac / _RAMP)
            elif frac > 1.0 - _RAMP:   _shp = P9((1.0 - frac) / _RAMP)
            else:                      _shp = 1.0
            vz_tgt = _wmax * _shp                                          # FLAT sink-rate feedforward (+down), peak ~2.0
            alt_tgt = _p9_alt0 - _H * frac                                 # ~linear alt (flat sink); the 0.008 fb trims
            # SPEED HELD AT 38 BY A DUAL bracket so it NEVER dips into stall (rpf122 dropped to 28 -> wing stall -> roll
            # osc P=34). THROTTLE rises when SLOW (authority to 0.75); SPOILER adds drag only when FAST. Together they pin 38.
            # [P9 true-38] the old bracket (thr 0.34, spoil -0.20) balanced at ~45 TAS in the descent (gravity PE drives it up).
            # Lower the throttle DRIVE (0.34->0.20: a descent needs little thrust to hold 38) + raise the spoiler DRAG baseline
            # (-0.20->-0.42) so the equilibrium drops to 38 at 152 m. Keep the strong slow-side throttle gain (stall safety).
            thr = max(0.05, min(0.75, 0.10 + 0.09*(P8_BRAKE_AS - cur_as)))            # [P9 V] low drive at 38 (the 0.34 baseline pushed V to 40.5); strong slow-gain = stall safety
            spoil = max(-0.95, min(0.0, -0.20 - 0.13*(cur_as - P8_BRAKE_AS)))         # [P9 speed] stronger speed-feedback to pull to true 38 (the 2.12 roll was a window artifact; clean roll stays <0.5)
            set_flap(spoil)
            # DESCENT via rotor support on a vz-RATE feedforward+feedback (NOT alt-position -> that oscillated vz -3..9.5) +
            # a gentle alt trim. Sink faster -> less lift; the FF tracks the smooth P9 sink profile.
            rs_tgt = max(0.10, min(0.60, 0.42 - 0.045*(vz_tgt - cur_vz) - 0.008*(cur_alt - alt_tgt)))
            if abs(rs_tgt - _p9_rs_last) >= 0.03:
                _p9_rs_last = rs_tgt; set_param(m, "VT_FW_MC_THR", round(rs_tgt, 3))
            pitch_c = math.radians(max(-3.0, min(3.0, 0.05*(alt_tgt - cur_alt))))     # mild alt trim; the rotors own the vertical
            # [1] TRACK-hold (course-based), NOT heading-hold. Control the GROUND TRACK [course] to West by banking to turn the
            # VELOCITY VECTOR; let the nose follow the velocity [yaw=course] so beta=0, vy_body=0 [coordinated]. The nose may
            # crab off West - that is fine; what matters is the TRACK. This carries vy_body~0 INTO the back-trans so the decel
            # cannot amplify beta=asin(vy/V) -> the entry transient source is removed at the source. [Replaces the heading-hold
            # West which held the NOSE West while the velocity track drifted south -> a persistent kinematic crab.]
            course = math.degrees(math.atan2(cur_vE, cur_vN)) % 360.0
            cerr = (P8_BRAKE_HDG - course + 540.0) % 360.0 - 180.0        # TRACK error to West(270)
            _hdg_i = max(-12.0, min(12.0, _hdg_i + cerr*0.1))
            roll_c = max(-15.0, min(15.0, 0.50*cerr + 0.12*_hdg_i))       # bank to steer the TRACK to West (coordinated turn)
            qd = quat_from_euler(math.radians(roll_c), pitch_c, math.radians(course))   # yaw=course -> nose tracks velocity, beta=0
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [P10a FW pre-bleed hook] decelerate 38->20 in FW @152 (rotor brake + spoiler + idle throttle, like P8) so the
        # subsequent back-trans hands over at a LOW speed -> small FW->MC reconfig transient. West held, alt-hold 152.
        if phase == "p10bleed" and _p10bleed_offb and cur_lat is not None:
            w = now - _p10bleed_t0
            rs_tgt = max(0.30, min(0.85, 0.42 + 0.015*(38.0 - cur_as) + 0.018*(GATE_ALT - cur_alt)))   # rotor support CONTINUES from the P9 ~0.4 (NO ramp-from-0 drop -> no lift-loss spike) + more as speed drops
            if abs(rs_tgt - _p9_rs_last) >= 0.03: _p9_rs_last = rs_tgt; set_param(m, "VT_FW_MC_THR", round(rs_tgt, 3))
            ramp_in = P9(min(1.0, w/12.0)); fade = P9(max(0.0, min(1.0, (cur_as - 18.0)/10.0)))
            set_flap(-0.45 * ramp_in * fade)                             # GENTLE spoiler [-0.45 not -0.90] -> the flaperons do NOT saturate at high speed -> roll authority kept (kills the +-28 roll spike); slower ramp 12 s
            pitch_c = math.radians(max(-3.0, min(8.0, 1.0 + 0.30*(GATE_ALT - cur_alt) + 0.55*cur_vz)))   # alt-hold 152 + vz-damp
            herr = (P8_BRAKE_HDG - (cur_hdg % 360.0) + 540.0) % 360.0 - 180.0
            roll_c = max(-12.0, min(12.0, 0.5*herr))                     # heading-hold WEST
            course = math.degrees(math.atan2(cur_vE, cur_vN))
            thr = max(0.03, min(0.60, 0.45 + 0.05*(16.0 - cur_as)))      # airspeed-hold 16 (idle while fast -> decelerate to ~18)
            qd = quat_from_euler(math.radians(roll_c), pitch_c, math.radians(course))
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111, qd, 0.0, 0.0, 0.0, thr)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # [P10 DIAGONAL hook] CLEAN 1:4 MC descent: command the VELOCITY VECTOR (vx forward + vz=vx/4) + yaw=West via
        # MASK_VEL_YAW -> NO position/alt hold (no pos-vs-vel fight, no alt-hold mask pinning it level). With the MPC
        # velocity limits raised at the back-trans, the MC FOLLOWS the diagonal -> it descends WHILE moving forward.
        # This is the direct proof + fix: the prior "MC cant descend fwd" was the MPC clamp (1.2/1.0 m/s), not physics.
        if phase == "p10diag" and cur_lat is not None:
            # SUSTAINED 1:4: hold a MODERATE forward speed (12 m/s, within the raised MPC_XY_VEL_MAX) + vz=fwd/4 (=3) for the
            # WHOLE descent 152->10 -> a clean constant 1:4 glide. The back-trans brakes the 44->12 entry; then this holds 12.
            # P9-ease the forward 20->12 at entry (snap-smooth) so the MC settles into the glide. At 10 m -> stop (hover).
            VFWD = 12.0
            if cur_alt > HOVER_ALT + 2.0:
                vcmd = VFWD + (min(_decel_as0, 20.0) - VFWD) * (1.0 - P9(min(1.0, (now - _decel_t0) / 8.0)))   # ease entry(clamp20)->12, P9
                vz_cmd = vcmd * 0.25                                      # 1:4 sink = vx/4 (VRS-safe) the whole way
            else:
                vcmd = 0.0; vz_cmd = 0.0                                  # reached 10 m -> stop, bleed to hover
            hd = math.radians(_g_hdg)                                    # velocity along WEST: vN=0 IS the cross-track null
            # [P10 (1)] velocity=West [vN=0] already nulls the cross-track once the MC is active [vstate=3] -> steady vy_body
            # ~0.15, hdg 269 [West], beta=0. TESTED an aggressive null [vN=-1.6*cur_vN, rpf149]: SAME steady, entry WORSE
            # [2.85 vs 2.31] -> reverted. The TRANSITION beta [+13, vstate=2] is the AUTOPILOT back-trans where the OFFBOARD
            # MC velocity is NOT active; vy_body GROWS -4 -> -6.4 as the decel amplifies it before the MC engages at bt+14 ->
            # the entry transient is UNREACHABLE by the OFFBOARD MC velocity [proven by the rpf149 vy_body trace].
            yaw_c = math.atan2(cur_vE, cur_vN) if (abs(cur_vE) + abs(cur_vN) > 1.0) else hd
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_VEL_YAW,
                0, 0, 0, vcmd*math.cos(hd), vcmd*math.sin(hd), vz_cmd, 0, 0, 0, yaw_c, 0)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)

        # ===== RP-1 OUT-AND-BACK: climbout -> cruiseout -> turn180 -> cruiseback -> descend (design shape) =====
        if phase == "climbout" and (cur_alt >= CRUISE_ALT_ABS - 30.0 or now-phase_t > 110.0):
            # [P6 SP-SMOOTH] hold the cruise under OFFBOARD ATTITUDE (roll=0, yaw=EAST, pitch=alt-hold) -> NO nav drift (the
            # nav/NPFG drifted the heading -16.7 deg). A P9 LEVEL-OFF (entry climb pitch -> alt-hold) kills the climb vz
            # momentum (no alt bounce). The THROTTLE follows a P9 airspeed target cur_as->72 (jerk-continuous accel via C).
            set_param(m, "FW_THR_MIN", 0.0); set_param(m, "VT_FW_MC_THR", 0.0); set_param(m, "FW_R_LIM", 15.0)
            _cruise_t0 = now; _cruise_as0 = cur_as; _cruise_pitch0 = cur_pitch; _cruise_hdg0 = cur_hdg; _cruise_offb = True
            _cb_lat0, _cb_lon0 = cur_lat, cur_lon
            print(f"  => RP-1 P6 CRUISE-OUT East (SP-SMOOTH attitude, P9 level-off + P9 accel {cur_as:.0f}->{CRUISE_AS:.0f}) at alt={cur_alt:.0f}", flush=True)
            gophase("cruiseout")
        elif phase == "cruiseout" and TEST_LEVELDECEL and (hdist(_cb_lat0, _cb_lon0, cur_lat, cur_lon) >= CRUISE_DIST or now-phase_t > 30.0):
            # TEST: rotor-assisted back-trans deceleration at CONSTANT 300 m, from cruise speed (>stall).
            set_flap(REV_FLAP); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 3)
            print(f"  *** TEST LEVEL-DECEL back-trans at alt={cur_alt:.0f} as={cur_as:.0f} (above stall 40) ***", flush=True)
            back_done = True; _g_hdg = cur_hdg; _decel_as0 = cur_as; _decel_t0 = now
            _decel_alt0 = cur_alt; _g_lat, _g_lon = cur_lat, cur_lon; _settle_t = None
            _ff_lat, _ff_lon = cur_lat, cur_lon; _ff_t = now; gophase("decel")
        elif phase == "cruiseout" and TEST_DECEL_SWEEP and (cur_as >= 68.0 or now-phase_t > 55.0):
            # [MOD-4 UNIT TEST] FW-GLIDE flaperon-drag test: pusher OFF, hold altitude (SPDWEIGHT=0, level repo),
            # deploy the flaperon to _flap rad. With the new FlaperonDrag plugin a 45deg deploy ADDS drag ->
            # the airspeed decays FASTER than flap=0. Isolates the decel attributable to the new flaperon drag.
            try: _flap = float(open("/tmp/mk7_decel_mcthr").read().strip())   # flaperon deflection (rad)
            except Exception: _flap = 0.0
            set_param(m, "FW_THR_MAX", 0.001); set_param(m, "FW_THR_MIN", 0.0)   # pusher OFF -> pure glide
            set_param(m, "FW_T_SPDWEIGHT", 0.0); set_param(m, "FW_AIRSPD_MIN", 18.0)  # pitch holds HEIGHT
            set_flap(_flap)                                                      # deploy flaperon (drag plugin acts)
            tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 9000.0); repo(m, tlat, tlon, CRUISE_ALT_ABS)
            _decel_t0 = now; _decel_alt0 = cur_alt; _decel_as0 = cur_as
            print(f"  *** GLIDE-DRAG TEST START flap={_flap:.3f}rad as={cur_as:.1f} alt={cur_alt:.0f} ***", flush=True)
            gophase("dsweep")
        elif phase == "dsweep":
            if now-_decel_t0 > 30.0 or cur_as < 24.0:
                print(f"  *** GLIDE-DRAG DONE flap={_flap:.3f} as0={_decel_as0:.1f}->as={cur_as:.1f} (dV={_decel_as0-cur_as:.1f} in {now-_decel_t0:.0f}s) alt={cur_alt:.0f} dAlt={cur_alt-_decel_alt0:+.0f} maxR={maxR:.0f} ***", flush=True)
                print(f"RESULT glidedrag flap={_flap:.3f} dV={_decel_as0-cur_as:.1f} dAlt={cur_alt-_decel_alt0:+.0f}", flush=True)
                break
        elif phase == "cruiseout" and not TEST_DECEL_SWEEP and cur_as >= CRUISE_AS - 2.0 and \
             (hdist(_cb_lat0, _cb_lon0, cur_lat, cur_lon) >= CRUISE_DIST or now-phase_t > 75.0):
            # [P7] only turn AFTER reaching 72 m/s (cur_as>=70) AND 300 m flight -- per spec "72m/s到達かつ300m飛行後"
            # [STAGE-2 P7 / SP-SMOOTH] ATTITUDE-COMMANDED CW coordinated turn 0->180 (replaces the nav loiter whose bank
            # command wiggled). One continuous bank profile for entry+turn+roll-out: roll = trapezoid (P9 ramp 0->45,
            # HOLD 45 = best turn rate, P9 ramp 45->0) keyed to the heading SWEPT; CW => +roll (right bank). The yaw
            # follows the natural coordinated turn (yaw_sp = cur_hdg, no rudder fight). roll_sp is smooth -> no wiggle.
            set_param(m, "FW_R_LIM", TURN_BANK + 3.0)
            _cruise_offb = False                                       # [P6] end the SP-SMOOTH cruise attitude hold -> P7 turn
            _turn_psi0 = cur_hdg; _turn_hdg0 = cur_hdg; _turn_offb = True
            print(f"  => RP-1 P7 TURN 180 CW (attitude-cmd coordinated bank<= {TURN_BANK:.0f}, best-rate) from hdg={cur_hdg:.0f}", flush=True)
            gophase("turn180")
        elif phase == "turn180":
            dh = (cur_hdg - _turn_psi0 + 360.0) % 360.0                 # CW heading swept 0 -> 180 (+)
            if (155.0 <= dh <= 270.0) or now-phase_t > 80.0:            # best-rate turn ~done -> hand to the PROVEN roll-out
                _turn_offb = False; set_param(m, "FW_R_LIM", 15.0)      # attitude turn did entry+mid; rollout does 45->0
                _turn_hdg0 = _turn_psi0                                 # out-leg heading (rollout->homecap uses +180=South)
                _rollout_t0 = now; _rollout_roll0 = cur_roll; _rollout_yaw0 = cur_hdg; _rollout_pitch0 = cur_pitch
                print(f"  => RP-1 P7 turn body done (hdg={cur_hdg:.0f} dh={dh:.0f}) -> attitude ROLL-OUT {cur_roll:+.0f}->0 (P9)", flush=True)
                gophase("rollout")
        elif phase == "rollout":
            # stream the OFFBOARD attitude: roll = P9 ramp bank->0. yaw = COURSE (velocity direction atan2(vE,vN)), NOT a
            # fixed/heading setpoint. ROOT-CAUSE [rpf82 dump]: commanding a yaw that differs from the velocity direction at
            # the turn exit forces SIDESLIP; the dihedral Cl_beta then drives a -50 deg roll the controller fights (rate-sp
            # saturates +70). Tracking the course keeps sideslip ~0 (coordinated) -> no Cl_beta roll -> no exit spike.
            w = min(1.0, (now - _rollout_t0) / ROLLOUT_T)
            roll_cmd = math.radians(_rollout_roll0) * (1.0 - P9(w))
            course = math.degrees(math.atan2(cur_vE, cur_vN))       # velocity direction (NED) = zero-sideslip yaw target
            qd = quat_from_euler(roll_cmd, math.radians(_rollout_pitch0), math.radians(course))
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111,
                qd, 0.0, 0.0, 0.0, ROLLOUT_THR)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            if w >= 1.0 and abs(cur_roll) < 8.0:                     # roll-out done & wings-level -> [P8] LEVEL spoiler brake
                _cb_lat0, _cb_lon0 = cur_lat, cur_lon
                _p8_t0 = now; _p8_as0 = cur_as; _p8_offb = True      # [P8 Cruise Brake] start the level spoiler brake 72->38 @300m West
                set_param(m, "FW_THR_MIN", 0.0); set_param(m, "FW_R_LIM", 15.0)
                set_param(m, "FW_RR_D", 0.10)                        # [#4] damp the brake roll oscillation
                set_param(m, "MC_ROLLRATE_P", 0.45); set_param(m, "MC_ROLLRATE_D", 0.06)   # [P8 roll] engage the rotor differential for roll (the deep spoiler took the flaperon roll authority)
                print(f"  => RP-1 P8 CRUISE-BRAKE (A): level spoiler brake {cur_as:.0f}->{P8_BRAKE_AS:.0f} m/s @300m West (course={course:.0f})", flush=True)
                gophase("p8brake")
        elif phase == "p8brake" and cur_as <= P8_BRAKE_AS + 3.0:      # the brake settles ~40 (spoiler fades at 38); P9 dials the last 2 m/s to 38
            if not _p8_reached:
                _p8_reached = True; _p8_reach_t = now
                print(f"  => [P8 brake done] reached {cur_as:.1f} m/s @alt={cur_alt:.0f} hdg={cur_hdg:.0f} (dV={_p8_as0-cur_as:.0f} in {now-_p8_t0:.0f}s)", flush=True)
            elif now - _p8_reach_t > 5.0:                             # brief settle, then START the P9 descent (P9 holds exactly 38)
                set_param(m, "FW_RR_D", 0.06); set_param(m, "FW_YD_GAIN", 6.5); set_param(m, "VT_FW_RDIFF", float(os.environ.get("MK7_RDIFF", "0.5")))   # [P9 ROLL-AUTH] lift-rotor diff roll (ailerons ~28% at 38 m/s; rotors already ~0.51). [DR mode-ID DONE] P9 roll osc = AIRFRAME low-speed roll divergence (38 m/s, below the 45 stable band); NOT control-fixable: yaw-damper 2x made it WORSE (0.26->2.14, adverse coupling), roll-rate-D 2x had NO effect (0.26->0.28, aileron authority ~28% at 38 m/s). Reverted to baseline. Already within spec (0.26<0.5).
                _p9_t0 = now; _p9_alt0 = cur_alt; _p9_offb = True; _p8_offb = False; _hdg_i = 0.0
                print(f"  => RP-1 P9 DESCEND X: 38-hold descent {cur_alt:.0f}->{GATE_ALT:.0f} m @{P8_BRAKE_AS:.0f} m/s West (P9 profile {P9_DESC_T:.0f}s)", flush=True)
                gophase("p9descend")
        elif phase == "p9descend" and cur_alt <= GATE_ALT + 1.0:
            # [P10 Descend 1/4] reached the 152 m gate at 38 m/s West -> BACK-TRANSITION (V rotors do the decel; pusher C OFF
            # = energy-minimal, adding C thrust would fight the decel) + the PROVEN velocity-FF decel machinery: level speed-
            # bleed 38->6 (hold 152, velocity-FF advancing target -> NO "stop now" -> no >8 m/s-fwd MC tumble), THEN the 1/4
            # GLIDE-GRADIENT descent (GRAD_DESC: sink=fwd/4, VRS-safe since vz->0 as vx->0) 152->12 -> hover-slam -> 10 m. West held (_g_hdg).
            # [P10] DIRECT back-trans (no FW pre-bleed: a FW decel below 45 m/s hits the known LOW-SPEED ROLL DIVERGENCE
            # [liftcruise-envelope], a +-27 deg actual-roll oscillation with smooth command -> airframe lateral instability,
            # NOT the rotors). MPC_XY_CRUISE=12 hands the back-trans over to MC at 12 [the 1:4 speed].
            print(f"  => [P9 done] {cur_alt:.0f}m as={cur_as:.1f} hdg={cur_hdg:.0f} -> [P10] BACK-TRANS (handover 12) + 1:4 (direct, V-only, C off)", flush=True)
            _p9_offb = False; set_flap(0.0)
            set_param(m, "FW_RR_D", 0.01); set_param(m, "FW_YD_GAIN", 5.0); set_param(m, "VT_FW_RDIFF", 0.0)   # [P9 ROLL-AUTH] OFF for the back-trans
            # [P10 entry (ii)] STRENGTHEN the MC roll (lift-rotor differential) during the transition: the direct back-trans
            # already engages the MC roll [rotorDiff~0.25] but its gain is too weak to reject the <45 m/s roll disturbance
            # [+14.5 deg overshoot vs +2.3 cmd]. Boost MC_ROLLRATE_P/D + faster mc_weight ramp [VT_B_TRANS_RAMP, the
            # _mc_roll_weight scaler, standard.cpp:262-272] so the rotor differential carries the roll continuously. (Param-side
            # allocation/gain change -> NO C++ rebuild.)
            set_param(m, "MC_ROLLRATE_P", 0.55); set_param(m, "MC_ROLLRATE_D", 0.09); set_param(m, "MC_ROLLRATE_I", 0.3)
            set_param(m, "MC_PITCHRATE_P", 0.75); set_param(m, "MC_PITCHRATE_D", 0.10)   # also strengthen the MC PITCH (the back-trans decel-tilt wobble)
            # [C] TESTED rotor-yaw boost (MC_YAW_P 0.65 / MC_YAWRATE_P 0.55 / I 0.25) -> the rotor yaw differential DID engage
            # [rotorYaw 0 -> 0.024] but beta got WORSE [13.4 -> 16.5] + roll-osc 2.3 -> 3.0. REASON: the crab beta is KINEMATIC
            # [the lateral velocity vy_body ~5 m/s amplified by the 44->12 decel, [B]], NOT a yaw-control deficit -> turning the
            # NOSE (yaw) does not change the velocity vector. The fix is to null the LATERAL VELOCITY in P9, not the yaw. Reverted.
            set_param(m, "VT_B_TRANS_RAMP", 1.5)
            set_param(m, "MPC_XY_VEL_MAX", 20.0); set_param(m, "MPC_XY_CRUISE", 12.0); set_param(m, "VT_B_TRANS_DUR", 25.0)
            set_param(m, "MPC_Z_VEL_MAX_DN", 9.0); set_param(m, "MPC_Z_V_AUTO_DN", 9.0); set_param(m, "MPC_TILTMAX_AIR", 28.0)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 3)
            back_done = True; _g_hdg = cur_hdg; _decel_as0 = cur_as; _decel_t0 = now
            _decel_alt0 = cur_alt; _g_lat, _g_lon = cur_lat, cur_lon; _settle_t = None
            _ff_lat, _ff_lon = cur_lat, cur_lon; _ff_t = now; gophase("p10diag")
        elif phase == "p10diag" and cur_alt <= HOVER_ALT + 1.5 and cur_as < 2.0:   # [Y-drift] tighter handoff (was 4.0) -> less residual coast-in
            _hl_lat, _hl_lon = cur_lat, cur_lon; _hy_yaw0 = cur_hdg; _hy_t0 = now; _hy_settled = False
            _hy_dpsi = (P11_HDG - cur_hdg + 540.0) % 360.0 - 180.0       # to P11_HDG (East 90 calm / upwind in wind)
            _fwdthr = int(os.environ.get("FWDTHR", "0"))                 # [WIND sep-ctrl] pusher (C) opposes the main flow: facing
            if _fwdthr:                                                  # upwind, the MC nose-down demand becomes pusher fwd thrust
                set_param(m, "VT_FWD_THRUST_EN", _fwdthr); set_param(m, "VT_FWD_THRUST_SC", 1.0)  # -> hold the flow w/o tilting (pitch ~const)
            print(f"  => [P10 diag 1:4 done @{cur_alt:.0f}m as={cur_as:.1f} hdg={cur_hdg:.0f}] -> [P11 Hover&set] yaw->P11_HDG (fwdthr={int(os.environ.get('FWDTHR','0'))})", flush=True)
            gophase("hoveryaw")
        elif phase == "homecap":
            # continue OFFBOARD attitude: a COORDINATED turn to the home bearing on a P9 yaw profile -> the bank is
            # DERIVED from the commanded yaw-rate (not nav cross-track) so the FW attitude loop TRACKS a snap-smooth
            # turn-to-home with no L1/NPFG overshoot ring. yaw=psi0+dpsi*P9(w); bank=atan(V*yaw_rate/g).
            w = min(1.0, (now - _hc_t0) / _hc_T)
            yaw_cmd = _hc_psi0 + _hc_dpsi * P9(w)
            dpsidt = _hc_dpsi * (630.0 * w**4 * (1.0-w)**4) / _hc_T   # deg/s (P9 derivative = 630 x^4 (1-x)^4)
            bank = math.atan(max(8.0, cur_as) * math.radians(dpsidt) / 9.81)
            bank = max(-math.radians(20.0), min(math.radians(20.0), bank))   # [SP-SMOOTH] cap gentle -> no roll-mode overshoot
            if w >= 1.0: bank = 0.0                                   # [STAGE-1] past the turn: command wings-level & HOLD
            qd = quat_from_euler(bank, math.radians(_hc_pitch0), math.radians(yaw_cmd))
            m.mav.set_attitude_target_send(0, m.target_system, m.target_component, 0b00000111,
                qd, 0.0, 0.0, 0.0, ROLLOUT_THR)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            if w >= 1.0 and abs(cur_roll) < 4.0:                     # wings-level & ON the parallel line -> hold under OFFBOARD velocity
                _cb_lat0, _cb_lon0 = cur_lat, cur_lon
                _ret_hdg = (_turn_hdg0 + 180.0) % 360.0               # the back-leg runs straight on this reverse-of-out-leg line
                # [SP-SMOOTH 2026-06-25 user] DO NOT hand the straight back-leg to nav. Diagnosis: the cruise-back ring is in
                # the SETPOINT, not the tracking - NPFG cross-track guidance generates a WIGGLING roll command that couples
                # with the lightly-damped 0.15 Hz roll mode; the FW faithfully TRACKS that wiggle -> it looks like a ring.
                # FIX: keep the leg under OFFBOARD velocity (MASK_ALTVEL: hold alt + velocity along the reverse heading). With
                # NO position/cross-track reference the FW flies straight and roll_sp stays ~0 (no nav-generated wiggle). The
                # streaming happens each loop iteration in the cruiseback hook above the phase machine. (Was: NPFG P=32 detune.)
                _cb_offb = True
                print(f"  => RP-1 CRUISE-BACK held under OFFBOARD wings-level velocity (ret_hdg={_ret_hdg:.0f}, roll={cur_roll:.0f}) -> roll_sp~0", flush=True)
                gophase("cruiseback")
        elif phase == "cruiseback" and PRE_LVL_DECEL and LEVEL_BT and cur_as > BT_LO_AS+2.0 and \
             ((_home_lat is not None and hdist(cur_lat, cur_lon, _home_lat, _home_lon) <= DESC_HOME_DIST)
              or hdist(_cb_lat0, _cb_lon0, cur_lat, cur_lon) >= CRUISE_DIST    # [STAGE-1] parallel back-leg: decel abeam home
              or now-phase_t > 170.0):
            # [TASK-1 FIX] LEVEL decel STAGE-1 = AERO BLEED. Idle the pusher + low airspeed target and let the speed
            # bleed (82 -> ROTSUP_BLEED_AS) on the WING alone -- NO lift-rotor support yet, NO heavy spoiler, roll I
            # HELD at 0.3 (full authority). The old code slammed VT_FW_MC_THR=0.6 (12 rotors) + spoiler + I-drop ON
            # as a STEP at 82 m/s -> a -50 deg left-roll disturbance (rotors out of regime at high q, authority cut at
            # the same instant). Bleeding first lets the rotors engage IN-REGIME (~58, near VT_ARSP_BLEND) on a P9 ramp.
            # [SP-SMOOTH] _cb_offb stays True: the wings-level attitude hook holds roll_sp=0 + heading through the bleed.
            set_param(m, "FW_T_SPDWEIGHT", 2.0); set_param(m, "FW_AIRSPD_TRIM", BT_LO_AS-3.0)
            set_param(m, "FW_AIRSPD_MIN", 44.0); set_param(m, "FW_THR_MAX", 0.05); set_param(m, "FW_THR_MIN", 0.0)
            print(f"  => [TASK-1] LEVEL decel STAGE-1 aero-bleed (idle, rotors OFF, roll I=0.3) {cur_as:.0f}->{ROTSUP_BLEED_AS:.0f} m/s, hold {CRUISE_ALT_ABS:.0f}m", flush=True)
            gophase("lvlbleed")
        elif phase == "lvlbleed" and (cur_as <= ROTSUP_BLEED_AS or now-phase_t > LVLBLEED_T):
            # [TASK-1 FIX] STAGE-2 = engage the lift-rotor support IN-REGIME. The streamed hook above P9-ramps
            # VT_FW_MC_THR 0->0.6 AND the spoiler 0->LVL_SPOIL over ROTSUP_RAMP_T, then drops roll I to 0.1 once the
            # ramp completes. Snap-smooth + in-regime -> no roll disturbance (replaces the 82 m/s simultaneous step).
            _rotsup_t0 = now; _rotsup_cur = 0.0; _rotsup_done = False; _pretrim_done = False
            print(f"  => [TASK-1] LEVEL decel STAGE-2 rotor-support P9 ramp start at as={cur_as:.0f} (in-regime, snap-smooth)", flush=True)
            gophase("lvldecel")
        elif PRETRIM_EN and phase == "lvldecel" and cur_as < BT_LO_AS + 5.0 and not _pretrim_done:
            # [optional safety] re-trim by restoring the high I a few s before the back-trans. With phase-scheduled I the
            # cruise-trimmed integrator VALUE is retained through the low-I decel, so this is usually unnecessary
            # (PRETRIM_EN=False). Kept as a fallback if the retained trim proves insufficient for a low spike.
            set_param(m, "FW_RR_I", 0.6); set_param(m, "FW_RR_D", 0.0); _pretrim_done = True
            print(f"  => RP-1f pre-trim FW_RR_I->0.3 at as={cur_as:.0f} (re-trim roll for low back-trans spike)", flush=True)
        elif phase == "lvldecel" and _rotsup_done and (cur_as <= BT_LO_AS or now-phase_t > 140.0):
            # [RP-1f (1)] decel done at altitude (AND the rotor-support ramp finished) -> NOW the LOW-SPEED back-trans.
            _cb_offb = False   # [SP-SMOOTH] end the wings-level attitude hold -> the back-trans logic owns the attitude now
            set_flap(LEVEL_BT_SPOIL0); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            set_param(m, "VT_FW_MC_THR", 0.6)   # [CCV-FIX] KEEP the lift-rotor support THROUGH the back-trans (was 0): the
            #   rotors carry the weight as the wing unloads so the FW does NOT pitch up +28 deg to decelerate -> no +54 m
            #   zoom. (Was: zeroed here -> a support gap -> the AUTO back-trans pitched up and zoomed the energy to altitude.)
            set_param(m, "FW_RR_I", 0.3); set_param(m, "FW_RR_D", 0.0)   # [phase-sched I] RESTORE the high cruise integral
            #   for the back-trans (trimmed entry -> low Dutch-roll spike) and for any later FW / wind-CG disturbance reject
            set_param(m, "VT_FW_LK_MC", 1); set_param(m, "MC_ROLLRATE_D", SPOIL_ROLL_D)
            set_param(m, "MPC_XY_VEL_P_ACC", 0.4); set_param(m, "MPC_XY_VEL_D_ACC", 0.6)
            if FT_OVERLAP: set_param(m, "VT_LIFT_HND_V", 0.0)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 3)
            print(f"  *** RP-1f LOW-SPEED back-trans at alt={cur_alt:.0f} as={cur_as:.0f} (after level decel) ***", flush=True)
            back_done = True; _g_hdg = cur_hdg; _decel_as0 = cur_as; _decel_t0 = now
            _decel_alt0 = cur_alt; _g_lat, _g_lon = cur_lat, cur_lon; _settle_t = None
            _spoil_cur = 0.0; _sdesc_t0 = None
            _ff_lat, _ff_lon = cur_lat, cur_lon; _ff_t = now; gophase("decel")
        elif phase == "cruiseback" and LEVEL_BT and ((_home_lat is not None and hdist(cur_lat, cur_lon, _home_lat, _home_lon) <= DESC_HOME_DIST)
                                                     or now-phase_t > 170.0):
            # SPOILER-LIFT-DUMP level back-trans over home: decel at constant 300 m; the decel phase below
            # schedules the flaperon spoiler dynamically (vz feedback) to cancel the nose-up zoom. Then a
            # VERTICAL hover-slam at home -> no FW descend circling -> a taller, cleaner U.
            set_flap(LEVEL_BT_SPOIL0); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            set_param(m, "VT_FW_LK_MC", 1)   # FORCE MC engagement (else at 71 m/s the pos-ctrl requests FW
                                             # -> vstate=2 -> rotors never spin up -> tumble). Now MC + spoiler.
            set_param(m, "MC_ROLLRATE_D", SPOIL_ROLL_D)   # [MOD-4c] more roll damping for the spoiler back-trans
            set_param(m, "MPC_XY_VEL_P_ACC", 0.4)         # [MOD-5] detune the MC velocity loop for the descent ->
            set_param(m, "MPC_XY_VEL_D_ACC", 0.6)         # smaller pitch overshoot -> tames the 0.12 Hz limit cycle
            if FT_OVERLAP: set_param(m, "VT_LIFT_HND_V", 0.0)   # [RP-1e] DISABLE the lift floor for the BT (a high-
                                                           # speed rotor floor = excess lift -> wrecks the back-trans)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 3)
            print(f"  *** LEVEL back-trans (spoiler lift-dump + LK_MC) at alt={cur_alt:.0f} as={cur_as:.0f} ***", flush=True)
            back_done = True; _g_hdg = cur_hdg; _decel_as0 = cur_as; _decel_t0 = now
            _decel_alt0 = cur_alt; _g_lat, _g_lon = cur_lat, cur_lon; _settle_t = None
            _spoil_cur = 0.0; _sdesc_t0 = None                  # [MOD-4b] reset the speed-sched spoiler + descent latch
            _ff_lat, _ff_lon = cur_lat, cur_lon; _ff_t = now; gophase("decel")
        elif phase == "cruiseback" and ((_home_lat is not None and hdist(cur_lat, cur_lon, _home_lat, _home_lon) <= DESC_HOME_DIST)
                                        or now-phase_t > 170.0):
            set_param(m, "FW_T_SPDWEIGHT", 2.0); set_param(m, "FW_AIRSPD_MIN", DESC_AS_MIN)
            set_param(m, "FW_AIRSPD_TRIM", DESC_AS_TRIM); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            if SPOIL_DESC:
                set_flap(SPOIL_DESC_FLAP)        # [MOD-4] deploy the spoiler -> steep STRAIGHT glide toward home
            # descend along a FRESHLY-FROZEN home bearing (straight glide) -> the final approach is a
            # clean straight line to the gate, no near-waypoint roll-setpoint thrash.
            if FROZEN_APPROACH and _home_lat is not None:           # straight frozen-bearing glide (thrash-free)
                _home_brg = bearing(cur_lat, cur_lon, _home_lat, _home_lon)
                hlat, hlon = ahead(cur_lat, cur_lon, _home_brg, 8000.0)
            else:                                                   # descend toward home (returns home)
                hlat, hlon = (_home_lat, _home_lon) if _home_lat is not None else ahead(cur_lat, cur_lon, cur_hdg, GATE_DESC_DIST)
            repo(m, hlat, hlon, GATE_ALT)
            print(f"  => RP-1 DESCEND to gate {GATE_ALT:.0f} m (toward home, frozen={FROZEN_APPROACH}, spoiler={SPOIL_DESC})", flush=True); gophase("descend")

        # P0: settle -> TURN   (legacy one-way profile, RP1=False only)
        if phase == "settle" and (now-t0)-fw_done_t > 18.0:
            if turn_rad > 1.0:
                perp = math.radians(cur_hdg) + math.pi/2.0
                clat = cur_lat + (turn_rad/111320.0)*math.cos(perp)
                clon = cur_lon + (turn_rad/(111320.0*math.cos(math.radians(cur_lat))))*math.sin(perp)
                repo(m, clat, clon, CRUISE, radius=turn_rad)
                print(f"  => START TURN r={turn_rad:.0f}", flush=True); gophase("turn")
            else:
                gophase("turn")  # no turn radius -> proceed straight
        # P1: CLIMB +300 m (max climb, const airspeed)
        elif phase == "turn" and now-phase_t > TURN_HOLD:
            set_param(m, "FW_THR_MAX", CLIMB_THR_MAX); set_param(m, "FW_T_CLMB_MAX", CLIMB_CLMB_MAX)
            set_param(m, "FW_T_SPDWEIGHT", 2.0)   # pitch holds airspeed -> excess thrust = climb (no zoom)
            tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 6000.0)
            repo(m, tlat, tlon, CRUISE+CLIMB_DELTA); gophase("climb")
        elif phase == "climb" and cur_alt >= CRUISE+CLIMB_DELTA-8.0:
            tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 6000.0)
            repo(m, tlat, tlon, CRUISE+CLIMB_DELTA); gophase("level")
        # P2: LEVEL -> SLOW descent to gate. Hold a LOW airspeed (~35) on the way down so the gate is
        # crossed slow (rotor-assist then only needs 35->33, minimal zoom). Pitch holds airspeed
        # (SPDWEIGHT=2), idle throttle, lowered FW_AIRSPD_MIN so the controller can reach ~33.
        elif phase == "level" and now-phase_t > LEVEL_HOLD:
            tlat, tlon = ahead(cur_lat, cur_lon, cur_hdg, 8000.0)
            set_param(m, "FW_T_SPDWEIGHT", 2.0)
            set_param(m, "FW_AIRSPD_MIN", DESC_AS_MIN)
            set_param(m, "FW_AIRSPD_TRIM", DESC_AS_TRIM)
            set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            repo(m, tlat, tlon, GATE_ALT); gophase("descend")
        # P3: at GATE (152 m) -> LEVEL-DECELERATE. FW can only decelerate in LEVEL flight (on a
        # descending glide gravity overpowers drag). Hold the gate altitude with flaps + idle +
        # low airspeed target and bleed speed off BELOW back-trans speed FIRST, so the subsequent
        # back-transition does NOT zoom-climb from high speed (v2 zoom was +186 m from 70 m/s).
        # P3: at GATE (152 m) -> command BACK-TRANSITION directly. The LIFT ROTORS (not aero drag,
        # too weak on this clean airframe: level-decel bled only 4 m/s in 40 s) do the deceleration.
        # Flaps assist. The back-trans zoom-climbs some; tolerated. Key fix vs v3: do NOT switch to
        # OFFBOARD until a fully settled hover (below).
        # P3: at GATE -> ROTOR-LED DECEL. Deploy REVERSE flap (airbrake) + command back-trans; the
        # rotors do the deceleration (aero is too weak). Decel + descent are mutually exclusive in FW
        # (decel needs pitch-up→zoom, descent needs pitch-down→dive), so we let it decel ~in place
        # (zooms some) and ONLY descend afterwards, in MC, where a glide slope IS achievable.
        elif phase == "descend" and cur_alt <= GATE_ALT+6.0 and cur_as > GATE_BT_AS_MAX:
            # [MOD-3 FIX] reached the gate altitude but STILL FAST (short descent didn't bleed) -> do NOT
            # back-trans yet (a high-speed back-trans dives). LEVEL-OFF here and keep bleeding: idle throttle +
            # low airspeed target (SPDWEIGHT=2 pitches up to slow) + flaps, flying straight ahead at the gate alt.
            set_flap(SPOIL_DESC_FLAP if SPOIL_DESC else FLAP_DEPLOY); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            set_param(m, "FW_T_SPDWEIGHT", 2.0); set_param(m, "FW_AIRSPD_TRIM", DESC_AS_TRIM)
            set_param(m, "FW_AIRSPD_MIN", DESC_AS_MIN)
            # bleed while flying TOWARD HOME (not straight ahead) so it does not overshoot home -> stays compact
            # (loiters/circles near home once there). Keeps the out-and-back returning to the origin.
            if _home_lat is not None: blat, blon = _home_lat, _home_lon
            else:                     blat, blon = ahead(cur_lat, cur_lon, cur_hdg, 3000.0)
            repo(m, blat, blon, GATE_ALT)
            if now - _last_bleed_print > 6.0:
                print(f"  ... GATE-BLEED: holding {GATE_ALT:.0f} m, as={cur_as:.1f} -> target<= {GATE_BT_AS_MAX:.0f} before back-trans", flush=True)
                _last_bleed_print = now
        elif phase == "descend" and cur_alt <= GATE_ALT+6.0:
            set_flap(REV_FLAP); set_param(m, "FW_THR_MAX", DESC_THR_MAX)
            # Decel-rate sweep concluded: dip is ~39 m at the optimum DECEL_FF_DEC=1.5 (slower=60 m,
            # faster=47 m). Lock/VT_LIFT_HND_V left OFF (no benefit). Best = velocity-FF 1.5 + RAMP=0.5.
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION, 3)
            print(f"  *** GATE-ENTRY airspeed = {cur_as:.1f} m/s (target {GATE_ENTRY_AS}) at alt={cur_alt:.0f} ***", flush=True)
            print(f"  => BACK-TRANSITION at gate (as={cur_as:.1f} alt={cur_alt:.0f}) revflap={REV_FLAP}", flush=True)
            back_done = True; _g_hdg = cur_hdg; _decel_as0 = cur_as; _decel_t0 = now
            _decel_alt0 = cur_alt; _g_lat, _g_lon = cur_lat, cur_lon; _settle_t = None
            _ff_lat, _ff_lon = cur_lat, cur_lon; _ff_t = now; gophase("decel")
        # P4 (A) ZERO-ALT-CHANGE ROTOR DECEL via OFFBOARD velocity-FEEDFORWARD. Hold the gate altitude
        # (pos-z=_decel_alt0, vd=0) while the commanded FORWARD speed ramps _decel_as0 -> 0 at DECEL_FF_DEC,
        # advancing the XY target ahead so the MC matches attitude smoothly (no "stop now" -> no tumble).
        # This removes the FW->MC transition FREE-FALL (the 19 m/s sink): the rotors hold altitude the whole
        # time instead of the wing unloading into a dive. Settle = commanded & actual fwd speed ~0.
        elif phase == "decel":
            if USE_VEL_FF:
                dt = min(0.5, max(1e-3, now - _ff_t)); _ff_t = now
                # [SNAP] P9-shaped forward decel _decel_as0 -> 0 instead of a LINEAR ramp. T_dec = as0/DECEL_FF_DEC
                # keeps the SAME duration & distance (integral of P9 over [0,1] = 0.5, same as the linear ramp) but
                # accel/jerk/snap are zero at both ends -> no step in decel at entry/exit (smoother pitch sweep).
                _T_dec = max(1.0, _decel_as0 / DECEL_FF_DEC)
                vcmd = _decel_as0 * (1.0 - P9((now - _decel_t0) / _T_dec))     # snap-continuous forward speed -> 0
                _ff_lat, _ff_lon = ahead(_ff_lat, _ff_lon, _g_hdg, vcmd*dt)    # advance the FF point forward
                hd = math.radians(_g_hdg)
                vd_cmd = -FF_CLIMB if (now - _decel_t0) < FF_CLIMB_T else 0.0  # anticipatory climb over the gap
                tgt_alt = _decel_alt0                          # default: hold altitude (production low-speed back-trans)
                if SDESC_MC:
                    # [MOD-4c] TAMER speed-scheduled zoom-cancel spoiler that FADES OUT before stall to keep the wing
                    # loaded (-> roll authority) through the vs==2 transition, then full spoiler in MC (vs==3).
                    # [RP-1f (6) TESTED -> NO EFFECT] a vz-feedback "zoom-cancel" spoiler does NOT cut the back-trans
                    # zoom: the zoom is KINEMATIC (pitch-up redirects forward KE -> vertical PE), not a lift increase,
                    # so dumping wing lift cannot cancel it (+52 m unchanged). The flaperon-cancel premise applies to
                    # the lift term only. Reverted to the plain full-MC spoiler.
                    if vs == 3:
                        set_flap(SPOIL_DESC_FLAP)
                    else:
                        # transition: MILD spoiler scaled by speed, fading to 0 by SPOIL_V0 (>stall) so it stops
                        # decelerating before 36 and the wing reloads -> authority kept. Gradual slew.
                        sched = SPOIL_HI * max(0.0, min(1.0, (cur_as - SPOIL_V0)/(72.0 - SPOIL_V0)))
                        sched += SPOIL_VZ_K * cur_vz
                        sched = max(SPOIL_HI, min(0.0, sched))
                        _spoil_cur += max(-SPOIL_RAMP*dt, min(SPOIL_RAMP*dt, sched - _spoil_cur))
                        set_flap(_spoil_cur)
                    if cur_as > 6.0 and not OVERLAP_DESC:
                        # [RP-1f user-arch] HOLD altitude (vz=0) through the ENTIRE MC forward decel until the
                        # forward speed is settled near hover (<6) -> NO forward-moving MC descent (which rolls
                        # over: OFFBOARD MC at >8 m/s forward tumbles). Decel(level) and descent are now cleanly
                        # separated: level decel -> hover at altitude -> THEN the vertical hover-slam descends.
                        tgt_alt = _decel_alt0; vd_cmd = 0.0       # HOLD altitude -> level decel, no descent
                    else:
                        if _sdesc_t0 is None: _sdesc_t0 = now; _sdesc_alt0 = cur_alt; _grad_alt = cur_alt
                        if GRAD_DESC:
                            # [RP-1f/3] 1/4 glide-gradient: sink rate = forward speed / 4 (P9-eased at entry), so the
                            # flight path holds 4:1 while the speed bleeds. Integrate the path-z so pos&vel stay
                            # consistent. vd ~ vx/4 keeps vz small as vx->0 (VRS-safe). Spoiler (above) adds the drag.
                            _tau = now - _sdesc_t0
                            vd_cmd = min(GRAD_VZ_CAP, vcmd*GRAD_RATIO) * P9(min(1.0, _tau/VZ_RAMP_T))
                            _grad_alt = max(SDESC_TARGET, _grad_alt - vd_cmd*dt)
                            tgt_alt = _grad_alt
                            if tgt_alt <= SDESC_TARGET+1.0: vd_cmd = 0.0
                        else:
                            # [SNAP] P9-ramp the sink rate 0->SDESC_VZ over VZ_RAMP_T; alt target = exact P9int integral.
                            _tau = now - _sdesc_t0
                            vd_cmd = SDESC_VZ * P9(_tau / VZ_RAMP_T)
                            _alt_drop = SDESC_VZ * (VZ_RAMP_T*P9int(_tau/VZ_RAMP_T) if _tau < VZ_RAMP_T
                                                    else (_tau - 0.5*VZ_RAMP_T))
                            tgt_alt = max(SDESC_TARGET, _sdesc_alt0 - _alt_drop)
                            if tgt_alt <= SDESC_TARGET+1.0: vd_cmd = 0.0           # arrived -> stop sinking
                # [MOD-6 TEST] DECEL_VEL_ONLY: command VELOCITY ONLY (MASK_VEL) like the clean rotor-borne climb,
                # instead of pos+vel (MASK_POSVEL). The dual pos(alt)+vel command makes the MC position loop and
                # velocity loop fight -> a candidate driver of the 0.12 Hz descent limit cycle.
                _mask = MASK_VEL if DECEL_VEL_ONLY else (MASK_ALTVEL if DECEL_ALTVEL else MASK_POSVEL)
                m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, _mask,
                    int(_ff_lat*1e7), int(_ff_lon*1e7), float(tgt_alt),
                    vcmd*math.cos(hd), vcmd*math.sin(hd), vd_cmd, 0, 0, 0, 0, 0)
                cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
                settled = (vcmd < 1.0 and cur_as < 4.0)   # [RP-1f] settle on forward-hover at ANY altitude (the level
                                                          # decel holds high) -> then vtail hover-slam descends vertically
            else:
                # AUTO loiter, HOLD the gate altitude. Stays in vtol_state 3 (TRANSITION_TO_MC) so the
                # VT_B_RAMP_MIN source floor (immediate rotor thrust) actually applies -> no free-fall.
                repo(m, _g_lat, _g_lon, _decel_alt0)
                settled = (vs in (1, 3) and 0.0 <= cur_as < 4.0)
            _settle_t = (_settle_t or now) if settled else None
            if (_settle_t is not None and now-_settle_t > 2.0) or (now-phase_t > 130.0):
                dv = _decel_as0 - cur_as; dtt = max(1e-3, now-_decel_t0)
                print(f"  => DECEL DONE (vel-FF) {_decel_as0:.1f}->{cur_as:.1f} m/s in {dtt:.0f}s alt={cur_alt:.0f} dAlt={cur_alt-_decel_alt0:+.0f} vs={vs}", flush=True)
                set_flap(0.0); _desc0 = cur_alt; _ldesc = now
                _vt_alt0 = cur_alt; _vt_lat, _vt_lon = cur_lat, cur_lon; _brake_t = None
                gophase("vtail")
                print(f"  => HOVER-SLAM DESCENT start alt={cur_alt:.0f} (brisk {DESC_SINK} m/s -> P9 brake band {FLARE_ALT:.0f}->{BRAKE_END_ALT:.0f} m -> settle {FLARE_SINK})", flush=True)
        # P4b HOVER-SLAM DESCENT + FLARE (SpaceX-booster-style). Descend briskly at DESC_SINK (a P9 ease-in
        # off the hover so the top has no corner), then below FLARE_ALT fire a DECISIVE P9 braking burn that
        # drives the sink DESC_SINK -> 0 over T_BRAKE so vz=0 EXACTLY at the pad. P9's 1st..4th derivatives
        # are zero at both ends -> accel/jerk/snap are continuous at brake ENTRY and at TOUCHDOWN (no corner,
        # no slam). The vertical velocity is fed FORWARD (MASK_POSVZ) so the MC velocity loop executes the
        # brake tightly (pos-only lagged and overshot the late flare -> hard touchdown).
        # Guard: if knocked out of pure MC with residual speed, HOLD until re-settled.
        elif phase == "vtail" and cur_alt <= HOVER_ALT + 2.0 and not _p11_done and (vs == 1 or cur_as < 6.0):
            # [STAGE-2 P10 done -> P11] reached the 10 m hover floor -> STOP descending, hover, yaw to EAST before landing
            _hl_lat, _hl_lon = cur_lat, cur_lon; _hy_yaw0 = cur_hdg; _hy_t0 = now
            _hy_dpsi = (P11_HDG - cur_hdg + 540.0) % 360.0 - 180.0   # to P11_HDG (East 90 calm / upwind in wind)
            print(f"  => [P10 Descend1/4 done @{cur_alt:.0f}m] -> [P11 Hover&set] yaw {cur_hdg:.0f} -> EAST 90 (P9 snap)", flush=True)
            gophase("hoveryaw")
        elif phase == "hoveryaw":
            # [P11 Hover and set] hold position + 10 m, P9-snap yaw to EAST (same MC yaw-set mechanism as P3). roll/pitch ~0.
            # [Y-drift FIX] SETTLE before the turn: the p10diag handoff still carries ~2 m/s -> holding a fixed point while
            # coasting overshoots it [~8 m drift]. First HOLD the hover point + current heading until the velocity bleeds
            # [vh<1.0, or 6 s timeout], THEN re-capture the hold point + the turn-from heading so the 174 deg turn + the land
            # happen at a SETTLED point. Reduces the P11/P12 XY drift [P12 needs XYZ osc < 0.3 m].
            if not _hy_settled:
                # wait for VELOCITY *and* ATTITUDE to settle: the p10diag decel leaves a big nose-up pitch [~+16 deg] at the
                # handoff; turning while it unwinds drove a ~8 m Y excursion + pitch-osc. Hold level until pitch/roll < 3 deg.
                if (math.hypot(cur_vN, cur_vE) < 0.8 and abs(cur_pitch) < 1.5 and abs(cur_roll) < 1.5) or now - _hy_t0 > 14.0:
                    _hy_settled = True; _hy_t0 = now; _hy_yaw0 = cur_hdg
                    _hl_lat, _hl_lon = cur_lat, cur_lon                  # re-capture the hold at the SETTLED point
                    _hy_dpsi = (P11_HDG - cur_hdg + 540.0) % 360.0 - 180.0  # recompute the turn to P11_HDG from here
                m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POS_YAW,
                    int(_hl_lat*1e7), int(_hl_lon*1e7), float(HOVER_ALT), 0, 0, 0, 0, 0, 0, math.radians(_hy_yaw0), 0)
                cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
                continue
            yaw_T = max(6.0, abs(_hy_dpsi) / 12.0)
            wy = min(1.0, (now - _hy_t0) / yaw_T)
            yawc = math.radians(_hy_yaw0 + _hy_dpsi * P9(wy))
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POS_YAW,
                int(_hl_lat*1e7), int(_hl_lon*1e7), float(HOVER_ALT), 0, 0, 0, 0, 0, 0, yawc, 0)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            _yaw_err_E = abs((cur_hdg - P11_HDG + 540.0) % 360.0 - 180.0)   # residual to P11_HDG
            if wy >= 1.0 and (_yaw_err_E < 1.5 or now - _hy_t0 > yaw_T + 25.0):   # CONVERGED to East (or timeout guard)
                _p11_done = True; _desc0 = cur_alt; _ldesc = now; _brake_t = None
                print(f"  => [P11 Hover&set done] yaw={cur_hdg:.0f} (East 90, err={_yaw_err_E:.1f}, {now-_hy_t0:.0f}s) -> [P12 Land] vertical descent", flush=True)
                gophase("land")
        elif phase == "land":
            # [P12 Land] final VERTICAL hover-slam from 10 m: P9-eased gentle sink braking to ~0 at the pad, XY + yaw held
            # (oscillation target < 0.5 deg). vz fed forward (MASK_POSVZ) so the MC velocity loop lands without overshoot.
            dt = min(0.5, max(1e-3, now - _ldesc))
            ease = P9(min(1.0, (now - phase_t) / TOP_EASE_T))
            sink = max(0.3, 1.2 * max(0.0, cur_alt / HOVER_ALT)) * ease   # ~1.2 m/s at 10 m -> 0.3 m/s soft settle
            tgt = max(0.0, _desc0 - sink * dt); _desc0 = tgt; _ldesc = now
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POSVZ_YAW,
                int(_hl_lat*1e7), int(_hl_lon*1e7), float(tgt), 0.0, 0.0, float(sink), 0, 0, 0, math.radians(P11_HDG), 0)   # HOLD P11_HDG through the land
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            if tgt <= 0.5 and cur_alt < 1.5:
                print(f"  => TOUCHDOWN t={now-t0:.0f} (P12 Land, yaw=East, P9 vz->0 at pad)", flush=True)
        elif phase == "vtail":
            dt = min(0.5, max(1e-3, now - _ldesc))
            if vs != 1 and cur_as > 6.0:
                tgt = _desc0; sink = 0.0
            else:
                ease = P9((now - phase_t) / TOP_EASE_T)                    # snap-cont ease-in off the hover
                if cur_alt > FLARE_ALT:
                    sink = DESC_SINK * ease                                # brisk steady descent
                elif cur_alt > BRAKE_END_ALT:
                    # HOVER-SLAM P9 BRAKE (altitude-scheduled): sink DESC_SINK -> FLARE_SINK as cur_alt
                    # goes FLARE_ALT -> BRAKE_END_ALT. P9 is flat (1st..4th derivs 0) at both band ends ->
                    # accel/jerk/snap continuous at brake entry AND burn-out. Self-correcting (keyed to the
                    # vehicle's actual altitude) and EKF-bias-robust (completes above the ~16.5 m bias).
                    s = (cur_alt - BRAKE_END_ALT) / (FLARE_ALT - BRAKE_END_ALT)
                    sink = FLARE_SINK + (DESC_SINK - FLARE_SINK) * P9(s)
                else:
                    sink = FLARE_SINK                                      # decisive soft settle to the pad
                tgt = max(0.0, _desc0 - sink * dt); _desc0 = tgt
            _ldesc = now
            # pos(XY+alt) + vertical-velocity FEEDFORWARD (vz=+sink, NED +down). XY held at current (pure
            # vertical). The vz FF is what lets the MC execute the hard hover-slam brake without overshoot.
            m.mav.set_position_target_global_int_send(0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, MASK_POSVZ,
                int(cur_lat*1e7), int(cur_lon*1e7), float(tgt), 0.0, 0.0, float(sink), 0, 0, 0, 0, 0)
            cmd(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
            if tgt <= 0.5 and cur_alt < 1.5:
                print(f"  => TOUCHDOWN t={now-t0:.0f} (hover-slam P9 brake, vz->0 at pad)", flush=True)
    print(f"RESULT maxRoll={maxR:.1f} maxPitch={maxP:.1f} maxAS={vmax:.1f} finalVtol={vs} phase={phase} fwDoneT={fw_done_t}", flush=True)

if __name__ == "__main__":
    main()

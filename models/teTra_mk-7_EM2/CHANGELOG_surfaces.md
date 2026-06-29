# teTra Mk-7 EM2 control-surface version log

## v7  (2026-06-15)  central TRANSITION high-lift flaps (linear-actuator, slow)
Backups: model.sdf.v6_preflap, airframe .v6_preflap, model.sdf.v7_phugoid_intermediate

Purpose (per user): central symmetric flaps used as a SLOW high-lift device for the
VTOL->FW transition, driven by a LINEAR ACTUATOR (trim-like, no fast servo). Deploy in
the 70-100 kt transition window, retract for 140 kt cruise (low drag). "Flap angle that
appears slowly" - actuator sized on thrust/stroke & holding force, not servo torque/rate.

Airfoil = LS(1)-0417 MOD (NASA TN D-7428 section; CR-2443/CR-2833 flaps). Clean cl_max(Re):
1.64@2M,2.05@6M,2.12@12M, MOD ~5% lower. 3D a=4.58/rad (cross-checks .stab 4.74). Chord-Re at
transition: 3.7M@70kt..7.3M@140kt -> clean CL_max_3D 1.50@70 / 1.64@100 / 1.72@140.
Transition lift balance (W=15304 N, S=12): CL_req 1.61@70 / 0.79@100 / 0.40@140.
With REAL airfoil the clean wing carries ~94% of weight at 70 kt (only ~6% on rotors;
clean stall ~72 kt) -> wing nearly self-sufficient. Flap's role = stall margin + early rotor
offload. Hinged SPLIT flap (user-confirmed; Fowler not used) dCL_max_3D +0.55 -> CL_max 2.05,
stall 72->62 kt (~10 kt margin at 70 kt). Schedule 70kt:45deg,85:35,100:20,120:8,140:0.
Linear actuator (horn arm ~0.06 m): peak hinge ~48 N.m -> thrust ~800 N, stroke ~47 mm,
slow/holding -> off-the-shelf electric linear actuator ~1 kN. No fast servo.
Flap-type comparison (2D dcl_max anchors): plain ~+0.80 (span 2.55/side), split ~+1.10
(span 2.0/side, CHOSEN), Fowler ~+1.70 (span 1.28/side, needs translation - NOT used).

Model (gz): left_flap/right_flap links 0.405 x 2.000 (hinge 80%c, mass~=0) at y=+-1.65
(span ~0.65..2.65, clear of fuselage and outboard ailerons; lift rotors on booms, stopped in
cruise -> no wake interference); servo_5/6 deflect DOWN 0..45 deg (limit -0.087..0.785 rad);
2 LiftDrag (area 3.00, upward 0 0 1, rad_to_cl 1.4) + 2 JointPositionControllers. Airframe:
SIM_GZ_SV_FUNC6/7=206/207; CA_SV_CS_COUNT 7; CA_SV_CS5_TYPE 9 (LeftFlap), CS6_TYPE 10 (RightFlap).
Driven by flaps setpoint (default 0 = retracted) -> existing flight undisturbed.

Verification: XML OK (38 links / 37 joints / 7 servos); build clean; no-wind MC hover
regression PASS (flaps retracted in hover, mass~=0). Pitch stability/short-period unchanged
(fixed surfaces & Vh/Cm_alpha untouched).

PX4 integration: GEOMETRY + CA registration DONE. Airspeed/transition-scheduled deployment
(deploy 70-100 kt during front transition, retract by cruise) is the proposed control step -
tie to VTOL transition state (VT_*) or an airspeed gate feeding the flaps setpoint; the slow
applyFlaps slew rate + linear actuator suit the "slow" requirement. Stock PX4 drives these
flaps only from manual/takeoff/landing setpoints today.
(Earlier intermediate explored these flaps for phugoid damping; superseded by the high-lift role.)

## v6-surf  (2026-06-15)  control-surface area resize for servo feasibility
Backup of pre-change model: `model.sdf.v5_presurf`

Driver: at real cruise VC=140 kt (72 m/s, VNE~81 m/s) the v5 surfaces gave
excessive hinge moments (aileron ~2300 kgf.cm, elevator ~1000 kgf.cm @VNE SF1.5).
Control-authority sizing (from .stab derivatives, m-units: Sref=12, b=8, c=1.5)
showed all surfaces were oversized: roll x6.2, pitch x2.9, yaw x1.7.

Changes (movable surfaces only; fixed wing/H-tail/V-tail UNCHANGED -> Vh=0.70,
Cn_b, SM=10.6% preserved):
- Aileron (each): 0.506x2.400 -> 0.405x0.640 m, moved outboard to y=+-3.28,
  hinge 75%->80%c (x=-0.585), balance ratio 0.35 kept (cf_aft 0.375->0.300).
- Elevator: 0.305x2.900 -> 0.242x2.000 m, hinge 75%->80%c (x=-5.321),
  balance 0.35 kept (cf_aft 0.2255->0.179). H-tail fixed area unchanged (stability intact).
- Rudder: unchanged (already near-optimal, x1.7 authority).
- LiftDrag re-scaled to keep ~x2.0 control-authority margin:
  servo_0/1 area 1.4->0.330, cp y=+-3.28; servo_2 area 2.63->1.81, cp x=-5.321;
  rad_to_cl unchanged (-1.05 / -2.67 / +1.41); rudder LiftDrag unchanged.

Result (hinge moment @VNE, scheduled delta): aileron 2303->319 kgf.cm (-83%),
elevator 1006->355 (-57%), rudder 156->126.  SF1.5: aileron 479, elevator 533, rudder 190 kgf.cm.
Verification: no-wind MC hover regression PASS (v6 xy-hold 0.256 vs v5 0.302; equivalent).
Control authority by design: roll x2.0, pitch x2.0, yaw x1.7 (empirical FW confirm deferred to objective 2).

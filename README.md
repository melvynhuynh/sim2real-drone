# Sim2real-drone

> Autonomous drone racing through gates, from simulation to real hardware.

## Demos

### Simulation (Webots)

https://github.com/user-attachments/assets/ccf32584-7456-4e23-9bdb-82d128ffee7f

### Real Hardware (Crazyflie 2.1)

https://github.com/user-attachments/assets/ba1c3e28-7c76-4f28-91cf-aa1ea07819b3

## Overview

This project develops a fully autonomous navigation stack for a **Crazyflie 2.1** nano-quadrotor, capable of detecting and flying through a circuit of gates as fast as possible.

The system is built and validated in the **Webots simulator** and then transferred onto real hardware, a classic sim-to-real challenge in aerial robotics.

The drone must:
1. **Take off** autonomously from a fixed pad
2. **Detect 5 unknown gates** during a first exploration lap using computer vision
3. **Race through the gates** at maximum speed for the subsequent laps

## Project Structure

```
micro-502/
├── controllers/
│   └── main/
│       └── assignment/
│           └── my_assignment.py       # Simulation controller (Webots)
├── hardware/
│   ├── Lap1.py                        # Real drone: exploration lap (gate mapping)
│   └── Lap23.py                       # Real drone: fast racing laps
├── worlds/                            # Webots simulation environments
└── docs/                              # Course documentation
```

- **Simulation** runs `controllers/main/assignment/my_assignment.py` inside Webots.
- **Hardware** runs `hardware/Lap1.py` then `hardware/Lap23.py` on the real Crazyflie 2.1 through the Crazyradio link.

## Sim vs Real: Why the Control Strategy Changes

The simulation and the real hardware share the same high-level goal (map gates, then race) but need different perception pipelines because the real world is noisier, the camera is lower resolution and the state estimator drifts.

### Simulation (`my_assignment.py`)
- **Perception**: HSV color segmentation on the simulated RGB camera. Gates are a known color, so a direct color mask works.
- **Localization of gates**: single-frame bounding-box back-projection using camera intrinsics. The gate is big enough and the simulated image clean enough that one detection is sufficient to estimate 3D position.
- **State machine**: `TAKEOFF → FIRST_GATE_SCAN → SEARCH → ALIGN → TRAVERSE → FAST`, with confidence-weighted gate memory.
- **Command filtering**: per-axis rate limiter + exponential smoothing.

### Hardware: Exploration Lap (`Lap1.py`)
On the real drone we lose the color trick (indoor lighting varies, gate LEDs saturate) and single-frame depth is unreliable. The pipeline is rebuilt around two ideas:

1. **Brightness-based detection**: the real gates carry a green LED ring, so detection switches to a monochrome brightness threshold on the AI-deck grayscale stream (`detect_gate_yann`, `BRIGHT_THRESH = 110`). A quad-contour extractor then finds the gate corners and tracks their centroid between frames (`tracked_max_jump` guards against identity swaps).

2. **Two-view triangulation for 3D gate position**: a single bearing from one camera pose does not give depth. The drone instead performs a short controlled lateral strafe:
   - `obs1`: record bearing to the gate centroid and the drone pose.
   - Move `baseline_right = 0.1 m` sideways while keeping yaw.
   - `obs2`: record a second bearing from the new pose.
   - Triangulate the 3D intersection of the two bearing rays to recover the gate position `H`.

   A sanity check (`sanity_check_gate`) rejects any triangulation whose result lies more than `GATE_SANITY_RADIUS_M = 0.5 m` from a hardcoded prior average, which filters out spurious detections caused by reflections or background LEDs.

3. **State machine adapted for triangulation**: an extra phase `"triangulation"` is inserted between detection and traversal, with `reset_triangulation` restarting the baseline sweep whenever the second observation times out or fails sanity checks.

### Hardware: Racing Laps (`Lap23.py`)
Once the exploration lap has produced a 3D map of all five gates, perception is no longer needed: `Lap23.py` runs an open-loop trajectory through the mapped waypoints with yaw pre-alignment, trading robustness for speed.

The guiding principle of the sim2real adaptation: **replace any step that relied on sim-only assumptions (clean color, calibrated depth, perfect pose) with a geometrically grounded alternative (LED brightness, two-view triangulation, map-then-race).**

## Architecture

```
Camera + Sensors
      |
  Gate Detection  (sim: HSV mask  /  real: brightness + quad contour)
      |
  3D Estimation   (sim: 1-frame back-projection  /  real: 2-view triangulation)
      |
  State Machine   (TAKEOFF / SCAN / SEARCH / ALIGN / TRAVERSE / FAST)
      |
  Waypoint Generator
      |
  Command Filter  (rate limiter + exponential smoothing)
      |
  Cascaded PID Controller
      |
  PWM output to motors
```

## Stack

| Component | Technology |
|---|---|
| Simulator | Webots |
| Language | Python 3 |
| Vision | OpenCV |
| Numerics | NumPy, SciPy |
| Hardware link | `cflib` (Crazyradio PA) + AI-deck video stream |
| GUI (hardware debug) | PyQt6 |
| Platform | Bitcraze Crazyflie 2.1 |

## Credits

Base course material and simulation environment from [EPFL micro-502](https://github.com/lis-epfl/micro-502), Aerial Robotics, LIS Lab, EPFL.

Navigation logic, gate detection pipeline, state machine, command filtering (`my_assignment.py`) and the hardware triangulation stack (`hardware/Lap1.py`, `hardware/Lap23.py`) are original work developed as part of the course project.

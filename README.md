# sim2real-drone

> Autonomous drone racing through gates — from simulation to real hardware.

![Crazyflie 2.1](https://www.bitcraze.io/images/crazyflie2-1/crazyflie_2.1_585px.jpg)

*Bitcraze Crazyflie 2.1 — the platform used in this project.*

---

## Overview

This project develops a fully autonomous navigation stack for a **Crazyflie 2.1** nano-quadrotor, capable of detecting and flying through a circuit of gates as fast as possible.

The system is built and validated in the **Webots simulator** and then transferred onto real hardware — a classic sim-to-real challenge in aerial robotics.

The drone must:
1. **Take off** autonomously from a fixed pad
2. **Detect 5 unknown gates** during a first exploration lap using computer vision
3. **Race through the gates** at maximum speed for the subsequent laps

---

## Architecture

```
Camera + Sensors
      |
  Gate Detection (OpenCV)
      |
  State Machine (TAKEOFF / SCAN / SEARCH / ALIGN / TRAVERSE / FAST)
      |
  Waypoint Generator
      |
  Command Filter (rate limiter + exponential smoothing)
      |
  Cascaded PID Controller
      |
  PWM output to motors
```

---

## Key Technical Contributions

### Computer Vision Gate Detection
- HSV-based color segmentation with OpenCV to detect gate contours in real time
- Bounding box estimation and centroid tracking
- Camera intrinsics used to back-project 2D detections into 3D world coordinates

### Gate Memory and Tracking
- Exponential moving average filter on noisy 3D gate position estimates
- Confidence scoring system: detections committed only once a score threshold is reached
- Separate confirmed position estimate for robustness against outliers

### State Machine Navigation
- `FIRST_GATE_SCAN`: rotational scan + forward approach with multi-attempt retry logic
- `SEARCH`: spiral waypoint search with visual acquisition assist
- `ALIGN`: approach waypoint with pixel-level centering on the gate
- `TRAVERSE`: fast pass-through with vertical correction from bounding box
- `FAST`: high-speed laps using pre-mapped gate positions

### Command Filtering
- Per-axis rate limiting (XY, Z, yaw) to prevent abrupt setpoint jumps
- Exponential smoothing for all control axes
- Ensures stability across sim and real hardware

---

## Stack

| Component | Technology |
|---|---|
| Simulator | Webots |
| Language | Python 3 |
| Vision | OpenCV |
| Numerics | NumPy, SciPy |
| Platform | Bitcraze Crazyflie 2.1 |

---

## Project Structure

```
micro-502/
├── my_assignment.py       # Main controller (this work)
├── lib/                   # Provided utility libraries
├── controllers/           # Webots controller entry points
├── worlds/                # Webots simulation environments
└── docs/                  # Assignment documentation and demos
```

---

## Credits

Base course material and simulation environment from [EPFL micro-502](https://github.com/lis-epfl/micro-502) — Aerial Robotics, LIS Lab, EPFL.

Navigation logic, gate detection pipeline, state machine, and command filtering in `my_assignment.py` are original work developed as part of the course project.

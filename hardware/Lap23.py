# -*- coding: utf-8 -*-

import logging
import time
import threading
from threading import Timer

import numpy as np
from pynput import keyboard

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper


# ============================================================================
# CONFIGURATION
# ============================================================================

# Crazyflie radio URI (same as in test.py — adjust to your radio/channel)
URI = uri_helper.uri_from_env(default='radio://0/40/2M/E7E7E7E714')

# Hardcoded gates 
GATES_CSV = [
    (0.62, -0.80, 1.25,   -36.9),
    (1.65,  -0.8, 1.27,     12.3),
    (2.33,  0.00, 1.41,    85.1),
    (1.65,  0.85, 1.42,   156.5),
    (0.64,  0.84, 1.26, -156.88),
]

# Take-off altitude (m). Use roughly the average gate height.
Z_TAKEOFF = 1.3

# Control loop period (s). Crazyflie position setpoints are typically sent
# at 20-100 Hz; 50 Hz (20 ms) is a safe default.
CONTROL_DT = 0.02

# --- Trajectory speeds (lowered vs. simulation to be safe on real HW) ---
V_LONG = 1.3   # cruise speed between gates [m/s]
V_GATE = 1.2   # speed when passing through a gate [m/s]
V_START = 1.5  # initial cruise [m/s]
V_LIM   = 2.5  # hard cap on velocity [m/s]
A_LIM   = 12.0 # hard cap on acceleration [m/s^2]
MIN_LAP_TIME = 7.0  # minimum total trajectory time [s]

# Take-off / landing parameters
TAKEOFF_RAMP_S   = 2.5    # seconds to ramp from ground to Z_TAKEOFF
HOVER_BEFORE_RUN = 1.5    # hover this long before starting the lap [s]
HOVER_AFTER_RUN  = 1.0    # hover briefly after lap finishes [s]
RETURN_SPEED     = 0.8    # cruise speed during return-to-origin [m/s]
RETURN_MIN_S     = 3.0    # minimum duration of the return phase [s]
HOVER_AT_ORIGIN  = 1.0    # hover above origin before landing [s]
LAND_RAMP_S      = 2.5    # seconds to ramp down to the ground

# Logging
logging.basicConfig(level=logging.ERROR)


# ============================================================================
# MOTION PLANNER
# ============================================================================

class LapPlanner:
    def __init__(self, gates_csv, target_laps=1):
        self.n_gates = len(gates_csv)
        self.gate_positions = []
        self.gate_normals   = []
        for (x, y, z, theta_deg) in gates_csv:
            self.gate_positions.append(np.array([x, y, z], dtype=float))
            th = np.deg2rad(theta_deg)
            self.gate_normals.append(np.array([np.cos(th), np.sin(th), 0.0]))

        self.target_laps = target_laps

        # Per-gate waypoint cache (pre/gate/post)
        self.traj_pre   = [None] * self.n_gates
        self.traj_pre2  = [None] * self.n_gates
        self.traj_pre3  = [None] * self.n_gates
        self.traj_pre4  = [None] * self.n_gates
        self.traj_post4 = [None] * self.n_gates
        self.traj_post3 = [None] * self.n_gates
        self.traj_post2 = [None] * self.n_gates
        self.traj_post  = [None] * self.n_gates

        # Filled in by build_trajectory()
        self.race_traj_times  = None
        self.race_traj_coeffs = None
        self.race_traj_total  = None
        self.race_waypoints   = None

    # ---------- polynomial helpers ----------

    def compute_poly_matrix(self, t):
        return np.array([
            [t**5,        t**4,      t**3,    t**2, t, 1],
            [5*t**4,      4*t**3,    3*t**2,  2*t,  1, 0],
            [20*t**3,     12*t**2,   6*t,     2,    0, 0],
            [60*t**2,     24*t,      6,       0,    0, 0],
            [120*t,       24,        0,       0,    0, 0],
        ])

    def compute_poly_coefficients(self, waypoints, times,
                                  velocities=None,
                                  v_start=None, v_end=None,
                                  a_start=None, a_end=None):
        seg_times = np.diff(times)
        m         = len(waypoints)
        n_seg     = m - 1
        coeffs    = np.zeros((6 * n_seg, 3))

        if velocities is None:
            velocities = [None] * m
        v0_def = np.zeros(3) if v_start is None else np.asarray(v_start, float)
        vf_def = np.zeros(3) if v_end   is None else np.asarray(v_end,   float)
        a0_def = np.zeros(3) if a_start is None else np.asarray(a_start, float)
        af_def = np.zeros(3) if a_end   is None else np.asarray(a_end,   float)

        A_0 = self.compute_poly_matrix(0)

        for dim in range(3):
            A = np.zeros((6 * n_seg, 6 * n_seg))
            b = np.zeros(6 * n_seg)
            pos = np.array([p[dim] for p in waypoints])

            row = 0
            for i in range(n_seg):
                A_f      = self.compute_poly_matrix(seg_times[i])
                col_i    = slice(i*6,     (i+1)*6)
                col_ip1  = slice((i+1)*6, (i+2)*6)

                A[row, col_i] = A_0[0]; b[row] = pos[i];     row += 1
                A[row, col_i] = A_f[0]; b[row] = pos[i + 1]; row += 1

                if i == 0:
                    A[row, col_i] = A_0[1]; b[row] = v0_def[dim]; row += 1
                    A[row, col_i] = A_0[2]; b[row] = a0_def[dim]; row += 1

                if i < n_seg - 1:
                    v_spec = velocities[i + 1]
                    if v_spec is None:
                        A[row, col_i] =  A_f[1]; A[row, col_ip1] = -A_0[1]; b[row]=0; row += 1
                        A[row, col_i] =  A_f[2]; A[row, col_ip1] = -A_0[2]; b[row]=0; row += 1
                        A[row, col_i] =  A_f[3]; A[row, col_ip1] = -A_0[3]; b[row]=0; row += 1
                        A[row, col_i] =  A_f[4]; A[row, col_ip1] = -A_0[4]; b[row]=0; row += 1
                    else:
                        A[row, col_i]   = A_f[1]; b[row] = v_spec[dim]; row += 1
                        A[row, col_ip1] = A_0[1]; b[row] = v_spec[dim]; row += 1
                        A[row, col_i] =  A_f[2]; A[row, col_ip1] = -A_0[2]; b[row]=0; row += 1
                        A[row, col_i] =  A_f[3]; A[row, col_ip1] = -A_0[3]; b[row]=0; row += 1

                if i == n_seg - 1:
                    A[row, col_i] = A_f[1]; b[row] = vf_def[dim]; row += 1
                    A[row, col_i] = A_f[2]; b[row] = af_def[dim]; row += 1

            coeffs[:, dim] = np.linalg.solve(A, b)

        return coeffs

    def sample_max(self, coeffs, times, n_samples=600):
        v_max, a_max = 0.0, 0.0
        n_segs = len(times) - 1
        for t in np.linspace(0, times[-1], n_samples):
            seg_idx = int(min(max(np.searchsorted(times, t) - 1, 0), n_segs - 1))
            t_local = t - times[seg_idx]
            A_m = self.compute_poly_matrix(t_local)
            c   = coeffs[seg_idx*6:(seg_idx+1)*6, :]
            v_max = max(v_max, np.linalg.norm(A_m[1] @ c))
            a_max = max(a_max, np.linalg.norm(A_m[2] @ c))
        return v_max, a_max

    def evaluate_trajectory(self, t):
        t = float(np.clip(t, 0.0, self.race_traj_total))
        n_segs = len(self.race_traj_times) - 1
        seg_idx = int(min(max(np.searchsorted(self.race_traj_times, t) - 1, 0),
                          n_segs - 1))
        t_local = t - self.race_traj_times[seg_idx]
        A_m = self.compute_poly_matrix(t_local)
        seg_coeffs = self.race_traj_coeffs[seg_idx*6:(seg_idx+1)*6, :]
        pos = A_m[0] @ seg_coeffs
        vel = A_m[1] @ seg_coeffs
        return pos, vel

    # ---------- trajectory construction -----

    def build_trajectory(self, start_pos):
        v_long = V_LONG
        v_gate = V_GATE
        v_s    = V_START

        # Per-gate pre/post waypoints
        for k in range(self.n_gates):
            G = self.gate_positions[k]
            n = self.gate_normals[k] / (np.linalg.norm(self.gate_normals[k]) + 1e-9)

            G_next = self.gate_positions[(k + 1) % self.n_gates]
            dist_next = np.linalg.norm(G_next[:2] - G[:2])

            L_pre  = np.clip(0.40 * dist_next, 0.7, 2.0)
            L_post = np.clip(0.12 * dist_next, 0.4, 0.9)

            self.traj_pre[k]   = G - L_pre  * n
            self.traj_pre2[k]  = G - 0.60   * n
            self.traj_pre3[k]  = G - 0.40   * n
            self.traj_pre4[k]  = G - 0.20   * n
            self.traj_post4[k] = G + 0.05   * n
            self.traj_post3[k] = G + 0.15   * n
            self.traj_post2[k] = G + 0.30   * n
            self.traj_post[k]  = G + L_post * n

        # End point
        last_pt = self.gate_positions[self.n_gates - 1] + 2.0 * self.gate_normals[self.n_gates - 1]

        pts     = [tuple(float(v) for v in start_pos)]
        types   = ["start"]
        normals = [None]

        for lap_idx in range(self.target_laps):
            for k in range(self.n_gates):
                pre   = tuple(float(v) for v in self.traj_pre[k])
                pre2  = tuple(float(v) for v in self.traj_pre2[k])
                pre3  = tuple(float(v) for v in self.traj_pre3[k])
                pre4  = tuple(float(v) for v in self.traj_pre4[k])
                gate  = tuple(float(v) for v in self.gate_positions[k])
                post4 = tuple(float(v) for v in self.traj_post4[k])
                post3 = tuple(float(v) for v in self.traj_post3[k])
                post2 = tuple(float(v) for v in self.traj_post2[k])
                post  = tuple(float(v) for v in self.traj_post[k])

                pts.append(pre);   types.append("pre");   normals.append(None)
                pts.append(pre2);  types.append("pre2");  normals.append(self.gate_normals[k])
                pts.append(pre3);  types.append("pre3");  normals.append(self.gate_normals[k])
                pts.append(pre4);  types.append("pre4");  normals.append(self.gate_normals[k])
                pts.append(gate);  types.append("gate");  normals.append(self.gate_normals[k])
                pts.append(post4); types.append("post4"); normals.append(self.gate_normals[k])
                pts.append(post3); types.append("post3"); normals.append(self.gate_normals[k])
                pts.append(post2); types.append("post2"); normals.append(self.gate_normals[k])
                pts.append(post);  types.append("post");  normals.append(None)

        pts.append(tuple(float(v) for v in last_pt))
        types.append("transit")
        normals.append(None)

        # Iterative time/velocity solve with rate-limit safety check
        coeffs = None
        times  = None
        for _ in range(8):
            velocities = [None] * len(pts)
            for i, (t, n) in enumerate(zip(types, normals)):
                if t == "gate":
                    n_arr = np.asarray(n, dtype=float)
                    n_arr = n_arr / (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = v_gate * n_arr

            for i, t in enumerate(types):
                if t == "pre":
                    n_arr = np.asarray(normals[i + 4], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.95 * v_gate * n_arr
                elif t == "pre2":
                    n_arr = np.asarray(normals[i + 3], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.97 * v_gate * n_arr
                elif t == "pre3":
                    n_arr = np.asarray(normals[i + 2], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.99 * v_gate * n_arr
                elif t == "pre4":
                    n_arr = np.asarray(normals[i], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = v_gate * n_arr
                elif t == "post4":
                    n_arr = np.asarray(normals[i], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = v_gate * n_arr
                elif t == "post3":
                    n_arr = np.asarray(normals[i - 2], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.99 * v_gate * n_arr
                elif t == "post2":
                    n_arr = np.asarray(normals[i - 3], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.97 * v_gate * n_arr
                elif t == "post":
                    n_arr = np.asarray(normals[i - 4], dtype=float)
                    n_arr /= (np.linalg.norm(n_arr) + 1e-9)
                    velocities[i] = 0.95 * v_gate * n_arr

            def v_cap(t):
                if t in ("gate", "pre2", "pre3", "pre4", "post2", "post3", "post4"):
                    return v_gate
                if t in ("pre", "post"):
                    return 0.5 * (v_gate + v_long)
                return v_s

            pts_arr = np.array(pts, dtype=float)
            n_pts   = len(pts_arr)
            v_at_pt = np.empty(n_pts)
            v_at_pt[0]  = v_cap(types[0])
            v_at_pt[-1] = v_cap(types[-1])

            for i in range(1, n_pts - 1):
                d_in  = pts_arr[i]   - pts_arr[i-1]
                d_out = pts_arr[i+1] - pts_arr[i]
                n_in, n_out = np.linalg.norm(d_in), np.linalg.norm(d_out)
                if n_in < 1e-6 or n_out < 1e-6:
                    speed_factor = 1.0
                else:
                    cos_a = np.clip(np.dot(d_in, d_out) / (n_in * n_out), -1.0, 1.0)
                    angle = np.arccos(cos_a)
                    speed_factor = max(np.cos(angle / 2.0), 0.25)
                v_at_pt[i] = v_cap(types[i]) * speed_factor

            t_min      = 0.05
            t_min_gate = 0.10
            seg_durations = []
            for i in range(n_pts - 1):
                d = np.linalg.norm(pts_arr[i+1] - pts_arr[i])
                v_avg = max(0.5 * (v_at_pt[i] + v_at_pt[i+1]), 0.3)
                t_i = d / v_avg
                in_gate_zone = (types[i]   in ("pre","pre2","pre3","pre4","gate","post4","post3","post2","post")
                            and types[i+1] in ("pre","pre2","pre3","pre4","gate","post4","post3","post2","post"))
                tmin = t_min_gate if in_gate_zone else t_min
                seg_durations.append(max(t_i, tmin))

            times = np.concatenate([[0.0], np.cumsum(seg_durations)])

            min_total_time = self.target_laps * MIN_LAP_TIME
            if times[-1] < min_total_time:
                times = times * (min_total_time / times[-1])

            coeffs = self.compute_poly_coefficients(
                pts, times, velocities=velocities,
                v_start=np.zeros(3), v_end=np.zeros(3))

            v_max, a_max = self.sample_max(coeffs, times)
            if v_max <= V_LIM and a_max <= A_LIM:
                break

            scale = max(v_max / V_LIM, np.sqrt(a_max / A_LIM)) * 1.05
            v_long = v_long / np.sqrt(scale)
            v_gate = v_gate / scale
            v_s    = v_s    / np.sqrt(scale)

        self.race_traj_times  = times
        self.race_traj_coeffs = coeffs
        self.race_traj_total  = float(times[-1])
        self.race_waypoints   = pts

        v_max, a_max = self.sample_max(coeffs, times)
        print(f"[planner] {len(pts)} waypoints, T={self.race_traj_total:.2f}s, "
              f"v_max={v_max:.2f} m/s, a_max={a_max:.2f} m/s^2")


# ============================================================================
# CRAZYFLIE HARDWARE WRAPPER
# ============================================================================

class CrazyflieRunner:
    def __init__(self, link_uri):
        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self.state = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.state_lock = threading.Lock()

        self.is_connected = False
        self._connect_event = threading.Event()

        print('Connecting to %s' % link_uri)
        self._cf.open_link(link_uri)

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)
        self.is_connected = True

        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=20)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')

        try:
            self._cf.log.add_config(self._lg_stab)
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration, {} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        self._connect_event.set()

    def wait_until_connected(self, timeout=10.0):
        return self._connect_event.wait(timeout)

    def _stab_log_data(self, timestamp, data, logconf):
        with self.state_lock:
            self.state['x']   = data['stateEstimate.x']
            self.state['y']   = data['stateEstimate.y']
            self.state['z']   = data['stateEstimate.z']
            self.state['yaw'] = data['stabilizer.yaw']  

    def _stab_log_error(self, logconf, msg):
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _connection_failed(self, link_uri, msg):
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False
        self._connect_event.set()

    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

    def get_state(self):
        with self.state_lock:
            return dict(self.state)

    @property
    def cf(self):
        return self._cf


# ============================================================================
# EMERGENCY STOP
# ============================================================================

emergency_flag = threading.Event()


def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':
                print("Emergency stop triggered!")
                emergency_flag.set()
                cf.commander.send_stop_setpoint()
                cf.close_link()
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ============================================================================
# MAIN
# ============================================================================

def main():
    cflib.crtp.init_drivers()

    runner = CrazyflieRunner(URI)
    if not runner.wait_until_connected(timeout=10.0) or not runner.is_connected:
        print("Failed to connect.")
        return
    cf = runner.cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2.0)

    es_thread = threading.Thread(target=emergency_stop_callback, args=(cf,), daemon=True)
    es_thread.start()

    time.sleep(0.3)
    s0 = runner.get_state()
    print(f"[init] state after reset: x={s0['x']:.2f}, y={s0['y']:.2f}, "
          f"z={s0['z']:.2f}, yaw={s0['yaw']:.1f} deg")

    # ---------------- Build the racing trajectory ----------------
    takeoff_pos = (s0['x'], s0['y'], Z_TAKEOFF)
    planner = LapPlanner(GATES_CSV, target_laps=2)
    planner.build_trajectory(takeoff_pos)

    pos0, _ = planner.evaluate_trajectory(0.0)
    pos_ahead, _ = planner.evaluate_trajectory(min(0.5, planner.race_traj_total))
    init_yaw_deg = float(np.rad2deg(np.arctan2(pos_ahead[1] - pos0[1],
                                               pos_ahead[0] - pos0[0])))

    try:
        # ---------------- Take-off ramp ----------------
        print("[phase] take-off")
        n_ramp = max(int(TAKEOFF_RAMP_S / CONTROL_DT), 1)
        for i in range(n_ramp):
            if emergency_flag.is_set():
                return
            z_cmd = Z_TAKEOFF * (i + 1) / n_ramp
            cf.commander.send_position_setpoint(s0['x'], s0['y'], z_cmd, init_yaw_deg)
            time.sleep(CONTROL_DT)

        # ---------------- Hover briefly to stabilise ----------------
        print("[phase] hover")
        n_hover = max(int(HOVER_BEFORE_RUN / CONTROL_DT), 1)
        for _ in range(n_hover):
            if emergency_flag.is_set():
                return
            cf.commander.send_position_setpoint(s0['x'], s0['y'], Z_TAKEOFF, init_yaw_deg)
            time.sleep(CONTROL_DT)

        # ---------------- Execute the racing laps ----------------
        print(f"[phase] racing {planner.target_laps} laps "
              f"({planner.race_traj_total:.2f} s total)")
        t_start = time.time()
        while True:
            if emergency_flag.is_set():
                return
            t = time.time() - t_start
            if t >= planner.race_traj_total:
                break

            state = runner.get_state()
            speed_xy_est = 0.0 
            T_lead = 0.05 + 0.015 * speed_xy_est
            t_cmd = min(t + T_lead, planner.race_traj_total)

            pos_cmd, _ = planner.evaluate_trajectory(t_cmd)
            pos_yaw, _ = planner.evaluate_trajectory(min(t_cmd + 0.4, planner.race_traj_total))
            dx, dy = pos_yaw[0] - pos_cmd[0], pos_yaw[1] - pos_cmd[1]
            if dx*dx + dy*dy > 1e-4:
                yaw_deg = float(np.rad2deg(np.arctan2(dy, dx)))
            else:
                yaw_deg = init_yaw_deg

            cf.commander.send_position_setpoint(
                float(pos_cmd[0]), float(pos_cmd[1]),
                float(max(pos_cmd[2], 0.4)),  
                yaw_deg)
            time.sleep(CONTROL_DT)

        # ---------------- Hover at end ----------------
        print("[phase] hover (end)")
        end_pos, _ = planner.evaluate_trajectory(planner.race_traj_total)
        x_end, y_end, z_end = float(end_pos[0]), float(end_pos[1]), float(max(end_pos[2], 0.4))
        n_end = max(int(HOVER_AFTER_RUN / CONTROL_DT), 1)
        for _ in range(n_end):
            if emergency_flag.is_set():
                return
            cf.commander.send_position_setpoint(x_end, y_end, z_end, init_yaw_deg)
            time.sleep(CONTROL_DT)

        # ---------------- Return to origin (x=0, y=0, z=Z_TAKEOFF) ----------------
        print("[phase] return to origin")
        x_origin, y_origin, z_origin = 0.0, 0.0, Z_TAKEOFF
        dist_return = float(np.linalg.norm([x_origin - x_end,
                                            y_origin - y_end,
                                            z_origin - z_end]))
        return_time = max(dist_return / RETURN_SPEED, RETURN_MIN_S)
        n_return = max(int(return_time / CONTROL_DT), 1)

        yaw_start = init_yaw_deg
        yaw_end   = 0.0
        dyaw = ((yaw_end - yaw_start + 180.0) % 360.0) - 180.0

        for i in range(n_return):
            if emergency_flag.is_set():
                return
            frac = (i + 1) / n_return
            s = frac * frac * (3.0 - 2.0 * frac)
            x_cmd = x_end + (x_origin - x_end) * s
            y_cmd = y_end + (y_origin - y_end) * s
            z_cmd = z_end + (z_origin - z_end) * s
            yaw_cmd = yaw_start + dyaw * s
            cf.commander.send_position_setpoint(x_cmd, y_cmd,
                                                max(z_cmd, 0.4), yaw_cmd)
            time.sleep(CONTROL_DT)

        # ---------------- Hover above origin ----------------
        print("[phase] hover above origin")
        n_hover_origin = max(int(HOVER_AT_ORIGIN / CONTROL_DT), 1)
        for _ in range(n_hover_origin):
            if emergency_flag.is_set():
                return
            cf.commander.send_position_setpoint(x_origin, y_origin,
                                                z_origin, yaw_end)
            time.sleep(CONTROL_DT)

        # ---------------- Land at origin (0, 0, 0) ----------------
        print("[phase] landing at origin")
        n_land = max(int(LAND_RAMP_S / CONTROL_DT), 1)
        for i in range(n_land):
            if emergency_flag.is_set():
                return
            frac = 1.0 - (i + 1) / n_land
            z_cmd = max(z_origin * frac, 0.05)
            cf.commander.send_position_setpoint(x_origin, y_origin,
                                                z_cmd, yaw_end)
            time.sleep(CONTROL_DT)

        cf.commander.send_stop_setpoint()
        print("[done] landed at origin.")

    finally:
        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass
        try:
            cf.close_link()
        except Exception:
            pass


if __name__ == '__main__':
    main()
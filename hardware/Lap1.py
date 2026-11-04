#!/usr/bin/env python3
import contextlib
import os
import socket
import struct
import sys
import time

import cflib.crtp
import cv2
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from PyQt6 import QtCore, QtGui, QtWidgets


@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


# --- Configure these for your setup ---
URI         = 'radio://0/40/2M/E7E7E7E714'
AIDECK_IP   = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT  = 5001
START_MAGIC = b'FER'

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 200

CTRL_PERIOD_MS = 50  # 20 Hz control loop


# =============================================================================
#         Average gate position for sanity check
# =============================================================================
GATES_CSV = [
    (0.55, -0.81, 1.25,   -36.9),
    (1.65,  -0.8, 1.3,     12.3),
    (2.33,  0.00, 1.41,    89.7),
    (1.65,  0.85, 1.42,   157.7),
    (0.56,  0.89, 1.30, -162.88),
]
GATE_SANITY_RADIUS_M = 0.5  


# =============================================================================
#                    GATE DETECTION (Yann, FPV gate racer)
# =============================================================================
BRIGHT_THRESH     = 110            # mono intensity threshold for green-LED ring
MIN_GATE_AREA_PX  = 400
MAX_GATE_AREA_PX  = int(0.55 * IMG_WIDTH * IMG_HEIGHT)
BORDER_REJECT_PX  = 2


def order_corners(pts):
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    sorted_pts = pts[np.argsort(angles)]
    tl = int(np.argmin(sorted_pts.sum(axis=1)))
    return np.roll(sorted_pts, -tl, axis=0)


def detect_gate_yann(image):
    if image is None:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < MIN_GATE_AREA_PX or area > MAX_GATE_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if (x <= BORDER_REJECT_PX or y <= BORDER_REJECT_PX
                or x + w >= IMG_WIDTH  - BORDER_REJECT_PX
                or y + h >= IMG_HEIGHT - BORDER_REJECT_PX):
            continue
        ar = w / max(h, 1)
        if ar < 0.45 or ar > 2.2:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)

    if best is None:
        return None
    return order_corners(best)


# =============================================================================
#                                VIDEO THREAD
# =============================================================================
class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack(
                    '<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                self._decode_and_emit(buffer)
                receiving = False

    def _decode_and_emit(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_ready.emit(img)


# =============================================================================
#                       AUTONOMOUS CONTROLLER 
# =============================================================================
class Lap1Controller:
    def __init__(self):
        # Phase
        self.phase = "take_off"
        self.lap1_phase = "goto_wp"  

        # Take-off height
        self.z_take_off = 1.2 

        # Camera 
        self.fov = 1.5
        self.f   = None
        self.u0  = None
        self.v0  = None
        self.gate_height = 0.4

        self.R_cam_to_body = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0],
        ])
        self.cam_offset_body = np.array([0.03, 0.0, 0.01])

        # Scan
        self.scan_yaw_offset    = 0.0
        self.scan_direction     = 1
        self.scan_max_offset    = np.pi          
        self.scan_yaw_rate_rad  = 0.05 #np.deg2rad(35.0)  
        self.scan_t_last        = None

        # Gates info
        self.n_gates = 5
        self.gate_idx = 0
        self.gate_zone = [
            -np.pi/4,
             np.pi/6,
             np.pi/2,
             np.pi,
            -(5*np.pi)/6,
        ]
        self.search_wp = [
            [0, -0.3],
            [0.95, -1.25],
            [2.6, -1.2],
            [2.75, 0.6],
            [1.2, 1.35],
        ]
        
        self.gate_ref_xyz = [
            np.array([g[0], g[1], g[2]], dtype=float) for g in GATES_CSV
        ]
        self.gate_positions = [None] * self.n_gates
        self.gate_normals   = [None] * self.n_gates
        self.approach_pt    = [None] * self.n_gates
        self.exit_pt        = [None] * self.n_gates
        self.approach_yaw   = None

        # Gate tracking
        self.tracked_pixel = None
        self.tracked_max_jump = 80

        # Observation / triangulation
        self.baseline_right = 0.1
        self.obs1 = None
        self.obs2 = None
        self.baseline_target = None
        self.obs1_image = None
        self.obs2_image = None
        self.second_obs_t_in_phase = 0.0
        self.second_obs_timeout = 4.0
        self.frame_buffer = []
        self.N_frames_obs = 1

        # Misc
        self.start_pos = None
        self.last_gates_drawing = []

        # Landing
        self.land_x = 0.0
        self.land_y = 0.0
        self.land_yaw = 0.0
        self.land_target_z = 0.0
        self.land_rate = 0.3   
        self.land_cutoff_z = 0.12  

    # -------------------------------------------------------------------------
    #                  CAMERA / GATE DETECTION
    # -------------------------------------------------------------------------
    def init_camera_par(self, camera_data):
        height, width = camera_data.shape[:2]
        self.f  = width / (2 * np.tan(self.fov / 2))
        self.u0 = width  / 2
        self.v0 = height / 2

    def gate_detection(self, camera_data, tracked_pixel=None):
        if camera_data is None:
            self.last_gates_drawing = []
            return None

        corners = detect_gate_yann(camera_data)
        if corners is None:
            self.last_gates_drawing = []
            return None

        self.last_gates_drawing = [corners.astype(np.int32)]

        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))

        if tracked_pixel is not None:
            ref = np.array(tracked_pixel, dtype=float)
            dist = float(np.linalg.norm(np.array([cx, cy]) - ref))
            if dist > self.tracked_max_jump:
                return None

        return (cx, cy)

    # -------------------------------------------------------------------------
    #              SANITY CHECK : Gates in a correct range
    # -------------------------------------------------------------------------
    def sanity_check_gate(self, H, gate_idx, radius=GATE_SANITY_RADIUS_M):
        """
        Verifie que la position 3D H (porte estimee par triangulation) est
        dans un rayon `radius` de la position moyenne hardcodee
        `self.gate_ref_xyz[gate_idx]`. Retourne (ok, dist).
        """
        if gate_idx < 0 or gate_idx >= len(self.gate_ref_xyz):
            return True, 0.0  # pas de reference -> on laisse passer
        ref = self.gate_ref_xyz[gate_idx]
        dist = float(np.linalg.norm(np.asarray(H, dtype=float) - ref))
        return (dist <= radius), dist

    # -------------------------------------------------------------------------
    #                      LAP1 STATE MACHINE
    # -------------------------------------------------------------------------
    def compute_command(self, sensor_data, camera_data, dt):

        if camera_data is not None and self.f is None:
            self.init_camera_par(camera_data)

        # ----- Phase : Take Off -----
        if self.phase == "take_off":
            if self.start_pos is None:
                self.start_pos = (sensor_data['x_global'],
                                  sensor_data['y_global'],
                                  self.z_take_off)
            if sensor_data['z_global'] < 0.95 * self.z_take_off:
                return [sensor_data['x_global'],
                        sensor_data['y_global'],
                        self.z_take_off,
                        sensor_data['yaw']]
            self.phase = "lap1"
            self.lap1_phase = "goto_wp"
            self.scan_yaw_offset = 0.0
            self.scan_direction = 1

        # ----- Phase : Lap 1 -----
        if self.phase == "lap1":

            if self.gate_idx >= self.n_gates:
                self.phase = "end"
                return [sensor_data['x_global'],
                        sensor_data['y_global'],
                        sensor_data['z_global'],
                        sensor_data['yaw']]

            wp = self.search_wp[self.gate_idx]
            base_yaw = self.gate_zone[self.gate_idx]

            # --- 1. GO TO WAYPOINT ---
            if self.lap1_phase == "goto_wp":
                control_command = [float(wp[0]), float(wp[1]),
                                   self.z_take_off, float(base_yaw)]
                if self.pos_reached(sensor_data, control_command,
                                    pos_tol=0.10, yaw_tol=0.15):
                    self.lap1_phase = "search"
                    self.scan_yaw_offset = 0.0
                    self.scan_direction = 1
                    self.scan_t_last = None
                    print(f"GATE {self.gate_idx}: WP REACHED, ROTATING TO FIND GATE")
                return control_command

            # --- 2. SEARCH ---
            if self.lap1_phase == "search":
                #self.scan_yaw_offset += self.scan_direction * self.scan_yaw_rate_rad * dt
                self.scan_yaw_offset += self.scan_direction * self.scan_yaw_rate_rad
                if abs(self.scan_yaw_offset) > self.scan_max_offset:
                    self.scan_direction *= -1
            
                    self.scan_yaw_offset = float(np.clip(self.scan_yaw_offset,
                                                         -self.scan_max_offset,
                                                          self.scan_max_offset))

                target_yaw = base_yaw + self.scan_yaw_offset
                target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))

                control_command = [float(wp[0]), float(wp[1]),
                                   self.z_take_off, float(target_yaw)]

                det = self.gate_detection(camera_data, tracked_pixel=None)
                if det is not None:
                    self.obs1 = self.capture_observation(det, sensor_data)
                    self.obs1_image = camera_data.copy() if camera_data is not None else None

                    right_dir = np.array([
                        np.sin(sensor_data['yaw']),
                        -np.cos(sensor_data['yaw']),
                    ])
                    self.baseline_target = [
                        sensor_data['x_global'] + self.baseline_right * right_dir[0],
                        sensor_data['y_global'] + self.baseline_right * right_dir[1],
                        self.z_take_off,
                        sensor_data['yaw'],
                    ]

                    self.lap1_phase = "move_right_for_second_image"
                    self.tracked_pixel = det
                    self.second_obs_t_in_phase = 0.0
                    self.frame_buffer = []
                    print(f"GATE {self.gate_idx}: FIRST IMAGE SAVED at pix={det}, "
                          f"MOVING {self.baseline_right:.2f} m RIGHT")
                    return self.baseline_target

                return control_command

            # --- 3. MOVE RIGHT BASELINE ---
            if self.lap1_phase == "move_right_for_second_image":
                control_command = self.baseline_target
                if self.pos_reached(sensor_data, control_command, pos_tol=0.03):
                    self.lap1_phase = "search_second_image"
                    self.second_obs_t_in_phase = 0.0
                    self.frame_buffer = []
                    print("RIGHT BASELINE REACHED, SEARCHING SECOND IMAGE")
                return control_command

            # --- 4. SEARCH 2nd IMAGE ---
            if self.lap1_phase == "search_second_image":
                control_command = self.baseline_target
                self.second_obs_t_in_phase += dt

                det = self.gate_detection(camera_data, tracked_pixel=self.tracked_pixel)
                if det is not None:
                    self.tracked_pixel = det
                    self.frame_buffer.append((det, sensor_data.copy()))

                if len(self.frame_buffer) >= self.N_frames_obs:
                    pxs = [d[0][0] for d in self.frame_buffer]
                    pys = [d[0][1] for d in self.frame_buffer]
                    px_med, py_med = float(np.median(pxs)), float(np.median(pys))

                    mid_sensor = self.frame_buffer[len(self.frame_buffer)//2][1]
                    self.obs2 = self.capture_observation((px_med, py_med), mid_sensor)
                    self.obs2_image = camera_data.copy() if camera_data is not None else None

                    self.frame_buffer = []
                    self.lap1_phase = "triangulation"
                    print("SECOND IMAGE SAVED, TRIANGULATING")

                elif self.second_obs_t_in_phase > self.second_obs_timeout:
                    print("SECOND IMAGE NOT FOUND, BACK TO SEARCH")
                    self.reset_triangulation()
                    self.lap1_phase = "search"
                    self.scan_yaw_offset = 0.0
                    self.scan_direction = 1

                return control_command

            # --- 5. TRIANGULATION ---
            if self.lap1_phase == "triangulation":
                if self.obs1 is None or self.obs2 is None:
                    print("TRIANGULATION WITHOUT TWO OBSERVATIONS, BACK TO SEARCH")
                    self.reset_triangulation()
                    self.lap1_phase = "search"
                    return [float(wp[0]), float(wp[1]),
                            self.z_take_off, float(base_yaw)]

                P, r = self.obs1
                Q, s = self.obs2
                r = r / (np.linalg.norm(r) + 1e-9)
                s = s / (np.linalg.norm(s) + 1e-9)
                A = np.column_stack([r, -s])
                b = Q - P
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                lam, mu = x[0], x[1]
                F = P + lam * r
                G = Q + mu * s
                H = (F + G) / 2

                if lam <= 0 or mu <= 0:
                    print("BAD TRIANGULATION: intersection behind camera")
                    self.reset_triangulation()
                    self.lap1_phase = "search"
                    return [float(wp[0]), float(wp[1]),
                            self.z_take_off, float(base_yaw)]

                ray_gap = float(np.linalg.norm(F - G))
                if ray_gap > 0.5:
                    print(f"BAD TRIANGULATION: rays miss by {ray_gap:.2f} m")
                    self.reset_triangulation()
                    self.lap1_phase = "search"
                    return [float(wp[0]), float(wp[1]),
                            self.z_take_off, float(base_yaw)]

                # *** SANITY CHECK ***
                ok, dist_ref = self.sanity_check_gate(H, self.gate_idx)
                if not ok:
                    ref = self.gate_ref_xyz[self.gate_idx]
                    print(f"SANITY CHECK FAILED: gate {self.gate_idx} "
                          f"H={np.round(H, 2).tolist()} too far from "
                          f"ref={np.round(ref, 2).tolist()} "
                          f"(dist={dist_ref:.2f} m > {GATE_SANITY_RADIUS_M:.2f} m). "
                          f"REJECTING -> back to SEARCH.")
                    self.reset_triangulation()
                    self.lap1_phase = "search"
                    self.scan_yaw_offset = 0.0
                    self.scan_direction = 1
                    return [float(wp[0]), float(wp[1]),
                            self.z_take_off, float(base_yaw)]
                else:
                    print(f"SANITY CHECK OK: gate {self.gate_idx} "
                          f"dist={dist_ref:.2f} m (<= {GATE_SANITY_RADIUS_M:.2f} m)")

                self.gate_positions[self.gate_idx] = H

                r_h = np.array([r[0], r[1], 0.0])
                r_h = r_h / (np.linalg.norm(r_h) + 1e-9)
                self.gate_normals[self.gate_idx] = r_h

                self.approach_pt[self.gate_idx] = H - 0.7 * r_h
                self.exit_pt[self.gate_idx]     = H + 0.7 * r_h
                self.approach_yaw = np.arctan2(r_h[1], r_h[0])
                self.approach_yaw = np.arctan2(np.sin(self.approach_yaw),
                                               np.cos(self.approach_yaw))
                self.lap1_phase = "approach"
                print(f"GATE {self.gate_idx} TRIANGULATED at H={H}, "
                      f"approach_yaw={np.rad2deg(self.approach_yaw):.1f} deg")

            # --- 6. APPROACH ---
            if self.lap1_phase == "approach":
                control_command = [
                    float(self.approach_pt[self.gate_idx][0]),
                    float(self.approach_pt[self.gate_idx][1]),
                    float(max(self.approach_pt[self.gate_idx][2], 0.4)),
                    float(self.approach_yaw),
                ]
                if self.pos_reached(sensor_data, control_command):
                    self.lap1_phase = "through"
                return control_command

            # --- 7. THROUGH ---
            if self.lap1_phase == "through":
                control_command = [
                    float(self.exit_pt[self.gate_idx][0]),
                    float(self.exit_pt[self.gate_idx][1]),
                    float(max(self.exit_pt[self.gate_idx][2], 0.4)),
                    float(self.approach_yaw),
                ]
                H = self.gate_positions[self.gate_idx]
                n = self.gate_normals[self.gate_idx]
                drone_pos = np.array([sensor_data['x_global'],
                                      sensor_data['y_global'],
                                      sensor_data['z_global']])
                along = np.dot(drone_pos - H, n)

                if along > 0.4:
                    print(f"GATE {self.gate_idx} PASSED")
                    self.gate_idx += 1
                    self.reset_triangulation()
                    self.lap1_phase = "goto_wp"
                    self.scan_yaw_offset = 0.0
                    self.scan_direction = 1

                return control_command

        # ----- Phase : end -----
        if self.phase == "end":
            control_command = [0.0, 0.0, self.z_take_off, sensor_data['yaw']]

            if self.pos_reached(sensor_data, control_command):
                self.phase = "land"
                print("ALL GATES DONE -> LANDING")
            return control_command
            

        # ----- Phase : land -----
        if self.phase == "land":
            if abs(sensor_data['z_global'] - self.land_target_z) < 0.03:
                self.land_target_z -= 0.1
            control_command = [0.0, 0.0, self.land_target_z, sensor_data['yaw']]
            if (sensor_data['z_global'] < self.land_cutoff_z):
                self.phase = "shutdown"
                print("LANDED -> SHUTDOWN")
                return None

            return control_command

        # ----- Phase : shutdown -----
        if self.phase == "shutdown":
            return None
        
        # fallback 
        return [sensor_data['x_global'],
                sensor_data['y_global'],
                sensor_data['z_global'],
                sensor_data['yaw']]

    # -------------------------------------------------------------------------
    #                              HELPERS
    # -------------------------------------------------------------------------
    def capture_observation(self, det, sensor_data):
        px, py = det
        P, r = self.pixel_to_world_pos(px, py, sensor_data.copy())
        return P, r

    def reset_triangulation(self):
        self.obs1 = None
        self.obs2 = None
        self.obs1_image = None
        self.obs2_image = None
        self.baseline_target = None
        self.frame_buffer = []
        self.tracked_pixel = None
        self.second_obs_t_in_phase = 0.0

    def pos_reached(self, sensor_data, goal, pos_tol=0.1, yaw_tol=0.1):
        dx = sensor_data['x_global'] - goal[0]
        dy = sensor_data['y_global'] - goal[1]
        dz = sensor_data['z_global'] - goal[2]
        dyaw = np.arctan2(np.sin(sensor_data['yaw'] - goal[3]),
                          np.cos(sensor_data['yaw'] - goal[3]))
        return (dx*dx + dy*dy + dz*dz) ** 0.5 < pos_tol and abs(dyaw) < yaw_tol

    def pixel_to_world_pos(self, px, py, sensor_data):
        v_cam = np.array([px - self.u0, py - self.v0, self.f])
        v_body = self.R_cam_to_body @ v_cam
        R_b2w  = self.R_body_to_world(sensor_data)
        v_inertial = R_b2w @ v_body
        drone_pos = np.array([sensor_data['x_global'],
                              sensor_data['y_global'],
                              sensor_data['z_global']])
        cam_pos = drone_pos + R_b2w @ self.cam_offset_body
        return cam_pos, v_inertial

    def R_body_to_world(self, sensor_data):
        cr, sr = np.cos(sensor_data['roll']),  np.sin(sensor_data['roll'])
        cp, sp = np.cos(sensor_data['pitch']), np.sin(sensor_data['pitch'])
        cy, sy = np.cos(sensor_data['yaw']),   np.sin(sensor_data['yaw'])
        R_roll  = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_yaw   = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return R_yaw @ R_pitch @ R_roll


# =============================================================================
#                               MAIN WINDOW
# =============================================================================
class Lap1Window(QtWidgets.QWidget):
    _connected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crazyflie Lap1 (Hybride : Yann + vision.py)')

        self.image_label  = QtWidgets.QLabel('Waiting for video...')
        self.status_label = QtWidgets.QLabel(f'Connecting to {URI}...')
        self.phase_label  = QtWidgets.QLabel('phase: -- / lap1_phase: --')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.phase_label)

        self.latest_frame = None
        self.sensor_data = {
            'x_global': 0.0, 'y_global': 0.0, 'z_global': 0.0,
            'v_x': 0.0,      'v_y': 0.0,      'v_z': 0.0,
            'ax_global': 0.0,'ay_global': 0.0,'az_global': 0.0,
            'roll': 0.0,     'pitch': 0.0,    'yaw': 0.0,
            'q_x': 0.0,      'q_y': 0.0,      'q_z': 0.0, 'q_w': 1.0,
        }
        self.have_sensors = False

        self.controller = Lap1Controller()
        self.running = True
        self.last_tick = time.time()

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._on_frame)
        self.video.start()

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache='cache')
        self._connected_signal.connect(self._on_connected)
        self.cf.connected.add_callback(self._connected_signal.emit)
        self.cf.open_link(URI)

        # Control loop
        self._ctrl_timer = QtCore.QTimer(self)
        self._ctrl_timer.timeout.connect(self._control_step)
        self._ctrl_timer.setInterval(CTRL_PERIOD_MS)

    # ---- Crazyflie callbacks ----
    def _on_connected(self, uri):
        self.status_label.setText(f'Connected to {uri}')
        self.cf.supervisor.send_arming_request(True)
        self._start_log()
        self._ctrl_timer.start()

    def _start_log(self):
        log_pose = LogConfig(name='Pose', period_in_ms=20)
        log_pose.add_variable('stateEstimate.x', 'float')
        log_pose.add_variable('stateEstimate.y', 'float')
        log_pose.add_variable('stateEstimate.z', 'float')
        log_pose.add_variable('stateEstimate.yaw',   'float')
        log_pose.add_variable('stateEstimate.roll',  'float')
        log_pose.add_variable('stateEstimate.pitch', 'float')

        log_vel = LogConfig(name='Vel', period_in_ms=20)
        log_vel.add_variable('stateEstimate.vx', 'float')
        log_vel.add_variable('stateEstimate.vy', 'float')
        log_vel.add_variable('stateEstimate.vz', 'float')
        log_vel.add_variable('stateEstimate.ax', 'float')
        log_vel.add_variable('stateEstimate.ay', 'float')
        log_vel.add_variable('stateEstimate.az', 'float')

        try:
            self.cf.log.add_config(log_pose)
            log_pose.data_received_cb.add_callback(self._on_pose)
            log_pose.start()

            self.cf.log.add_config(log_vel)
            log_vel.data_received_cb.add_callback(self._on_vel)
            log_vel.start()
        except KeyError as e:
            print(f"Log variable not found: {e}")
        except AttributeError as e:
            print(f"Bad log config: {e}")

    def _on_pose(self, ts, data, cfg):
        self.sensor_data['x_global'] = data['stateEstimate.x']
        self.sensor_data['y_global'] = data['stateEstimate.y']
        self.sensor_data['z_global'] = data['stateEstimate.z']
        self.sensor_data['yaw']   = np.deg2rad(data['stateEstimate.yaw'])
        self.sensor_data['roll']  = np.deg2rad(data['stateEstimate.roll'])
        self.sensor_data['pitch'] = np.deg2rad(data['stateEstimate.pitch'])
        self.have_sensors = True

    def _on_vel(self, ts, data, cfg):
        self.sensor_data['v_x'] = data['stateEstimate.vx']
        self.sensor_data['v_y'] = data['stateEstimate.vy']
        self.sensor_data['v_z'] = data['stateEstimate.vz']
        self.sensor_data['ax_global'] = data['stateEstimate.ax']
        self.sensor_data['ay_global'] = data['stateEstimate.ay']
        self.sensor_data['az_global'] = data['stateEstimate.az']

    # ---- Frame video ----
    def _on_frame(self, img):
        self.latest_frame = img
        self._draw_frame(img)

    def _draw_frame(self, img):
        display_img = img.copy()
        if display_img.ndim == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

        for corners in self.controller.last_gates_drawing:
            contour = corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)

        if self.controller.tracked_pixel is not None:
            cx, cy = self.controller.tracked_pixel
            cv2.drawMarker(display_img, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)

        cv2.drawMarker(display_img, (IMG_WIDTH // 2, IMG_HEIGHT // 2),
                       (255, 0, 0), cv2.MARKER_TILTED_CROSS, 16, 1)

        h, w, _ = display_img.shape
        qimg = QtGui.QImage(display_img.data, w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    # ---- Control loop ----
    def _control_step(self):
        if not self.running:
            return
        if not self.have_sensors:
            return

        now = time.time()
        dt = max(now - self.last_tick, 1e-3)
        self.last_tick = now

        cmd = self.controller.compute_command(self.sensor_data,
                                              self.latest_frame, dt)
        if cmd is None:
            if self.controller.phase == "shutdown":
                print("MISSION COMPLETE -> CLOSING APP")
                self.running = False
                self._ctrl_timer.stop()
                try:
                    self.cf.commander.send_stop_setpoint()
                except Exception:
                    pass
                try:
                    self.cf.close_link()
                except Exception:
                    pass
                QtWidgets.QApplication.quit()
            return

        x_cmd, y_cmd, z_cmd, yaw_cmd_rad = cmd
        yaw_cmd_deg = float(np.rad2deg(yaw_cmd_rad))

        if self.controller.phase == "land":
            z_cmd = float(np.clip(z_cmd, 0.0, 2.0))
        else:
            z_cmd = float(np.clip(z_cmd, 0.1, 2.0))

        self.cf.commander.send_position_setpoint(
            float(x_cmd), float(y_cmd), z_cmd, yaw_cmd_deg)

        self.phase_label.setText(
            f"phase: {self.controller.phase} / "
            f"lap1_phase: {self.controller.lap1_phase} / "
            f"gate_idx: {self.controller.gate_idx} / "
            f"cmd: x={x_cmd:.2f} y={y_cmd:.2f} z={z_cmd:.2f} "
            f"yaw={yaw_cmd_deg:.1f}deg"
        )

    # ---- Emergency stop ----
    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == QtCore.Qt.Key.Key_Space:
            print("EMERGENCY STOP")
            self.running = False
            self._ctrl_timer.stop()
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

    def closeEvent(self, event):
        self.running = False
        self._ctrl_timer.stop()
        try:
            self.cf.commander.send_stop_setpoint()
            self.cf.close_link()
        except Exception:
            pass
        event.accept()


# =============================================================================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Lap1Window()
    win.show()
    sys.exit(app.exec())
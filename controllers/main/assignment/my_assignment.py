import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# The available ground truth state measurements can be accessed by calling sensor_data[item].
# All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# "v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravitational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value


class MyAssignment:
    def __init__(self):
        # ============================================================
        # DEBUG
        # ============================================================
        self.DEBUG = False
        self.DEBUG_VISION = False
        self.DEBUG_STATE = True
        self.DEBUG_PASS = True
        self.DEBUG_GATE_PROGRESS = True
        self.DEBUG_EVERY = 80
        self.DEBUG_PASS_EVERY = 25
        self.DEBUG_REJECT_EVERY = 200
        self._reject_counter = 0

        # ============================================================
        # ARENA
        # ============================================================
        self.center = np.array([4.0, 4.0], dtype=float)
        self.home = np.array([1.0, 4.0, 1.0], dtype=float)
        self.num_gates = 5
        self.num_segments = 6
        self.segment_angular_size = np.pi / self.num_segments
        self.angular_bounds = []

        for i in range(self.num_segments):
            a0 = ((2 * i - 0.5) * self.segment_angular_size) % (2 * np.pi)
            a1 = ((2 * i + 0.5) * self.segment_angular_size) % (2 * np.pi)
            self.angular_bounds.append((a0, a1))

        # ============================================================
        # LAPS / STRATEGY
        # ============================================================
        # Priority: pass the gates reliably.
        # FAST mode is kept in the code, but disabled by default.
        self.lap_count = 1
        self.safe_total_laps = 3
        self.use_fast_laps = False
        self.fast_laps_left = 0

        # ============================================================
        # FLIGHT PARAMETERS
        # ============================================================
        self.takeoff_z = 1.0
        self.cruise_z = 1.10
        self.search_radii = [2.1, 2.7, 3.2]
        self.search_zs = [1.00, 1.18, 1.35]

        self.wp_tol = 0.24
        self.align_tol = 0.25
        self.traverse_tol = 0.23
        self.traverse_gate_tol = 0.16  # tighter tolerance specifically for gate-center wp

        self.approach_d = 1.15
        self.center_d = 0.10
        self.exit_d = 1.15

        # ============================================================
        # CAMERA MODEL
        # ============================================================
        w = h = 300
        fov = 1.5
        self.fx = self.fy = (w / 2.0) / np.tan(fov / 2.0)
        self.cx = w / 2.0
        self.cy = h / 2.0
        self.img_w = w
        self.img_h = h

        # ============================================================
        # HEIGHT / ALIGNMENT TUNING
        # ============================================================
        self.align_max_ticks = 280
        self.align_ticks = 0
        self.align_center_count = 0
        self.align_required_count = 5
        self.align_px_tol_x = 18
        self.align_px_tol_y = 16
        self.align_vertical_gain = 0.0030
        self.align_vertical_step_clip = 0.12
        self.align_z_blend = 0.45
        self.align_bbox_min_area = 140
        self.height_margin = 0.02

        # Extra safety before crossing: do not start TRAVERSE unless the
        # drone is facing the gate axis and is already near the gate altitude.
        self.align_yaw_max_deg = 20.0
        self.pre_gate_z_tol = 0.12
        self.align_z_required_count = 5
        self.align_z_ok_count = 0
        self.align_dz_lift_trigger = 0.15
        self.align_visual_z_max_correction = 0.05

        # ============================================================
        # SEARCH VISUAL ACQUISITION TUNING
        # ============================================================
        self.search_visual_px_trigger = 70
        self.search_visual_hold_xy = True
        self.search_visual_yaw_blend = 0.70
        self.search_visual_z_gain = 0.0022
        self.search_visual_z_clip = 0.08

        # ============================================================
        # FIRST GATE SCAN
        # ============================================================
        self.first_gate_scan_done = False
        self.first_gate_scan_ticks = 0
        self.first_gate_scan_max_ticks = 240
        self.first_gate_scan_base_yaw = None
        self.first_gate_scan_total_angle = np.deg2rad(180)
        self.first_gate_scan_pos = None

        self.first_gate_scan_attempt = 0
        self.first_gate_scan_max_attempts = 4
        self.first_gate_forward_distance = 0.35
        self.first_gate_forward_waypoint = None

        self.first_gate_candidates = []
        self.first_gate_candidate_min_area = 260
        self.first_gate_candidate_consistency_tol = 0.45
        self.first_gate_candidate_min_count = 3
        self.first_gate_candidate_min_radius = 1.35
        self.first_gate_candidate_max_radius = 3.65
        self.first_gate_candidate_max_drone_dist = 3.35
        self.predicted_gate_radius = None

        # ============================================================
        # ROBUST PASS DETECTION
        # ============================================================
        # Slightly more permissive than the original version.
        # The internal pass check is now used mostly for debug.
        # We do NOT finish a gate early just because this check is true,
        # because the simulator can validate the official gate crossing a bit later.
        self.pass_proj_threshold = -0.03
        self.pass_lateral_tol = 0.42
        self.pass_z_tol = 0.35
        self.pass_dist_tol = 1.05

        # ============================================================
        # STATE
        # ============================================================
        self.state = "TAKEOFF"
        self.current_segment = 1

        self.search_waypoints = []
        self.search_wp_idx = 0

        self.align_waypoints = []
        self.align_wp_idx = 0

        self.traverse_waypoints = []
        self.traverse_wp_idx = 0
        # Snapshot of gate position captured at start of each traverse.
        # Reverted-to after the gate is done to freeze a known-good position.
        self.traverse_gate_start_pos = {}
        # Gates that have been fully traversed; their mapped positions are frozen.
        self.traversed_gates = set()
        # Initial confirmed position for each gate (set on first confirmation, before
        # any ALIGN-phase blending drift).  Used for the first-lap traverse so the
        # approach axis is aimed at the physically observed gate, not a drifted estimate.
        self.gate_first_confirmed_pos = {}
        # Visual lock-on / commit phase: once gate fills the image, fly straight forward
        # in the current heading instead of toward a (possibly biased) waypoint.
        self.traverse_committed = set()
        self.traverse_commit_area = 1500   # bbox area threshold to commit
        self.commit_max_cx_err = 25        # px, gate must be visually centered before commit
        self.commit_max_cy_err = 35        # px, gate must be at near-correct height before commit
        self.visual_nudge_gain = 0.0012    # m per pixel of cx_error
        self.visual_nudge_max = 0.10       # m, clip per tick

        self.fast_waypoints = []
        self.fast_wp_idx = 0

        # ============================================================
        # MAPPING / MEMORY
        # ============================================================
        self.mapped_gates = {}
        # Remap helper for gate 1.
        # Some simulations make the first immobile scan pick a plausible but wrong pink gate.
        # If later, while actively targeting gate 1, we repeatedly see another coherent gate-1
        # estimate not too far away, we allow gate 1 to correct itself.
        self.gate1_remap_candidate = None
        self.gate1_remap_count = 0
        self.trackers = {
            1: GateMemory(confirm_threshold=4.0, gate_id=1, debug_fn=self.dbg_tracker),
            **{i: GateMemory(confirm_threshold=2.5, gate_id=i, debug_fn=self.dbg_tracker) for i in range(2, 6)},
        }
        self.last_detection = None

        # ============================================================
        # GATE NORMALS + ACTIVE INSPECTION MEMORY
        # ============================================================
        # gate_normals stores the actual traversal vector through the gate
        # (pre-gate side -> post-gate side). get_tangent() uses it first,
        # then falls back to the old circular-course tangent.
        self.gate_normals = {}
        self.gate_normal_confidence = {}
        self.gate_normal_min_conf = 0.40

        # First-lap active mapping: before committing a gate, hover in front
        # of it, collect clean samples, then store a robust median model.
        self.gate_models = {}
        self.inspect_segment = None
        self.inspect_samples = []
        self.inspect_ticks = 0
        self.inspect_stage = "GOTO"
        self.inspect_staging_wp = None
        self.inspect_max_ticks = 220
        self.inspect_min_samples = 8
        self.inspect_stable_std_xy = 0.18
        self.inspect_stable_std_z = 0.12
        self.inspect_distance = 1.35
        self.inspect_lateral_gain = 0.0025
        self.inspect_vertical_gain = 0.0025
        self.inspect_max_xy_correction = 0.18
        self.inspect_max_z_correction = 0.10
        self.inspect_bbox_min_area = 220
        self.inspect_px_tol_x = 35
        self.inspect_px_tol_y = 35
        self.inspect_yaw_max_deg = 15.0

        # ============================================================
        # COMMAND FILTERING
        # ============================================================
        # More conservative command filter for safe laps.
        self._last_cmd = None
        self._last_dt = 0.0
        self.max_xy_step = 0.10
        self.max_z_step = 0.12
        self.max_yaw_step = 0.14
        self.alpha_xy = 0.26
        self.alpha_z = 0.36
        self.alpha_yaw = 0.12

        self.tick = 0

        # ============================================================
        # ROBUST RECOVERY / DE-SYNC DETECTION
        # ============================================================
        # Pattern observed in failed runs: while searching/inspecting gate k,
        # the camera repeatedly sees gate k+1. If we keep moving with the normal
        # search pattern, the drone stays too close/too far along the course and
        # can map the wrong gate, then the official progress and internal state
        # desynchronize. These counters trigger a deliberate backtrack / wide
        # recovery viewpoint before accepting a rough map.
        self.search_ticks = 0
        self.next_gate_seen_count = 0
        self.consecutive_next_gate_seen = 0
        self.wrong_gate_seen_count = 0
        self.frames_since_current_gate_seen = 0
        self.recovery_active = False
        self.recovery_waypoint = None
        self.recovery_attempts = {i: 0 for i in range(1, 6)}
        self.recovery_post_cross_hits = {i: 0 for i in range(1, 6)}
        self.post_cross_exit_active = False
        self.post_cross_exit_waypoint = None
        self.suppress_post_cross_break_once = set()
        self.valid_crossing_candidate_gates = set()
        self.max_search_ticks_before_recovery = 520
        self.next_gate_seen_recovery_threshold = 7
        self.max_recovery_attempts_per_gate = 3
        self.gate5_crossed_plane = False

    # ============================================================
    # DEBUG HELPERS
    # ============================================================
    def dbg(self, msg, kind="INFO", force=False):
        if self.DEBUG or force:
            print(f"[t={self.tick:04d}] [{kind}] {msg}")

    def dbg_state(self, old, new, extra=""):
        if self.DEBUG_STATE:
            suffix = f" | {extra}" if extra else ""
            print(f"[t={self.tick:04d}] [STATE] {old} -> {new}{suffix}")

    def set_state(self, new_state, extra=""):
        old = getattr(self, "state", None)
        self.state = new_state
        if old != new_state:
            self.dbg_state(old, new_state, extra)

    def dbg_tracker(self, msg):
        if self.DEBUG_VISION:
            print(f"[t={self.tick:04d}] [TRACKER] {msg}")

    def periodic_debug(self, sensor_data):
        if not self.DEBUG:
            return
        if self.tick % self.DEBUG_EVERY != 0:
            return

        pos = self._pos(sensor_data)
        mapped = sorted(list(self.mapped_gates.keys()))
        det = "none"

        if self.last_detection is not None:
            det = (
                f"seg={self.last_detection['segment']} "
                f"area={self.last_detection['bbox']['area']:.0f} "
                f"cxerr={self.last_detection['cx_error']:.1f} "
                f"cyerr={self.last_detection['cy_error']:.1f}"
            )

        self.dbg(
            f"lap={self.lap_count} state={self.state} gate={self.current_segment} "
            f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
            f"mapped={mapped} last_det={det}",
            "PERIODIC",
        )

    def reject_debug(self, reason, det=None, world_gate=None, seg=None):
        if not self.DEBUG_VISION:
            return

        self._reject_counter += 1
        if self._reject_counter % self.DEBUG_REJECT_EVERY != 0:
            return

        pieces = [reason]
        if seg is not None:
            pieces.append(f"seg={seg}")
        if det is not None:
            pieces.append(
                f"bbox_area={det.get('area', -1):.0f} "
                f"cx={det.get('cx', -1):.1f} cy={det.get('cy', -1):.1f} "
                f"near_edge={det.get('near_edge', False)}"
            )
        if world_gate is not None:
            r = np.linalg.norm(world_gate[:2] - self.center)
            pieces.append(f"world=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f}) r={r:.2f}")

        self.dbg(" | ".join(pieces), "REJECT")

    def dbg_gate_progress(self, sensor_data, label=""):
        if not self.DEBUG_GATE_PROGRESS:
            return
        if self.tick % self.DEBUG_PASS_EVERY != 0:
            return

        pos = self._pos(sensor_data)

        if self.current_segment in self.mapped_gates:
            gate = self.mapped_gates[self.current_segment]
            tangent = self.get_tangent(gate)
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            rel = pos[:2] - gate[:2]

            proj = rel[0] * tangent[0] + rel[1] * tangent[1]
            lateral = rel[0] * normal[0] + rel[1] * normal[1]
            dist = np.linalg.norm(rel)
            z_err = abs(pos[2] - gate[2])

            print(
                f"[t={self.tick:04d}] [GATE_DEBUG] lap={self.lap_count} "
                f"state={self.state} gate={self.current_segment} {label} | "
                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"gate=({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}) | "
                f"proj={proj:.2f} lat={lateral:.2f} dist={dist:.2f} zerr={z_err:.2f}"
            )
        else:
            print(
                f"[t={self.tick:04d}] [GATE_DEBUG] lap={self.lap_count} "
                f"state={self.state} gate={self.current_segment} {label} | "
                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) gate=NOT_MAPPED"
            )

    # ============================================================
    # MAIN CONTROL
    # ============================================================
    def compute_command(self, sensor_data, camera_data, dt):
        self.tick += 1
        self._last_dt = dt

        if camera_data is not None and self.state in {
            "FIRST_GATE_SCAN",
            "FIRST_GATE_FORWARD",
            "SEARCH",
            "INSPECT_GATE",
            "ALIGN",
            "TRAVERSE",
            "FAST",
        }:
            self.update_vision(sensor_data, camera_data)

        self.periodic_debug(sensor_data)
        pos = self._pos(sensor_data)

        # ------------------------------------------------------------
        # TAKEOFF
        # ------------------------------------------------------------
        if self.state == "TAKEOFF":
            control_command = [pos[0], pos[1], self.takeoff_z, sensor_data["yaw"]]

            if pos[2] > 0.95:
                if self.current_segment == 1 and not self.first_gate_scan_done:
                    self.start_first_gate_scan(sensor_data)
                else:
                    self.start_search(self.current_segment)

            return self.apply_command_filter(control_command, dt)

        # ------------------------------------------------------------
        # FIRST GATE SCAN
        # ------------------------------------------------------------
        if self.state == "FIRST_GATE_SCAN":
            scan_cmd = self.first_gate_initial_scan(sensor_data)

            if scan_cmd is None:
                best = self.select_best_first_gate_candidate()

                if best is not None:
                    self.mapped_gates[1] = best.copy()
                    self.predicted_gate_radius = float(np.linalg.norm(best[:2] - self.center))
                    self.first_gate_scan_done = True

                    print(
                        f"[t={self.tick:04d}] [GATE] Gate 1 selected from candidates: "
                        f"pos=({best[0]:.2f},{best[1]:.2f},{best[2]:.2f}) "
                        f"radius={self.predicted_gate_radius:.2f} candidates={len(self.first_gate_candidates)}"
                    )

                    if self.lap_count == 1 and 1 not in self.gate_models:
                        self.start_inspect_gate(sensor_data, 1)
                        return self.command_to_waypoint(self.inspect_staging_wp)

                    self.start_align(sensor_data, 1)
                    return self.command_to_waypoint(self.align_waypoints[0])

                print(
                    f"[t={self.tick:04d}] [MISS] Gate 1 scan failed "
                    f"attempt={self.first_gate_scan_attempt}/{self.first_gate_scan_max_attempts} "
                    f"candidates={len(self.first_gate_candidates)}"
                )

                if self.first_gate_scan_attempt >= self.first_gate_scan_max_attempts:
                    print(f"[t={self.tick:04d}] [MISS] Gate 1 scan exhausted. Starting dedicated recovery SEARCH.")
                    self.start_search(1)
                    return self.start_or_continue_search_recovery(sensor_data, force=True)

                self.start_first_gate_forward(sensor_data)
                return self.command_to_waypoint(self.first_gate_forward_waypoint)

            return self.apply_command_filter(scan_cmd, dt)

        # ------------------------------------------------------------
        # FIRST GATE FORWARD
        # ------------------------------------------------------------
        if self.state == "FIRST_GATE_FORWARD":
            if self.reached_waypoint(sensor_data, self.first_gate_forward_waypoint, tol=0.18):
                self.first_gate_scan_attempt += 1
                print(
                    f"[t={self.tick:04d}] [GATE] Reached first-gate forward waypoint. "
                    f"New scan attempt={self.first_gate_scan_attempt}"
                )
                self.start_first_gate_scan(sensor_data)
                hold_cmd = [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"], sensor_data["yaw"]]
                return self.apply_command_filter(hold_cmd, dt)

            return self.command_to_waypoint(self.first_gate_forward_waypoint)

        # ------------------------------------------------------------
        # SEARCH
        # ------------------------------------------------------------
        if self.state == "SEARCH":
            self.search_ticks += 1

            # If we are trying to find gate k but repeatedly see gate k+1,
            # move deliberately backward/outward to create a wider viewpoint.
            # This prevents the rare loop where gate 4 is missed and gate 5 is
            # seen forever from too close a pose.
            if self._search_recovery_needed(sensor_data):
                return self.start_or_continue_search_recovery(sensor_data)

            if self.current_segment in self.mapped_gates:
                gate = self.mapped_gates[self.current_segment]

                if self.should_inspect_gate(self.current_segment):
                    print(
                        f"[t={self.tick:04d}] [GATE] Gate {self.current_segment} rough-mapped. "
                        f"Entering INSPECT_GATE at ({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f})"
                    )
                    self.start_inspect_gate(sensor_data, self.current_segment)
                    if self.current_segment == 5:
                        self.inspect_staging_wp = self.make_safe_gate5_waypoint(sensor_data, self.inspect_staging_wp)
                    return self.command_to_waypoint(self.inspect_staging_wp)

                print(
                    f"[t={self.tick:04d}] [GATE] Gate {self.current_segment} already mapped/inspected. "
                    f"Starting ALIGN at ({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f})"
                )
                self.start_align(sensor_data, self.current_segment)
                return self.command_to_waypoint(self.align_waypoints[0])

            search_acq_cmd = self.search_visual_acquisition_command(sensor_data)
            if search_acq_cmd is not None:
                if self.current_segment == 5:
                    search_acq_cmd = self.make_safe_gate5_waypoint(sensor_data, search_acq_cmd)
                return self.apply_command_filter(search_acq_cmd, dt)

            target = self.search_waypoints[self.search_wp_idx]
            if self.current_segment == 5:
                target = self.make_safe_gate5_waypoint(sensor_data, target)

            if self.reached_waypoint(sensor_data, target, tol=self.wp_tol):
                self.dbg(f"SEARCH wp reached gate={self.current_segment} wp={self.search_wp_idx}", "WP")
                self.search_wp_idx = (self.search_wp_idx + 1) % len(self.search_waypoints)
                target = self.search_waypoints[self.search_wp_idx]
                if self.current_segment == 5:
                    target = self.make_safe_gate5_waypoint(sensor_data, target)

            return self.command_to_waypoint(target)

        # ------------------------------------------------------------
        # INSPECT_GATE: first-lap active mapping before ALIGN/TRAVERSE
        # ------------------------------------------------------------
        if self.state == "INSPECT_GATE":
            return self.run_inspect_gate(sensor_data)

        # ------------------------------------------------------------
        # ALIGN
        # ------------------------------------------------------------
        if self.state == "ALIGN":
            self.align_ticks += 1
            self.dbg_gate_progress(sensor_data, label="ALIGN")

            if self.current_segment not in self.mapped_gates:
                print(f"[t={self.tick:04d}] [MISS] Lost mapped gate {self.current_segment} during ALIGN. Returning SEARCH.")
                self.start_search(self.current_segment)
                return self.command_to_waypoint(self.search_waypoints[0])

            # Refresh approach waypoints regularly because the tracker can refine gate position.
            if self.align_ticks % 8 == 1:
                self.start_align(sensor_data, self.current_segment, preserve_counter=True)

            det_ok = (
                self.last_detection is not None
                and self.last_detection.get("segment") == self.current_segment
                and self.last_detection.get("bbox", {}).get("area", 0) >= self.align_bbox_min_area
            )
            centered = det_ok and self.last_detection.get("centered", False)
            close_to_approach = self.reached_waypoint(sensor_data, self.align_waypoints[0], tol=self.align_tol)

            if centered or close_to_approach:
                self.align_center_count += 1
            else:
                self.align_center_count = max(0, self.align_center_count - 1)

            if self.align_ticks % 25 == 0:
                print(
                    f"[t={self.tick:04d}] [ALIGN] gate={self.current_segment} "
                    f"det_ok={det_ok} centered={centered} close={close_to_approach} "
                    f"center_count={self.align_center_count} ticks={self.align_ticks}"
                )

            if self.align_center_count >= self.align_required_count:
                # Check geometric lateral, yaw, and altitude alignment before committing.
                gate = self.mapped_gates[self.current_segment]
                tangent = self.get_tangent(gate)
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                pos_now = self._pos(sensor_data)
                rel = pos_now[:2] - gate[:2]
                lateral = abs(rel[0] * normal[0] + rel[1] * normal[1])

                target_yaw = float(np.arctan2(tangent[1], tangent[0]))
                yaw_err_deg = abs(float(np.degrees(self.wrap_angle(target_yaw - sensor_data["yaw"]))))
                target_z = self.get_gate_target_z(self.current_segment)
                z_err = abs(float(sensor_data["z_global"] - target_z))
                z_ok = z_err < self.pre_gate_z_tol
                yaw_ok = yaw_err_deg < self.align_yaw_max_deg

                if self.align_ticks % 25 == 0:
                    print(
                        f"[t={self.tick:04d}] [ALIGN_ANGLE] gate={self.current_segment} "
                        f"angle_deg={yaw_err_deg:.1f}<{self.align_yaw_max_deg:.1f} "
                        f"zerr={z_err:.2f}<{self.pre_gate_z_tol:.2f} "
                        f"yaw_ok={yaw_ok} z_ok={z_ok}"
                    )

                if (lateral > 0.38 or not yaw_ok or not z_ok) and self.align_ticks < self.align_max_ticks - 30:
                    if self.align_ticks % 25 == 0:
                        print(
                            f"[t={self.tick:04d}] [ALIGN_HOLD] gate={self.current_segment} "
                            f"lateral={lateral:.2f} yaw_err={yaw_err_deg:.1f} zerr={z_err:.2f}"
                        )
                    self.align_center_count = max(0, self.align_center_count - 1)
                    hold = np.asarray(self.align_waypoints[0], dtype=float).copy()
                    hold[2] = target_z
                    hold[3] = target_yaw
                    return self.command_to_waypoint(hold)

                print(
                    f"[t={self.tick:04d}] [GATE] Gate {self.current_segment} aligned. "
                    f"Starting TRAVERSE. lateral={lateral:.2f} yaw_err={yaw_err_deg:.1f} zerr={z_err:.2f}"
                )
                self.start_traverse(self.current_segment)
                return self.command_to_waypoint(self.traverse_waypoints[0])

            if self.align_ticks > self.align_max_ticks:
                print(
                    f"[t={self.tick:04d}] [TIMEOUT] ALIGN timeout on gate {self.current_segment}. "
                    f"Forcing TRAVERSE. center_count={self.align_center_count}"
                )
                self.start_traverse(self.current_segment)
                return self.command_to_waypoint(self.traverse_waypoints[0])

            target = self.align_waypoints[min(self.align_wp_idx, len(self.align_waypoints) - 1)]

            if self.reached_waypoint(sensor_data, target, tol=self.align_tol) and self.align_wp_idx + 1 < len(self.align_waypoints):
                self.dbg(f"ALIGN wp reached gate={self.current_segment} wp={self.align_wp_idx}", "WP")
                self.align_wp_idx += 1
                target = self.align_waypoints[self.align_wp_idx]

            target = self.refine_align_target_with_detection(target, sensor_data)
            return self.command_to_waypoint(target)

        # ------------------------------------------------------------
        # TRAVERSE
        # ------------------------------------------------------------
        if self.state == "TRAVERSE":
            self.dbg_gate_progress(sensor_data, label=f"TRAVERSE wp={self.traverse_wp_idx}")

            target = self.traverse_waypoints[self.traverse_wp_idx]

            # Visual commit: lock on only when the gate is BOTH close AND visually
            # centered.  Without the cx/cy checks, a still-rotating yaw at commit time
            # makes the drone fly straight in the wrong direction (misses the gate).
            if (
                self.traverse_wp_idx == 1
                and self.current_segment not in self.traverse_committed
                and self.last_detection is not None
                and self.last_detection.get("segment") == self.current_segment
                and self.last_detection["bbox"]["area"] >= self.traverse_commit_area
                and not self.last_detection["bbox"].get("near_edge", False)
                and abs(self.last_detection["cx_error"]) <= self.commit_max_cx_err
                and abs(self.last_detection["cy_error"]) <= self.commit_max_cy_err
            ):
                self.traverse_committed.add(self.current_segment)
                print(
                    f"[t={self.tick:04d}] [COMMIT] gate={self.current_segment} "
                    f"bbox_area={self.last_detection['bbox']['area']:.0f} "
                    f"cx_err={self.last_detection['cx_error']:+.0f} "
                    f"cy_err={self.last_detection['cy_error']:+.0f} - flying straight through"
                )

            # If committed, override target: project ~0.7 m forward along the gate
            # tangent direction (not sensor yaw, which may still be rotating).
            # Keep small lateral visual nudges active so cx_err drift can still
            # correct trajectory in the final approach.
            # Advance to wp2 once we cross the (mapped) gate plane.
            if self.traverse_wp_idx == 1 and self.current_segment in self.traverse_committed:
                pos = self._pos(sensor_data)
                # Use the gate-center waypoint yaw (= gate tangent direction).
                gate_yaw = float(self.traverse_waypoints[1][3])
                forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw)])
                commit_xy = pos[:2] + 0.70 * forward
                # Z trim from cy_error
                commit_z = pos[2]
                if (
                    self.last_detection is not None
                    and self.last_detection.get("segment") == self.current_segment
                ):
                    cy_err = float(self.last_detection["cy_error"])
                    commit_z = pos[2] + float(np.clip(-0.0022 * cy_err, -0.05, 0.05))
                    # Small lateral nudge from cx_error during commit (half the regular gain).
                    cx_err = float(self.last_detection["cx_error"])
                    right_world = np.array([np.sin(gate_yaw), -np.cos(gate_yaw)])
                    commit_lat_nudge = float(
                        np.clip(0.5 * self.visual_nudge_gain * cx_err, -0.05, 0.05)
                    )
                    commit_xy[0] += commit_lat_nudge * right_world[0]
                    commit_xy[1] += commit_lat_nudge * right_world[1]
                if self.current_segment in self.mapped_gates:
                    gate_z = float(self.mapped_gates[self.current_segment][2])
                    commit_z = float(np.clip(commit_z, max(gate_z - 0.05, 0.80), 1.90))
                target = np.array([commit_xy[0], commit_xy[1], commit_z, gate_yaw])

                # Advance to wp2 when we've crossed the mapped gate plane.
                if self.current_segment in self.mapped_gates:
                    gate = self.mapped_gates[self.current_segment]
                    tangent = self.get_tangent(gate)
                    rel = pos[:2] - gate[:2]
                    proj = rel[0] * tangent[0] + rel[1] * tangent[1]
                    if proj > 0.10:
                        if self.current_segment == 5 and not self.gate5_crossed_plane:
                            self.gate5_crossed_plane = True
                            print(
                                f"[t={self.tick:04d}] [GATE5_GUARD] disabled: "
                                f"gate 5 plane crossed, allowing exit waypoints"
                            )
                        print(
                            f"[t={self.tick:04d}] [POST_COMMIT] gate={self.current_segment} "
                            f"crossed plane proj={proj:.2f}, advancing to post-cross"
                        )
                        self.traverse_wp_idx += 1
                        target = self.traverse_waypoints[self.traverse_wp_idx]
            elif self.traverse_wp_idx <= 1:
                # Not committed: refine via the existing geometric+visual nudge function.
                target = self.refine_traverse_target_with_detection(target, sensor_data)

            # Call pass check for debug only; we never finish early from it.
            if self.current_segment in self.mapped_gates:
                _ = self.has_passed_current_gate(sensor_data)

            # Use tighter tolerance for the gate-center waypoint (wp idx 1).
            tol = self.traverse_gate_tol if self.traverse_wp_idx == 1 else self.traverse_tol

            if self.reached_waypoint(sensor_data, target, tol=tol):
                if self.traverse_wp_idx == 1:
                    print(
                        f"[t={self.tick:04d}] [CROSSING] Gate {self.current_segment} center reached. "
                        f"Continuing to post-cross waypoint."
                    )
                elif self.traverse_wp_idx == 2:
                    print(
                        f"[t={self.tick:04d}] [POST_CROSS] Gate {self.current_segment} near-exit reached. "
                        f"Continuing to far exit."
                    )
                else:
                    print(
                        f"[t={self.tick:04d}] [WP] TRAVERSE wp reached gate={self.current_segment} "
                        f"wp={self.traverse_wp_idx}"
                    )

                self.traverse_wp_idx += 1

                if self.traverse_wp_idx >= len(self.traverse_waypoints):
                    print(
                        f"[t={self.tick:04d}] [GATE_SAFE] Gate {self.current_segment} fully traversed. "
                        f"All waypoints done."
                    )
                    return self.finish_gate(sensor_data)

                target = self.traverse_waypoints[self.traverse_wp_idx]
                if self.current_segment == 5 and self.gate5_crossed_plane:
                    print(
                        f"[t={self.tick:04d}] [GATE5_GUARD] allowing gate 5 exit waypoint "
                        f"old=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})"
                    )

            return self.command_to_waypoint(target)

        # ------------------------------------------------------------
        # RETURN HOME
        # ------------------------------------------------------------
        if self.state == "RETURN_HOME":
            control_command = [self.home[0], self.home[1], self.home[2], 0.0]

            if self.reached_xyz(sensor_data, self.home, tol=0.30):
                if len(self.mapped_gates) == 5:
                    print(
                        f"[t={self.tick:04d}] [LAP] Finished lap {self.lap_count}. "
                        f"mapped={sorted(self.mapped_gates.keys())}"
                    )

                    if self.lap_count >= self.safe_total_laps:
                        self.set_state("HOVER", "all safe laps complete")
                        return self.apply_command_filter(
                            [self.home[0], self.home[1], self.home[2], sensor_data["yaw"]],
                            dt,
                        )

                    self.lap_count += 1
                    self.current_segment = 1

                    if self.use_fast_laps:
                        print(f"[t={self.tick:04d}] [LAP] Starting FAST lap {self.lap_count}, conservative mode.")
                        self.start_fast_laps(reset_laps=True)
                        return self.command_to_waypoint(self.fast_waypoints[0])

                    print(f"[t={self.tick:04d}] [LAP] Starting SAFE lap {self.lap_count}.")
                    self.start_search(1)
                    return self.command_to_waypoint(self.search_waypoints[0])

                self.set_state("HOVER", f"only mapped={sorted(self.mapped_gates.keys())}")

            return self.apply_command_filter(control_command, dt)

        # ------------------------------------------------------------
        # FAST MODE KEPT, BUT NOT USED BY DEFAULT
        # ------------------------------------------------------------
        if self.state == "FAST":
            target = self.fast_waypoints[self.fast_wp_idx]

            seg_guess = min(self.fast_wp_idx // 3 + 1, 5)
            phase = self.fast_wp_idx % 3

            if seg_guess in self.mapped_gates:
                gate = self.mapped_gates[seg_guess]

                if phase <= 1 and self.last_detection is not None:
                    if self.last_detection.get("segment") == seg_guess:
                        bbox = self.last_detection["bbox"]
                        if bbox["area"] >= self.align_bbox_min_area:
                            cy_error = self.last_detection["cy_error"]
                            dz = -0.0020 * cy_error
                            desired_z = target[2] + float(np.clip(dz, -0.06, 0.06))
                            current_z = sensor_data["z_global"]
                            target = np.asarray(target, dtype=float).copy()
                            target[2] = float(np.clip(current_z + np.clip(desired_z - current_z, -0.10, 0.10), 0.80, 1.90))

                if phase >= 1:
                    tangent = self.get_tangent(gate)
                    normal = np.array([-tangent[1], tangent[0]], dtype=float)
                    pos_fast = self._pos(sensor_data)
                    rel = pos_fast[:2] - gate[:2]
                    proj = rel[0] * tangent[0] + rel[1] * tangent[1]
                    lateral = rel[0] * normal[0] + rel[1] * normal[1]
                    dist = np.linalg.norm(rel)
                    z_err = abs(pos_fast[2] - gate[2])

                    passed = (
                        proj > self.pass_proj_threshold
                        and abs(lateral) < self.pass_lateral_tol
                        and dist < self.pass_dist_tol
                        and z_err < self.pass_z_tol
                    )

                    if self.DEBUG_PASS and self.tick % self.DEBUG_PASS_EVERY == 0:
                        print(
                            f"[t={self.tick:04d}] [FAST_PASS_CHECK] gate={seg_guess} passed={passed} "
                            f"proj={proj:.2f} lat={lateral:.2f} dist={dist:.2f} zerr={z_err:.2f}"
                        )

                    if passed and phase == 1:
                        print(f"[t={self.tick:04d}] [FAST_PASS] pass detected for gate {seg_guess}")
                        self.fast_wp_idx += 1

                        if self.fast_wp_idx >= len(self.fast_waypoints):
                            self.fast_laps_left -= 1

                            if self.fast_laps_left > 0:
                                self.start_fast_laps(reset_laps=False)
                            else:
                                self.set_state("HOVER", "fast laps complete")
                                return self.apply_command_filter([self.home[0], self.home[1], self.home[2], sensor_data["yaw"]], dt)

                        target = self.fast_waypoints[self.fast_wp_idx]

            if self.reached_waypoint(sensor_data, target, tol=0.22):
                self.fast_wp_idx += 1

                if self.fast_wp_idx >= len(self.fast_waypoints):
                    self.fast_laps_left -= 1

                    if self.fast_laps_left > 0:
                        self.start_fast_laps(reset_laps=False)
                    else:
                        self.set_state("HOVER", "fast laps complete")
                        return self.apply_command_filter([self.home[0], self.home[1], self.home[2], sensor_data["yaw"]], dt)

                target = self.fast_waypoints[self.fast_wp_idx]

            return self.command_to_waypoint(target)

        # ------------------------------------------------------------
        # DEFAULT HOLD
        # ------------------------------------------------------------
        control_command = [pos[0], pos[1], pos[2], sensor_data["yaw"]]
        return self.apply_command_filter(control_command, dt)

    # ============================================================
    # VISION
    # ============================================================
    def update_vision(self, sensor_data, camera_data):
        det = self.detect_gate(camera_data)
        self.last_detection = None

        if det is None:
            if self.state in {"SEARCH", "INSPECT_GATE", "ALIGN"} and self.current_segment in [1, 2, 3, 4, 5]:
                self.frames_since_current_gate_seen += 1
                self.consecutive_next_gate_seen = 0
            self.reject_debug("no pink gate contour detected")
            return

        world_gate = self.estimate_gate_world(det, sensor_data)
        if world_gate is None:
            self.reject_debug("world estimate failed", det=det)
            return

        raw_seg = self.classify_segment(world_gate[0], world_gate[1])
        seg = self.classify_segment_soft(world_gate[0], world_gate[1])
        if seg not in [1, 2, 3, 4, 5]:
            if self.state == "FIRST_GATE_SCAN":
                # During initial yaw scan the drone is far from all gates and direction
                # matters. Never force-assign a gap/unclassifiable detection to gate 1 —
                # that is exactly how (2.76,5.14) was wrongly accepted as Gate 1.
                return
            if self.current_segment in [1, 2, 3, 4, 5]:
                if self._gate_sector_check(self.current_segment, world_gate, tol_deg=28):
                    self.reject_debug(
                        f"strong current-sector fallback: raw_seg={seg} -> current_segment={self.current_segment}",
                        det=det,
                        world_gate=world_gate,
                        seg=seg,
                    )
                    seg = self.current_segment
                else:
                    print(
                        f"[t={self.tick:04d}] [SECTOR_REJECT] expected={self.current_segment} "
                        f"detected={seg} candidate=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f})"
                    )
                    return
            else:
                self.reject_debug("invalid segment", det=det, world_gate=world_gate, seg=seg)
                return

        radius = np.linalg.norm(world_gate[:2] - self.center)
        if radius < 1.2 or radius > 3.9:
            self.reject_debug("radius rejected", det=det, world_gate=world_gate, seg=seg)
            return

        # During FIRST_GATE_SCAN, only collect gate 1 candidates.
        # Do not update trackers or confirm any other gate.
        if self.state == "FIRST_GATE_SCAN":
            if seg == 1 and self._gate_sector_check(1, world_gate):
                self.collect_first_gate_candidate(sensor_data, det, world_gate, seg)
            elif seg != 1:
                print(
                    f"[t={self.tick:04d}] [SEQUENCE_REJECT] expected=1 detected={seg} "
                    f"state={self.state}"
                )
            else:
                print(
                    f"[t={self.tick:04d}] [SECTOR_REJECT] expected=1 detected={raw_seg} "
                    f"candidate=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f})"
                )
            return

        # During first-gate forward/rescan, never let a random detection confirm
        # gate 1 unless it passes the same hard sector guard as the yaw scan.
        if self.state == "FIRST_GATE_FORWARD" and self.current_segment == 1:
            if seg != 1 or not self._gate_sector_check(1, world_gate):
                return

        # ------------------------------------------------------------
        # SEQUENCE GUARD: only allow mapping the current gate k.
        # Gate k+1 is a recovery signal, never permission to skip k.
        # ------------------------------------------------------------
        if not self._vision_segment_allowed(seg):
            if self.state in {"SEARCH", "INSPECT_GATE", "ALIGN"} and seg == self.current_segment + 1:
                self.next_gate_seen_count += 1
                self.consecutive_next_gate_seen += 1
                self.frames_since_current_gate_seen += 1
                print(
                    f"[t={self.tick:04d}] [SEQUENCE_REJECT] expected={self.current_segment} "
                    f"detected={seg} state={self.state} next_gate_seen={self.next_gate_seen_count} "
                    f"consecutive_next={self.consecutive_next_gate_seen}"
                )
            elif self.state in {"SEARCH", "INSPECT_GATE", "ALIGN"}:
                self.wrong_gate_seen_count += 1
                self.consecutive_next_gate_seen = 0
                self.frames_since_current_gate_seen += 1
                print(
                    f"[t={self.tick:04d}] [SEQUENCE_REJECT] expected={self.current_segment} "
                    f"detected={seg} state={self.state} wrong_gate_seen={self.wrong_gate_seen_count}"
                )
            return

        if not self._gate_sector_check(self.current_segment, world_gate):
            if self.state in {"SEARCH", "INSPECT_GATE", "ALIGN"}:
                self.consecutive_next_gate_seen = 0
                self.frames_since_current_gate_seen += 1
            print(
                f"[t={self.tick:04d}] [SECTOR_REJECT] expected={self.current_segment} detected={raw_seg} "
                f"candidate=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f})"
            )
            return

        centered = abs(det["cx"] - self.cx) < self.align_px_tol_x and abs(det["cy"] - self.cy) < self.align_px_tol_y
        noisy = det["near_edge"] or det["area"] < 150
        if self.state in {"SEARCH", "INSPECT_GATE", "ALIGN"} and seg == self.current_segment:
            self.frames_since_current_gate_seen = 0
            self.consecutive_next_gate_seen = 0
            if len(getattr(self, "inspect_samples", [])) > 0:
                self.next_gate_seen_count = max(0, self.next_gate_seen_count - 2)

        self.last_detection = {
            "segment": seg,
            "centered": centered,
            "bbox": det,
            "world": world_gate,
            "cy_error": det["cy"] - self.cy,
            "cx_error": det["cx"] - self.cx,
        }

        if self.DEBUG_VISION and self.tick % self.DEBUG_EVERY == 0:
            self.dbg(
                f"detected seg={seg} current={self.current_segment} "
                f"area={det['area']:.0f} centered={centered} noisy={noisy} "
                f"world=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f}) r={radius:.2f}",
                "VISION",
            )

        # ------------------------------------------------------------
        # INSPECTED GATES: after active inspection, freeze the model strongly.
        # Allow only tiny smoothing while inspecting the same gate; otherwise
        # do not let close/partial detections drift the robust first-lap model.
        # ------------------------------------------------------------
        if seg in self.gate_models and self.state != "INSPECT_GATE":
            old_gate = self.mapped_gates.get(seg, self.gate_models[seg]["center"])
            jump = np.linalg.norm(world_gate[:2] - old_gate[:2])
            z_jump = abs(world_gate[2] - old_gate[2])
            if jump < 0.25 and z_jump < 0.12 and not noisy:
                self.mapped_gates[seg] = 0.98 * old_gate + 0.02 * world_gate
            return

        # ------------------------------------------------------------
        # IMPORTANT SAFETY GUARD
        # ------------------------------------------------------------
        # Once a gate is mapped, do NOT let random later detections overwrite it
        # unless the new estimate is close to the already mapped position.
        # In your logs, gate 1 was selected near (1.69, 1.59), then overwritten
        # around (4.3, 5.5). That destroys the expected CCW order.
        if seg in self.mapped_gates:
            old_gate = self.mapped_gates[seg]
            jump = np.linalg.norm(world_gate[:2] - old_gate[:2])
            z_jump = abs(world_gate[2] - old_gate[2])

            if jump > 0.85 or z_jump > 0.45:
                if self.tick % self.DEBUG_PASS_EVERY == 0:
                    print(
                        f"[t={self.tick:04d}] [MAP_REJECT] gate={seg} ignored overwrite "
                        f"jump={jump:.2f} z_jump={z_jump:.2f} "
                        f"old=({old_gate[0]:.2f},{old_gate[1]:.2f},{old_gate[2]:.2f}) "
                        f"new=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f})"
                    )
                return

        self.trackers[seg].add(world_gate, centered=centered, noisy=noisy)

        if self.trackers[seg].confirmed is not None:
            was_mapped = seg in self.mapped_gates

            if was_mapped:
                # Blend only when: not traversing (noisy close-up estimates) AND
                # gate not yet in traversed_gates (position locked after first successful pass).
                if self.state != "TRAVERSE" and seg not in self.traversed_gates:
                    old_gate = self.mapped_gates[seg]
                    new_gate = self.trackers[seg].confirmed.copy()
                    self.mapped_gates[seg] = 0.92 * old_gate + 0.08 * new_gate
            else:
                # Do not accept a new rough map if it is not in the expected sector.
                # This catches rare false confirmations caused by seeing the next gate
                # while the internal state is still waiting for the current one.
                if not self._gate_sector_check(seg, self.trackers[seg].confirmed, tol_deg=58):
                    g_bad = self.trackers[seg].confirmed
                    print(
                        f"[t={self.tick:04d}] [MAP_REJECT] gate={seg} rough confirmation outside sector "
                        f"candidate=({g_bad[0]:.2f},{g_bad[1]:.2f},{g_bad[2]:.2f})"
                    )
                    self.reset_gate_tracker(seg)
                    return
                self.mapped_gates[seg] = self.trackers[seg].confirmed.copy()
                # Snapshot the very first confirmed position, before any ALIGN drift.
                self.gate_first_confirmed_pos[seg] = self.mapped_gates[seg].copy()
                g = self.mapped_gates[seg]
                print(
                    f"[t={self.tick:04d}] [GATE] Gate {seg} CONFIRMED at "
                    f"({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) score={self.trackers[seg].score:.2f}"
                )

    def _vision_segment_allowed(self, seg):
        if self.current_segment > 5:
            return False
        k = self.current_segment
        return seg == k

    def expected_gate_sector_score(self, seg, world_gate):
        if seg < 1 or seg > 5:
            return -1.0

        strict = self.classify_segment(world_gate[0], world_gate[1])
        if strict != -1 and strict != seg:
            return -1.0

        rel = np.array(world_gate[:2], dtype=float) - self.center
        if np.linalg.norm(rel) < 1e-6:
            return -1.0

        ang_shifted = np.arctan2(rel[1], rel[0]) + np.pi
        a0, a1 = self.angular_bounds[seg]
        center = 0.5 * (a0 + a1)
        diff = abs(self.wrap_angle(ang_shifted - center))
        base_tol = 58.0 if seg in {1, 5} else 46.0
        return 1.0 - diff / np.deg2rad(base_tol)

    def _gate_sector_check(self, seg, world_gate, tol_deg=58):
        """True if world_gate is in (or near) the angular sector for gate seg (1–5).

        Uses the same shifted-angle coordinate as classify_segment.
        angular_bounds[seg] is the strict sector for gate seg (seg 1→bounds[1], etc.).
        For positions that land in a gap between sectors (strict returns -1), a generous
        angular proximity check is applied so edge cases near the sector boundary still pass.
        Positions that strictly classify as a *different* gate are always rejected.
        """
        if seg < 1 or seg > 5:
            return False
        strict = self.classify_segment(world_gate[0], world_gate[1])
        if strict == seg:
            return True
        if strict != -1:
            # Clearly classified as a different segment — wrong sector.
            return False
        # Position is in a gap: verify angular proximity to the expected sector center.
        rel = np.array(world_gate[:2], dtype=float) - self.center
        ang_shifted = np.arctan2(rel[1], rel[0]) + np.pi  # 0–2π, matches angular_bounds
        a0, a1 = self.angular_bounds[seg]  # seg 1-5 → bounds[1-5]
        center = 0.5 * (a0 + a1)
        diff = abs(self.wrap_angle(ang_shifted - center))
        if seg in {1, 5}:
            tol_deg += 12
        return diff < np.deg2rad(tol_deg)

    def detect_gate(self, image_bgra):
        bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([135, 35, 45], dtype=np.uint8)
        upper = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        if self.state in {"FIRST_GATE_SCAN", "SEARCH"}:
            kernel = np.ones((3, 3), np.uint8)
            min_area = 75
            edge_penalty = 0.92
            border_margin = 6
            near_edge_margin = 10
            ratio_low, ratio_high = 0.18, 4.2
        else:
            kernel = np.ones((5, 5), np.uint8)
            min_area = 110
            edge_penalty = 0.70
            border_margin = 12
            near_edge_margin = 18
            ratio_low, ratio_high = 0.25, 3.2

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            ratio = w / float(h)
            if not (ratio_low <= ratio <= ratio_high):
                continue

            score = float(area)

            if x < border_margin or y < border_margin or x + w > self.img_w - border_margin or y + h > self.img_h - border_margin:
                score *= edge_penalty

            if self.state in {"FIRST_GATE_SCAN", "SEARCH"}:
                score *= 1.0 + 0.15 * min(w / 40.0, 1.0)

            if score > best_score:
                best_score = score
                best = (x, y, w, h, area, near_edge_margin)

        if best is None:
            return None

        x, y, w, h, area, near_edge_margin = best

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + 0.5 * w,
            "cy": y + 0.5 * h,
            "area": area,
            "near_edge": x < near_edge_margin
            or y < near_edge_margin
            or x + w > self.img_w - near_edge_margin
            or y + h > self.img_h - near_edge_margin,
        }

    def estimate_gate_world(self, det, sensor_data):
        depth = self.fy * 0.40 / max(det["h"], 1.0)

        if self.state in {"FIRST_GATE_SCAN", "SEARCH"}:
            depth = float(np.clip(depth, 0.6, 5.5))
        else:
            depth = float(np.clip(depth, 0.6, 5.0))

        x_cam = (det["cx"] - self.cx) * depth / self.fx
        y_cam = (det["cy"] - self.cy) * depth / self.fy

        t_body = np.array([depth, -x_cam, -y_cam], dtype=float)
        rot = R.from_euler("xyz", [sensor_data["roll"], sensor_data["pitch"], sensor_data["yaw"]]).as_matrix()
        drone_world = np.array([sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]], dtype=float)

        gate_world = drone_world + rot @ t_body

        rel = gate_world[:2] - self.center
        r = np.linalg.norm(rel)

        if r < 1e-6:
            return None

        if self.state in {"FIRST_GATE_SCAN", "SEARCH"}:
            r_clip = np.clip(r, 1.4, 3.6)
        else:
            r_clip = np.clip(r, 1.5, 3.5)

        gate_world[:2] = self.center + rel / r * r_clip
        gate_world[2] = np.clip(gate_world[2], 0.90, 1.85)

        return gate_world

    # ============================================================
    # FIRST GATE SCAN
    # ============================================================
    def start_first_gate_scan(self, sensor_data):
        self.set_state("FIRST_GATE_SCAN", f"attempt={self.first_gate_scan_attempt}")
        self.first_gate_scan_ticks = 0
        self.first_gate_scan_base_yaw = float(sensor_data["yaw"])
        self.first_gate_scan_pos = np.array(
            [float(sensor_data["x_global"]), float(sensor_data["y_global"]), float(sensor_data["z_global"])],
            dtype=float,
        )
        self.last_detection = None
        self.first_gate_candidates = []

        print(
            f"[t={self.tick:04d}] [GATE] Start first gate yaw scan at "
            f"pos=({self.first_gate_scan_pos[0]:.2f},{self.first_gate_scan_pos[1]:.2f},{self.first_gate_scan_pos[2]:.2f}) "
            f"yaw={self.first_gate_scan_base_yaw:.2f}"
        )

    def select_best_first_gate_candidate(self):
        if len(self.first_gate_candidates) < self.first_gate_candidate_min_count:
            self.dbg(
                f"Gate 1 candidate selection failed: only {len(self.first_gate_candidates)}/{self.first_gate_candidate_min_count} candidates",
                "MISS",
            )
            return None

        candidates = [np.asarray(c, dtype=float) for c in self.first_gate_candidates]
        best_group = []

        for c in candidates:
            group = [p for p in candidates if np.linalg.norm(p - c) < self.first_gate_candidate_consistency_tol]
            if len(group) > len(best_group):
                best_group = group

        if len(best_group) < self.first_gate_candidate_min_count:
            self.dbg(
                f"Gate 1 candidate selection failed: best consistent group {len(best_group)}/{self.first_gate_candidate_min_count}",
                "MISS",
            )
            return None

        group_arr = np.asarray(best_group)
        mean_pos = np.mean(group_arr, axis=0)

        # Reject group if mean z is suspiciously close to the lower clamp (0.90)
        if mean_pos[2] < 0.96:
            self.dbg(
                f"Gate 1 candidate group rejected: mean_z={mean_pos[2]:.2f} too close to lower clamp",
                "MISS",
            )
            return None

        if not self._gate_sector_check(1, mean_pos):
            print(
                f"[t={self.tick:04d}] [SECTOR_REJECT] expected=1 detected={self.classify_segment(mean_pos[0], mean_pos[1])} "
                f"candidate=({mean_pos[0]:.2f},{mean_pos[1]:.2f},{mean_pos[2]:.2f})"
            )
            return None

        return mean_pos

    def first_gate_initial_scan(self, sensor_data):
        if self.first_gate_scan_ticks >= self.first_gate_scan_max_ticks:
            return None

        self.first_gate_scan_ticks += 1
        progress = self.first_gate_scan_ticks / float(self.first_gate_scan_max_ticks)
        yaw_cmd = self.wrap_angle(self.first_gate_scan_base_yaw + progress * self.first_gate_scan_total_angle)

        return [
            float(self.first_gate_scan_pos[0]),
            float(self.first_gate_scan_pos[1]),
            float(self.first_gate_scan_pos[2]),
            float(yaw_cmd),
        ]

    def collect_first_gate_candidate(self, sensor_data, det, world_gate, seg):
        if seg != 1:
            return

        # Hard angular-sector guard: candidate must be in or near Gate 1's sector.
        # Catches cases where the soft classifier or fallback assigned seg=1 to a
        # detection that is geometrically in a different gate's direction.
        # Example rejection: (2.76,5.14) → strict_seg=-1, gap at 317° ≠ Gate1 sector ~60°.
        if not self._gate_sector_check(1, world_gate):
            strict = self.classify_segment(world_gate[0], world_gate[1])
            print(
                f"[t={self.tick:04d}] [REGION_REJECT] current_gate=1 "
                f"candidate=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f}) "
                f"reason=wrong_sector strict_seg={strict}"
            )
            return

        drone_pos = self._pos(sensor_data)
        radius = np.linalg.norm(world_gate[:2] - self.center)
        drone_dist = np.linalg.norm(world_gate[:2] - drone_pos[:2])
        cx_err_abs = abs(det["cx"] - self.cx)

        plausible = (
            det["area"] >= self.first_gate_candidate_min_area
            and not det["near_edge"]
            and self.first_gate_candidate_min_radius <= radius <= self.first_gate_candidate_max_radius
            and 0.60 <= drone_dist <= self.first_gate_candidate_max_drone_dist
            and cx_err_abs < 125
            and 0.75 <= world_gate[2] <= 1.85
            and det["h"] >= 18
            and det["w"] >= 12
        )

        if plausible:
            self.first_gate_candidates.append(world_gate.copy())
            if len(self.first_gate_candidates) in [1, 3, 6, 10]:
                print(
                    f"[t={self.tick:04d}] [REGION_ACCEPT] current_gate=1 "
                    f"n={len(self.first_gate_candidates)} "
                    f"candidate=({world_gate[0]:.2f},{world_gate[1]:.2f},{world_gate[2]:.2f}) "
                    f"area={det['area']:.0f} r={radius:.2f} dist={drone_dist:.2f}"
                )
        else:
            reasons = []
            if det["area"] < self.first_gate_candidate_min_area:
                reasons.append(f"area {det['area']:.0f}<min")
            if det["near_edge"]:
                reasons.append("near_edge")
            if not (self.first_gate_candidate_min_radius <= radius <= self.first_gate_candidate_max_radius):
                reasons.append(f"radius {radius:.2f}")
            if not (0.60 <= drone_dist <= self.first_gate_candidate_max_drone_dist):
                reasons.append(f"drone_dist {drone_dist:.2f}")
            if cx_err_abs >= 125:
                reasons.append(f"cx_err {cx_err_abs:.1f}")
            if not (0.75 <= world_gate[2] <= 1.85):
                reasons.append(f"z {world_gate[2]:.2f}")
            if det["h"] < 18:
                reasons.append(f"h {det['h']}")
            if det["w"] < 12:
                reasons.append(f"w {det['w']}")
            self.reject_debug("gate1 candidate rejected: " + ", ".join(reasons), det=det, world_gate=world_gate, seg=seg)

    def start_first_gate_forward(self, sensor_data):
        self.set_state("FIRST_GATE_FORWARD")

        pos = self._pos(sensor_data)
        yaw = self.first_gate_scan_base_yaw
        if yaw is None:
            yaw = sensor_data["yaw"]

        x = pos[0] + self.first_gate_forward_distance * np.cos(yaw)
        y = pos[1] + self.first_gate_forward_distance * np.sin(yaw)
        z = pos[2]

        self.first_gate_forward_waypoint = np.array([x, y, z, yaw], dtype=float)
        print(f"[t={self.tick:04d}] [GATE] Move forward before rescan to ({x:.2f},{y:.2f},{z:.2f})")

    # ============================================================
    # SEARCH VISUAL ACQUISITION
    # ============================================================
    def search_visual_acquisition_command(self, sensor_data):
        if self.last_detection is None:
            return None

        if self.last_detection.get("segment") != self.current_segment:
            return None

        det = self.last_detection["bbox"]
        world = self.last_detection["world"]
        pos = self._pos(sensor_data)

        excentered = abs(self.last_detection["cx_error"]) > self.search_visual_px_trigger or det["near_edge"]
        if not excentered:
            return None

        yaw_to_gate = np.arctan2(world[1] - pos[1], world[0] - pos[0])
        cmd_yaw = self.wrap_angle(sensor_data["yaw"] + self.search_visual_yaw_blend * self.wrap_angle(yaw_to_gate - sensor_data["yaw"]))

        dz = -self.search_visual_z_gain * self.last_detection["cy_error"]
        dz = float(np.clip(dz, -self.search_visual_z_clip, self.search_visual_z_clip))
        cmd_z = float(np.clip(pos[2] + dz, 0.85, 1.85))

        target = self.search_waypoints[self.search_wp_idx]

        blend_gate = 0.22
        blend_search = 0.12

        gate_xy = np.array([world[0], world[1]], dtype=float)
        cmd_xy = (1.0 - blend_gate - blend_search) * pos[:2] + blend_gate * gate_xy + blend_search * target[:2]

        return [float(cmd_xy[0]), float(cmd_xy[1]), cmd_z, cmd_yaw]

    # ============================================================
    # ACTIVE FIRST-LAP GATE INSPECTION
    # ============================================================
    def get_gate_target_z(self, segment):
        gate = self.mapped_gates[segment]
        return float(np.clip(gate[2] + self.height_margin, 0.80, 1.82))

    def start_inspect_gate(self, sensor_data, segment):
        self.current_segment = segment
        self.inspect_segment = segment
        self.inspect_ticks = 0
        self.inspect_samples = []
        self.frames_since_current_gate_seen = 0
        self.consecutive_next_gate_seen = 0
        self.next_gate_seen_count = 0
        self.inspect_stage = "GOTO"
        self.inspect_staging_wp = self.get_inspection_staging_waypoint(segment, sensor_data)
        if segment == 5:
            self.inspect_staging_wp = self.make_safe_gate5_waypoint(sensor_data, self.inspect_staging_wp)
        self.set_state("INSPECT_GATE", f"lap={self.lap_count} gate={segment}")
        g = self.mapped_gates[segment]
        wp = self.inspect_staging_wp
        print(
            f"[t={self.tick:04d}] [INSPECT_START] gate={segment} "
            f"rough=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) "
            f"staging=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) yaw={wp[3]:.2f}"
        )

    def get_inspection_staging_waypoint(self, segment, sensor_data):
        gate = self.mapped_gates[segment]
        tangent = self.get_tangent(gate)
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        z = self.get_gate_target_z(segment)
        staging_xy = gate[:2] - self.inspect_distance * tangent
        return np.array([staging_xy[0], staging_xy[1], z, yaw], dtype=float)

    def run_inspect_gate(self, sensor_data):
        seg = self.inspect_segment if self.inspect_segment is not None else self.current_segment
        self.inspect_ticks += 1

        if seg not in self.mapped_gates:
            self.start_search(seg)
            return self.command_to_waypoint(self.search_waypoints[0])

        # Recompute staging because the rough gate can still be refined before finalization.
        self.inspect_staging_wp = self.get_inspection_staging_waypoint(seg, sensor_data)
        staging = self.inspect_staging_wp.copy()
        pos = self._pos(sensor_data)
        target_z = staging[2]
        target_yaw = staging[3]
        yaw_err = abs(float(np.degrees(self.wrap_angle(target_yaw - sensor_data["yaw"]))))
        z_err = abs(float(pos[2] - target_z))

        # Vertical and yaw stabilization first: avoid diagonal height changes into the gate.
        if z_err > self.pre_gate_z_tol:
            if self.inspect_ticks % 25 == 0:
                print(
                    f"[t={self.tick:04d}] [HEIGHT_ALIGN] gate={seg} "
                    f"current_z={pos[2]:.2f} target_z={target_z:.2f} z_err={pos[2]-target_z:+.2f}"
                )
            return self.command_to_waypoint([pos[0], pos[1], target_z, target_yaw])

        if yaw_err > self.inspect_yaw_max_deg:
            if self.inspect_ticks % 25 == 0:
                print(
                    f"[t={self.tick:04d}] [INSPECT_YAW] gate={seg} "
                    f"yaw_err={yaw_err:.1f}>{self.inspect_yaw_max_deg:.1f}"
                )
            return self.command_to_waypoint([pos[0], pos[1], target_z, target_yaw])

        # Move to the pre-gate staging point.
        if not self.reached_waypoint(sensor_data, staging, tol=0.22):
            return self.command_to_waypoint(staging)

        # At staging: use visual errors to hover more squarely in front of the gate.
        cmd = staging.copy()
        if self.last_detection is not None and self.last_detection.get("segment") == seg:
            det = self.last_detection["bbox"]
            tangent = self.get_tangent(self.mapped_gates[seg])
            lateral_axis = np.array([-tangent[1], tangent[0]], dtype=float)
            cx_err = float(self.last_detection["cx_error"])
            cy_err = float(self.last_detection["cy_error"])
            lat_corr = float(np.clip(self.inspect_lateral_gain * cx_err,
                                     -self.inspect_max_xy_correction,
                                     self.inspect_max_xy_correction))
            z_corr = float(np.clip(-self.inspect_vertical_gain * cy_err,
                                   -self.inspect_max_z_correction,
                                   self.inspect_max_z_correction))
            cmd[0] += lat_corr * lateral_axis[0]
            cmd[1] += lat_corr * lateral_axis[1]
            cmd[2] = float(np.clip(target_z + z_corr, 0.80, 1.90))

            if self._inspection_detection_is_clean(sensor_data):
                self.collect_inspect_sample(sensor_data)
                self.frames_since_current_gate_seen = 0
                self.consecutive_next_gate_seen = 0
                self.next_gate_seen_count = max(0, self.next_gate_seen_count - 3)

        if self.inspect_ticks % 25 == 0:
            print(
                f"[t={self.tick:04d}] [INSPECT] gate={seg} "
                f"samples={len(self.inspect_samples)}/{self.inspect_min_samples} "
                f"stage={self.inspect_stage} yaw_err={yaw_err:.1f} zerr={z_err:.2f}"
            )

        if self.inspect_ready_to_finalize():
            self.finalize_inspect_gate(seg, timeout=False)
            self.start_align(sensor_data, seg)
            return self.command_to_waypoint(self.align_waypoints[0])

        if self.inspect_ticks > self.inspect_max_ticks:
            if len(self.inspect_samples) >= max(3, self.inspect_min_samples // 2):
                self.finalize_inspect_gate(seg, timeout=True)
                self.start_align(sensor_data, seg)
                return self.command_to_waypoint(self.align_waypoints[0])

            trusted = self.gate_models.get(seg, None)
            has_trusted_model = (
                trusted is not None
                and trusted.get("samples", 0) >= max(4, self.inspect_min_samples // 2)
                and trusted.get("confidence", 0.0) >= 0.45
                and self._gate_sector_check(seg, np.asarray(trusted["center"], dtype=float))
            )
            print(
                f"[t={self.tick:04d}] [INSPECT_TIMEOUT] gate={seg} "
                f"only {len(self.inspect_samples)} samples; "
                f"{'using previous trusted model' if has_trusted_model else 'clearing rough map and recovering'}"
            )

            if has_trusted_model:
                self.mapped_gates[seg] = np.asarray(trusted["center"], dtype=float).copy()
                self.start_align(sensor_data, seg)
                return self.command_to_waypoint(self.align_waypoints[0])

            # No clean samples and no trusted model: the rough map is unsafe.
            # Clear it and backtrack/widen the viewpoint instead of traversing a
            # wrong gate and desynchronizing the official validator.
            self.clear_gate_mapping(seg, reset_tracker=True)
            self.start_search(seg)
            return self.start_or_continue_search_recovery(sensor_data, force=True)

        return self.command_to_waypoint(cmd)

    def _inspection_detection_is_clean(self, sensor_data):
        if self.last_detection is None:
            return False
        seg = self.inspect_segment
        if self.last_detection.get("segment") != seg:
            return False
        bbox = self.last_detection["bbox"]
        if bbox.get("near_edge", False):
            return False
        if bbox.get("area", 0) < self.inspect_bbox_min_area:
            return False
        if abs(float(self.last_detection["cx_error"])) > self.inspect_px_tol_x:
            return False
        if abs(float(self.last_detection["cy_error"])) > self.inspect_px_tol_y:
            return False
        world = self.last_detection.get("world")
        if world is None:
            return False
        if not self._gate_sector_check(seg, world):
            return False
        dist = float(np.linalg.norm(world[:2] - self._pos(sensor_data)[:2]))
        return 0.75 <= dist <= 2.30

    def collect_inspect_sample(self, sensor_data):
        seg = self.inspect_segment
        world = np.asarray(self.last_detection["world"], dtype=float).copy()
        pos = self._pos(sensor_data)
        los = world[:2] - pos[:2]
        n = np.linalg.norm(los)
        if n < 1e-6:
            return
        normal = los / n
        fallback = self.get_tangent(world)
        if np.dot(normal, fallback) < 0:
            normal = -normal
        self.inspect_samples.append({"pos": world, "normal": normal})
        if len(self.inspect_samples) in {1, 3, 5, 8, 12, 16}:
            print(
                f"[t={self.tick:04d}] [INSPECT_SAMPLE] gate={seg} n={len(self.inspect_samples)} "
                f"world=({world[0]:.2f},{world[1]:.2f},{world[2]:.2f}) "
                f"normal=({normal[0]:+.2f},{normal[1]:+.2f})"
            )

    def inspect_ready_to_finalize(self):
        if len(self.inspect_samples) < self.inspect_min_samples:
            return False
        pts = np.asarray([s["pos"] for s in self.inspect_samples[-self.inspect_min_samples:]], dtype=float)
        std_xy = float(np.mean(np.std(pts[:, :2], axis=0)))
        std_z = float(np.std(pts[:, 2]))
        return std_xy <= self.inspect_stable_std_xy and std_z <= self.inspect_stable_std_z

    def finalize_inspect_gate(self, segment, timeout=False):
        pts = np.asarray([s["pos"] for s in self.inspect_samples], dtype=float)
        normals = np.asarray([s["normal"] for s in self.inspect_samples], dtype=float)
        center = np.median(pts, axis=0)
        center[2] = float(np.clip(np.median(pts[:, 2]), 0.80, 1.85))
        normal = np.mean(normals, axis=0)
        nn = float(np.linalg.norm(normal))
        if nn < 1e-6:
            normal = self.get_tangent(center)
        else:
            normal = normal / nn
        fallback = self.get_tangent(center)
        if np.dot(normal, fallback) < 0:
            normal = -normal

        std_xy = float(np.mean(np.std(pts[:, :2], axis=0)))
        std_z = float(np.std(pts[:, 2]))
        confidence = float(np.clip(0.45 + 0.04 * len(self.inspect_samples)
                                   - 0.8 * std_xy - 0.8 * std_z, 0.35, 0.99))
        if timeout:
            confidence = min(confidence, 0.75)

        self.mapped_gates[segment] = center.copy()
        self.gate_first_confirmed_pos[segment] = center.copy()
        self.gate_normals[segment] = normal.copy()
        self.gate_normal_confidence[segment] = confidence
        self.gate_models[segment] = {
            "center": center.copy(),
            "z": float(center[2]),
            "normal": normal.copy(),
            "confidence": confidence,
            "samples": len(self.inspect_samples),
            "std_xy": std_xy,
            "std_z": std_z,
        }
        if segment in self.trackers:
            self.trackers[segment].pos = center.copy()
            self.trackers[segment].confirmed = center.copy()
            self.trackers[segment].score = max(self.trackers[segment].score,
                                               self.trackers[segment].confirm_threshold)
        tag = "INSPECT_TIMEOUT_MODEL" if timeout else "INSPECT_MODEL"
        print(
            f"[t={self.tick:04d}] [{tag}] gate={segment} "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
            f"normal=({normal[0]:+.2f},{normal[1]:+.2f}) "
            f"samples={len(self.inspect_samples)} std_xy={std_xy:.2f} std_z={std_z:.2f} conf={confidence:.2f}"
        )


    # ============================================================
    # RECOVERY HELPERS
    # ============================================================
    def should_inspect_gate(self, segment):
        """Return True when the gate should be actively inspected before traverse."""
        if segment not in self.mapped_gates:
            return False
        model = self.gate_models.get(segment)
        if self.lap_count == 1 and model is None:
            return True
        # Gate 5 is the most fragile one in the logs: low altitude, close to
        # home/return path, and easy to miss when traversed only geometrically.
        # Reinspect it on every lap unless we already have a strong model from
        # the current run.
        if segment == 5:
            return model is None or model.get("samples", 0) < self.inspect_min_samples
        return False

    def reset_gate_tracker(self, seg):
        if seg in self.trackers:
            self.trackers[seg] = GateMemory(
                confirm_threshold=4.0 if seg == 1 else 2.5,
                gate_id=seg,
                debug_fn=self.dbg_tracker,
            )

    def clear_gate_mapping(self, seg, reset_tracker=False):
        self.mapped_gates.pop(seg, None)
        self.gate_models.pop(seg, None)
        self.gate_normals.pop(seg, None)
        self.gate_normal_confidence.pop(seg, None)
        self.gate_first_confirmed_pos.pop(seg, None)
        self.traverse_gate_start_pos.pop(seg, None)
        if seg in self.traverse_committed:
            self.traverse_committed.remove(seg)
        if seg in self.traversed_gates:
            self.traversed_gates.remove(seg)
        if reset_tracker:
            self.reset_gate_tracker(seg)

    def _current_gate_geometry_plausible(self, sensor_data):
        seg = self.current_segment
        if seg not in self.mapped_gates:
            return False

        pos = self._pos(sensor_data)
        gate = np.asarray(self.mapped_gates[seg], dtype=float)
        dist_gate = float(np.linalg.norm(pos[:2] - gate[:2]))
        close_to_gate = dist_gate < 1.45

        close_to_staging = False
        if self.state == "INSPECT_GATE":
            try:
                staging = self.get_inspection_staging_waypoint(seg, sensor_data)
                close_to_staging = self.reached_waypoint(sensor_data, staging, tol=0.65)
            except Exception:
                close_to_staging = False
        elif self.state == "ALIGN" and len(self.align_waypoints) > 0:
            close_to_staging = self.reached_waypoint(sensor_data, self.align_waypoints[0], tol=0.70)

        return close_to_gate or close_to_staging

    def _next_gate_recovery_evidence(self, sensor_data):
        seg = self.current_segment
        samples = len(self.inspect_samples) if self.state == "INSPECT_GATE" else 0
        mapped = seg in self.mapped_gates
        plausible = self._current_gate_geometry_plausible(sensor_data)

        if samples > 0 or plausible or mapped:
            print(
                f"[t={self.tick:04d}] [RECOVERY_BLOCKED] current gate still plausible "
                f"gate={seg} mapped={mapped} samples={samples} plausible={plausible} "
                f"consecutive_next={self.consecutive_next_gate_seen} "
                f"frames_since_current={self.frames_since_current_gate_seen}"
            )
            return False

        threshold = max(14, self.next_gate_seen_recovery_threshold * 2)
        stale_timeout = 120 if self.state == "SEARCH" else max(90, self.inspect_max_ticks // 2)
        enough_next = self.consecutive_next_gate_seen >= threshold
        stale_current = self.frames_since_current_gate_seen >= stale_timeout
        timed_out = self.search_ticks > self.max_search_ticks_before_recovery

        if enough_next and stale_current:
            print(
                f"[t={self.tick:04d}] [RECOVERY_REASON] no samples, timeout, far from estimate, "
                f"next gate seen consistently | gate={seg} consecutive_next={self.consecutive_next_gate_seen} "
                f"frames_since_current={self.frames_since_current_gate_seen} threshold={threshold}"
            )
            return True

        if timed_out:
            print(
                f"[t={self.tick:04d}] [RECOVERY_REASON] no samples, timeout, far from estimate "
                f"| gate={seg} search_ticks={self.search_ticks}>{self.max_search_ticks_before_recovery}"
            )
            return True

        if self.next_gate_seen_count > 0 and self.tick % self.DEBUG_PASS_EVERY == 0:
            print(
                f"[t={self.tick:04d}] [RECOVERY_BLOCKED] waiting for stronger evidence "
                f"gate={seg} consecutive_next={self.consecutive_next_gate_seen}/{threshold} "
                f"frames_since_current={self.frames_since_current_gate_seen}/{stale_timeout}"
            )
        return False

    def _search_recovery_needed(self, sensor_data):
        if self.current_segment not in [1, 2, 3, 4, 5]:
            return False
        if self.current_segment in self.mapped_gates:
            return False
        if self.recovery_active:
            return True
        return self._next_gate_recovery_evidence(sensor_data)

    def gate5_guard_active(self):
        if self.current_segment != 5:
            return False
        if self.state in ["RETURN_HOME", "HOVER"] or 5 in self.traversed_gates:
            return False
        if self.state == "TRAVERSE":
            return False
        if self.post_cross_exit_active:
            return False
        if self.state in ["SEARCH", "INSPECT_GATE", "ALIGN", "FIRST_GATE_FORWARD"]:
            return True
        return False

    def is_unsafe_gate5_waypoint(self, wp):
        if not self.gate5_guard_active():
            return False

        arr = np.asarray(wp, dtype=float)
        xy = arr[:2]
        near_home = np.linalg.norm(xy - self.home[:2]) < 1.8
        start_corridor = arr[0] < 2.2 and 2.7 < arr[1] < 5.5
        segment0 = self.classify_segment(arr[0], arr[1]) == 0
        return bool(near_home or start_corridor or segment0)

    def make_safe_gate5_waypoint(self, sensor_data, wp):
        arr = np.asarray(wp, dtype=float).copy()
        had_yaw = arr.shape[0] >= 4
        if arr.shape[0] < 4:
            arr = np.array([arr[0], arr[1], arr[2], 0.0], dtype=float)

        if not self.is_unsafe_gate5_waypoint(arr):
            return arr if had_yaw else arr[:3]

        print("[GATE5_GUARD] gate 5 not done, refusing unsafe waypoint near spawn")
        old = arr.copy()
        if sensor_data is not None:
            current_z = float(sensor_data["z_global"])
        else:
            current_z = float(arr[2])

        gate = self.mapped_gates.get(5, None)
        candidates = []
        if gate is not None:
            gate = np.asarray(gate, dtype=float)
            away_home = gate[:2] - self.home[:2]
            n = np.linalg.norm(away_home)
            if n < 1e-6:
                away_home = np.array([0.0, 1.0], dtype=float)
            else:
                away_home = away_home / n
            tangent = self.get_tangent(gate)
            candidates.extend([
                gate[:2] + 1.05 * away_home,
                gate[:2] + 0.70 * away_home + 0.35 * np.array([-tangent[1], tangent[0]], dtype=float),
                gate[:2] + 0.70 * away_home - 0.35 * np.array([-tangent[1], tangent[0]], dtype=float),
            ])

        candidates.extend([
            np.array([3.6, 6.2], dtype=float),
            np.array([4.0, 6.2], dtype=float),
            np.array([3.4, 6.4], dtype=float),
            np.array([2.8, 5.8], dtype=float),
        ])

        best = None
        best_yaw = float(arr[3])
        for xy in candidates:
            z_ref = current_z
            if gate is not None:
                z_ref = max(z_ref, float(gate[2]), 1.50)
            else:
                z_ref = max(z_ref, 1.50)
            z = float(np.clip(z_ref, 1.20, 1.85))
            if gate is not None:
                yaw = float(np.arctan2(gate[1] - xy[1], gate[0] - xy[0]))
            else:
                yaw = float(np.arctan2(6.0 - xy[1], 3.6 - xy[0]))
            candidate = np.array([xy[0], xy[1], z, yaw], dtype=float)
            if not self.is_unsafe_gate5_waypoint(candidate):
                best = candidate
                best_yaw = yaw
                break

        if best is None:
            best = np.array([3.6, 6.2, float(np.clip(max(current_z, 1.50), 1.20, 1.85)), best_yaw], dtype=float)

        print(
            f"[t={self.tick:04d}] [GATE5_GUARD] blocked unsafe recovery waypoint "
            f"old=({old[0]:.2f},{old[1]:.2f},{old[2]:.2f}) "
            f"replaced=({best[0]:.2f},{best[1]:.2f},{best[2]:.2f})"
        )
        return best if had_yaw else best[:3]

    def get_recovery_gate_estimate(self, segment):
        if segment in self.mapped_gates:
            return np.asarray(self.mapped_gates[segment], dtype=float).copy()
        model = self.gate_models.get(segment)
        if model is not None and "center" in model:
            return np.asarray(model["center"], dtype=float).copy()
        tracker = self.trackers.get(segment)
        if tracker is not None:
            if tracker.confirmed is not None:
                return np.asarray(tracker.confirmed, dtype=float).copy()
            if tracker.pos is not None and tracker.score >= 1.5:
                return np.asarray(tracker.pos, dtype=float).copy()

        a = self.segment_center_angle(segment)
        radii = [np.linalg.norm(g[:2] - self.center) for g in self.mapped_gates.values()]
        if self.predicted_gate_radius is not None:
            r = float(self.predicted_gate_radius)
        elif len(radii) > 0:
            r = float(np.median(radii))
        else:
            r = 2.75
        r = float(np.clip(r, 2.0, 3.45))
        z = self.search_zs[self.recovery_attempts.get(segment, 0) % len(self.search_zs)]
        return np.array([
            self.center[0] - r * np.cos(a),
            self.center[1] - r * np.sin(a),
            z,
        ], dtype=float)

    def select_pre_crossing_recovery_waypoint(self, segment, sensor_data, base_wp):
        gate = self.get_recovery_gate_estimate(segment)
        if gate is None:
            return base_wp

        pos = self._pos(sensor_data)
        raw_normal = self.get_tangent(gate)
        if np.linalg.norm(raw_normal) < 1e-6:
            return base_wp
        raw_normal = raw_normal / np.linalg.norm(raw_normal)

        rel = gate[:2] - self.center
        expected_normal = np.array([-rel[1], rel[0]], dtype=float)
        if np.linalg.norm(expected_normal) < 1e-6:
            expected_normal = raw_normal.copy()
        else:
            expected_normal = expected_normal / np.linalg.norm(expected_normal)

        if np.dot(raw_normal, expected_normal) < 0:
            oriented_normal = -raw_normal
        else:
            oriented_normal = raw_normal.copy()

        current_proj = float(np.dot(pos[:2] - gate[:2], oriented_normal))
        attempt = self.recovery_attempts.get(segment, 0)
        recovery_distance = float(np.clip(1.15 + 0.25 * attempt, 1.10, 2.10))
        base = np.asarray(base_wp, dtype=float).copy()
        z = float(np.clip(base[2], 0.85, 1.85))
        if segment in self.mapped_gates:
            z = self.get_gate_target_z(segment)

        outward = gate[:2] - self.center
        if np.linalg.norm(outward) > 1e-6:
            outward = outward / np.linalg.norm(outward)
        else:
            outward = np.array([0.0, 1.0], dtype=float)

        candidates = []
        side_distance = recovery_distance + 0.30
        for normal in (raw_normal, -raw_normal, oriented_normal, -oriented_normal):
            for lateral in (0.0, 0.25, -0.25):
                side = np.array([-normal[1], normal[0]], dtype=float)
                xy = gate[:2] - side_distance * normal + lateral * side + 0.18 * outward
                yaw = float(np.arctan2(gate[1] - xy[1], gate[0] - xy[0]))
                candidates.append(np.array([xy[0], xy[1], z, yaw], dtype=float))

        def score(candidate):
            xy = candidate[:2]
            proj = float(np.dot(xy - gate[:2], oriented_normal))
            inside_penalty = 0.0 if (0.15 <= xy[0] <= 7.85 and 0.15 <= xy[1] <= 7.85) else 200.0
            pre_penalty = 0.0 if proj < -recovery_distance else 80.0 + 30.0 * max(0.0, proj + recovery_distance)
            home_dist = float(np.linalg.norm(xy - self.home[:2]))
            home_penalty = 45.0 if home_dist < 1.8 else 0.0
            seg0_penalty = 35.0 if self.classify_segment(xy[0], xy[1]) == 0 else 0.0
            base_penalty = 0.10 * float(np.linalg.norm(xy - base[:2]))
            return pre_penalty + inside_penalty + home_penalty + seg0_penalty + base_penalty + proj

        best = min(candidates, key=score)
        best_proj = float(np.dot(best[:2] - gate[:2], oriented_normal))
        print(
            f"[t={self.tick:04d}] [REVERSE_RECOVERY_SIDE_CHECK] gate={segment} "
            f"current_proj={current_proj:+.2f} target_proj={best_proj:+.2f} "
            f"dist={recovery_distance:.2f}"
        )
        print(
            f"[t={self.tick:04d}] [REVERSE_RECOVERY_PRE_SIDE_SELECTED] gate={segment} "
            f"wp=({best[0]:.2f},{best[1]:.2f},{best[2]:.2f})"
        )
        return best

    def gate_plane_metrics(self, segment, sensor_data):
        gate = self.get_recovery_gate_estimate(segment)
        if gate is None:
            return None

        normal = self.get_tangent(gate)
        n = np.linalg.norm(normal)
        if n < 1e-6:
            return None
        normal = normal / n

        if segment in self.gate_first_confirmed_pos:
            ref_gate = np.asarray(self.gate_first_confirmed_pos[segment], dtype=float)
            ref_normal = self.get_tangent(ref_gate)
            if np.linalg.norm(ref_normal) > 1e-6 and np.dot(normal, ref_normal) < 0:
                normal = -normal

        pos = self._pos(sensor_data)
        rel = pos[:2] - gate[:2]
        lateral_axis = np.array([-normal[1], normal[0]], dtype=float)
        proj = float(np.dot(rel, normal))
        lateral = abs(float(np.dot(rel, lateral_axis)))
        z_err = abs(float(pos[2] - gate[2]))
        return {
            "gate": gate,
            "normal": normal,
            "proj": proj,
            "lateral": lateral,
            "z_err": z_err,
            "pos": pos,
        }

    def should_break_recovery_post_cross_loop(self, segment, sensor_data):
        metrics = self.gate_plane_metrics(segment, sensor_data)
        if metrics is None:
            return False

        far_exit_distance = max(self.exit_d, 1.15)
        proj_limit = max(0.80, far_exit_distance * 0.70)
        post_side = metrics["proj"] > proj_limit
        aligned = metrics["lateral"] < 0.45 and metrics["z_err"] < 0.35

        if post_side:
            self.recovery_post_cross_hits[segment] = self.recovery_post_cross_hits.get(segment, 0) + 1
        else:
            self.recovery_post_cross_hits[segment] = max(0, self.recovery_post_cross_hits.get(segment, 0) - 1)

        had_candidate = (
            segment in self.valid_crossing_candidate_gates
            or segment in self.traverse_committed
            or self.recovery_attempts.get(segment, 0) >= self.max_recovery_attempts_per_gate
        )
        repeated = (
            self.recovery_attempts.get(segment, 0) >= self.max_recovery_attempts_per_gate
            and self.recovery_post_cross_hits.get(segment, 0) >= 2
        )

        if post_side and aligned and (had_candidate or repeated):
            print(
                f"[t={self.tick:04d}] [POST_CROSS_SUSPECTED] gate={segment} "
                f"proj={metrics['proj']:.2f}>{proj_limit:.2f} "
                f"lat={metrics['lateral']:.2f} zerr={metrics['z_err']:.2f} "
                f"attempts={self.recovery_attempts.get(segment, 0)} hits={self.recovery_post_cross_hits.get(segment, 0)}"
            )
            return True

        return False

    def start_forward_exit_retry(self, segment, sensor_data):
        metrics = self.gate_plane_metrics(segment, sensor_data)
        if metrics is None:
            return None

        gate = metrics["gate"]
        normal = metrics["normal"]
        exit_distance = max(1.65 * self.exit_d, 1.55)
        z = float(np.clip(gate[2] + self.height_margin, 0.80, 1.85))
        xy = gate[:2] + exit_distance * normal
        yaw = float(np.arctan2(normal[1], normal[0]))
        self.post_cross_exit_waypoint = np.array([xy[0], xy[1], z, yaw], dtype=float)
        self.post_cross_exit_active = True
        self.recovery_active = False
        print(
            f"[t={self.tick:04d}] [RECOVERY_LOOP_BREAK] gate={segment} "
            f"switching from reverse recovery to forward exit retry"
        )
        print(
            f"[t={self.tick:04d}] [FORWARD_EXIT_RETRY] gate={segment} "
            f"wp=({xy[0]:.2f},{xy[1]:.2f},{z:.2f}) proj={metrics['proj']:.2f}"
        )
        return self.command_to_waypoint(self.post_cross_exit_waypoint)

    def supervisor_recheck_gate_done(self, segment, sensor_data):
        supervisor_keys = ["segment", "current_segment", "official_segment", "lap", "gate_progress"]
        available = [k for k in supervisor_keys if isinstance(sensor_data, dict) and k in sensor_data]
        if not available:
            print(f"[t={self.tick:04d}] [SUPERVISOR_RECHECK] gate={segment} unavailable; using local geometry")
        else:
            print(f"[t={self.tick:04d}] [SUPERVISOR_RECHECK] gate={segment} keys={available}")
            official_segment = sensor_data.get("segment", sensor_data.get("current_segment", sensor_data.get("official_segment", None)))
            if official_segment is not None and int(official_segment) != segment:
                return True

        metrics = self.gate_plane_metrics(segment, sensor_data)
        if metrics is None:
            return False
        local_segment = self.classify_segment(metrics["pos"][0], metrics["pos"][1])
        far_exit_distance = max(self.exit_d, 1.15)
        return (
            metrics["proj"] > 1.15 * far_exit_distance
            and metrics["lateral"] < 0.50
            and metrics["z_err"] < 0.40
            and (local_segment == segment + 1 or (segment == 5 and local_segment == 0))
        )

    def build_recovery_waypoint(self, segment, sensor_data):
        """Build a wide/backward viewpoint for gate `segment`.

        The waypoint is deliberately farther outward and slightly before the
        expected gate sector. This gives the camera enough distance to separate
        gate k from gate k+1 after the rare case where the drone gets too close
        to the next segment and keeps seeing the wrong gate.
        """
        pos = self._pos(sensor_data)
        a = self.segment_center_angle(segment)
        attempt = self.recovery_attempts.get(segment, 0)
        # Alternate around the sector center and go farther/wider on retries.
        if attempt <= self.max_recovery_attempts_per_gate:
            angle_offsets = [0.0, -0.34, 0.34, -0.58, 0.58]
            radius_base = 2.95
            radius_step = 0.22
        else:
            angle_offsets = [0.0, -0.50, 0.50, -0.82, 0.82, -1.05, 1.05]
            radius_base = 3.35
            radius_step = 0.12
        da = angle_offsets[min(attempt, len(angle_offsets) - 1)]
        r = min(3.75, radius_base + radius_step * attempt)
        if segment == 4:
            r = min(3.82, r + 0.28)
            da *= 1.25
        if segment in {1, 5} and attempt >= 2:
            da *= 1.35
        ang = a + da
        x = self.center[0] - r * np.cos(ang)
        y = self.center[1] - r * np.sin(ang)
        z = float(np.clip(pos[2], 0.95, 1.55))
        if segment in self.mapped_gates:
            z = self.get_gate_target_z(segment)
        else:
            if segment == 5:
                z_levels = [0.90, 1.02, 1.18, 0.84, 1.34]
            elif segment == 1:
                z_levels = [1.00, 1.18, 0.92, 1.35, 0.86]
            else:
                z_levels = self.search_zs
            z = z_levels[attempt % len(z_levels)]
        yaw = self.wrap_angle(ang - np.pi / 2.0)
        base_wp = np.array([x, y, z, yaw], dtype=float)
        return self.select_pre_crossing_recovery_waypoint(segment, sensor_data, base_wp)


    def start_or_continue_search_recovery(self, sensor_data, force=False):
        seg = self.current_segment
        if self.post_cross_exit_active and self.post_cross_exit_waypoint is not None:
            if self.reached_waypoint(sensor_data, self.post_cross_exit_waypoint, tol=0.30):
                if self.supervisor_recheck_gate_done(seg, sensor_data):
                    print(f"[t={self.tick:04d}] [SUPERVISOR_RECHECK] gate={seg} advanced; finishing gate")
                    self.post_cross_exit_active = False
                    self.post_cross_exit_waypoint = None
                    self.valid_crossing_candidate_gates.add(seg)
                    return self.finish_gate(sensor_data)

                print(
                    f"[t={self.tick:04d}] [BACKTRACK_RESUME] gate={seg} "
                    f"forward exit not confirmed; returning to pre-cross recovery"
                )
                self.post_cross_exit_active = False
                self.post_cross_exit_waypoint = None
                self.recovery_active = False
                self.recovery_waypoint = None
                self.recovery_post_cross_hits[seg] = 0
                self.suppress_post_cross_break_once.add(seg)
                return self.start_or_continue_search_recovery(sensor_data, force=True)

            print(
                f"[t={self.tick:04d}] [FORWARD_EXIT_RETRY] gate={seg} continuing "
                f"wp=({self.post_cross_exit_waypoint[0]:.2f},{self.post_cross_exit_waypoint[1]:.2f},{self.post_cross_exit_waypoint[2]:.2f})"
            )
            return self.command_to_waypoint(self.post_cross_exit_waypoint)

        if seg in self.suppress_post_cross_break_once:
            self.suppress_post_cross_break_once.discard(seg)
        else:
            if self.should_break_recovery_post_cross_loop(seg, sensor_data):
                retry_cmd = self.start_forward_exit_retry(seg, sensor_data)
                if retry_cmd is not None:
                    return retry_cmd

        if force or not self.recovery_active or self.recovery_waypoint is None:
            self.recovery_active = True
            self.recovery_attempts[seg] = self.recovery_attempts.get(seg, 0) + 1
            self.recovery_waypoint = self.build_recovery_waypoint(seg, sensor_data)
            if seg == 5:
                self.recovery_waypoint = self.make_safe_gate5_waypoint(sensor_data, self.recovery_waypoint)
            wp = self.recovery_waypoint
            print(
                f"[t={self.tick:04d}] [RECOVERY_WP] gate={seg} "
                f"attempt={self.recovery_attempts[seg]} next_gate_seen={self.next_gate_seen_count} "
                f"search_ticks={self.search_ticks} wp=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})"
            )

        if self.reached_waypoint(sensor_data, self.recovery_waypoint, tol=0.28):
            # Reset local evidence and perform a fresh search sweep from the wider pose.
            self.recovery_active = False
            self.next_gate_seen_count = 0
            self.consecutive_next_gate_seen = 0
            self.wrong_gate_seen_count = 0
            self.search_ticks = 0
            self.frames_since_current_gate_seen = 0
            self.last_detection = None
            self.recovery_waypoint = None
            self.post_cross_exit_active = False
            self.post_cross_exit_waypoint = None
            self.suppress_post_cross_break_once.discard(seg)
            self.search_waypoints = self.build_search_waypoints(seg)
            if seg == 5:
                self.search_waypoints = [self.make_safe_gate5_waypoint(sensor_data, wp) for wp in self.search_waypoints]
            self.search_wp_idx = 0
            print(f"[t={self.tick:04d}] [RECOVERY_DONE] gate={seg} restarting search sweep")
            return self.command_to_waypoint(self.search_waypoints[0])

        return self.command_to_waypoint(self.recovery_waypoint)

    # ============================================================
    # STATE TRANSITIONS
    # ============================================================
    def start_search(self, segment):
        self.set_state("SEARCH", f"lap={self.lap_count} gate={segment}")
        self.current_segment = segment
        if segment == 5:
            self.gate5_crossed_plane = False
            print(f"[t={self.tick:04d}] [GATE5_GUARD] active before gate 5 traversal")
        self.search_waypoints = self.build_search_waypoints(segment)
        if segment == 5:
            self.search_waypoints = [self.make_safe_gate5_waypoint(None, wp) for wp in self.search_waypoints]
        self.search_wp_idx = 0
        self.align_center_count = 0
        self.align_ticks = 0
        self.search_ticks = 0
        self.next_gate_seen_count = 0
        self.consecutive_next_gate_seen = 0
        self.wrong_gate_seen_count = 0
        self.frames_since_current_gate_seen = 0
        self.recovery_active = False
        self.recovery_waypoint = None
        self.post_cross_exit_active = False
        self.post_cross_exit_waypoint = None
        self.suppress_post_cross_break_once.discard(segment)

        print(
            f"[t={self.tick:04d}] [SEARCH] lap={self.lap_count} gate={segment} "
            f"waypoints={len(self.search_waypoints)}"
        )

    def start_align(self, sensor_data, segment, preserve_counter=False):
        if segment == 5 and not preserve_counter:
            self.gate5_crossed_plane = False
        gate = self.mapped_gates[segment]
        tangent = self.get_tangent(gate)
        yaw = np.arctan2(tangent[1], tangent[0])
        z = self.get_gate_target_z(segment)
        pos = self._pos(sensor_data)

        waypoints = []
        dz_to_gate = z - pos[2]
        if abs(dz_to_gate) > self.align_dz_lift_trigger:
            yaw_lift = self.wrap_angle(sensor_data["yaw"] + 0.5 * self.wrap_angle(yaw - sensor_data["yaw"]))
            waypoints.append(np.array([pos[0], pos[1], z, yaw_lift], dtype=float))

        approach_far = np.array([
            gate[0] - self.approach_d * tangent[0],
            gate[1] - self.approach_d * tangent[1],
            z,
            yaw,
        ])
        approach_near = np.array([
            gate[0] - 0.55 * self.approach_d * tangent[0],
            gate[1] - 0.55 * self.approach_d * tangent[1],
            z,
            yaw,
        ])
        waypoints.extend([approach_far, approach_near])

        self.align_waypoints = waypoints
        self.align_wp_idx = 0
        self.set_state("ALIGN", f"lap={self.lap_count} gate={segment}")

        if not preserve_counter:
            self.align_center_count = 0
            self.align_ticks = 0
            self.align_z_ok_count = 0
            print(
                f"[t={self.tick:04d}] [ALIGN_START] lap={self.lap_count} gate={segment} "
                f"gate_pos=({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}) yaw={yaw:.2f} "
                f"target_z={z:.2f} dz_to_gate={dz_to_gate:+.2f} wps={len(waypoints)}"
            )

    def start_traverse(self, segment):
        if segment == 5:
            self.gate5_crossed_plane = False
            print(f"[t={self.tick:04d}] [GATE5_GUARD] disabled during TRAVERSE")
        # First traversal: use the initial confirmed position (before ALIGN blending
        # drift).  ALIGN-phase updates can shift XY by 10-20 cm, enough to miss the
        # official gate center.  Subsequent laps use the frozen locked position.
        if segment not in self.traversed_gates and segment in self.gate_first_confirmed_pos:
            gate = self.gate_first_confirmed_pos[segment].copy()
        else:
            gate = self.mapped_gates[segment].copy()
        self.traverse_gate_start_pos[segment] = gate.copy()
        # Reset commit state for this traversal (so a fresh visual lock-on can fire)
        self.traverse_committed.discard(segment)
        tangent = self.get_tangent(gate)
        yaw = np.arctan2(tangent[1], tangent[0])
        # Use gate[2] directly (no height_margin uplift) so low gates aren't
        # pushed above their physical frame.  Lower clamp 0.88 allows flying at
        # the confirmed gate z even when it's near the arena floor.
        # Traverse at the inspected/mapped gate center height.
        z = float(np.clip(gate[2] + self.height_margin, 0.80, 1.82))

        self.traverse_waypoints = [
            np.array([
                gate[0] - 0.70 * self.approach_d * tangent[0],
                gate[1] - 0.70 * self.approach_d * tangent[1],
                z,
                yaw,
            ]),
            # Gate center: drone must pass through here with tight tolerance.
            np.array([gate[0], gate[1], z, yaw]),
            # Near-exit: keeps drone on gate axis ~0.63 m after center before curving away.
            np.array([
                gate[0] + 0.55 * self.exit_d * tangent[0],
                gate[1] + 0.55 * self.exit_d * tangent[1],
                z,
                yaw,
            ]),
            # Far exit: full clearance before next-gate search.
            np.array([
                gate[0] + 1.75 * self.exit_d * tangent[0],
                gate[1] + 1.75 * self.exit_d * tangent[1],
                z,
                yaw,
            ]),
        ]

        self.traverse_wp_idx = 0
        self.set_state("TRAVERSE", f"lap={self.lap_count} gate={segment}")

        print(
            f"[t={self.tick:04d}] [TRAVERSE_START] lap={self.lap_count} gate={segment} "
            f"gate=({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}) yaw={yaw:.2f}"
        )

    def finish_gate(self, sensor_data):
        finished = self.current_segment

        if self.current_segment == 1:
            self.first_gate_scan_done = True

        # Revert to the pre-traverse position snapshot and permanently freeze it.
        # This discards any tracker drift accumulated during close-approach/traverse,
        # ensuring future laps use the same axis the official pass was registered on.
        if finished in self.traverse_gate_start_pos:
            locked = self.traverse_gate_start_pos[finished]
            self.mapped_gates[finished] = locked.copy()
            print(
                f"[t={self.tick:04d}] [GATE_LOCK] gate={finished} "
                f"locked=({locked[0]:.2f},{locked[1]:.2f},{locked[2]:.2f})"
            )
        self.traversed_gates.add(finished)
        self.valid_crossing_candidate_gates.discard(finished)
        self.post_cross_exit_active = False
        self.post_cross_exit_waypoint = None
        self.suppress_post_cross_break_once.discard(finished)
        if finished == 5:
            self.gate5_crossed_plane = True
            print(f"[t={self.tick:04d}] [GATE5_GUARD] guard disabled: gate 5 completed / RETURN_HOME")

        self.current_segment += 1
        self.align_center_count = 0
        self.align_ticks = 0

        print(
            f"[t={self.tick:04d}] [GATE_DONE] lap={self.lap_count} finished_gate={finished} "
            f"next_gate={self.current_segment}"
        )

        if self.current_segment > 5:
            self.set_state("RETURN_HOME", f"lap={self.lap_count} all gates traversed")
            return self.apply_command_filter([self.home[0], self.home[1], self.home[2], sensor_data["yaw"]], self._last_dt)

        self.start_search(self.current_segment)
        return self.command_to_waypoint(self.search_waypoints[0])

    def start_fast_laps(self, reset_laps=True):
        if reset_laps:
            self.fast_laps_left = 1

        wps = []

        for seg in range(1, 6):
            gate = self.mapped_gates[seg]
            tangent = self.get_tangent(gate)
            yaw = np.arctan2(tangent[1], tangent[0])
            z = float(np.clip(gate[2] + self.height_margin, 0.95, 1.82))

            local_wps = [
                np.array([
                    gate[0] - 0.70 * self.approach_d * tangent[0],
                    gate[1] - 0.70 * self.approach_d * tangent[1],
                    z,
                    yaw,
                ]),
                np.array([gate[0], gate[1], z, yaw]),
                np.array([
                    gate[0] + 1.60 * self.exit_d * tangent[0],
                    gate[1] + 1.60 * self.exit_d * tangent[1],
                    z,
                    yaw,
                ]),
            ]

            wps.extend(local_wps)

        wps.append(np.array([self.home[0], self.home[1], self.home[2], 0.0]))

        self.fast_waypoints = wps
        self.fast_wp_idx = 0
        self.set_state("FAST", f"laps_left={self.fast_laps_left}")

    # ============================================================
    # GEOMETRY
    # ============================================================
    def build_search_waypoints(self, segment):
        a = self.segment_center_angle(segment)
        if segment == 5:
            offsets = np.deg2rad(np.array([-52.0, -34.0, -18.0, 0.0, 18.0, 38.0, 58.0]))
            z_cycle = [0.86, 0.96, 1.08, 1.20, 0.90, 1.34, 0.82]
        elif segment == 1:
            offsets = np.deg2rad(np.array([-50.0, -32.0, -16.0, 0.0, 16.0, 34.0, 54.0]))
            z_cycle = [0.92, 1.04, 1.18, 1.32, 0.88, 1.40, 1.00]
        else:
            offsets = np.deg2rad(np.array([-34.0, -18.0, 0.0, 18.0, 34.0]))
            z_cycle = self.search_zs

        if self.recovery_attempts.get(segment, 0) > self.max_recovery_attempts_per_gate:
            offsets = np.deg2rad(np.array([-70.0, -48.0, -28.0, -10.0, 10.0, 30.0, 52.0, 72.0]))
            z_cycle = [0.84, 0.96, 1.10, 1.25, 1.42, 0.90, 1.32, 1.02]

        wps = []

        for k, da in enumerate(offsets):
            r = self.search_radii[k % len(self.search_radii)]
            if self.recovery_attempts.get(segment, 0) > self.max_recovery_attempts_per_gate:
                r = min(3.75, r + 0.45)
            z = z_cycle[k % len(z_cycle)]
            ang = a + da

            x = self.center[0] - r * np.cos(ang)
            y = self.center[1] - r * np.sin(ang)
            yaw = self.wrap_angle(ang - np.pi / 2.0)

            wps.append(np.array([x, y, z, yaw], dtype=float))

        return wps

    def segment_center_angle(self, segment_idx):
        # Gates are numbered 1..5 and main.py places gate i in angular_bounds[i]
        # because angular_bounds[0] is the start/home segment. The old -1 shift
        # made search waypoints look in the previous sector, causing rare loops
        # where gate k+1 was seen while searching gate k.
        idx = int(segment_idx)
        idx = int(np.clip(idx, 0, len(self.angular_bounds) - 1))

        a0, a1 = self.angular_bounds[idx]

        if idx == 0 and a0 > a1:
            a1 += 2 * np.pi

        return 0.5 * (a0 + a1)

    def classify_segment(self, x, y):
        rel = np.array([x, y], dtype=float) - self.center
        n = np.linalg.norm(rel)

        if n < 1e-9:
            return -1

        rel /= n
        angle = np.arctan2(rel[1], rel[0]) + np.pi

        for i in range(self.num_segments):
            a0, a1 = self.angular_bounds[i]

            if i == 0:
                if angle >= a0 or angle <= a1:
                    return i
            else:
                if a0 <= angle <= a1:
                    return i

        return -1

    def classify_segment_soft(self, x, y):
        raw = self.classify_segment(x, y)

        if raw in [1, 2, 3, 4, 5]:
            return raw

        if self.current_segment in [1, 2, 3, 4, 5]:
            candidate = np.array([x, y, self.cruise_z], dtype=float)
            if self.expected_gate_sector_score(self.current_segment, candidate) > 0.38:
                return self.current_segment

        return raw

    def segment_world_angle(self, segment):
        a = self.segment_center_angle(segment)
        x = self.center[0] - np.cos(a)
        y = self.center[1] - np.sin(a)
        return np.arctan2(y - self.center[1], x - self.center[0])

    def get_tangent(self, gate):
        # If this gate has been actively inspected, use the stored traversal
        # normal as the actual pre->post direction.  The function name is kept
        # for compatibility with the existing controller.
        gate_arr = np.asarray(gate, dtype=float)
        for seg, model in self.gate_models.items():
            center = np.asarray(model.get("center", self.mapped_gates.get(seg, gate_arr)), dtype=float)
            if np.linalg.norm(center[:2] - gate_arr[:2]) < 0.35:
                normal = np.asarray(model.get("normal", self.gate_normals.get(seg, None)), dtype=float)
                if normal.shape == (2,) and np.linalg.norm(normal) > 1e-6:
                    return normal / np.linalg.norm(normal)
        for seg, normal in self.gate_normals.items():
            if seg in self.mapped_gates and np.linalg.norm(self.mapped_gates[seg][:2] - gate_arr[:2]) < 0.35:
                normal = np.asarray(normal, dtype=float)
                if np.linalg.norm(normal) > 1e-6:
                    return normal / np.linalg.norm(normal)

        rel = gate_arr[:2] - self.center
        tangent = np.array([-rel[1], rel[0]], dtype=float)

        n = np.linalg.norm(tangent)

        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=float)

        return tangent / n

    def has_passed_current_gate(self, sensor_data):
        gate = self.mapped_gates[self.current_segment]
        tangent = self.get_tangent(gate)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        pos = self._pos(sensor_data)
        rel = pos[:2] - gate[:2]

        proj = rel[0] * tangent[0] + rel[1] * tangent[1]
        lateral = rel[0] * normal[0] + rel[1] * normal[1]
        dist = np.linalg.norm(rel)
        z_err = abs(pos[2] - gate[2])

        passed = (
            proj > self.pass_proj_threshold
            and abs(lateral) < self.pass_lateral_tol
            and dist < self.pass_dist_tol
            and z_err < self.pass_z_tol
        )

        if self.DEBUG_PASS and self.tick % self.DEBUG_PASS_EVERY == 0:
            print(
                f"[t={self.tick:04d}] [PASS_CHECK] gate={self.current_segment} passed={passed} | "
                f"proj={proj:.2f}>{self.pass_proj_threshold:.2f}, "
                f"lat={abs(lateral):.2f}<{self.pass_lateral_tol:.2f}, "
                f"dist={dist:.2f}<{self.pass_dist_tol:.2f}, "
                f"zerr={z_err:.2f}<{self.pass_z_tol:.2f}"
            )

        if passed:
            self.valid_crossing_candidate_gates.add(self.current_segment)

        return passed

    # ============================================================
    # ALIGN / TRAVERSE REFINEMENT
    # ============================================================
    def refine_align_target_with_detection(self, target, sensor_data):
        refined = np.asarray(target, dtype=float).copy()

        if self.last_detection is None:
            return refined

        if self.last_detection.get("segment") != self.current_segment:
            return refined

        bbox = self.last_detection["bbox"]

        if bbox["area"] < self.align_bbox_min_area:
            return refined

        cy_error = self.last_detection["cy_error"]
        dz_from_bbox = -self.align_vertical_gain * cy_error
        dz_from_bbox = float(np.clip(
            dz_from_bbox,
            -self.align_visual_z_max_correction,
            self.align_visual_z_max_correction,
        ))

        mapped_z = self.get_gate_target_z(self.current_segment)
        desired_z = mapped_z + dz_from_bbox

        current_z = sensor_data["z_global"]
        blended_z = (1.0 - self.align_z_blend) * refined[2] + self.align_z_blend * desired_z
        blended_z = current_z + float(np.clip(blended_z - current_z, -0.18, 0.18))

        z_floor_align = max(float(self.mapped_gates[self.current_segment][2]) - 0.05, 0.80)
        refined[2] = float(np.clip(blended_z, z_floor_align, 1.90))

        return refined

    def refine_traverse_target_with_detection(self, target, sensor_data):
        refined = np.asarray(target, dtype=float).copy()

        if self.last_detection is None:
            return refined

        if self.last_detection.get("segment") != self.current_segment:
            return refined

        bbox = self.last_detection["bbox"]

        if bbox["area"] < self.align_bbox_min_area:
            return refined

        # Z refinement from vertical pixel error.
        # z_floor: don't let the servo push the drone more than 5 cm below the
        # mapped gate center.  For very low gates (z≈0.90) this prevents the drone
        # from diving below the physical gate frame and missing the official crossing.
        if self.current_segment in self.mapped_gates:
            z_floor = max(self.mapped_gates[self.current_segment][2] - 0.05, 0.80)
        else:
            z_floor = 0.80

        cy_error = self.last_detection["cy_error"]
        dz = -0.0022 * cy_error
        desired_z = refined[2] + float(np.clip(dz, -0.07, 0.07))
        current_z = sensor_data["z_global"]
        refined[2] = float(np.clip(current_z + np.clip(desired_z - current_z, -0.12, 0.12), z_floor, 1.90))

        # Lateral correction: combine geometric (mapped-axis) + visual (pixel-space).
        # Geometric handles drift along a correct axis; visual handles systematic
        # world-coordinate error that no geometric correction can fix.
        if not bbox.get("near_edge", False) and bbox["area"] >= 250:
            # 1. Geometric nudge — push back toward the SAME axis the waypoints use.
            #    CRITICAL: this must use traverse_gate_start_pos (the snapshot used to
            #    build wp1), NOT mapped_gates which has been drifted by ALIGN-phase
            #    blending.  If they disagree, the geometric nudge fights the waypoint.
            ref_gate = None
            if self.current_segment in self.traverse_gate_start_pos:
                ref_gate = self.traverse_gate_start_pos[self.current_segment]
            elif self.current_segment in self.mapped_gates:
                ref_gate = self.mapped_gates[self.current_segment]
            if ref_gate is not None:
                tangent = self.get_tangent(ref_gate)
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                pos = self._pos(sensor_data)
                rel = pos[:2] - ref_gate[:2]
                lateral_err = rel[0] * normal[0] + rel[1] * normal[1]
                # Slightly stronger gain since the reference axis is now consistent.
                geom_nudge = float(np.clip(-0.35 * lateral_err, -0.10, 0.10))
                refined[0] += geom_nudge * normal[0]
                refined[1] += geom_nudge * normal[1]

            # 2. Visual nudge — bypass world-coord systematic error using pixel cx_error.
            #    cx_error > 0 ⇒ gate visibly to the right of image center
            #    ⇒ move target in the body-right direction ⇒ drone approaches actual gate.
            #    Body-right in world frame (yaw rotation only): (sin(yaw), -cos(yaw)).
            cx_error_px = float(self.last_detection["cx_error"])
            yaw = float(sensor_data["yaw"])
            right_world_x = np.sin(yaw)
            right_world_y = -np.cos(yaw)
            visual_nudge = float(
                np.clip(self.visual_nudge_gain * cx_error_px, -self.visual_nudge_max, self.visual_nudge_max)
            )
            refined[0] += visual_nudge * right_world_x
            refined[1] += visual_nudge * right_world_y

        return refined

    # ============================================================
    # UTILS
    # ============================================================
    def _pos(self, sd):
        return np.array([sd["x_global"], sd["y_global"], sd["z_global"]], dtype=float)

    def reached_waypoint(self, sensor_data, waypoint, tol=0.22):
        return np.linalg.norm(self._pos(sensor_data) - np.asarray(waypoint[:3], dtype=float)) < tol

    def reached_xyz(self, sensor_data, xyz, tol=0.25):
        return np.linalg.norm(self._pos(sensor_data) - np.asarray(xyz[:3], dtype=float)) < tol

    def command_to_waypoint(self, wp):
        if self.gate5_guard_active() and self.is_unsafe_gate5_waypoint(wp):
            wp = self.make_safe_gate5_waypoint(None, wp)
        return self.apply_command_filter([float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])], self._last_dt)

    def apply_command_filter(self, cmd, dt):
        target = np.array(cmd, dtype=float)

        if self._last_cmd is None:
            if self.gate5_guard_active() and self.is_unsafe_gate5_waypoint(target):
                target = self.make_safe_gate5_waypoint(None, target)
            self._last_cmd = target.copy()
            return [float(v) for v in target]

        last = self._last_cmd.copy()

        dxy = target[:2] - last[:2]
        nxy = np.linalg.norm(dxy)

        if nxy > self.max_xy_step:
            dxy = dxy / nxy * self.max_xy_step

        limited_xy = last[:2] + dxy

        dz = float(np.clip(target[2] - last[2], -self.max_z_step, self.max_z_step))
        limited_z = last[2] + dz

        dyaw = self.wrap_angle(target[3] - last[3])
        dyaw = float(np.clip(dyaw, -self.max_yaw_step, self.max_yaw_step))
        limited_yaw = self.wrap_angle(last[3] + dyaw)

        limited = np.array([limited_xy[0], limited_xy[1], limited_z, limited_yaw], dtype=float)

        filtered_xy = last[:2] + self.alpha_xy * (limited[:2] - last[:2])
        filtered_z = last[2] + self.alpha_z * (limited[2] - last[2])
        filtered_yaw = self.wrap_angle(last[3] + self.alpha_yaw * self.wrap_angle(limited[3] - last[3]))

        filtered = np.array([filtered_xy[0], filtered_xy[1], filtered_z, filtered_yaw], dtype=float)
        if self.gate5_guard_active() and self.is_unsafe_gate5_waypoint(filtered):
            filtered = self.make_safe_gate5_waypoint(None, filtered)

        self._last_cmd = filtered.copy()

        return [float(v) for v in filtered]

    @staticmethod
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


class GateMemory:
    def __init__(self, alpha=0.22, confirm_threshold=5.0, match_distance=0.80, confirmed_alpha=0.12, gate_id=None, debug_fn=None):
        self.alpha = alpha
        self.confirm_threshold = confirm_threshold
        self.match_distance = match_distance
        self.confirmed_alpha = confirmed_alpha
        self.gate_id = gate_id
        self.debug_fn = debug_fn
        self.pos = None
        self.score = 0.0
        self.confirmed = None

    def add(self, pos, centered=False, noisy=False):
        pos = np.asarray(pos, dtype=float).reshape(3,)

        weight = 1.0

        if centered:
            weight += 0.8

        if noisy:
            weight *= 0.45

        if self.pos is None:
            self.pos = pos.copy()
            self.score = weight
        else:
            dist = np.linalg.norm(pos - self.pos)
            dz = abs(pos[2] - self.pos[2])

            if dist < self.match_distance and dz < 0.28:
                mixed = (1.0 - self.alpha) * self.pos + self.alpha * pos
                mixed[2] = 0.88 * self.pos[2] + 0.12 * pos[2]

                self.pos = mixed
                self.score += weight
            else:
                if self.debug_fn is not None:
                    self.debug_fn(
                        f"Gate {self.gate_id}: tracker reset because detection jumped "
                        f"dist={dist:.2f}, dz={dz:.2f}, old_score={self.score:.2f}"
                    )
                self.pos = pos.copy()
                self.score = weight

        if self.debug_fn is not None and int(self.score * 10) % 10 == 0:
            self.debug_fn(
                f"Gate {self.gate_id}: score={self.score:.2f}/{self.confirm_threshold:.2f} "
                f"pos=({self.pos[0]:.2f},{self.pos[1]:.2f},{self.pos[2]:.2f}) centered={centered} noisy={noisy}"
            )

        if self.score >= self.confirm_threshold and self.confirmed is None:
            self.confirmed = self.pos.copy()
            if self.debug_fn is not None:
                self.debug_fn(
                    f"Gate {self.gate_id}: CONFIRMED at "
                    f"({self.confirmed[0]:.2f},{self.confirmed[1]:.2f},{self.confirmed[2]:.2f})"
                )

        elif self.confirmed is not None:
            dconf = np.linalg.norm(self.pos - self.confirmed)

            if dconf < self.match_distance:
                new_conf = (1.0 - self.confirmed_alpha) * self.confirmed + self.confirmed_alpha * self.pos
                new_conf[2] = 0.90 * self.confirmed[2] + 0.10 * self.pos[2]

                self.confirmed = new_conf


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

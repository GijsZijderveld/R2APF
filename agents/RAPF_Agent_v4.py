import numpy as np
from planners.rapf_global_planner import RAPFGlobalPlanner

# Default parameters for V4 Agent logic
RAPF_V4_PARAMS = {
    # Sensing
    'SENSE_RANGE': 3.0,            # Perception radius
    'SENSE_USE_FOV': False,        # Toggle Field-of-View restriction
    'SENSE_FOV_DEG': 62.0,         # FOV width in degrees
    'SENSE_PAD': 0.05,             # Small buffer for discrete step sensing

    # Replanning
    'REPLAN_CORRIDOR_MARGIN': 0.2, # (unused here but kept)
    'GOAL_TOLERANCE': 0.5,         # Distance to consider goal reached

    # RAPF Constants (forwarded to planner)
    'RHO_L': 0.30,                 # Safety radius (agent body)
    'RHO_U': 1.50,                 # Repulsive influence range
}


class RAPF_Agent_v4:
    """
    RAPF v4 Agent: High-level controller responsible for sensing,
    path tracking, and reactive replanning.
    Strict path-following (Pure Pursuit removed).
    """
    def __init__(self, agent_id, position, goal=None, **params):
        self.id = agent_id
        self.q = np.array(position, dtype=float)
        self.goal_pos = np.array(goal, dtype=float) if goal is not None else None
        self.heading = 0.0  # Radians

        # Merge parameters
        self.p = {**RAPF_V4_PARAMS, **params}

        # Internal State
        self.done = False
        self.planned_path = []
        self.path_index = 0

        self.perceived_obstacles = []
        self._known_obstacle_keys = set()

        # Debug / metrics
        self.planning_effort = 0.0
        self.replan_count = 0
        self.plan_fail_count = 0
        self.last_plan_fail_step = None
        self.blocked_steps = 0

        # Virtual obstacles (latest + counters)
        self.virtual_obstacles = []
        self.last_virtual_obstacles = []

        # Executed path tracking for minima-escape (RAPF v3 style)
        self.executed_path_all = [self.q.copy()]
        self.executed_path_opt = [self.q.copy()]  # will be pruned based on movement
        # self._last_exec_store_q = self.q.copy()
        self._last_exec_store_q_all = self.q.copy()
        # self._last_store_q_opt = self.q.copy()

        #escape bookkeeping
        self._escape_remaining = 0
        self._escape_active = False

        # Agent-side backtracking (when planner fails / start-local-minima)
        self._agent_backtrack_active = False
        self._agent_backtrack_tries = 0
        self._agent_backtrack_last_start_step = None

        # Backtracking params (can be overridden via **params)
        self.p.setdefault("AGENT_BACKTRACK_POINTS", 3)          # how many OPT points to walk back
        self.p.setdefault("AGENT_BACKTRACK_MAX_TRIES", 50)       # cap before we give up on backtracking
        self.p.setdefault("AGENT_BACKTRACK_COOLDOWN", 2)        # ticks before allowing another backtrack
        self.p.setdefault("AGENT_BACKTRACK_WP_TOL", 0.15)        # should be >= waypoint reach radius (0.1)
        self.p.setdefault("AGENT_REPLAN_AFTER_ESCAPE", True)

        # # How far the agent must move before recording another executed point
        # self.EXEC_STORE_MIN_DIST = 0.05
        # # Cap for executed path length to avoid unbounded memory use
        # self.EXEC_PATH_MAX_LEN = 2000

        # Agent-side counters (approximate until planner exposes true counts)
        self.virtual_total_seen = 0          # sum of "active virtual obstacles" over replans
        self.virtual_max_active = 0          # max len(virtual_obstacles) observed
        self.virtual_total_created = 0       # sum of "total_virtual_obstacles_created" over replans

        # Initialize the global planner module
        self.planner = RAPFGlobalPlanner(**self.p)
        self.backtrack_failed = False  # Flag to indicate if backtracking has failed

        #debug stuff
        self.debug_status = "init"
    @property
    def position(self):
        return self.q

    @property
    def goal(self):
        return self.goal_pos

    def step(self, env, agents, dt, speed_limit, time_step, bounds=None):
        if self.done or self.goal_pos is None:
            return

        self.debug_status = "step"

        # --- Sense + validity ---
        _ = self._update_perception(env)
        blocked = self._check_path_validity()

        # If path consumed -> force replan
        if self.planned_path and self.path_index >= len(self.planned_path):
            self.debug_status = "path_consumed"
            self.planned_path = []

        # --- Replan if needed ---
        # If we are currently executing an escape/backtrack, do not call planner.
        if not getattr(self, "_escape_active", False):
            needs_replan = blocked
            if needs_replan or not self.planned_path:
                self.debug_status = "replanning"
                ok = self._replan(time_step=time_step)
                if not ok:
                    self.debug_status = "replan_failed"
                    # Planner failed: let the agent backtrack on its OPT spine and try again.

        moved = False

        # --- Move along path if available ---
        if isinstance(self.planned_path, list) and len(self.planned_path) > 0 and self.path_index < len(self.planned_path):
            target_pos = self.planned_path[self.path_index]
            d = float(np.linalg.norm(target_pos - self.q))
            self.debug_status = f"moving to wp {self.path_index} (d={d:.2f})"
            moved = self._move_towards(target_pos, speed_limit, dt)

            # Waypoint reached?
            if moved and np.linalg.norm(self.q - target_pos) < 0.1:
                reached_wp = np.array(target_pos, dtype=float).copy()

                # Capture escape state BEFORE decrementing remaining
                escape_active_before = bool(getattr(self, "_escape_active", False))

                # Advance along the plan
                self.path_index += 1

                # Ensure containers exist
                if not hasattr(self, "executed_path_opt") or self.executed_path_opt is None:
                    self.executed_path_opt = []
                if not hasattr(self, "executed_path_all") or self.executed_path_all is None:
                    self.executed_path_all = [self.q.copy()]

                # ---- Update OPT spine ----
                if escape_active_before:
                    # We are backtracking: remove progress from opt spine.
                    if self.executed_path_opt:
                        # Must be consistent with waypoint reach radius (~0.1)
                        tol = float(self.p.get("AGENT_BACKTRACK_WP_TOL", 0.15))

                        dists = [np.linalg.norm(p - reached_wp) for p in self.executed_path_opt]
                        i_min = int(np.argmin(dists))

                        if dists[i_min] <= tol:
                            # We have returned near an earlier OPT point; everything after it is undone.
                            # Keep points strictly BEFORE i_min (so we don't keep the undone prefix point twice)
                            self.executed_path_opt = self.executed_path_opt[:i_min +1]

                    # Escape bookkeeping: consume escape-prefix waypoints
                    if escape_active_before and self.path_index >= len(self.planned_path):
                        self._escape_active = False
                        self.debug_status = "escape_finished"
                        if bool(self.p.get("AGENT_REPLAN_AFTER_ESCAPE", True)):
                            self.planned_path = []
                            self.path_index = 0

                else:
                    # Normal forward progress: append reached waypoint to opt spine (dedup)
                    is_escaping = getattr(self, "_escape_active", False)
                    if not is_escaping:
                        if (len(self.executed_path_opt) == 0) or (np.linalg.norm(self.executed_path_opt[-1] - reached_wp) > 0.1):
                            self.executed_path_opt.append(reached_wp)

        else:
            # No path or out-of-bounds index: do nothing this tick
            if not self.planned_path:
                self.debug_status = "no_path"
            else:
                self.debug_status = "no_move_pathindex_oob"

            self.plan_fail_count = getattr(self, "plan_fail_count", 0) + 1
            self.last_plan_fail_step = time_step
            # print(f"[RAPF_Agent_v4] PLAN FAILED at t={time_step} | sensed={len(self.perceived_obstacles)}")

        # --- Record executed ALL pose trace (debug/replay only) ---
        if not hasattr(self, "executed_path_all") or self.executed_path_all is None:
            self.executed_path_all = [self.q.copy()]
        if not hasattr(self, "_last_exec_store_q_all") or self._last_exec_store_q_all is None:
            self._last_exec_store_q_all = self.q.copy()

        # Decimate pose trace a bit so it doesn't explode in size
        store_min_dist_all = getattr(self, "EXEC_STORE_MIN_DIST_ALL", 0.02)
        if moved and np.linalg.norm(self.q - self._last_exec_store_q_all) >= store_min_dist_all:
            self.executed_path_all.append(self.q.copy())
            self._last_exec_store_q_all = self.q.copy()

        # --- Goal reached ---
        if np.linalg.norm(self.q - self.goal_pos) < self.p['GOAL_TOLERANCE']:
            self.done = True

    def _start_agent_backtrack(self, *, time_step=None, reason: str = ""):
        """Agent-side escape: walk back along executed_path_opt for N points, then replan."""
        # Cooldown to avoid spamming backtracks every tick
        cooldown = int(self.p.get("AGENT_BACKTRACK_COOLDOWN", 2))
        if self._agent_backtrack_last_start_step is not None and time_step is not None:
            if (time_step - self._agent_backtrack_last_start_step) < cooldown:
                return

        if getattr(self, "_escape_active", False):
            return

        opt = getattr(self, "executed_path_opt", None)
        if opt is None or len(opt) < 2:
            # Not enough history to backtrack. Let the planner keep trying via VOs.
            return

        max_tries = int(self.p.get("AGENT_BACKTRACK_MAX_TRIES", 5))
        if self._agent_backtrack_tries >= max_tries:
            print(f"[RAPF_Agent_v4] Backtrack give-up after {self._agent_backtrack_tries} tries at t={time_step} | reason={reason}")
            self.backtrack_failed = True
            return

        k = int(self.p.get("AGENT_BACKTRACK_POINTS", 6))
        k = max(1, min(k, len(opt) - 1))

        # Build backtrack targets: most recent OPT points first (closest behind us)
        back_targets = [np.array(p, dtype=float).copy() for p in opt[-k:]]
        back_targets.reverse()

        # Ensure we don't target something we're already basically on.
        while back_targets and float(np.linalg.norm(back_targets[0] - self.q)) < 1e-6:
            back_targets.pop(0)

        if not back_targets:
            return

        # Activate escape mode so OPT spine gets pruned while we walk back.
        self._escape_active = True
        self._escape_remaining = len(back_targets)
        self._agent_backtrack_active = True
        self._agent_backtrack_tries += 1
        self._agent_backtrack_last_start_step = time_step

        # Replace current plan with the backtrack plan.
        self.planned_path = back_targets
        self.path_index = 0
        self.debug_status = f"agent_backtrack({reason}) k={k}"

    def _update_perception(self, env):
        """Detect obstacles within SENSE_RANGE and update internal map."""
        new_found = False
        for obs in env.obstacles:
            o_pos = np.array([obs.x, obs.y]) if hasattr(obs, 'x') else np.array(obs.center)
            dist = np.linalg.norm(self.q - o_pos) - float(obs.radius)

            if dist < (self.p['SENSE_RANGE'] + self.p['SENSE_PAD']):
                obs_key = (float(o_pos[0]), float(o_pos[1]), float(obs.radius))
                if obs_key not in self._known_obstacle_keys:
                    self.perceived_obstacles.append(obs)
                    self._known_obstacle_keys.add(obs_key)
                    new_found = True
        return new_found

    def _check_path_validity(self, lookahead_steps: int = 8, corridor_scale: float = 1.0) -> bool:
        """
        Returns True if replanning is needed.
        Only checks a limited number of upcoming waypoints (local lookahead).
        """
        if self.planned_path is None or len(self.planned_path) < 2:
            return True
        if self.path_index >= len(self.planned_path) - 1:
            return False

        i0 = max(0, self.path_index)
        i1 = min(len(self.planned_path) - 1, self.path_index + lookahead_steps)

        agent_radius = float(self.p["RHO_L"])

        for obs in self.perceived_obstacles:
            obs_pos = np.array([obs.x, obs.y]) if hasattr(obs, 'x') else np.array(obs.center)
            obs_r = float(obs.radius)

            corridor_w = corridor_scale * (agent_radius + obs_r)

            for i in range(i0, i1):
                a = self.planned_path[i]
                b = self.planned_path[i + 1]

                ab = b - a
                denom = float(np.dot(ab, ab)) + 1e-12
                t = float(np.dot(obs_pos - a, ab)) / denom
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                d = float(np.linalg.norm(obs_pos - closest))

                if d < corridor_w:
                    return True

        return False

    def _replan(self, time_step=None):
        self.replan_count += 1

        # if self.replan_count % 5 == 0:
        #     self.virtual_obstacles = []
        #     self.last_virtual_obstacles = []

        # Call planner statelessly (optional: keep virtual obstacles across replans)
        result = self.planner.plan(
            self.q, self.goal_pos, self.perceived_obstacles,
            virtual_obstacles=self.virtual_obstacles
        )
        success = bool(result.get("success", False))


        self.planning_effort += float(result.get("planning_effort", 0.0))

        # Always store VOs returned
        self.last_virtual_obstacles = result.get("virtual_obstacles", []) or []
        self.virtual_obstacles = self.last_virtual_obstacles

        self.virtual_total_seen += len(self.virtual_obstacles)
        self.virtual_max_active = max(self.virtual_max_active, len(self.virtual_obstacles))
        self.virtual_total_created = result.get("total_virtual_obstacles_created", 0)

        if not success:
            self.plan_fail_count += 1
            self.last_plan_fail_step = time_step
            self.planned_path = []
            self.path_index = 0

            # Agent handles backtracking
            self._start_agent_backtrack(time_step=time_step, reason="planner_failed")
            return False

        path = result.get("path", []) or []

        # Drop first point if it's basically our current pose
        if len(path) > 1 and np.linalg.norm(np.array(path[0]) - self.q) < 1e-6:
            path = path[1:]

        self.planned_path = [np.array(p, dtype=float) for p in path]
        self.path_index = 0

        return True

    def _move_towards(self, target_pos, speed_limit, dt):
        """Update position based on discrete kinematics. Includes a cheap collision gate."""
        max_step = float(speed_limit) * float(dt)
        displacement = target_pos - self.q
        dist = float(np.linalg.norm(displacement))

        if dist <= 0:
            return False

        heading = float(np.arctan2(displacement[1], displacement[0]))
        step_size = min(dist, max_step)
        q_next = self.q + (displacement / dist) * step_size

        # Cheap safety: prevent clipping through perceived obstacles
        agent_radius = float(0.1)
        for obs in self.perceived_obstacles:
            o_pos = np.array([obs.x, obs.y]) if hasattr(obs, 'x') else np.array(obs.center)
            o_r = float(obs.radius)
            d = self._point_to_segment_dist(o_pos, self.q, q_next)
            if d < (agent_radius + o_r + 1e-9):
                self.blocked_steps += 1
                # invalidate current plan so we replan next tick
                self.planned_path = []
                self.path_index = 0
                print(f"[RAPF_Agent_v4] BLOCKED MOVE at | sensed={len(self.perceived_obstacles)} | obs_radius={o_r} | obs_pos={o_pos} | d={d:.3f}" )
                return False

        # commit move
        self.heading = heading
        self.q = q_next
        return True

    def _point_to_segment_dist(self, p, a, b):
        """Distance from point p to segment ab."""
        ap = p - a
        ab = b - a
        t = float(np.dot(ap, ab)) / (float(np.dot(ab, ab)) + 1e-6)
        t = max(0.0, min(1.0, t))
        nearest = a + t * ab
        return float(np.linalg.norm(p - nearest))
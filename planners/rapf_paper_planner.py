import numpy as np
from simulation.env.runtime import CircularObstacle


class RAPFGlobalPlanner:
    """
    Standalone Global Planner implementing a paper-closer RAPF variant.

    This version keeps the same external structure as the newer planner, but:
    - removes repair / rewind / backtrack behavior
    - removes midpoint collision checking
    - when stuck, places a virtual obstacle and restarts planning from the
      current stuck position with a cleared simulated path
    """

    DEFAULT_PARAMS = {
        'ALPHA_A': 12.093740785111983,
        'MU_A': 0.00034084819962926796,
        'ALPHA_O': 10.38835966989917,
        'MU_O': 89.30139204021756,
        'RHO_L': 0.30,
        'RHO_U': 1.5024832235762904,
        'N_B': 13,
        'RHO_B': 0.583260830441854,
        'ARTIFICIAL_R': 0.6515262510741205,
        'MAX_PLANNING_STEPS': 5000,
        'GOAL_TOLERANCE': 0.8,
        'MAX_RESTARTS': 50,
    }

    def __init__(self, **params):
        self.p = {**self.DEFAULT_PARAMS, **params}
        self.total_virtual_obstacles_created = 0
        self.total_plan_calls = 0
        self.total_plan_failures = 0
        self._bacteria_angles = (
            2.0 * np.pi * np.arange(int(self.p['N_B'])) / int(self.p['N_B'])
        )

    def plan(self, start_pos, goal_pos, real_obstacles, *, virtual_obstacles=None) -> dict:
        """
        Executes the internal RAPF simulation loop to find a trap-free path.
        """
        self.total_plan_calls += 1
        start_pos = np.array(start_pos, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        dbg = bool(self.p.get("DEBUG_PRINTS", False))

        caller_start = start_pos.copy()
        effective_start = start_pos.copy()

        active_vos = virtual_obstacles if virtual_obstacles is not None else []

        if dbg:
            print(
                f"[Planner] plan() start={caller_start} goal={goal_pos} "
                f"obs={len(real_obstacles)} vos_in={len(active_vos)}"
            )

        path_found = False
        total_internal_steps = 0
        restart_count = 0
        break_reason = "max_restarts"

        # Current candidate path for this planning attempt
        sim_path = [effective_start.copy()]

        while not path_found and restart_count < self.p['MAX_RESTARTS']:
            restart_count += 1
            sim_q = sim_path[-1].copy()
            current_obstacles = self._prepare_obstacles(real_obstacles + active_vos)
            inner_success = False
            break_reason = "max_steps"

            remaining_steps = self.p['MAX_PLANNING_STEPS'] - total_internal_steps
            if remaining_steps <= 0:
                break_reason = "max_steps"
                if dbg:
                    print("[Planner] WARNING: MAX_PLANNING_STEPS reached.")
                break

            for _ in range(remaining_steps):
                total_internal_steps += 1

                # Goal check
                if np.linalg.norm(sim_q - goal_pos) < self.p['GOAL_TOLERANCE']:
                    inner_success = True
                    break_reason = "success"
                    break

                next_pos, is_stuck = self._compute_internal_step(sim_q, goal_pos, current_obstacles)

                if is_stuck:
                    # Paper-like RAPF behavior:
                    # place a virtual obstacle at the local minimum and restart
                    # planning from the current stuck position with a cleared path
                    vo_radius = self.p['ARTIFICIAL_R']
                    vo = CircularObstacle(center=sim_q.copy(), radius=vo_radius, kind="virtual")
                    active_vos.append(vo)
                    self.total_virtual_obstacles_created += 1

                    if len(sim_path) <= 1 or np.linalg.norm(sim_q - caller_start) < 1e-6:
                        return {
                            "success": False,
                            "break_reason": "agent_backtrack",
                            "virtual_obstacles": active_vos,
                        }

                    effective_start = sim_q.copy()
                    sim_path = [effective_start.copy()]

                    if dbg:
                        print(f"[Planner] STUCK at {sim_q}. Added VO and restarting from current position.")

                    break_reason = "stuck"
                    break

                sim_q = next_pos
                sim_path.append(sim_q.copy())

            if inner_success:
                if np.linalg.norm(sim_path[-1] - goal_pos) > 1e-3:
                    sim_path.append(goal_pos.copy())
                path_found = True

            if not path_found and break_reason not in ("stuck",):
                self.total_plan_failures += 1

        if not path_found and break_reason in ("max_restarts", "max_steps"):
            self.total_plan_failures += 1

        return {
            "path": sim_path,
            "success": path_found,
            "planning_effort": total_internal_steps,
            "virtual_obstacles": active_vos,
            "break_reason": break_reason,
            "total_virtual_obstacles_created": self.total_virtual_obstacles_created,
            "total_plan_calls": self.total_plan_calls,
            "total_plan_failures": self.total_plan_failures,
        }

    @staticmethod
    def _prepare_obstacles(obstacles):
        """Extract immutable circle geometry once per planning restart."""
        prepared = []
        for obstacle in obstacles:
            if hasattr(obstacle, "x") and hasattr(obstacle, "y"):
                center = np.array([obstacle.x, obstacle.y], dtype=float)
            else:
                center = np.asarray(obstacle.center, dtype=float)
            prepared.append((center, float(obstacle.radius)))
        return prepared

    def _compute_internal_step(self, pos, goal, obstacles):
        j_robot = self._compute_total_potential(pos, goal, obstacles)
        bacteria_points = self._generate_bacteria_points(pos, goal)

        best_point = pos
        min_potential = j_robot

        for b_point in bacteria_points:
            # Paper-closer behavior:
            # only check the candidate endpoint itself for collision
            if self._is_collision(b_point, obstacles):
                continue

            j_bac = self._compute_total_potential(b_point, goal, obstacles)
            if j_bac < min_potential:
                min_potential = j_bac
                best_point = b_point

        is_stuck = np.linalg.norm(best_point - pos) < 1e-4
        return best_point, is_stuck

    def _generate_bacteria_points(self, center_pos, goal_pos):
        vector_to_goal = goal_pos - center_pos
        base_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        angles = base_angle + self._bacteria_angles
        return np.column_stack(
            (
                center_pos[0] + self.p['RHO_B'] * np.cos(angles),
                center_pos[1] + self.p['RHO_B'] * np.sin(angles),
            )
        )
    def _compute_total_potential(self, pos, target, obstacles):
        target_delta = pos - target
        d_t_sq = float(np.dot(target_delta, target_delta))
        j_a = -self.p['ALPHA_A'] * np.exp(-self.p['MU_A'] * d_t_sq)

        j_r = 0.0
        for center, radius in obstacles:
            dist_surface = float(np.linalg.norm(pos - center)) - radius
            if dist_surface < self.p['RHO_L']:
                j_r += 1e6 * (1.0 / max(dist_surface, 1e-6))
            elif dist_surface < self.p['RHO_U']:
                d_calc = max(0.0, dist_surface)
                j_r += self.p['ALPHA_O'] * np.exp(-self.p['MU_O'] * (d_calc ** 2))
        return j_a + j_r
    def _is_collision(self, check_pos, obstacles):
        safety_radius = self.p['RHO_L']
        for center, radius in obstacles:
            if float(np.linalg.norm(check_pos - center)) - radius < safety_radius:
                return True
        return False

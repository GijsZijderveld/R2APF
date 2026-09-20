import numpy as np
from simulation.env.runtime import CircularObstacle

class RAPFGlobalPlanner:
    """
    Standalone Global Planner implementing the Robust Artificial Potential Field (RAPF) v3.
    Decoupled from agent state to allow for flexible replanning from any start configuration.
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
        # When stuck near the agent's actual executed path, remove recent executed
        # points within this radius to avoid replanning loops back into minima.
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
        # Initialize working variables
        self.total_plan_calls += 1
        start_pos = np.array(start_pos, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        dbg = bool(self.p.get("DEBUG_PRINTS", False))

        # Keep the caller's start (agent's current position at the time of replanning)
        caller_start = start_pos.copy()

        # If we need to escape a start-local-minimum, we will plan forward from an
        # for the agent to execute first.
        effective_start = start_pos.copy()

        active_vos = virtual_obstacles if virtual_obstacles is not None else []

        # NOTE (v4): executed_path backtracking is handled agent-side.
        # We keep these args for compatibility, but the planner itself stays stateless.
        # Optional entry debug print
        if dbg:
            print(f"[Planner] plan() start={caller_start} goal={goal_pos} obs={len(real_obstacles)} vos_in={len(active_vos)}")
        

        path_found = False
        total_internal_steps = 0
        restart_count = 0
        break_reason = "max_restarts"
        
        # Current simulated path starts at the provided start_pos
        sim_path = [effective_start.copy()]

        while not path_found and restart_count < self.p['MAX_RESTARTS']:
            restart_count += 1
            sim_q = sim_path[-1].copy()
            current_obstacles = self._prepare_obstacles(real_obstacles + active_vos)
            inner_success = False
            break_reason = "max_steps"
            
            # Remaining steps in the global budget
            remaining_steps = self.p['MAX_PLANNING_STEPS'] - total_internal_steps
            if remaining_steps <= 0:
                break_reason = "max_steps"
                print(f"[Planner] WARNING: MAX_PLANNING_STEPS reached at start of iteration (remaining_steps={remaining_steps}). Consider increasing MAX_PLANNING_STEPS or check for logical errors.")
                break

            for _ in range(remaining_steps):
                total_internal_steps += 1

                # Goal Check
                if np.linalg.norm(sim_q - goal_pos) < self.p['GOAL_TOLERANCE']:
                    inner_success = True
                    break_reason = "success"
                    break
                
                next_pos, is_stuck = self._compute_internal_step(sim_q, goal_pos, current_obstacles)
                
                if is_stuck:
                    # 1. Place the Virtual Obstacle at the local minimum
                    vo_radius = self.p['ARTIFICIAL_R']
                    vo = CircularObstacle(center=sim_q, radius=vo_radius, kind="virtual")
                    active_vos.append(vo)
                    self.total_virtual_obstacles_created += 1
                    
                    # 2. Define the 'Danger Zone' radius
                    influence_threshold = vo_radius + self.p['RHO_U']
                    
                    # 3. Try to backtrack within the CURRENT simulation path first
                    cut_index = -1
                    for i in range(len(sim_path) - 1, -1, -1):
                        if np.linalg.norm(sim_path[i] - sim_q) > influence_threshold:
                            cut_index = i
                            break
                    
                    if cut_index != -1:
                        # Found a safe point in the current simulated path
                        sim_path = sim_path[:cut_index + 1]
                        effective_start = sim_path[-1].copy()
                    else:
                        # CURRENT sim path is fully inside the danger zone.
                        # Agent-side logic will decide whether to backtrack.
                        return {"success": False, "break_reason": "agent_backtrack", "virtual_obstacles": active_vos}

                    if dbg:
                        print(f"[Planner] STUCK at {sim_q}. Rewinding to {effective_start}.")
                    
                    break_reason = "stuck"
                    break # Restart inner loop with new start and new VO
                
                sim_q = next_pos
                sim_path.append(sim_q.copy())
            
            if inner_success:
                # Avoid appending duplicate if final sim_q is already at goal
                if np.linalg.norm(sim_path[-1] - goal_pos) > 1e-3:
                    sim_path.append(goal_pos.copy())
                path_found = True

            if not path_found:
                self.total_plan_failures += 1

        
        # print(f"[Planner] plan() complete: success={path_found} total_steps={total_internal_steps} virtual_obstacles={len(active_vos)} break_reason={break_reason}")
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
            # Collision check including Midpoint Check to prevent "tunneling" 
            collision = False
            for t in [0.25, 0.5, 0.75]:
                check_p = pos + t * (b_point - pos)
                if self._is_collision(check_p, obstacles):
                    collision = True
                    break
            
            if collision:
                continue

            j_bac = self._compute_total_potential(b_point, goal, obstacles)
            if j_bac < min_potential:
                min_potential = j_bac
                best_point = b_point
        
        # Stuck if no better point is found [cite: 2192]
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
                j_r += self.p['ALPHA_O'] * np.exp(-self.p['MU_O'] * (d_calc**2))
        return j_a + j_r
    def _is_collision(self, check_pos, obstacles):
        safety_radius = self.p['RHO_L']
        for center, radius in obstacles:
            if float(np.linalg.norm(check_pos - center)) - radius < safety_radius:
                return True
        return False

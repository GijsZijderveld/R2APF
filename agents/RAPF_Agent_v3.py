import numpy as np
from simulation.env.runtime import CircularObstacle 

# --- RAPF Parameters ---
RAPF_PARAMS = {
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

class RAPF_Agent:
    def __init__(self, agent_id, position, goal=None, **params):
        self.id = agent_id
        self.start_pos = np.array(position, dtype=float)
        self.q = self.start_pos.copy()
        self.goal_pos = np.array(goal, dtype=float) if goal is not None else None
        
        self.p = {**RAPF_PARAMS, **params}
        self.planned_path = [] 
        self.path_index = 0
        self.is_planning_complete = False
        self.planning_effort = 0
        self.restart_count = 0
        self.virtual_obstacles = [] 

    @property
    def position(self):
        return self.q

    @property
    def goal(self):
        return self.goal_pos

    def step(self, env, agents, dt, speed_limit, time_step, bounds=None):
        if not self.is_planning_complete:
            self.plan_trajectory(env.obstacles)
            self.is_planning_complete = True
            
        if self.path_index < len(self.planned_path):
            target_pos = self.planned_path[self.path_index]
            displacement = target_pos - self.q
            dist = np.linalg.norm(displacement)
            max_step = speed_limit * dt
            
            if dist > max_step:
                self.q = self.q + (displacement / dist) * max_step
            else:
                self.q = target_pos
                self.path_index += 1 

    def plan_trajectory(self, real_obstacles, start_pos=None):
        self.virtual_obstacles = []
        path_found = False
        total_internal_steps = 0
        self.restart_count = 0
        limit_restarts = self.p['MAX_RESTARTS']
        rho_u = self.p['RHO_U']

        # [FIX 1] Determine the actual start point (Self.q or Start)
        current_start = start_pos if start_pos is not None else self.start_pos
        
        # Initialize path with the CORRECT start
        sim_path = [current_start.copy()]

        while not path_found and self.restart_count < limit_restarts:
            self.restart_count += 1
            
            sim_q = sim_path[-1].copy()
            current_obstacles = real_obstacles + self.virtual_obstacles
            inner_success = False
            break_reason = "max_steps" 
            
            remaining_steps = self.p['MAX_PLANNING_STEPS'] - len(sim_path)
            
            for _ in range(max(remaining_steps, 100)): 
                total_internal_steps += 1

                if np.linalg.norm(sim_q - self.goal_pos) < self.p['GOAL_TOLERANCE']:
                    inner_success = True
                    break_reason = "success"
                    break
                
                next_pos, _ = self._compute_internal_step(sim_q, current_obstacles)
                
                # Check for stuck condition
                if np.linalg.norm(next_pos - sim_q) < 1e-4:
                    vo_radius = self.p['ARTIFICIAL_R']
                    vo = CircularObstacle(center=sim_q, radius=vo_radius, kind="virtual")
                    self.virtual_obstacles.append(vo)
                    
                    influence_threshold = vo_radius + rho_u
                    cut_index = 0
                    found_safe_point = False
                    
                    # Backtrack to find safe point
                    for i in range(len(sim_path) - 1, -1, -1):
                        dist_to_stuck = np.linalg.norm(sim_path[i] - sim_q)
                        if dist_to_stuck > influence_threshold:
                            cut_index = i
                            found_safe_point = True
                            break
                    
                    if found_safe_point:
                        sim_path = sim_path[:cut_index + 1]
                    else:
                        # [FIX 2 - CRITICAL] 
                        # If we pruned everything, restart from CURRENT POSITION, not global start
                        sim_path = [current_start.copy()]
                    
                    break_reason = "stuck"
                    break 
                
                sim_q = next_pos
                sim_path.append(sim_q.copy())
            
            if inner_success:
                sim_path.append(self.goal_pos.copy())
                self.planned_path = sim_path
                path_found = True
            elif break_reason == "stuck":
                pass 
            else:
                break 

        if not path_found:
            self.planned_path = sim_path

        self.planning_effort = total_internal_steps
        return path_found

    def _compute_internal_step(self, pos, obstacles):
        j_robot = self._compute_total_potential(pos, self.goal_pos, obstacles)
        bacteria_points = self._generate_bacteria_points(pos)

        best_point = pos
        min_potential = j_robot
        
        for b_point in bacteria_points:
            j_bac = self._compute_total_potential(b_point, self.goal_pos, obstacles)
            
            # Midpoint Check
            safe_step = True
            for t in [0.25, 0.5, 0.75]: # Check 25%, 50%, and 75%
                check_p = pos + t * (b_point - pos)
                if self._is_collision(check_p, obstacles):
                    j_bac = float('inf')
                    safe_step = False
                    break

            if j_bac < min_potential:
                min_potential = j_bac
                best_point = b_point
        
        return best_point, (best_point is pos)

    def _is_collision(self, check_pos, obstacles):
        safety_radius = self.p['RHO_L']
        for o in obstacles:
            if hasattr(o, 'distance_to_surface'):
                dist_surface = o.distance_to_surface(check_pos)
            else:
                o_pos = np.array([o.x, o.y]) if hasattr(o, 'x') else o.center
                dist_surface = np.linalg.norm(check_pos - o_pos) - o.radius

            if dist_surface < safety_radius:
                return True
        return False

    def _generate_bacteria_points(self, center_pos):
        points = []
        n_b = self.p['N_B']
        rho_b = self.p['RHO_B']
        
        # 1. Calculate angle to goal (The "Alignment" improvement from RAPF)
        # This ensures one bacteria always points directly at the target.
        vector_to_goal = self.goal_pos - center_pos
        base_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        
        for i in range(n_b):
            # 2. Distribute points evenly (Structured, Eq. 13)
            # We add the base_angle to rotate the whole structure toward the goal.
            angle = base_angle + (2 * np.pi * i / n_b)
            
            # Note: Manteaux uses a fixed radius (rho_b) in Eq 13.
            # Your code used random radius. To strictly follow RAPF, 
            # use the fixed ring, or a fixed set of concentric rings.
            # Here we use the fixed outer ring as per Eq 13.
            b_x = center_pos[0] + rho_b * np.cos(angle)
            b_y = center_pos[1] + rho_b * np.sin(angle)
            
            points.append(np.array([b_x, b_y]))
            
        return points

    def _compute_total_potential(self, pos, target, obstacles):
        d_t_sq = np.linalg.norm(pos - target)**2
        j_a = -self.p['ALPHA_A'] * np.exp(-self.p['MU_A'] * d_t_sq)

        j_r = 0
        for o in obstacles:
            if hasattr(o, 'distance_to_surface'):
                dist_surface = o.distance_to_surface(pos)
            else:
                o_pos = np.array([o.x, o.y]) if hasattr(o, 'x') else o.center
                dist_surface = np.linalg.norm(pos - o_pos) - o.radius

            if dist_surface < self.p['RHO_L']:
                j_r += 1e6 * (1.0 / max(dist_surface, 1e-6))
            elif dist_surface < self.p['RHO_U']:
                d_calc = max(0.0, dist_surface)
                j_r += self.p['ALPHA_O'] * np.exp(-self.p['MU_O'] * (d_calc**2))
            
        return j_a + j_r
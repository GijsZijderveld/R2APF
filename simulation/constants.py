# simulation/constants.py
N_AGENTS       = 5
DT             = 0.1            # seconds
MAX_STEPS      = 1000
GOAL_RADIUS    = 1            # m; success radius
K_STABLE       = 10             # consecutive steps inside goal to count success
ROBOT_RADIUS   = 0.2           # m; used for agent-agent collisions
ROBOT_CLEARANCE= 0.2           # m; clearance to obstacle edge
VEL_CLAMP      = 1.0          # m/s max commanded speed (safety clamp)

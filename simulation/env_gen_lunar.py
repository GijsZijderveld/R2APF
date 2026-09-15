from __future__ import annotations

# env_gen_lunar.py
import math, random
import numpy as np
from dataclasses import dataclass




SCENARIOS = {
    "A": dict(rock_count=42,  crater_count=38),
    "D": dict(rock_count=65,  crater_count=35),
    "B": dict(rock_count=88,  crater_count=32),
    "E": dict(rock_count=112, crater_count=28),
    "C": dict(rock_count=137, crater_count=24),
}

BASE = dict(
    L=30.0,          # 30x30 m map (per spec)
    rock_cov=0.018,  # 1.8% of ROI
    crater_cov=0.11, # 15% of ROI  (set to 0.11 if you intended 11%)
    q_r=1.6,
    k_r=0.02,
    D_crit=0.065,
    allow_overlap=True,
    min_sep_factor=1.0,
    roi=(5.0, 25.0, 5.0, 25.0),  # (xmin, xmax, ymin, ymax)
)


def make_config(scn: str, seed: int, **overrides) -> EnvConfig:
    if scn not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scn}', expected one of {list(SCENARIOS)}")
    return EnvConfig(rng_seed=seed, **BASE, **SCENARIOS[scn], **overrides)

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float     # meters
    kind: str         # "rock" | "crater"

    @property
    def center(self):
        # matches what your agent expects
        return (self.x, self.y)


@dataclass
class EnvConfig:
    L: float
    rock_cov: float
    crater_cov: float
    q_r: float
    k_r: float
    rock_count: int
    crater_count: int
    D_crit: float = 0.065
    allow_overlap: bool = True
    min_sep_factor: float = 1.0
    rng_seed: int = 0
    roi: tuple[float, float, float, float] = (5.0, 25.0, 5.0, 25.0)

def _sample_exp_diameters(n, q_r, D_min):
    """Sample n diameters >= D_min from an exponential tail with rate q_r."""
    # if D ~ D_min + Exp(q_r)
    u = np.random.rand(n)
    return D_min - (1.0/q_r)*np.log(1.0 - u)  # inverse-CDF with shift

def _fit_to_coverage(diams, target_area, kind):
    """Uniformly scale diameters so sum of areas equals target_area."""
    areas = np.pi * (diams/2.0)**2
    S = areas.sum()
    if S <= 1e-12:
        return diams
    scale = math.sqrt(target_area / S) * 1.0  # uniform radius scale
    return diams * scale

def _place_disks_rect(xmin, xmax, ymin, ymax, radii, allow_overlap, min_sep_factor, rng):
    """Place disks inside a rectangle with optional non-overlap."""
    obs = []
    for r in radii:
        for _ in range(2000):
            x = rng.uniform(xmin + r, xmax - r)
            y = rng.uniform(ymin + r, ymax - r)
            if allow_overlap:
                obs.append((x, y, r)); break
            ok = True
            for (px, py, pr) in obs:
                if math.hypot(x - px, y - py) < (r + pr) * min_sep_factor:
                    ok = False; break
            if ok:
                obs.append((x, y, r)); break
        else:
            # as a last resort, place within bounds even if near others
            obs.append((rng.uniform(xmin + r, xmax - r), rng.uniform(ymin + r, ymax - r), r))
    return obs

def generate_lunar_env(cfg: EnvConfig):
    rng = random.Random(cfg.rng_seed)
    np.random.seed(cfg.rng_seed)

    xmin, xmax, ymin, ymax = cfg.roi
    A_roi = (xmax - xmin) * (ymax - ymin)  # coverage is defined over the 20x20 area

    rock_area_budget = cfg.rock_cov * A_roi
    crater_area_budget = cfg.crater_cov * A_roi

    rock_d = _sample_exp_diameters(cfg.rock_count, cfg.q_r, cfg.D_crit)
    crater_d = _sample_exp_diameters(cfg.crater_count, cfg.q_r, cfg.D_crit)

    rock_d = _fit_to_coverage(rock_d, rock_area_budget, "rock")
    crater_d = _fit_to_coverage(crater_d, crater_area_budget, "crater")

    # place only inside ROI
    rocks_xy = _place_disks_rect(xmin, xmax, ymin, ymax, rock_d/2.0, cfg.allow_overlap, cfg.min_sep_factor, rng)
    crats_xy = _place_disks_rect(xmin, xmax, ymin, ymax, crater_d/2.0, cfg.allow_overlap, cfg.min_sep_factor, rng)

    rocks = [Obstacle(x, y, r, "rock") for (x, y, r) in rocks_xy]
    craters = [Obstacle(x, y, r, "crater") for (x, y, r) in crats_xy]
    return rocks + craters

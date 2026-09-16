# simulation/env_suite.py
from .env_gen_lunar import make_config  # the helper you had
from dataclasses import asdict

def sample_batch(split: str, per_scenario: int = 8, seed_base: int | None = None):
    """
    Returns a list of EnvConfig for a balanced A/B/C batch.
    Seeds are partitioned by split for determinism.
    """
    seed_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    if seed_base is not None:
        seed_offset += seed_base

    confs = []
    for scn in ["A", "B", "C"]:
        for i in range(per_scenario):
            confs.append(make_config(scn=scn, seed=seed_offset + i))
            confs[-1].split = split  # if EnvConfig has this field; else omit
    return confs

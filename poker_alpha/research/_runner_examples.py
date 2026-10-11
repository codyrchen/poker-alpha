"""Tiny picklable task functions used by the runner tests and smoke checks."""

import time


def square(config, seed):
    return {"value": (config.get("x", 0) + (seed or 0)) ** 2}


def boom(config, seed):
    raise RuntimeError(f"deliberate failure for seed {seed}")


def sleepy(config, seed):
    time.sleep(config.get("seconds", 5))
    return {"slept": True}

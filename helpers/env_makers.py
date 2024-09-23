import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from beartype import beartype

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from helpers import logger


# Farama Foundation Gymnasium MuJoCo
FARAMA_MUJOCO_STEM = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "HumanoidStandup",
    "Humanoid",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "Swimmer",
    "Walker2d",
]
FARAMA_MUJOCO = []
FARAMA_MUJOCO.extend([f"{name}-v4"
    for name in FARAMA_MUJOCO_STEM])
FARAMA_MUJOCO.extend([f"{name}-v5"
    for name in FARAMA_MUJOCO_STEM])

# DeepMind Control Suite (DMC) MuJoCo
DEEPMIND_MUJOCO_STEM = [
    "Hopper-Hop",
    "Cheetah-Run",
    "Walker-Walk",
    "Walker-Run",
    "Stacker-Stack_2",
    "Stacker-Stack_4",
    "Humanoid-Walk",
    "Humanoid-Run",
    "Humanoid-Run_Pure_State",
    "Humanoid_CMU-Stand",
    "Humanoid_CMU-Run",
    "Quadruped-Walk",
    "Quadruped-Run",
    "Quadruped-Escape",
    "Quadruped-Fetch",
    "Dog-Run",
    "Dog-Fetch",
]
DEEPMIND_MUJOCO = []
DEEPMIND_MUJOCO.extend([f"{name}-Feat-v0"
    for name in DEEPMIND_MUJOCO_STEM])
DEEPMIND_MUJOCO.extend([f"{name}-Pix-v0"
    for name in DEEPMIND_MUJOCO_STEM])

# Flag benchmarks that are not available yet
BENCHMARKS = {"farama": FARAMA_MUJOCO, "dmc": DEEPMIND_MUJOCO}
AVAILABLE_FLAGS = dict.fromkeys(BENCHMARKS, False)
AVAILABLE_FLAGS["dmc"] = False  # TODO(lionel): integrate with the DMC suite


@beartype
def get_benchmark(env_id: str):
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    assert not AVAILABLE_FLAGS[benchmark], "unavailable benchmark"
    return benchmark


@beartype
def make_env(
    env_id: str,
    horizon: int,
    *,
    vectorized: bool,
    multi_proc: bool,
    num_env: Optional[int] = None,
    wrap_absorb: bool,
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], float]:

    # create an environment
    bench = get_benchmark(env_id)  # at this point benchmark is valid

    if bench == "farama_mujoco":
        # remove the lockfile if it exists (maybe not a thing in Farama's Gymnasium anymore?)
        lockfile = (Path(os.environ["CONDA_PREFIX"]) / "lib" / "python3.7" / "site-packages" /
                    "mujoco_py" / "generated" / "mujocopy-buildlock.lock")
        try:
            Path(lockfile).unlink()
            logger.info("[WARN] removed mujoco lockfile")
        except OSError:
            pass

        return make_farama_mujoco_env(
            env_id,
            horizon,
            vectorized=vectorized,
            multi_proc=multi_proc,
            num_env=num_env,
            wrap_absorb=wrap_absorb,
            record=record,
            render=render,
        )
    raise ValueError(f"invalid benchmark: {bench}")


@beartype
def make_farama_mujoco_env(
    env_id: str,
    horizon: int,
    *,
    vectorized: bool,
    multi_proc: bool,
    num_env: Optional[int],
    wrap_absorb: bool,
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], float]:

    # not ideal for code golf, but clearer for debug

    assert sum([record, vectorized]) <= 1, "not both same time"
    assert sum([render, vectorized]) <= 1, "not both same time"
    assert (not vectorized) or (num_env is not None), "must give num_envs when vectorized"

    # create env
    # normally the windowed one is "human" .other option for later: "rgb_array", but prefer:
    # the following: `from gymnasium.wrappers.pixel_observation import PixelObservationWrapper`
    if record:  # overwrites render
        assert horizon is not None
        env = TimeLimit(gym.make(env_id, render_mode="rgb_array_list"), max_episode_steps=horizon)
    elif render:
        assert horizon is not None
        env = TimeLimit(gym.make(env_id, render_mode="human"), max_episode_steps=horizon)
        # reference: https://younis.dev/blog/render-api/
    elif vectorized:
        assert num_env is not None
        assert horizon is not None
        env = (AsyncVectorEnv if multi_proc else SyncVectorEnv)([
            lambda: TimeLimit(gym.make(env_id), max_episode_steps=horizon)
            for _ in range(num_env)
        ])
        assert isinstance(env, (AsyncVectorEnv, SyncVectorEnv))
        logger.info("using vectorized envs")
    else:
        assert horizon is not None
        env = TimeLimit(gym.make(env_id), max_episode_steps=horizon)

    # build shapes for nets and replay buffer
    net_shapes = {}
    erb_shapes = {}

    # for the nets
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box)  # for due diligence
    ob_shape = ob_space.shape
    assert ob_shape is not None
    ac_space = env.action_space  # used now and later to get max action
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError(f"env ({env}) is discrete: out of scope here")
    assert isinstance(ac_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_shape = ac_space.shape
    assert ac_shape is not None
    net_shapes.update({"ob_shape": ob_shape, "ac_shape": ac_shape})

    # for the replay buffer
    ob_dim = ob_dim_orig = ob_shape[-1]
    ac_dim = ac_dim_orig = ac_shape[-1]  # for both: num dims
    if wrap_absorb:
        ob_dim += 1
        ac_dim += 1
        erb_shapes.update({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
            "obs0_orig": (ob_dim_orig,),
            "obs1_orig": (ob_dim_orig,),
            "acs_orig": (ac_dim_orig,),
        })
    else:
        erb_shapes.update({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
        })

    # max value for action
    max_ac = max(
        np.abs(np.amax(ac_space.high.astype("float32"))),
        np.abs(np.amin(ac_space.low.astype("float32"))),
    ).item()  # return it not as an ndarray but a standard Python scalar
    # despite the fact that we use the max, the actions are clipped with min and max
    # during interaction in the orchestrator

    return env, net_shapes, erb_shapes, max_ac

import os
import time
import tqdm
from pathlib import Path
from typing import Union, Callable, Generator
from contextlib import contextmanager
from collections import deque

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import rearrange
from termcolor import colored
import wandb
from wandb.errors import CommError
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from gymnasium.core import Env
from gymnasium.vector.vector_env import VectorEnv

from helpers import logger
from agents.on_policy_agent import OnPolicyAgent


@contextmanager
@beartype
def timed(op: str, timer: Callable[[], float]):
    logger.info(colored(
        f"starting timer | op: {op}",
        "magenta", attrs=["underline", "bold"]))
    tstart = timer()
    yield
    tot_time = timer() - tstart
    logger.info(colored(
        f"stopping timer | op took {tot_time}secs",
        "magenta"))


@beartype
def segment(env: Union[Env, VectorEnv],
            agent: OnPolicyAgent,
            seed: int,
            segment_len: int,
    ) -> Generator[None, None, None]:

    assert agent.rb is not None

    obs = torch.zeros((agent.hps.segment_len, agent.hps.num_envs) + env.single_observation_space.shape).to(agent.device)
    next_obs = torch.zeros((agent.hps.segment_len, agent.hps.num_envs) + env.single_observation_space.shape).to(agent.device)
    actions = torch.zeros((agent.hps.segment_len, agent.hps.num_envs) + env.single_action_space.shape).to(agent.device)
    logprobs = torch.zeros((agent.hps.segment_len, agent.hps.num_envs)).to(agent.device)
    rewards = torch.zeros((agent.hps.segment_len, agent.hps.num_envs)).to(agent.device)
    terminations = torch.zeros((agent.hps.segment_len, agent.hps.num_envs)).to(agent.device)
    values = torch.zeros((agent.hps.segment_len, agent.hps.num_envs)).to(agent.device)

    ob, _ = env.reset(seed=seed)  # for the very first reset, we give a seed (and never again)
    ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)
    action = None  # as long as r is init at 0: ac will be written over

    t = 0

    while True:
        with torch.no_grad():
            # predict action
            action, logprob, _, q_value = agent.predict(
                TensorDict(
                    {
                        "observations": ob,
                    },
                    device=agent.device,
                ),
            )

        # interact with env
        next_ob, reward, termination, truncation, infos = env.step(action.cpu().numpy())

        next_ob = torch.as_tensor(next_ob, device=agent.device, dtype=torch.float)
        real_next_ob = next_ob.clone()

        for idx, trunc in enumerate(np.array(truncation)):
            if trunc:
                real_next_ob[idx] = torch.as_tensor(
                    infos["final_observation"][idx], device=agent.device, dtype=torch.float)

        reward = torch.as_tensor(reward, device=agent.device, dtype=torch.float)
        termination = torch.as_tensor(termination, device=agent.device, dtype=torch.bool)

        obs[t] = ob
        next_obs[t] = next_ob
        actions[t] = action
        rewards[t] = reward
        terminations[t] = termination
        logprobs[t] = logprob
        values[t] = q_value.flatten()

        if t > 0 and t % (segment_len-1) == 0:
            t = 0
            # advantage estimation
            advantages, returns = agent.advantage_estimation(next_obs, rewards, terminations, values)

            agent.rb.extend(
                TensorDict(
                    {
                        "observations": obs.reshape((-1,) + env.single_observation_space.shape),
                        "next_observations": next_obs.reshape((-1,) + env.single_observation_space.shape),
                        "actions": actions.reshape((-1,) + env.single_action_space.shape),
                        "rewards": rewards.reshape(-1),
                        "terminations": terminations.reshape(-1),
                        "logprobs": logprobs.reshape(-1),
                        "values": values.reshape(-1),
                        "advantages": advantages.reshape(-1),
                        "returns": returns.reshape(-1),
                    },
                    batch_size=agent.hps.segment_len*agent.hps.num_envs,
                    device=agent.device,
                ),
            )
            
            yield

        else:
            ob = next_ob
            t += 1


@beartype
def episode(env: Env,
            agent: OnPolicyAgent,
            seed: int,
            *,
            need_lists: bool = False,
    ) -> Generator[dict[str, np.ndarray], None, None]:
    # generator that spits out a trajectory collected during a single episode

    # `append` operation is significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible

    def randomize_seed() -> int:
        return seed + rng.integers(2**32 - 1, size=1).item()
        # seeded Generator: deterministic -> reproducible

    obs_list = []
    next_obs_list = []
    actions_list = []
    rewards_list = []
    terminations_list = []
    dones_list = []

    ob, _ = env.reset(seed=randomize_seed())
    if need_lists:
        obs_list.append(ob)
    ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)

    while True:
        with torch.no_grad():
            # predict action
            action, log_probs, ent, q_values = agent.predict(
                TensorDict(
                    {
                        "observations": ob,
                    },
                    device=agent.device,
                ),
            )

        new_ob, reward, termination, truncation, infos = env.step(action.cpu().numpy())

        done = termination or truncation

        if need_lists:
            next_obs_list.append(new_ob)
            actions_list.append(action)
            rewards_list.append(reward)
            terminations_list.append(termination)
            dones_list.append(done)
            if not done:
                obs_list.append(new_ob)

        new_ob = torch.as_tensor(new_ob, device=agent.device, dtype=torch.float)
        ob = new_ob

        if "final_info" in infos:
            # we have len(infos["final_info"]) == 1
            for info in infos["final_info"]:
                ep_len = float(info["episode"]["l"].item())
                ep_ret = float(info["episode"]["r"].item())

            if need_lists:
                out = {
                    "observations": np.array(obs_list),
                    "actions": np.array(actions_list),
                    "next_observations": np.array(next_obs_list),
                    "rewards": np.array(rewards_list),
                    "terminations": np.array(terminations_list),
                    "dones": np.array(dones_list),
                    "length": np.array(ep_len),
                    "return": np.array(ep_ret),
                }
            else:
                out = {
                    "length": np.array(ep_len),
                    "return": np.array(ep_ret),
                }
            yield out

            if need_lists:
                obs_list = []
                next_obs_list = []
                actions_list = []
                rewards_list = []
                terminations_list = []
                dones_list = []

            ob, _ = env.reset(seed=randomize_seed())
            if need_lists:
                obs_list.append(ob)
            ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)


@beartype
def train(cfg: DictConfig,
          env: Union[Env, VectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], OnPolicyAgent],
          name: str):

    assert isinstance(cfg, DictConfig)

    agent = agent_wrapper()

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save the model as a dry run, to avoid bad surprises at the end
    agent.save(ckpt_dir, sfx="dryrun")
    logger.info(f"dry run -- saved model @: {ckpt_dir}")

    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    group = ".".join(name.split(".")[:-1])  # everything in name except seed
    logger.warn(f"{name=}")
    logger.warn(f"{group=}")
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
                save_code=True,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    # create segment generator for training
    seg_gen = segment(env, agent, cfg.seed, cfg.segment_len)
    # create episode generator for evaluating
    ep_gen = episode(eval_env, agent, cfg.seed)

    lr_updates = 0
    start_time = None
    measure_burnin = None
    pbar = tqdm.tqdm(range(cfg.num_timesteps))
    time_spent_eval = 0
    num_updates = cfg.num_timesteps // (cfg.segment_len * cfg.num_envs)

    tlog = TensorDict({})
    maxlen = 20 * cfg.eval_steps
    len_buff = deque(maxlen=maxlen)
    ret_buff = deque(maxlen=maxlen)

    mode = None
    tc_update_nets = agent.update_nets
    if cfg.compile:
        tc_update_nets = torch.compile(tc_update_nets, mode=mode)
    if cfg.cudagraphs:
        tc_update_nets = CudaGraphModule(tc_update_nets, in_keys=[], out_keys=[])

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if ((agent.timesteps_so_far >= (cfg.measure_burnin) and
             start_time is None)):
            start_time = time.time()
            measure_burnin = agent.timesteps_so_far

        if cfg.anneal_lr:
            frac = 1.0 - lr_updates / num_updates
            lrnow = frac * cfg.lr
            agent.optimizer.param_groups[0]["lr"] = lrnow

        logger.info(("interact").upper())
        next(seg_gen)
        agent.timesteps_so_far += (increment := cfg.segment_len * cfg.num_envs)
        pbar.update(increment)

        logger.info(("train").upper())

        # minibatch update

        # Initialize accumulators for losses
        policy_losses = torch.zeros(cfg.update_epochs, cfg.num_minibatches, device=agent.device)
        value_losses = torch.zeros(cfg.update_epochs, cfg.num_minibatches, device=agent.device)
        entropy_losses = torch.zeros(cfg.update_epochs, cfg.num_minibatches, device=agent.device)
        total_losses = torch.zeros(cfg.update_epochs, cfg.num_minibatches, device=agent.device)
        approx_kls = torch.zeros(cfg.update_epochs, cfg.num_minibatches, device=agent.device)

        mini_batch_size = cfg.segment_len * cfg.num_envs // cfg.num_minibatches
        for epoch in range(cfg.update_epochs):
            indices = torch.randperm(len((agent.rb)))
            for j, i in enumerate(range(0, len(agent.rb), mini_batch_size)):
                batch_indices = indices[i:i + mini_batch_size]
                batch = agent.rb[batch_indices]
                losses = tc_update_nets(batch)

                # Store values in accumulators
                policy_losses[epoch, j] = losses["policy_loss"]
                value_losses[epoch, j] = losses["value_loss"]
                entropy_losses[epoch, j] = losses["entropy_loss"]
                total_losses[epoch, j] = losses["loss"]
                approx_kls[epoch, j] = losses["approx_kl"]
                # tlog.update(tc_update_nets(batch))

        # Compute the mean for each loss type across all epochs and mini-batches
        mean_policy_loss = policy_losses.mean()
        mean_value_loss = value_losses.mean()
        mean_entropy_loss = entropy_losses.mean()
        mean_loss = total_losses.mean()
        mean_approx_kl = approx_kls.mean()

        # Output as a TensorDict
        tlog.update(TensorDict(
            {
                "losses/policy_loss": mean_policy_loss,
                "losses/value_loss": mean_value_loss,
                "losses/entropy_loss": mean_entropy_loss,
                "losses/loss": mean_loss,
                "losses/approx_kl": mean_approx_kl,
            },
            batch_size=[],
            device=agent.device,
        ))
        agent.rb.empty()
        
        y_pred, y_true = agent.rb["values"].cpu().numpy(), agent.rb["returns"].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if agent.timesteps_so_far % cfg.eval_every == 0:
            logger.info(("eval").upper())
            eval_start = time.time()

            for _ in range(cfg.eval_steps):
                ep = next(ep_gen)
                len_buff.append(ep["length"])
                ret_buff.append(ep["return"])

            with torch.no_grad():
                eval_metrics = {
                    "length": torch.tensor(np.array(list(len_buff)), dtype=torch.float).mean(),
                    "return": torch.tensor(np.array(list(ret_buff)), dtype=torch.float).mean(),
                }

            if (new_best := eval_metrics["return"].item()) > agent.best_eval_ep_ret:
                # save the new best model
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
                logger.info(f"new best eval! -- saved model @: {ckpt_dir}")

            # log with logger
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for k, v in eval_metrics.items():
                logger.record_tabular(k, v.numpy())
            logger.dump_tabular()

            # log with wandb
            wandb.log(
                {
                    **tlog.to_dict(),
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    "losses/explained_variance": explained_var,
                    "vitals/replay_buffer_numel": len(agent.rb),
                    "vitals/learning_rate": agent.optimizer.param_groups[0]["lr"],
                },
                step=agent.timesteps_so_far,
            )

            time_spent_eval += time.time() - eval_start

            if start_time is not None:
                # compute the speed in steps per second
                speed = (
                    (agent.timesteps_so_far - measure_burnin) /
                    (time.time() - start_time - time_spent_eval)
                )
                desc = f"speed={speed: 4.4f} sps"
                pbar.set_description(desc)
                wandb.log(
                    {
                        "vitals/speed": speed,
                    },
                    step=agent.timesteps_so_far,
                )

        lr_updates += 1
        tlog.clear()

    # save once we are done
    agent.save(ckpt_dir, sfx="done")
    logger.info(f"we are done -- saved model @: {ckpt_dir}")
    # mark a run as finished, and finish uploading all data (from docs)
    wandb.finish()
    logger.warn("bye")


@beartype
def evaluate(cfg: DictConfig,
             env: Env,
             agent_wrapper: Callable[[], OnPolicyAgent],
             name: str):

    assert isinstance(cfg, DictConfig)

    trajectory_path = None
    if cfg.gather_trajectories:
        trajectory_path = Path(cfg.trajectory_dir) / name
        trajectory_path.mkdir(parents=True, exist_ok=True)

    agent = agent_wrapper()

    agent.load(cfg.load_ckpt)

    # create episode generator
    ep_gen = episode(
        env, agent, cfg.seed, need_lists=cfg.gather_trajectories)

    pbar = tqdm.tqdm(range(cfg.num_episodes))
    pbar.set_description("evaluating")

    len_list = []
    ret_list = []

    for i in pbar:

        ep = next(ep_gen)
        len_list.append(ep_len := ep["length"])
        ret_list.append(ep_ret := ep["return"])

        if trajectory_path is not None:
            # verify that all the arrays have the same length
            for k, v in ep.items():
                assert v.shape[0] == ep_len, f"wrong array length for {k=}"
            name = f"{str(i).zfill(3)}_L{int(ep_len)}_R{int(ep_ret)}"
            td = TensorDict(ep)
            for k, v in td.items():
                if v.dtype == torch.float64:
                    # convert from float64 to float32
                    td[k] = v.float()
            fname = trajectory_path / f"{name}.h5"
            td.to_h5(fname)  # can then easily load with `from_h5`

    if trajectory_path is not None:
        logger.warn(f"saved trajectories @: {trajectory_path}")

    with torch.no_grad():
        eval_metrics = {
            "length": torch.tensor(np.array(len_list), dtype=torch.float).mean(),
            "return": torch.tensor(np.array(ret_list), dtype=torch.float).mean(),
        }

    # log with logger
    for k, v in eval_metrics.items():
        logger.record_tabular(k, v.numpy())
    logger.dump_tabular()

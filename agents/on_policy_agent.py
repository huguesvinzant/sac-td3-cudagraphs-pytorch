import tempfile
from pathlib import Path
from typing import Optional, Union, Any

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
import wandb
import numpy as np
import torch
from torch.optim.adam import Adam
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from helpers import logger
from agents.nets import PPOActor, PPOCritic, log_module_info


class OnPolicyAgent(object):

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 device: torch.device,
                 hps: DictConfig,
                 rb: Optional[ReplayBuffer] = None):
        ob_shape, ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]

        self.device = device

        assert isinstance(hps, DictConfig)
        self.hps = hps

        self.timesteps_so_far = 0
        self.updates_so_far = 0

        self.best_eval_ep_ret = -float("inf")  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay buffer
        self.rb = rb

        # create actor nets

        actor_net_args = [ob_shape, ac_shape, (64, 64)]
        actor_net_kwargs = {"layer_norm": self.hps.layer_norm}
        self.actor = PPOActor(*actor_net_args, **actor_net_kwargs, device=self.device)
        
        # create critic net

        qnet_net_args = [ob_shape, (64, 64)]
        qnet_net_kwargs = {"layer_norm": self.hps.layer_norm}
        self.qnet = PPOCritic(*qnet_net_args, **qnet_net_kwargs, device=self.device)

        # set up the optimizers

        self.optimizer = Adam(
            list(self.qnet.parameters()) + list(self.actor.parameters()),
            lr=self.hps.lr,
            eps=1e-5,
            capturable=self.hps.cudagraphs and not self.hps.compile,
        )

        # log module architectures
        log_module_info(self.actor)
        log_module_info(self.qnet)

    @beartype
    def predict(self, in_td: TensorDict, action: torch.Tensor = None) -> np.ndarray:
        """Predict with policy"""
        obs = in_td["observations"]
        out_actions, log_prob, entropy = self.actor(obs, action)
        q_values = self.qnet(obs)
        return out_actions, log_prob, entropy, q_values

    @beartype
    def save(self, path: Path, sfx: Optional[str] = None):
        """Save the agent to disk and wandb servers"""
        # prep checkpoint
        fname = (f"ckpt_{sfx}"
                 if sfx is not None
                 else f".ckpt_{self.timesteps_so_far}ts")
        # design choice: hide the ckpt saved without an extra qualifier
        path = (parent := path) / f"{fname}.pth"
        checkpoint = {
            "hps": self.hps,  # handy for archeology
            "timesteps_so_far": self.timesteps_so_far,
            # and now the state_dict objects
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # save checkpoint to filesystem
        torch.save(checkpoint, path)
        logger.info(f"{sfx} model saved to disk")
        if sfx == "best":
            # upload the model to wandb servers
            wandb.save(str(path), base_path=parent)
            logger.info("model saved to wandb")

    @beartype
    def load_from_disk(self, path: Path):
        """Load another agent into this one"""
        checkpoint = torch.load(path)
        if "timesteps_so_far" in checkpoint:
            self.timesteps_so_far = checkpoint["timesteps_so_far"]
        # the "strict" argument of `load_state_dict` is True by default
        self.actor.load_state_dict(checkpoint["actor"])
        self.qnet1.load_state_dict(checkpoint["qnet1"])
        self.qnet2.load_state_dict(checkpoint["qnet2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

    @staticmethod
    @beartype
    def compare_dictconfigs(
        dictconfig1: DictConfig,
        dictconfig2: DictConfig,
    ) -> dict[str, dict[str, Union[str, int, list[int], dict[str, Union[str, int, list[int]]]]]]:
        """Compare two DictConfig objects of depth=1 and return the differences.
        Returns a dictionary with keys "added", "removed", and "changed".
        """
        differences = {"added": {}, "removed": {}, "changed": {}}

        keys1 = set(dictconfig1.keys())
        keys2 = set(dictconfig2.keys())

        # added keys
        for key in keys2 - keys1:
            differences["added"][key] = dictconfig2[key]

        # removed keys
        for key in keys1 - keys2:
            differences["removed"][key] = dictconfig1[key]

        # changed keys
        for key in keys1 & keys2:
            if dictconfig1[key] != dictconfig2[key]:
                differences["changed"][key] = {
                    "from": dictconfig1[key], "to": dictconfig2[key]}

        return differences

    @beartype
    def load(self, wandb_run_path: str, model_name: str = "ckpt_best.pth"):
        """Download a model from wandb and load it"""
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # compare the current cfg with the cfg of the loaded model
        wandb_cfg_dict: dict[str, Any] = run.config
        wandb_cfg: DictConfig = OmegaConf.create(wandb_cfg_dict)
        a, r, c = self.compare_dictconfigs(wandb_cfg, self.hps).values()
        # N.B.: in Python 3.7 and later, dicts preserve the insertion order
        logger.warn(f"added  : {a}")
        logger.warn(f"removed: {r}")
        logger.warn(f"changed: {c}")
        # create a temporary directory to download to
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file = run.file(model_name)
            # download the model file from wandb servers
            file.download(root=tmp_dir_name, replace=True)
            logger.warn("model downloaded from wandb to disk")
            tmp_file_path = Path(tmp_dir_name) / model_name
            # load the agent stored in this file
            self.load_from_disk(tmp_file_path)
            logger.warn("model loaded")

    @beartype
    def advantage_estimation(self):
        """Compute the advantage estimation and store it in the replay buffer"""
        envs_indices = [list(range(k, k + self.hps.num_envs)) for k in range(self.hps.segment_len * self.hps.num_envs - self.hps.num_envs, -1, -self.hps.num_envs)]
        nextvalue = self.qnet(self.rb["next_observations"][-1]).reshape(-1, 1)
        if self.hps.gae:
            advantages = torch.zeros_like(self.rb["rewards"])
            lastgaelam = 0
            for t in envs_indices:
                nextnonterminal = ~self.rb["terminations"][t]
                delta = self.rb["rewards"][t] + self.hps.gamma * nextvalue * nextnonterminal - self.rb["values"][t]
                advantages[t] = lastgaelam = delta + self.hps.gamma * self.hps.gae_lambda * nextnonterminal * lastgaelam
                nextvalue = self.rb["values"][t]
            returns = advantages + self.rb["values"]
        else:
            returns = torch.zeros_like(self.rb["rewards"])
            for t in envs_indices:
                nextnonterminal = ~self.rb["terminations"][t]
                next_return = nextvalue
                returns[t] = self.rb["rewards"][t] + self.hps.gamma * nextnonterminal * next_return
                nextvalue = returns[t]
            advantages = returns - self.rb["values"]

        self.rb.extend(
            TensorDict(
                {
                    "advantages": advantages,
                    "returns": returns,
                },
                batch_size=advantages.shape[0],
                device=self.device,
            )
        )

    @beartype
    def update_nets(self, batch: TensorDict) -> TensorDict:
        
        _, newlogprobs, entropies, newvalue = self.predict(batch, batch["actions"])
        logratio = newlogprobs - batch["logprobs"]
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()

        # advantage normalization
        mb_advantages = batch["advantages"]
        mb_advantages_mean = mb_advantages.mean()
        if self.hps.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # policy loss (clipped objective)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.hps.clip_coef, 1 + self.hps.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # value loss clipping
        mb_returns = batch["returns"]
        if self.hps.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = batch["values"] + torch.clamp(
                newvalue - batch["values"],
                -self.hps.clip_coef,
                self.hps.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        # entropy loss
        entropy_loss = entropies.mean()
        loss = pg_loss - self.hps.ent_coef * entropy_loss + v_loss * self.hps.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hps.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.hps.max_grad_norm)
        self.optimizer.step()
        self.updates_so_far += 1

        return TensorDict(
            {
                "debug/mean_advantage": mb_advantages_mean,
                "debug/mean_return": mb_returns.mean(),
                "losses/policy_loss": pg_loss.detach(),
                "losses/value_loss": v_loss.detach(),
                "losses/entropy_loss": entropy_loss.detach(),
                "losses/loss": loss.detach(),
                "losses/old_approx_kl": old_approx_kl.detach(),
                "losses/approx_kl": approx_kl.detach(),
            },
        )

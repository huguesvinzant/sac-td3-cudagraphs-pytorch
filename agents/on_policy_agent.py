import tempfile
from pathlib import Path
from typing import Optional, Union, Any

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
import wandb
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.nn import functional as ff
from torch.nn.utils import clip_grad
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
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
        self.actor_updates_so_far = 0
        self.qnet_updates_so_far = 0

        self.best_eval_ep_ret = -float("inf")  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay buffer
        self.rb = rb

        # create online and target nets

        actor_net_args = [ob_shape, ac_shape, (64, 64)]
        actor_net_kwargs = {"layer_norm": self.hps.layer_norm}

        self.actor = PPOActor(*actor_net_args, **actor_net_kwargs, device=self.device)
        self.actor_params = TensorDict.from_module(self.actor, as_module=True)
        self.actor_target = self.actor_params.data.clone()

        # discard params of net
        self.actor = PPOActor(*actor_net_args, **actor_net_kwargs, device="meta")
        self.actor_params.to_module(self.actor)
        # self.actor_detach = PPOActor(*actor_net_args, **actor_net_kwargs, device=self.device)

        # # copy params to actor_detach without grad
        # TensorDict.from_module(self.actor).data.to_module(self.actor_detach)

        qnet_net_args = [ob_shape, (64, 64)]
        qnet_net_kwargs = {"layer_norm": self.hps.layer_norm}

        self.qnet = PPOCritic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        self.qnet_params = TensorDict.from_module(self.qnet, as_module=True)
        self.qnet_target = self.qnet_params.data.clone()

        # discard params of net
        self.qnet = PPOCritic(*qnet_net_args, **qnet_net_kwargs, device="meta")
        self.qnet_params.to_module(self.qnet)

        # set up the optimizers

        self.optimizer = Adam(
            self.qnet.parameters(),
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
        out_actions, log_probs, ent = self.actor(obs, action)
        q_values = self.qnet(obs)
        return out_actions, log_probs, ent, q_values

        if (self.hps.prefer_td3_over_sac or (
            self.qnet_updates_so_far % self.hps.crit_targ_update_freq == 0)):

            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y

            self.qnet_target.lerp_(self.qnet_params.data, self.hps.polyak)
            if self.hps.prefer_td3_over_sac:
                # using TD3 (SAC does not use a target actor)
                self.actor_target.lerp_(self.actor_params.data, self.hps.polyak)

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

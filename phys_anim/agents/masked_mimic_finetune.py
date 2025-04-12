from phys_anim.agents.mimic_vae_dagger import MimicVAEDagger
from phys_anim.envs.humanoid.common import Humanoid
from phys_anim.agents.models.actor import ActorFixedSigmaVAE

import torch
from torch import Tensor
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule

from typing import Tuple, Dict, Optional, Union


class MimicFinetune(MimicVAEDagger):  # TODO inherit from PPO
    env: Humanoid
    actor: ActorFixedSigmaVAE

    def __init__(self, fabric: Fabric, env: Humanoid, config):
        super().__init__(fabric, env, config)

    def setup(self):
        super().setup()
        self.config.freeze_actor = not getattr(self.config, "dont_freeze_actor", False)
        actor_state_dict = self.actor.state_dict()
        if self.config.pre_trained_maskedmimic_path is not None:
            pre_trained_masked_mimic_state_dict = torch.load(
                self.config.pre_trained_maskedmimic_path, map_location=self.device
            )
            pre_trained_actor_state_dict = pre_trained_masked_mimic_state_dict['actor']
            for param_name, param_val in self.actor.state_dict().items():
                pre_trained_param_val = pre_trained_actor_state_dict.get(param_name)
                if pre_trained_param_val is not None:
                    if param_val.shape == pre_trained_param_val.shape:
                        actor_state_dict[param_name] = pre_trained_param_val
                    else:
                        print(f"Shape mismatch: {param_name}")
                        print(f"Actor shape: {param_val.shape}, Pre-trained shape: {pre_trained_param_val.shape}")

            self.actor.load_state_dict(actor_state_dict,
                                       strict=False)  # strict=False to allow loading partial state_dict
            for name, param in self.actor.named_parameters():
                fixed_name = name.replace("_forward_module.", "")
                if fixed_name in pre_trained_actor_state_dict and self.config.freeze_actor:
                    param.requires_grad = False

        print_param_summary(self.actor)

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        if not self.actor.training:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            if getattr(self.env, "_current_successes", None) is not None:
                success_mask = self.env._current_successes.to(bool)
            else:
                success_mask = torch.zeros_like(all_env_ids)
            end_episode_mask = self.env.progress_buf == self.env.config.max_episode_length - 1
            reset_ids = all_env_ids[success_mask | end_episode_mask]
            self.env.reset_envs(reset_ids)
        self.eval()
        results = getattr(self.env, 'results')
        if len(results) > 0:
            log_dict = results
            success_rate = results["reach_success"]
        else:
            log_dict = {}
            success_rate = None
        return log_dict, success_rate


def print_param_summary(model: Union[torch.nn.Module, _FabricModule]) -> None:
    # print the number of trainable parameters, and the number of total parameters
    # format them in a human-readable way
    if isinstance(model, _FabricModule):
        model = model.module

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100
    print(f"Total parameters: {human_readable_count(total_params)}")
    print(f"Trainable parameters: {human_readable_count(trainable_params)} ({trainable_percentage:.2f}%)")


# Helper function to format large numbers with K, M, G
def human_readable_count(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)

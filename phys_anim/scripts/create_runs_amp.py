import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    debug: bool = False
    envs: List[str] = field(default_factory=lambda: ["steering", "direction_facing", "strike"])
    use_disable_discriminator: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class WandbConfig:
    entity: str = "task_tokens"
    project: str = "amp"


@dataclass
class Config:
    training: RunConfig = RunConfig()
    wandb: WandbConfig = WandbConfig()


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    debug = cfg.training.debug
    output_file_path = "amp_runs.sh" if not debug else "amp_debug_runs.sh"

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    base_run_command = "python phys_anim/train_agent.py +robot=smpl +backbone=isaacgym"
    extra_args = []
    max_epochs = 20 if debug else 4000

    if not debug:
        base_run_command += " seed=${seed}"
        extra_args += [f"wandb.wandb_entity={cfg.wandb.entity} wandb.wandb_project={cfg.wandb.project}"]
    else:
        base_run_command += " auto_load_latest=False"

    opts = ["small_run", "wdb"] if debug else ["full_run", "wdb", "combined_callbacks"]

    for env in cfg.training.envs:
        for disable_discriminator in cfg.training.use_disable_discriminator:
            current_opts = opts.copy()
            current_extra_args = extra_args.copy()
            current_run_command = ""
            current_run_command += base_run_command
            current_run_command += f' +exp=amp_inversion/{env}'
            current_experiment_name = f"{env}_disable_discriminator_{disable_discriminator}"
            # add with ++ all the flags for easier filtering
            if disable_discriminator:
                algo_type = "PureRL"
            else:
                algo_type = "AMP"
            current_run_command += f" ++algo_type={algo_type}"
            if debug:
                current_experiment_name += '_DEBUG'

            current_extra_args += [f"algo.config.max_epochs={max_epochs}"]
            current_run_command += (
                    f" experiment_name={current_experiment_name}" + "_${seed}" +
                    f" ++clean_exp_name={algo_type}_{current_experiment_name}"
            )

            if disable_discriminator:
                current_opts += ["disable_discriminator", "disable_discriminator_weights"]

            opt_string = "+opt=[" + ','.join(current_opts) + "]"
            extra_args_string = ' '.join(current_extra_args)
            current_run_command = " ".join(
                [current_run_command, opt_string, extra_args_string, ]
            )

            with open(output_file_path, "a") as f:
                f.write(current_run_command + "\n")


if __name__ == "__main__":
    main()

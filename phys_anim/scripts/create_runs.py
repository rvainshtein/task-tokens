import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    debug: bool = False
    envs: List[str] = field(default_factory=lambda: ["steering", "direction_facing"])
    use_hand_crafted_prior: List[bool] = field(default_factory=lambda: [True, False])
    use_text: List[bool] = field(default_factory=lambda: [False])
    use_current_pose_obs: List[bool] = field(default_factory=lambda: [True])
    use_bigger_model: List[bool] = field(default_factory=lambda: [True])
    train_actor: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class WandbConfig:
    entity: str = "task_tokens"
    project: str = "all_runs"


@dataclass
class Config:
    training: RunConfig = RunConfig()
    wandb: WandbConfig = WandbConfig()


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    debug = cfg.training.debug
    output_file_path = "runs.sh" if not debug else "debug_runs.sh"

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

    # create runs for each combination of hyperparameters
    for env in cfg.training.envs:
        for prior_flag in cfg.training.use_hand_crafted_prior:
            for text_flag in cfg.training.use_text:
                for current_pose in cfg.training.use_current_pose_obs:
                    for bigger_model in cfg.training.use_bigger_model:
                        for train_actor_flag in cfg.training.train_actor:
                            current_opts = opts.copy()
                            current_extra_args = extra_args.copy()
                            current_run_command = ""
                            current_run_command += base_run_command
                            current_run_command += f' +exp=inversion/{env}'
                            current_experiment_name = (
                                f"{env}_prior_{prior_flag}_text_{text_flag}_current_pose_{current_pose}_bigger_{bigger_model}_train_actor_{train_actor_flag}"
                            )
                            # add with ++ all the flags for easier filtering

                            current_run_command += (
                                f" ++prior={prior_flag} ++text={text_flag} ++current_pose={current_pose} ++bigger={bigger_model}"
                            )
                            if train_actor_flag:
                                algo_type = "MaskedMimic_Inversion"
                            else:
                                algo_type = "MaskedMimic_Finetune"
                            current_run_command += f" ++algo_type={algo_type}"
                            if debug:
                                current_experiment_name += '_DEBUG'
                            current_extra_args += [f"algo.config.max_epochs={max_epochs}"]
                            current_run_command += (
                                    f" experiment_name={current_experiment_name}" + "_${seed}" +
                                    f" ++clean_exp_name={algo_type}_{current_experiment_name}"
                            )
                            current_extra_args += [f"env.config.use_hand_crafted_prior={prior_flag}"]
                            current_extra_args += [f"env.config.use_text={text_flag}"]
                            if current_pose:
                                current_opts += ["masked_mimic/inversion/current_pose_obs"]
                            if bigger_model:
                                current_extra_args += [
                                    "algo.config.models.extra_input_model_for_transformer.config.units=[512,512,512]"]
                            if train_actor_flag:
                                current_opts += ["masked_mimic/inversion/train_actor"]

                            opt_string = "+opt=[" + ','.join(current_opts) + "]"
                            extra_args_string = ' '.join(current_extra_args)
                            current_run_command = " ".join(
                                [current_run_command, opt_string, extra_args_string, ]
                            )

                            with open(output_file_path, "a") as f:
                                f.write(current_run_command + "\n")


if __name__ == "__main__":
    main()

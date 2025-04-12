import glob
import subprocess
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax

from phys_anim.eval_agent import resolve_config_path


@dataclass
class WandbConfig:
    entity: str = "task_tokens"
    project: str = "eval_results"


@dataclass
class PerturbationsConfig:
    gravity_z: float = field(default=-9.81)
    friction: float = field(default=1.0)
    complex_terrain: bool = field(default=False)
    mass_multiplier: dict = field(default_factory=lambda: {})


@dataclass
class EvalConfig:
    checkpoint_paths: List[str] = field(default_factory=lambda: ["results/long_jump_pose/last.ckpt"])
    checkpoint_paths_glob: str = field(default="")  # Use this to override and glob paths dynamically
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    more_options: str = field(default="")
    log_eval_results: bool = field(default=True)
    wandb: WandbConfig = WandbConfig()
    opt: List[str] = field(default_factory=lambda: ["wdb"])
    num_envs: int = field(default=1024)
    games_per_env: int = field(default=5)
    prior_only: bool = field(default=False)
    use_perturbations: bool = field(default=False)
    perturbations: PerturbationsConfig = field(default_factory=lambda: PerturbationsConfig())
    record_video: bool = field(default=False)
    record_dir: str = field(default="output/eval_videos")
    termination: bool = field(default=False)


def build_command(config: DictConfig, checkpoint: Path, gpu_id: int, base_dir: Path):
    opt = config.opt
    more_options = config.more_options

    if config.record_video:
        config.log_eval_results = False
        headless = False
    else:
        headless = True
    cmd = []
    cmd.extend(
        [
            f"python phys_anim/eval_agent.py +robot=smpl +backbone=isaacgym +headless={headless}",
            f"+checkpoint={checkpoint} +device={gpu_id}",
            f"+wandb.wandb_entity={config.wandb.entity} +wandb.wandb_project={config.wandb.project} +wandb.wandb_id=null",
            f"+opt=[{','.join(opt)}]",
            f"+env.config.log_output=False",
            f"+base_dir={base_dir}",
            f"++num_envs={config.num_envs} +algo.config.num_games={config.num_envs * config.games_per_env}",
            f"{more_options}",
        ]
    )
    if config.record_video:
        cmd.append(f"++algo.config.eval_callbacks.export_video_cb.config.record_dir={config.record_dir}")
    if config.termination:
        cmd.append("++env.config.enable_height_termination=True")
        if 'strike' in str(checkpoint):
            cmd.append("++env.config.enable_success_termination=True")
    if config.log_eval_results:
        cmd.append("++algo.config.log_eval_results=True")
    cmd.append(f"++use_perturbations={config.use_perturbations}")
    if config.use_perturbations:
        for key, value in config.perturbations.items():
            if key == "complex_terrain":
                if value is True:
                    cmd.extend(["+terrain=complex",
                                # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping,
                                # poles, flat]
                                "+terrain.config.terrain_proportions=[0.3,0.25,0.2,0.2,0.05,0.,0.,0.]"])

            else:
                cmd.append(f"++env.config.perturbations.{key}={value}")
    return cmd


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: DictConfig):
    console = Console()

    # Merge default config with CLI arguments
    default_cfg = OmegaConf.structured(EvalConfig())
    config = OmegaConf.merge(default_cfg, config)

    # Resolve checkpoint paths
    if config.checkpoint_paths_glob:
        checkpoint_paths = glob.glob(config.checkpoint_paths_glob, recursive=False)
    else:
        checkpoint_paths = config.checkpoint_paths

    gpu_ids = config.gpu_ids
    gpu_cycle = cycle(gpu_ids) if len(gpu_ids) > 1 else None

    if config.record_video:
        config.opt.append("record_video")
        config.opt = [opt for opt in config.opt if opt != "wdb"]
    if config.prior_only:
        print("Using prior only, make sure checkpoint is an inversion model.")
        config.opt.append("masked_mimic/inversion/disable_inversion_obs")
        config.more_options += (
            " +env.config.use_hand_crafted_prior=True"
            " +env.config.prior_only=True"
            " ++algo_type=MaskedMimic_Prior_Only"
            " ++bigger=null"
            " ++current_pose=null"
            " ++prior=True"
        )
    processes = []

    for checkpoint in checkpoint_paths:
        checkpoint = Path(checkpoint).resolve()
        base_dir = resolve_config_path(checkpoint)[0].parent.parent
        gpu_id = next(gpu_cycle) if gpu_cycle else gpu_ids[0]
        cmd = build_command(config, checkpoint, gpu_id, base_dir)

        cmd_print = Syntax(' '.join(cmd), "bash", theme="monokai", line_numbers=False, word_wrap=True)

        if len(gpu_ids) == 1:
            console.print(f"[bold blue]Running sequentially on GPU {gpu_id}:[/bold blue]")
            console.print(cmd_print)
            subprocess.run(' '.join(cmd), shell=True)
        else:
            while len(processes) >= len(gpu_ids):
                processes = [p for p in processes if p.poll() is None]  # Remove finished processes
                time.sleep(1)
            console.print(f"[bold blue]Running on GPU {gpu_id}:[/bold blue]")
            console.print(cmd_print)
            processes.append(subprocess.Popen(' '.join(cmd), shell=True))

    # Wait for all remaining processes to finish
    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()

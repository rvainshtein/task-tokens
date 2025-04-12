from pathlib import Path

import lightning as L

from phys_anim.agents.ppo import PPO
from phys_anim.envs.humanoid.common import BaseHumanoid


class SaveBestModelCallback:
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self, warmup: int = 100):
        """
        Callback to save the model when rewards exceed the previous best.
        """
        self.best_reward = float('-inf')
        self.warmup = warmup

    def after_train(self, training_loop: PPO):
        """
        Check and save the model at the end of each epoch if the reward is higher.

        Args:
            training_loop (PPO): The trainer instance.
        """
        if training_loop.current_epoch < self.warmup:
            # Skip the first few epochs
            return

        current_reward = training_loop.experience_buffer.total_rewards.mean().item()
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            print(f"New best reward: {self.best_reward}. Saving model...")

            # Save the model to the specified directory
            ckpt_name = "best_model.ckpt"
            training_loop.save(name=ckpt_name)  # the path is None, so it will save to the default directory
            # print the version of the run (model path save folder)
            print(f"Model saved to: {Path(training_loop.fabric.loggers[0].log_dir) / ckpt_name}")
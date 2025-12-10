import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class SaveOnBestEpLenCallback(BaseCallback):
    """
    Save the model when rollout/ep_len_mean reaches a new maximum.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_ep_len_model")
        self.best_ep_len = -np.inf

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            df_log = load_results(self.log_dir)

            l_log = df_log["l"].values
            mean_len = np.mean(l_log[-10:])
            if self.verbose >= 1:
                print(
                    f"Timesteps: {self.num_timesteps} | "
                    f"Best ep_len_mean: {self.best_ep_len:.2f} | "
                    f"Current ep_len_mean: {mean_len:.2f}"
                )

            # ✅ 최대값 갱신 시 저장
            if mean_len > self.best_ep_len:
                self.best_ep_len = mean_len
                if self.verbose >= 1:
                    print(f"✅ New best ep_len_mean! Saving model to {self.save_path}")
                self.model.save(self.save_path)

        return True

class EarlyStopping_by_avg():
    def __init__(self, patience=10, verbose=0):
        super().__init__()
        self.best_avg = 0
        self.step = 0
        self.patience = patience
        self.verbose = verbose

    def check(self, avg, avg_scores):
        if avg >= self.best_avg:
            self.best_avg = avg
            self.step = 0
        elif len(avg_scores) > 1 and avg > avg_scores[-2]:
            self.step = 0
        else:
            self.step += 1
            if self.step > self.patience:
                if self.verbose:
                    print('Early stopping !!!')
                return True
        return False


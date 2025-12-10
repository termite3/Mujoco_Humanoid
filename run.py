import os, time
import gymnasium as gym
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from callback import SaveOnBestTrainingRewardCallback

# def train(env_id, log_base_dir="logs", model_base_dir="models", model_name=None, total_timesteps=100000):
#     """
#     Train a PPO agent on the specified environment.
    
#     Args:
#         env_id (str): Environment ID to train on
#         log_base_dir (str): Base directory for logs
#         model_base_dir (str): Base directory for saving models
#         model_name (str): Name for the model file
#         total_timesteps (int): Total timesteps for training
#     """   
#     # Create log directory and model directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     log_dir = os.path.join(script_dir, log_base_dir, env_id)
#     model_dir = os.path.join(script_dir, model_base_dir)
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
    
#     # # Environment
#     # n_envs = 4
#     # env = make_vec_env(env_id, n_envs=n_envs)
#     # env = VecMonitor(env, log_dir)

#     n_envs = 64

#     def make_env():
#         def _init():
#             env = gym.make(env_id)
#             env = Monitor(env)   # ✅ 이게 핵심
#             return env
#         return _init

#     env = DummyVecEnv([make_env() for _ in range(n_envs)])

#     # Agent Model
#     if model_name is None:    
#         model_name = env_id + "_PPO"
    
#     policy_kwargs = {
#         'log_std_init':-2,
#         'ortho_init': False,
#     }
    
#     model = PPO(
#         policy="MlpPolicy",
#         env=env,
#         # -------------------------------------------------
#         learning_rate=1e-3,
#         n_steps = 1024,
#         gamma=0.9,
#         gae_lambda=0.95,
#         batch_size=64,
#         n_epochs=10,
#         ent_coef=0.0,
#         clip_range=0.2,
#         clip_range_vf=None,
#         normalize_advantage=True,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         use_sde=True,
#         sde_sample_freq=4,
#         rollout_buffer_class=None,
#         rollout_buffer_kwargs=None,
#         target_kl=None,
#         stats_window_size=100,
#         policy_kwargs=policy_kwargs,
#         # -------------------------------------------------
#         tensorboard_log = log_dir,
#         verbose=1,
#         seed=None,
#         device='auto',
#     )


#     # Train
#     callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)   
#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=callback,
#         log_interval=4,
#         tb_log_name="PPO",
#         reset_num_timesteps=True,
#         progress_bar=True,
#     )

#     # Save the trained model
#     save_path = os.path.join(model_dir, model_name)
#     model.save(save_path)

#     # close the environment
#     env.close()


def run(env_id, model_base_dir="models", model_name=None, n_episodes=1, total_timesteps=100000):
    """
    Run a trained PPO agent on the specified environment.
    
    Args:
        env_id (str): Environment ID to run on
        model_base_dir (str): Base directory for loading models
        model_name (str): Name of the model file
        n_episodes (int): Number of episodes to run
    """
    # Environment
    env = gym.make(env_id,
                   xml_file='humanoid.xml',
                   forward_reward_weight=1.0,
                   ctrl_cost_weight=0.1,
                   contact_cost_weight=5e-7,
                   healthy_reward=5.0,
                   terminate_when_unhealthy=True,
                   healthy_z_range=(1.0, 2.0),
                   reset_noise_scale=1e-2,
                   exclude_current_positions_from_observation=True,
                   render_mode='human'
                   )
    
    # Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name is None:
        model_name = env_id + "_PPO"
    # model_path = os.path.join(script_dir, model_base_dir, model_name)
    # model_path = './models/Humanoid-v5_PPO_100epo_x10000step_20251209_140044.zip'
    model_path = './models/Humanoid-v5_PPO_10epo_x5000000step_[256, 256]arch_20251209_171808_1000000step_[256, 256]arch_-3.0exp_20251210_104817'
    model = PPO.load(model_path, env)

    # Run    
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        while True:
            action, state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
            episode_reward += reward
                
            if terminated or truncated:
                time.sleep(3.0)
                print(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}")
                episode_reward = 0
                break
            
    # close the environment
    env.close()


if __name__ == "__main__":
    env_id = "Humanoid-v5"
    
    # train(env_id)
    run(env_id)
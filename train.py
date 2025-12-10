import os, time, datetime, math
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

from callback import SaveOnBestEpLenCallback, SaveOnBestTrainingRewardCallback

def train(env_id, log_base_dir="logs", model_base_dir="models", model_name="PPO", total_timesteps=5_000_000):
    """
    Train a PPO agent on the specified environment.
    
    Args:
        env_id (str): Environment ID to train on
        log_base_dir (str): Base directory for logs
        model_base_dir (str): Base directory for saving models
        model_name (str): Name for the model file
        total_timesteps (int): Total timesteps for training
    """   

    # Create log directory and model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, log_base_dir, env_id)
    model_dir = os.path.join(script_dir, model_base_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def cosine_schedule(initial_lr):
        def func(progress_remaining):
            return initial_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return func

    n_envs = 32
    n_steps = 2048
    n_epochs = 10
    batch_size = n_envs * n_steps
    total_timesteps = 5_000_000
    net_arch=[256, 256]

    log_std_init=-0.5
    gamma=0.9999
    target_kl=0.01
    LR = 5e-4
    learning_rate=cosine_schedule(LR)
    healthy_reward=5.0

    print(f"n_envs: {n_envs}, n_steps: {n_steps}, n_epochs: {n_epochs}, batch_size: {batch_size}, total_timesteps: {total_timesteps}")


    def make_env():
        def _init():
            # Environment
            env = gym.make(env_id,
                        xml_file='humanoid.xml',
                        forward_reward_weight=1.0,
                        ctrl_cost_weight=0.01,
                        contact_cost_weight=5e-7,
                        healthy_reward=healthy_reward,
                        terminate_when_unhealthy=True,
                        healthy_z_range=(1.0, 2.0),
                        reset_noise_scale=1e-2,
                        exclude_current_positions_from_observation=True,
                        )
            env = Monitor(env)   # ✅ 이게 핵심
            return env
        return _init

    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    env = VecMonitor(env, log_dir)

    # ✅ 타임스탬프 생성 (YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ✅ "n_epochs*n_steps, 생성시간" 형식으로 모델명 생성
    model_name = f"{model_name}_{total_timesteps}step_{net_arch}arch_{learning_rate.__qualname__.split('.')[2]}sche_{LR}LR_gamma{gamma}_{timestamp}"


    # Agent Model
    if model_name is None:    
        model_name = env_id + "_PPO"
    

    policy_kwargs = dict(
        net_arch=net_arch,
        log_std_init=log_std_init,
        ortho_init=False,
        )
    
    # Pre-Train
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # -------------------------------------------------
        learning_rate=learning_rate,
        n_steps = n_steps,
        gamma=gamma,
        gae_lambda=0.95,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=0.01,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=target_kl,
        stats_window_size=100,
        policy_kwargs=policy_kwargs,
        # -------------------------------------------------
        tensorboard_log = log_dir,
        verbose=1,
        seed=None,
        device='auto',
    )

    # # Post-Train: Load pre-trained model
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # model_name = "PPO_5000000step_[256, 256]arch_cosine_schedulesche_0.0005LR_20251210_113305_gamma0.9999"
    # if model_name is None:
    #     model_name = env_id + "_PPO"
    # model_path = os.path.join(script_dir, model_base_dir, model_name)
    # model = PPO.load(model_path, env)

    # Train
    callback = SaveOnBestEpLenCallback(check_freq=1000, log_dir=log_dir)   
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # Retrieve training reward
    df_log = load_results(log_dir)

    l_log = df_log["l"].values
    mean_len = np.mean(l_log[-10:])

    lifespan = f"span_{mean_len:.0f}"

    # Save the trained model
    save_path = os.path.join(model_dir, model_name, lifespan)
    model.save(save_path)

    # close the environment
    env.close()


# def run(env_id, model_base_dir="models", model_name=None, n_episodes=5):
#     """
#     Run a trained PPO agent on the specified environment.
    
#     Args:
#         env_id (str): Environment ID to run on
#         model_base_dir (str): Base directory for loading models
#         model_name (str): Name of the model file
#         n_episodes (int): Number of episodes to run
#     """
#     # Environment
#     env = gym.make(env_id,
#                    xml_file='humanoid.xml',
#                    forward_reward_weight=1.0,
#                    ctrl_cost_weight=0.1,
#                    contact_cost_weight=5e-7,
#                    healthy_reward=5.0,
#                    terminate_when_unhealthy=True,
#                    healthy_z_range=(1.0, 2.0),
#                    reset_noise_scale=1e-2,
#                    exclude_current_positions_from_observation=True,
#                    )
    
#     # Model
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     if model_name is None:
#         model_name = env_id + "_PPO"
#     model_path = os.path.join(script_dir, model_base_dir, model_name)
#     model = PPO.load(model_path, env)

#     # Run    
#     for episode in range(n_episodes):
#         obs, info = env.reset()

#         episode_reward = 0
#         while True:
#             action, state = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             time.sleep(0.01)
#             episode_reward += reward
                
#             if terminated or truncated:
#                 time.sleep(1.0)
#                 print(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}")
#                 episode_reward = 0
#                 break
            
#     # close the environment
#     env.close()


if __name__ == "__main__":
    env_id = "Humanoid-v5"
    
    train(env_id)
    # run(env_id)
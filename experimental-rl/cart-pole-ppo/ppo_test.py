import os
import sys
import time

import gymnasium as gym
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SAVE_BEST = "ppo_cartpole.pth"
NUM_EPISODES = 5
STEP_DELAY = 0.02


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def forward(self, x):
        return self.actor(x), self.critic(x)


def load_model(state_dim: int, action_dim: int, weights_path: str) -> ActorCritic | None:
    if not os.path.exists(weights_path):
        print(f"错误：找不到模型权重文件 {weights_path}")
        return None

    model = ActorCritic(state_dim, action_dim)
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
    except FileNotFoundError:
        print(f"错误：找不到模型权重文件 {weights_path}")
        return None

    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_action(model: ActorCritic, state) -> int:
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(state_tensor)
        return int(torch.argmax(logits, dim=-1).item())


def run_episodes(env, model: ActorCritic, num_episodes: int) -> list[int]:
    steps_per_episode: list[int] = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = select_action(model, state)
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            time.sleep(STEP_DELAY)

        steps_per_episode.append(steps)
        print(f"Episode {episode}: survived {steps} steps")

    return steps_per_episode


def make_env(render_mode: str | None):
    return gym.make("CartPole-v1", render_mode=render_mode)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, SAVE_BEST)

    env = None
    render_mode = "human"
    try:
        env = make_env(render_mode)
    except Exception:
        print("注意：无法使用 render_mode=human，回退到 render_mode=None 进行无渲染测试。")
        render_mode = None
        env = make_env(render_mode)

    model = load_model(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        weights_path=weights_path,
    )
    if model is None:
        env.close()
        return

    print(f"Loaded {weights_path} (render_mode={render_mode!r})")
    run_episodes(env, model, NUM_EPISODES)
    env.close()


if __name__ == "__main__":
    main()

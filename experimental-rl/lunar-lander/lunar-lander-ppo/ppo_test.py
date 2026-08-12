"""Load discrete PPO and land LunarLander with human render."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

from ppo_train import ActorCritic, SAVE_BEST

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
NUM_EPISODES = 5
STEP_DELAY = 0.02


def load_model(env: gym.Env) -> ActorCritic:
    path = HERE / SAVE_BEST
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; run ppo_train.py first.")
    model = ActorCritic(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def select_action(model: ActorCritic, state) -> int:
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x)
        return int(torch.argmax(logits, dim=-1).item())


def main() -> None:
    try:
        env = gym.make("LunarLander-v3", render_mode="human")
        render_mode = "human"
    except Exception:
        print("注意：无法使用 render_mode=human，回退到无渲染。")
        env = gym.make("LunarLander-v3")
        render_mode = None

    model = load_model(env)
    print(f"Loaded {SAVE_BEST} (render_mode={render_mode!r})")

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = select_action(model, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if render_mode == "human":
                time.sleep(STEP_DELAY)
            done = bool(terminated or truncated)
        print(f"Episode {ep}: return={total:.1f}  steps={steps}")

    env.close()


if __name__ == "__main__":
    main()

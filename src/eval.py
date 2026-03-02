import argparse
import importlib.util
import json
import os
import numpy as np
import torch
import gymnasium as gym
import imageio
from tqdm.auto import tqdm

import sys
sys.path.extend(["..", "."])
from reinforce import NeuralPolicy


@torch.no_grad()
def evaluate_policy(env, policy, n_episodes=10, device='cpu', deterministic=False, gif_path=None):
    """Evaluate policy over n_episodes, return summary statistics."""
    returns = []
    saved_gif = False
    last_frames = None

    render_env = None
    if gif_path is not None:
        render_env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")

    for ep in tqdm(range(n_episodes), desc="Evaluate Episodes..."):
        obs, _ = env.reset()
        total, terminated, truncated = 0.0, False, False
        frames = []

        # Mirror episode in render_env using the same actions
        if render_env is not None and not saved_gif:
            render_env.reset()

        while not (terminated or truncated):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            if deterministic:
                action = policy.get_action_deterministic(obs_t).item()
            else:
                action = policy.sample_action(obs_t).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward

            if render_env is not None and not saved_gif:
                render_env.step(action)
                frames.append(render_env.render())

        returns.append(total)

        if render_env is not None and not saved_gif:
            last_frames = frames
            if total >= 200:
                imageio.mimsave(gif_path, last_frames, fps=30)
                print(f"GIF saved (successful, ep {ep+1}, reward={total:.1f}): {gif_path}")
                saved_gif = True

    if render_env is not None:
        if not saved_gif and last_frames:
            imageio.mimsave(gif_path, last_frames, fps=30)
            print(f"GIF saved (last episode, reward={returns[-1]:.1f}): {gif_path}")
        render_env.close()

    return {
        'mean': float(np.mean(returns)),
        'std':  float(np.std(returns)),
        'min':  float(np.min(returns)),
        'max':  float(np.max(returns)),
    }


def _load_policy(checkpoint_path: str, device: torch.device):

    raw = torch.load(checkpoint_path, map_location='cpu')
    state_dict = raw['policy'] if 'policy' in raw else raw

    n_features  = state_dict['network.0.weight'].shape[1]
    hidden_size = state_dict['network.0.weight'].shape[0]
    n_actions   = state_dict['network.4.weight'].shape[0]

    policy = NeuralPolicy(n_features, n_actions, hidden_size).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved LunarLander policy")
    parser.add_argument("--checkpoint",   required=True,  type=str)
    parser.add_argument("--results-dir",  required=True,  type=str)
    parser.add_argument("--n-episodes",   type=int,   default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--gif",          type=str,   default=None, help="Path to save the GIF")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    env = gym.make("LunarLander-v3", continuous=False, render_mode=None, enable_wind=False)

    policy = _load_policy(args.checkpoint, device)
    print(f"Loaded policy from: {args.checkpoint}")

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path  = os.path.join(args.results_dir, f"{ckpt_stem}.json")

    gif_path = args.gif if args.gif else os.path.join(args.results_dir, f"{ckpt_stem}.gif")

    results = evaluate_policy(
        env, policy,
        n_episodes=args.n_episodes,
        device=device,
        deterministic=args.deterministic,
        gif_path=gif_path,
    )

    payload = {
        'checkpoint':   os.path.abspath(args.checkpoint),
        'n_episodes':   args.n_episodes,
        'deterministic': args.deterministic,
        'device':       str(device),
        'results':      results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to: {out_path}")

    env.close()

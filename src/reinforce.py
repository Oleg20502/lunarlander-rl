import csv
import os
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse

class NeuralPolicy(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.network(x)

    def get_dist(self, x):
        return Categorical(logits=self.forward(x))

    @torch.no_grad()
    def sample_action(self, x):
        return self.get_dist(x).sample()

    @torch.no_grad()
    def get_action_deterministic(self, x):
        return torch.argmax(self.forward(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, n_features, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

def compute_returns(rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    T = len(rewards)
    advantages, gae = [], 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values)]
    return returns, advantages

def train_reinforce(
    env_maker,
    n_iter=300,
    n_episodes=64,
    num_envs=64,
    alpha=3e-4,
    gamma=0.99,
    use_entropy=True,
    entropy_coef=0.01,
    baseline='gae_nn',      # 'none' | 'nn' | 'gae_no_nn' | 'gae_nn'
    baseline_lr=1e-3,
    use_advantage_norm=True,
    adv_eps=1e-8,
    gae_lambda=0.95,
    hidden_size=128,
    device='cpu',
    seed=None,
    exp_dir=None,
    save_every_n=50, 
):
    """
    REINFORCE with vectorized environments, entropy regularization,
    and advantage normalization.

    baseline
    ---
    'none'      — No baseline

    'nn'        — Neural network value function as baseline, no GAE.

    'gae_no_nn' — GAE without a neural network. V(t) is estimated as the per-timestep average of returns across all episodes in the batch

    'gae_nn'    — Neural network value function + GAE.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        metrics_path = os.path.join(exp_dir, "metrics.csv")
        _CSV_FIELDS = [
            'iteration', 'mean_return', 'best_return',
            'policy_loss', 'value_loss', 'entropy',
            'ep_return_std', 'ep_return_min', 'ep_return_max',
        ]
        with open(metrics_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    envs = gym.vector.SyncVectorEnv([env_maker for _ in range(num_envs)], copy=False)
    n_features = envs.single_observation_space.shape[0]
    n_actions  = envs.single_action_space.n

    policy    = NeuralPolicy(n_features, n_actions, hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=alpha)

    use_value_net = baseline in ('nn', 'gae_nn')
    if use_value_net:
        value_net      = ValueNetwork(n_features, hidden_size).to(device)
        value_optimizer = optim.Adam(value_net.parameters(), lr=baseline_lr)
    else:
        value_net      = None
        value_optimizer = None

    mean_returns     = []
    best_return      = -np.inf
    best_policy_state = None

    pbar = tqdm(range(n_iter), desc="Train Iter...")

    for iteration in pbar:
        all_obs        = []   # list of tensors [n_features]
        all_actions    = []   # list of ints
        all_returns    = []   # MC returns  OR  GAE target returns
        all_advantages = []   # filled for gae_nn (inline) and gae_no_nn (post-rollout)
        raw_episodes   = []   # list of raw reward lists; used by gae_no_nn post-rollout
        episode_returns = []

        obs, _ = envs.reset(seed=seed if iteration == 0 else None)
        obs = torch.FloatTensor(obs).to(device)

        buf_obs     = [[] for _ in range(num_envs)]
        buf_actions = [[] for _ in range(num_envs)]
        buf_rewards = [[] for _ in range(num_envs)]
        completed   = 0

        while completed < n_episodes:
            actions = policy.sample_action(obs)  # no_grad inside
            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())

            for i in range(num_envs):
                buf_obs[i].append(obs[i])
                buf_actions[i].append(int(actions[i].item()))
                buf_rewards[i].append(float(rewards[i]))

                if terminated[i] or truncated[i]:
                    ep_obs     = buf_obs[i]
                    ep_rewards = buf_rewards[i]

                    if baseline == 'gae_nn':
                        # Compute value per episode, then GAE
                        with torch.no_grad():
                            ep_values = value_net(torch.stack(ep_obs)).tolist()
                        ep_returns, ep_adv = compute_gae(ep_rewards, ep_values, gamma, gae_lambda)
                        all_advantages.extend(ep_adv)
                    else:
                        ep_returns = compute_returns(ep_rewards, gamma)
                        if baseline == 'gae_no_nn':
                            raw_episodes.append(ep_rewards[:])

                    all_obs.extend(ep_obs)
                    all_actions.extend(buf_actions[i])
                    all_returns.extend(ep_returns)
                    episode_returns.append(sum(ep_rewards))

                    buf_obs[i]     = []
                    buf_actions[i] = []
                    buf_rewards[i] = []
                    completed += 1

            obs = torch.FloatTensor(next_obs).to(device)

        if baseline == 'gae_no_nn':
            # Per-timestep average of returns across all episodes:
            timestep_sums   = {}
            timestep_counts = {}
            for ep_rewards in raw_episodes:
                for t, G in enumerate(compute_returns(ep_rewards, gamma)):
                    timestep_sums[t]   = timestep_sums.get(t, 0.0) + G
                    timestep_counts[t] = timestep_counts.get(t, 0) + 1
            v_by_t = {t: timestep_sums[t] / timestep_counts[t] for t in timestep_sums}

            for ep_rewards in raw_episodes:
                T         = len(ep_rewards)
                ep_values = [v_by_t[t] for t in range(T)]
                _, ep_adv = compute_gae(ep_rewards, ep_values, gamma, gae_lambda)
                all_advantages.extend(ep_adv)

        obs_t     = torch.stack(all_obs)                                    # [N, n_features]
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=device)  # [N]
        returns_t = torch.FloatTensor(all_returns).to(device)               # [N]

        # Re-compute log_probs and entropy with gradient tracking
        dist        = policy.get_dist(obs_t)
        log_probs_t = dist.log_prob(actions_t)
        entropies_t = dist.entropy()

        if baseline == 'gae_nn':
            advantages = torch.FloatTensor(all_advantages).to(device)
            # Second forward pass with gradient to compute value loss
            values     = value_net(obs_t)
            value_loss = F.mse_loss(values, returns_t)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        elif baseline == 'gae_no_nn':
            advantages = torch.FloatTensor(all_advantages).to(device)
            value_loss = torch.tensor(0.0)

        elif baseline == 'nn':
            values     = value_net(obs_t)
            advantages = returns_t - values.detach()
            value_loss = F.mse_loss(values, returns_t)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        else:
            advantages = returns_t
            value_loss = torch.tensor(0.0)

        if use_advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + adv_eps)

        policy_loss = -(log_probs_t * advantages).mean()
        if use_entropy:
            policy_loss = policy_loss - entropy_coef * entropies_t.mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        mean_ret = float(np.mean(episode_returns))
        if mean_ret > best_return:
            best_return       = mean_ret
            best_policy_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        mean_returns.append(mean_ret)

        pbar.set_postfix({
            'mean_ret': f"{mean_ret:.1f}",
            'best':     f"{best_return:.1f}",
            'π_loss':   f"{policy_loss.item():.3f}",
            'V_loss':   f"{value_loss.item():.3f}" if use_value_net else "N/A",
        })

        if exp_dir is not None:
            step_metrics = {
                'iteration':     iteration,
                'mean_return':   round(mean_ret, 4),
                'best_return':   round(best_return, 4),
                'policy_loss':   round(policy_loss.item(), 6),
                'value_loss':    round(value_loss.item(), 6) if use_value_net else '',
                'entropy':       round(entropies_t.mean().item(), 6),
                'ep_return_std': round(float(np.std(episode_returns)), 4),
                'ep_return_min': round(float(np.min(episode_returns)), 4),
                'ep_return_max': round(float(np.max(episode_returns)), 4),
            }
            with open(metrics_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(step_metrics)

            if (iteration + 1) % save_every_n == 0 and best_policy_state is not None:
                ckpt = {
                    'iteration':    iteration,
                    'policy':       best_policy_state,
                    'optimizer':    optimizer.state_dict(),
                }
                if use_value_net:
                    ckpt['value_net']       = {k: v.cpu() for k, v in value_net.state_dict().items()}
                    ckpt['value_optimizer'] = value_optimizer.state_dict()
                torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_iter_{iteration + 1:04d}_best_return_{best_return:.2f}.pt"))

    envs.close()

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        if exp_dir is not None:
            torch.save(best_policy_state, os.path.join(exp_dir, "best_policy.pt"))

    return policy, mean_returns


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter",            type=int,   default=300)
    parser.add_argument("--n-episodes",        type=int,   default=128)
    parser.add_argument("--num-envs",          type=int,   default=64)
    parser.add_argument("--alpha",             type=float, default=1e-3)
    parser.add_argument("--gamma",             type=float, default=0.99)
    parser.add_argument("--entropy-coef",      type=float, default=0.01)
    parser.add_argument("--no-entropy",        action="store_true")
    parser.add_argument("--baseline",          type=str,   default="gae_nn", choices=["none", "nn", "gae_no_nn", "gae_nn"])
    parser.add_argument("--baseline-lr",       type=float, default=1e-3)
    parser.add_argument("--no-advantage-norm", action="store_true")
    parser.add_argument("--gae-lambda",        type=float, default=0.95)
    parser.add_argument("--hidden-size",       type=int,   default=128)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--exp-dir",           type=str,   default="runs/exp_001")
    parser.add_argument("--save-every-n",      type=int,   default=50)
    parser.add_argument("--device",            type=str,   default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    def make_env():
        return gym.make(
            "LunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode=None,
        )

    eval_env = make_env()
    policy, returns = train_reinforce(
        env_maker=make_env,
        n_iter=args.n_iter,
        n_episodes=args.n_episodes,
        num_envs=args.num_envs,
        alpha=args.alpha,
        gamma=args.gamma,
        use_entropy=not args.no_entropy,
        entropy_coef=args.entropy_coef,
        baseline=args.baseline,
        baseline_lr=args.baseline_lr,
        use_advantage_norm=not args.no_advantage_norm,
        gae_lambda=args.gae_lambda,
        hidden_size=args.hidden_size,
        device=device,
        seed=args.seed,
        exp_dir=args.exp_dir,
        save_every_n=args.save_every_n,
    )

    eval_env.close()

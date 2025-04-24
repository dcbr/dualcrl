import numpy as np
import gymnasium as gym

from collections import namedtuple
from dataclasses import dataclass, field
import itertools
from typing import Optional

from methods import Method
from metrics import Metric
import utils

Experience = namedtuple("Experience", ("s0", "a0", "r", "s1", "term"))


class Run:

    @dataclass
    class Parameters(utils.BaseDataClass):
        name: str = "run"
        seed: Optional[int] = None
        env_id: str = "CliffWalking-v0"
        env_kwargs: dict = field(default_factory=dict)
        method_id: str = "QLearning"
        method_kwargs: dict = field(default_factory=dict)
        train_steps: int = 50000
        eval_steps: int = 1000
        eval_stride: int = 500
        eval_metrics: tuple = ("return", "length")

    def __init__(self, job_path, params: Parameters):
        utils.set_seed(params.seed)
        self.env = create_env(params.env_id, params.env_kwargs, seed=params.seed+1)
        self.eval_env = create_env(params.env_id, params.env_kwargs, seed=params.seed+2)
        self.init_env = create_env(params.env_id, params.env_kwargs, seed=params.seed+3)
        self.method = Method.from_id(params.method_id, self.env.observation_space, self.env.action_space, **params.method_kwargs)
        self.train_steps = params.train_steps
        self.eval_steps = params.eval_steps
        self.eval_stride = params.eval_stride
        self.eval_metrics = tuple(Metric.from_id(metric_id, self) for metric_id in params.eval_metrics)
        self.job_path = job_path
        self.run_name = params.name
        self.run_path = self.job_path / self.run_name
        if self.run_path.exists():
            print(f"Run {self.run_name} already exists, loading from final checkpoint.")
            self.method.load_checkpoint(self.run_path / "final_ckpt.pth")
            self.metrics = {key: val for (key, val) in np.load(str(self.run_path / "evaluation_metrics.npz")).items()}
            self.stats = {key: val for (key, val) in np.load(str(self.run_path / "train_stats.npz")).items()}
            self.trained = True
        else:
            self.run_path.mkdir(parents=True)
            self.trained = False

    def train_env_step(self, exp):
        if "init" in self.method.buffers:
            obs, _ = self.init_env.reset()
            self.method.store_sample("init", obs)
        self.method.store_sample("exp", *exp)
        if "state" in self.method.buffers:
            self.method.store_sample("state", exp.s0)
            if exp.term:
                self.method.store_sample("state", exp.s1)
        self.method.train_step()
        utils.Schedule.trigger("step")

    def train(self, steps=-1):
        k = 0
        metric_keys = itertools.chain.from_iterable(metric.keys for metric in self.eval_metrics)
        metrics = {key: [] for key in itertools.chain(metric_keys, ["steps"])}
        stats = {key: [] for key in itertools.chain(self.method.stats.keys(), ["steps"])}
        if steps < 0:
            steps = self.train_steps
        while k < steps:
            delta = min(self.eval_stride, steps - k)
            # Train and register training stats
            self.method.reset_stats()
            rollout(self.env, self.method.behavior_policy, delta, collect_experience=False, on_step=self.train_env_step)
            train_data = self.method.fetch_stats()
            for key, value in train_data.items():
                stats[key].append(value)
            # Evaluate and register evaluation metrics
            eval_data = self.evaluate()
            for key, value in eval_data.items():
                metrics[key].append(value)
            k += delta
            metrics["steps"].append(k)
            stats["steps"].append(k)
            print(f"[Run {self.run_name}] Training progress: {100*k/steps:.2f}%")
        # Save final model, training stats and evaluation metrics
        self.method.save_checkpoint(self.run_path / "final_ckpt.pth")
        np.savez(str(self.run_path / "train_stats.npz"), **stats)
        np.savez(str(self.run_path / "evaluation_metrics.npz"), **metrics)
        self.metrics = metrics
        self.stats = stats
        self.trained = True
        return metrics, stats

    def evaluate(self, steps=-1):
        if steps < 0:
            steps = self.eval_steps
        batches = rollout(self.eval_env, self.method.policy, steps, collect_experience=True)
        if len(batches) > 1 and not batches[-1][-1].term:
            # For episodic environments (with multiple batches in the evaluation rollout), remove the last batch if it was truncated
            batches.pop(-1)
        metrics_data = {}
        for metric in self.eval_metrics:
            metric.reset()
            metric.process(batches)
            metrics_data.update(metric.results())

        return metrics_data


def create_env(env_id, env_kwargs, seed):
    env = gym.make(env_id, **env_kwargs)
    env.reset(seed=seed)
    return env


def rollout(env, policy, steps, collect_experience=False, on_step=None, on_episode=None):
    k0 = 0
    batches = []
    while k0 < steps:
        k = 0
        batch = []
        obs, _ = env.reset()
        done = False
        while not done and k0+k < steps:
            prev_obs = obs
            action = policy(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            exp = Experience(prev_obs, action, reward, obs, term)
            if collect_experience:
                batch.append(exp)
            if on_step is not None:
                on_step(exp)
            k += 1

        batches.append(batch)
        if on_episode is not None:
            on_episode(batch)
        k0 += k
    return batches

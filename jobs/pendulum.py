import numpy as np
import pandas as pd
import scipy.stats

from methods import DualCRL
from run import Run
from plotting import line_plots, shaded_line_plot, plot_pendulum, make_video
import utils

ENV_ID = "Pendulum-v1"
SEEDS = [2899433845,  891794536, 1204047055, 3650233223, 1169137882,
         2000243642,   55487360, 2302026220, 2613579794,  698561950]
TRAIN_STEPS = 50000
WARMUP_STEPS = 1000
EVAL_STEPS = 1000
EVAL_STRIDE = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
STATE_BUFFER_SIZE = 5000  # Keep small enough such that state buffer is as close to on-policy samples as possible (except for the extra exploration by the behavioural policy)
EXPLORATION_EPS = utils.LinearSchedule(1.0, 0.0, 10*WARMUP_STEPS, "step")
GAMMA = 0.99
HIDDEN_DIMS = (50, 10)
# All seed values above are randomly generated using:
# >>> import numpy as np
# >>> rng = np.random.default_rng()
# >>> rng.integers(2**32, size=10)

OPTIMIZER_PARAMS = {
    "optimizer": "adam", "eta_critic": 4e-4, "eta_actor": 4e-4, "eta_modifiers": 4e-4,
    "stride_actor": 2, "stride_modifiers": 10, "stride_targets": 2,
}

# Evaluation grid of states:
angles = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
vels = np.linspace(-8, 8, 100, dtype=np.float32)
A, V = np.meshgrid(angles, vels, indexing='ij')
X, Y = np.cos(A), np.sin(A)
s = np.stack([X.flatten(), Y.flatten(), V.flatten()], axis=1)
a = np.zeros((1,), dtype=np.float32)  # Only evaluate one action to limit memory usage

# Entropy regularization parameters (SAC-like training):
alpha = utils.ExponentialSchedule(0.1, 1e-3, TRAIN_STEPS, "step")
er = DualCRL.EntropyRegularizationParameters(alpha=alpha)

# Specify state visititation density bounds
def dlb(s):
    return np.zeros_like(s[:, [0]])
def dub(s):
    X_TH = -0.1
    Y_TH = -0.1  # Allow some room for slowing down around upward position
    return np.where((s[:, [0]] > X_TH) & (s[:, [1]] < Y_TH), 1e-5, 1.0)  # Avoid upper right quadrant
db = DualCRL.DensityConstraintParameters(lower_bound=dlb, upper_bound=dub)

# Specify value constraint reward function
def vcr(s0, a, s1):
    return -(s0[:, [2]] / 8.0)**2  # Extra cost for fast movements
vc = DualCRL.ValueConstraintParameters(reward=vcr, lower_bound=-1.0 / 9 / (1 - GAMMA))  # On average, move slower than a third of the maximum velocity for the whole trajectory

# Specify transition constraint cost function
def tcc(s0, a, s1):
    dv = s1[:, [2]] - s0[:, [2]]
    return (dv / 8.0)**2  # Extra cost for high accelerations
tc = DualCRL.TransitionConstraintParameters(cost=tcc, upper_bound=2e-3)  # Enforce small accelerations (smooth state transitions)


def params_generator():
    experiments = {
        "raw": {"er": er},
        "db_vc": {"er": er, "db": db, "vc": vc},
        # "tc": {"er": er, "tc": tc},
    }
    for name, params in experiments.items():
        for seed in SEEDS:
            yield name, params, seed


def setup_runs():
    method_id = DualCRL.method_id
    method_params = DualCRL.Parameters(
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, state_buffer_size=STATE_BUFFER_SIZE, warm_up=WARMUP_STEPS,
        **OPTIMIZER_PARAMS, gamma=GAMMA, eps=EXPLORATION_EPS, use_actor=True, use_targets=True, hidden_dims=HIDDEN_DIMS,
        eval_states=s, eval_actions=a,
    )
    run_params = Run.Parameters(
        env_id=ENV_ID,
        method_id=method_id,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
        eval_stride=EVAL_STRIDE,
        eval_metrics=("return", "length", "state", "roughness")
    )
    for name, params, seed in params_generator():
        yield run_params.copy(
            name=f"{name}/{seed}",
            seed=seed,
            method_kwargs={
                "params": method_params.copy(**params)
            }
        )


def process_run(run, metrics, stats):
    line_plots(stats["steps"], stats["eps"], x_label="Training steps", y_label="$\epsilon$", title="Exploration epsilon", save_path=run.run_path / "eps.svg")
    line_plots(stats["steps"], stats["alpha"], x_label="Training steps", y_label="$\\alpha$", title="Entropy regularization temperature", save_path=run.run_path / "alpha.svg")
    line_plots(stats["steps"], stats["critic_loss"], x_label="Training steps", y_label="$L_Q$", title="Critic loss", save_path=run.run_path / "critic_loss.svg")
    line_plots(stats["steps"], stats["actor_loss"], x_label="Training steps", y_label=r"$L_\pi$", title="Actor loss", save_path=run.run_path / "actor_loss.svg")

    shaded_line_plot(metrics["steps"], metrics["Rmean"], metrics["Rmin"], metrics["Rmax"], x_label="Training steps", y_label="Returns", title="Evaluation returns", save_path=run.run_path / "returns.svg")
    line_plots(metrics["steps"], metrics["state_roughness"], x_label="Training steps", y_label="$\\rho$", title="Average Roughness", save_path=run.run_path / "roughness.svg")

    make_video(
        (plot_state_visits(eval_states.reshape((-1, 3)), step, close=False) for step, eval_states in zip(metrics["steps"], metrics["state"])),
        save_path=run.run_path / "states.mp4", fps=4, save_frames=True
    )

    make_video(
        (plot_pendulum(A, V, a_mean, title=f"Pendulum Actions Mean (step {step})", close=False) for step, a_mean in zip(stats["steps"], stats["a_mean"])),
        save_path=run.run_path / "actions_mean.mp4", fps=4, save_frames=True
    )
    make_video(
        (plot_pendulum(A, V, a_std, title=f"Pendulum Actions Standard Deviation (step {step})", close=False) for step, a_std in zip(stats["steps"], stats["a_std"])),
        save_path=run.run_path / "actions_std.mp4", fps=4, save_frames=True
    )

    policy = run.method.policy
    critic = run.method.critic
    plot_pendulum(A, V, critic(policy.batch(s), policy.batch(policy(s))).detach().numpy(), title="Pendulum Values", save_path=run.run_path / "values.svg")

    if run.method.params.db is not None:
        line_plots(stats["steps"], stats["modifiers_db_loss"], x_label="Training steps", y_label=r"$L_r$", title="Modifiers Loss", save_path=run.run_path / "modifiers_db_loss.svg")
        make_video(
            (plot_pendulum(A, V, log_d, vmin=-15, title=f"Pendulum Estimated State Visitation (step {step})", close=False) for step, log_d in zip(stats["steps"], stats["d_logits"])),
            save_path=run.run_path / "estimated_visitation.mp4", fps=4, save_frames=True
        )
        make_video(
            (plot_pendulum(A, V, rl-ru, title=f"Pendulum Reward Modifiers (step {step})", close=False) for step, rl, ru in zip(stats["steps"], stats["db_rl"], stats["db_ru"])),
            save_path=run.run_path / "reward_modifiers_db.mp4", fps=4, save_frames=True
        )
    if run.method.params.vc is not None:
        line_plots(stats["steps"], stats["modifiers_vc_loss"], x_label="Training steps", y_label=r"$L_r$", title="Modifiers Loss", save_path=run.run_path / "modifiers_vc_loss.svg")
        line_plots(stats["steps"], np.array(stats["vc_w"])[:, 0], x_label="Training steps", y_label="w", title="VC Weight", save_path=run.run_path / "vcw.svg")
        line_plots(metrics["steps"], mean_eval_reward(metrics["state"], vcr), x_label="Training steps", y_label="$\\bar{r}$", title="Average VC Reward", save_path=run.run_path / "vcr.svg")
    if run.method.params.tc is not None:
        line_plots(stats["steps"], stats["modifiers_tc_loss"], x_label="Training steps", y_label=r"$L_r$", title="Modifiers Loss", save_path=run.run_path / "modifiers_tc_loss.svg")
        line_plots(metrics["steps"], mean_eval_reward(metrics["state"], tcc), x_label="Training steps", y_label="$\\bar{c}$", title="Average TC Cost", save_path=run.run_path / "tcc.svg")


def plot_state_visits(states, step=None, save_path=None, close=True):
    # angles = np.arctan2(states[:, 1], states[:, 0])
    x, y = states[:, 0], states[:, 1]
    vels = states[:, 2]
    dist = scipy.stats.gaussian_kde(np.stack([x, y, vels], axis=0))
    log_visits = dist.logpdf(s.T)
    title = "Pendulum State Visitation"
    if step is not None:
        title = f"{title} (step {step})"
    return plot_pendulum(A, V, log_visits, vmin=-6, title=title, save_path=save_path, close=close)  # vmin=-15 shows up to ~1e-7, vmin=-6 shows up to ~1e-3


def mean_eval_reward(states, reward):
    r_means = []
    for eval_states in states:
        s0 = eval_states[:, :-1, :].reshape((-1, 3))
        s1 = eval_states[:, 1:, :].reshape((-1, 3))
        r_means.append(np.mean(reward(s0, None, s1)))
    return r_means


def analyze_runs(job_path, runs):
    metrics = {}
    for run, (name, *_) in zip(runs.values(), params_generator()):
        run_metrics = {key: metric for (key, metric) in np.load(str(run.run_path / "evaluation_metrics.npz")).items()}
        if name not in metrics:
            metrics[name] = {"state": []}
        steps = run_metrics.pop("steps")
        metrics[name]["state"].append(run_metrics.pop("state"))
        for key, metric in run_metrics.items():
            if key not in metrics[name]:
                metrics[name][key] = []
            metrics[name][key].append(pd.Series(metric, steps))
    names = [*metrics.keys()]
    aggregations = {
        "Rmin": ("min", "mean"), "Rmean": ("mean",), "Rmax": ("max", "mean"),
        "state_roughness": ("min", "mean", "max"),
    }
    agg_metrics = {key: {agg: [] for agg in aggs} for key, aggs in aggregations.items()}  # Aggregated metrics over different seeds for each experiment
    agg_metrics["steps"] = []
    agg_metrics["state"] = []
    for name in metrics.keys():
        agg_metrics["state"].append(np.concatenate(metrics[name]["state"], axis=1))
        agg_steps = None
        for key, aggs in aggregations.items():
            for agg in aggs:
                agg_series = aggregate_metrics(metrics[name][key], agg, alpha=0.4)
                agg_metrics[key][agg].append(agg_series.to_numpy())
                agg_steps = agg_series.index.to_numpy()
        if agg_steps is not None:
            agg_metrics["steps"].append(agg_steps)
    shaded_line_plot(agg_metrics["steps"], agg_metrics["Rmean"]["mean"], agg_metrics["Rmin"]["mean"], agg_metrics["Rmax"]["mean"], names,
                     x_label="Training steps", y_label="Returns", title="Evaluation returns", save_path=job_path / "returns.svg")
    shaded_line_plot(agg_metrics["steps"], agg_metrics["state_roughness"]["mean"],  agg_metrics["state_roughness"]["min"], agg_metrics["state_roughness"]["max"], names,
                     x_label="Training steps", y_label="$\\rho$", title="Average Roughness", save_path=job_path / "roughness.svg")
    vcrs = [mean_eval_reward(eval_states, vcr) for eval_states in agg_metrics["state"]]
    line_plots(agg_metrics["steps"], vcrs, names, x_label="Training steps", y_label="$\\bar{r}$", title="Average VC Reward", save_path=job_path / "vcr.svg")

    for i, name in enumerate(names):
        eval_states = agg_metrics["state"][i][-1]
        plot_state_visits(eval_states.reshape((-1, 3)), save_path=job_path / name / "states.svg")
        # make_video(
        #     (plot_state_visits(eval_states.reshape((-1, 3)), step, close=False) for step, eval_states in zip(agg_metrics["steps"][i], agg_metrics["state"][i])),
        #     save_path=job_path / name / "states.mp4", fps=4, save_frames=True
        # )


def aggregate_metrics(metrics, agg='mean', k_delta=1000, alpha=0.1):
    """Preprocess the given metrics array (each metric being a pandas Series)."""
    # Concatenate them into 1 dataframe:
    metrics_original = pd.concat(metrics, axis=1)
    metrics = metrics_original
    # Create empty dataframe with regularly spaced index:
    N = int(np.ceil(metrics.index.max() / k_delta))
    df_delta = pd.DataFrame(index=k_delta*np.arange(N+1))
    # Concatenate it with original metrics:
    metrics = pd.concat([metrics, df_delta], axis=1)
    # Interpolate missing data:
    metrics = metrics.interpolate(method="index", limit_area="inside")
    # Reindex using the regularly spaced index of df_delta:
    metrics = metrics.reindex(df_delta.index)
    # We now have a dataframe with regular timestep data (every k_delta timesteps), drop the initial missing datapoints:
    metrics = metrics.dropna()
    # Aggregate metrics, calculating the minimum, maximum and mean values across the different columns:
    metrics = metrics.agg(agg, axis=1)
    # Apply exponential smoothing filter to each column:
    metrics = metrics.ewm(alpha=alpha).mean()
    return metrics

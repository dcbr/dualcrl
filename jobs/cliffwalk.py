import numpy as np

from methods import DualCRL
from models import DiscretePolicy, TabularCritic
from run import Run
from plotting import line_plots, shaded_line_plot, plot_cliff, plot_cliff_arrows, make_video, PowerNorm, MaskedLocator, MaskedFormatter
import utils

ENV_ID = "CliffWalking-v0"
SEED = 2081880523
TRAIN_STEPS = 50000
EVAL_STEPS = 1000
EVAL_STRIDE = 500
BATCH_SIZE = 32
BUFFER_SIZE = 10000
STATE_BUFFER_SIZE = 5000  # Keep small enough such that state buffer is as close to on-policy samples as possible (except for the extra exploration by the behavioural policy)
EXPLORATION_EPS = utils.ExponentialSchedule(1.0, 0.1, TRAIN_STEPS, "step")
GAMMA = 0.99
# All seed values above are randomly generated using:
# >>> import numpy as np
# >>> rng = np.random.default_rng()
# >>> rng.integers(2**32, size=1)

OPTIMIZER_PARAMS = {
    "optimizer": "adam", "eta_critic": 2e-2, "eta_actor": 4e-3, "eta_visitation": 4e-3, "eta_modifiers": 1e-2,
    "stride_actor": 2, "stride_visitation": 1, "stride_modifiers": 10, "stride_targets": 2,
}

# Specify state visititation density bounds
dl = np.zeros((48,))
du = np.ones((48,))
du[[26, 27]] = 1e-5  # Avoid these two unstable squares near the cliff
def dlb(s):
    return dl[s]
def dub(s):
    return du[s]
db = DualCRL.DensityConstraintParameters(lower_bound=dlb, upper_bound=dub)

# Specify teacher policy
def teacher_policy(eps):
    Q = np.zeros((48, 4))
    s_up = [*range(12, 23)] + [*range(24, 35)] + [36]  # Move up in entire second and third row (except in last column) and in the initial state
    s_right = [*range(11)]  # Move right in entire first row (except in last column)
    s_down = [11, 23, 35]  # Move down towards goal in last column
    Q[s_up, 0] = 1
    Q[s_right, 1] = 1
    Q[s_down, 2] = 1
    critic = TabularCritic(48, 4)
    critic.load_state_dict({'Q': Q})
    return DiscretePolicy(critic=critic, eps=eps)
alpha = utils.ExponentialSchedule(1, 0.01, TRAIN_STEPS, "step")
er = DualCRL.EntropyRegularizationParameters(teacher=teacher_policy(0.01), alpha=alpha)

# Specify action density bounds
pil = np.zeros((48, 4))
piu = np.ones((48, 4))
pil[1:11, 2] = 0.5  # Move down from the top row at least 50% of the time
def pilb(s, a):
    return pil[s, :] if a is None else pil[s, a]
def piub(s, a):
    return piu[s, :] if a is None else piu[s, a]
ab = DualCRL.DensityConstraintParameters(lower_bound=pilb, upper_bound=piub)

# Specify transition constraint cost function
s_ridge = [6, 18, 21, 33]
tcost = np.zeros((48, 4))
tcost[s_ridge, 1] = 1
tcost[s_ridge, 3] = 1
def tcc(s0, a, s1):
    return tcost[s0, a] + tcost[s1, a]
tc = DualCRL.TransitionConstraintParameters(cost=tcc, upper_bound=1e-5)  # No horizontal movements over the ridges


def params_generator():
    experiments = {
        "db_tc": {"db": db, "tc": tc, "use_actor": False},
        # "db": {"db": db, "use_actor": False},
        # "tc": {"tc": tc, "use_actor": False},
        "er_ab": {"er": er, "ab": ab, "use_actor": True},
        # "er": {"er": er, "use_actor": True},
        # "ab": {"ab": ab, "use_actor": True},
    }
    for name, params in experiments.items():
        yield name, params


def setup_runs():
    method_id = DualCRL.method_id
    method_params = DualCRL.Parameters(
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, state_buffer_size=STATE_BUFFER_SIZE,
        **OPTIMIZER_PARAMS, gamma=GAMMA, eps=EXPLORATION_EPS, use_targets=True,
    )
    run_params = Run.Parameters(
        env_id=ENV_ID,
        method_id=method_id,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
        eval_stride=EVAL_STRIDE,
        eval_metrics=("return", "length", "density")
    )
    for name, params in params_generator():
        yield run_params.copy(
            name=f"{name}/{SEED}",
            seed=SEED,
            method_kwargs={
                "params": method_params.copy(**params)
            }
        )


def process_run(run, metrics, stats):
    use_actor = run.method.actor is not None
    use_visitation = run.method.visitation is not None
    Q = stats["Q"][-1]
    logits = stats["a_logits"][-1] if use_actor else None
    visitations = np.exp(stats["d_logits"][-1]) if use_visitation else None
    np.save(str(run.run_path / "values.npy"), Q)
    if use_actor:
        np.save(str(run.run_path / "logits.npy"), logits)
    if use_visitation:
        np.save(str(run.run_path / "visitations.npy"), visitations)

    line_plots(stats["steps"], stats["eps"], x_label="Training steps", y_label="$\epsilon$", title="Exploration epsilon", save_path=run.run_path / "eps.svg")
    line_plots(stats["steps"], stats["critic_loss"], x_label="Training steps", y_label="$L_Q$", title="Critic loss", save_path=run.run_path / "critic_loss.svg")
    if use_actor:
        line_plots(stats["steps"], stats["actor_loss"], x_label="Training steps", y_label=r"$L_\pi$", title="Actor loss", save_path=run.run_path / "actor_loss.svg")
    if use_visitation:
        line_plots(stats["steps"], stats["visitation_loss"], x_label="Training steps", y_label="$L_d$", title="Visitation loss", save_path=run.run_path / "visitation_loss.svg")
        make_video(
            (plot_cliff(np.exp(d), np.nan, title=f"CliffWalking Densities (step {step})", close=False)
                for step, d in zip(stats["steps"], stats["d_logits"])),
            save_path=run.run_path / "learned_densities.mp4", fps=4, save_frames=True)

    make_video(
        (plot_cliff(np.max(Q, axis=1), np.nan, title=f"CliffWalking Values (step {step})", close=False)
            for step, Q in zip(stats["steps"], stats["Q"])),
        save_path=run.run_path / "learned_values.mp4", fps=4, save_frames=True)
    make_video(
        (plot_cliff(Q, np.nan, norm=PowerNorm(3), colorbar_kwargs={"ticks": MaskedLocator(8), "format": MaskedFormatter([0, 2, 3, 4, 5, 7])}, title=f"CliffWalking Values (step {step})", close=False)
            for step, Q in zip(stats["steps"], stats["Q"])),
        save_path=run.run_path / "learned_values_q.mp4", fps=4, save_frames=True)
    # make_video(
    #     (plot_cliff(d, np.nan, title=f"CliffWalking Densities (step {step})", close=False)
    #         for step, d in zip(metrics["steps"], metrics["density"])),
    #     save_path=run.run_path / "estimated_densities.mp4", fps=4, save_frames=True)
    make_video(
        (plot_cliff_arrows(l, not use_actor, d, title=f"CliffWalking Densities & Actions (step {step})", close=False)
            for step, d, l in zip(stats["steps"], metrics["density"], stats["a_logits" if use_actor else "Q"])),
        save_path=run.run_path / "estimated_densities.mp4", fps=4, save_frames=True)
    make_video(
        (plot_cliff_arrows(l, not use_actor, title=f"CliffWalking Actions (step {step})", close=False)
            for step, l in zip(stats["steps"], stats["a_logits" if use_actor else "Q"])),
        save_path=run.run_path / "actions.mp4", fps=4, save_frames=True)
    if run.method.params.db is not None:
        make_video(
            (plot_cliff(rl-ru, np.nan, title=f"CliffWalking Reward Modifiers (step {step})", close=False)
                for step, rl, ru in zip(stats["steps"], stats["db_rl"], stats["db_ru"])),
            save_path=run.run_path / "reward_modifiers_db.mp4", fps=4, save_frames=True)
    if run.method.params.ab is not None:
        make_video(
            (plot_cliff(rl-ru, np.nan, title=f"CliffWalking Reward Modifiers (step {step})", close=False)
                for step, rl, ru in zip(stats["steps"], stats["ab_rl"], stats["ab_ru"])),
            save_path=run.run_path / "reward_modifiers_ab.mp4", fps=4, save_frames=True)
    if run.method.params.tc is not None:
        make_video(
            (plot_cliff(-rc, np.nan, title=f"CliffWalking Reward Modifiers (step {step})", close=False)
                for step, rc in zip(stats["steps"], stats["tc_r"])),
            save_path=run.run_path / "reward_modifiers_tc.mp4", fps=4, save_frames=True)


def analyze_runs(job_path, runs):
    metrics = {}
    for run in runs.values():
        run_metrics = np.load(str(run.run_path / "evaluation_metrics.npz"))
        for key, metric in run_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(metric)
    names = [f"{name}" for name, *_ in params_generator()]
    shaded_line_plot(metrics["steps"], metrics["Rmean"], metrics["Rmin"], metrics["Rmax"], names, x_label="Training steps",
                     y_label="Returns", title="Evaluation returns", save_path=job_path / "returns.svg")
    shaded_line_plot(metrics["steps"], metrics["Lmean"], metrics["Lmin"], metrics["Lmax"], names, x_label="Training steps",
                     y_label="Lengths", title="Evaluation episode lengths", save_path=job_path / "lengths.svg")

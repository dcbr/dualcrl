from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from models import Model, ContinuousPolicy, DiscretePolicy, MaxEntropyTeacher, Policy, DeepActor, DeepCritic, DeepRewardModifier, KernelDensity, TabularActor, TabularCritic, TabularRewardModifier, TabularVisitationDensity
import utils


class Buffer:
    """Buffer used to store collected experience and fetch random samples.
    We use numpy buffers, as indexing from pytorch Tensors is much slower (see pytorch/pytorch#29973)."""

    def __init__(self, capacity, fields):
        assert capacity > 0
        self._capacity = int(capacity)
        self._idx = 0
        self._filled = False
        self._data = {}
        for i, field in enumerate(fields):
            dim = field.get("dim", 1)
            dim = (dim,) if not isinstance(dim, (list, tuple)) else dim
            cont = field.get("cont", True)
            shape = dim if cont else (len(dim),)
            dtype = np.float32 if cont else np.int32
            name = field.get("name", f"field:{i}")
            self._data[name] = np.empty((self._capacity, *shape), dtype=dtype)

    def store(self, row):
        for d, r in zip(self._data.values(), row):
            d[self._idx, :] = np.asarray(r, dtype=d.dtype)
        self._idx = (self._idx + 1) % self._capacity
        self._filled |= (self._idx == 0)

    def clear(self):
        self._idx = 0
        self._filled = False

    @property
    def size(self):
        return self._capacity if self._filled else self._idx

    def sample(self, size, unique=True):
        if unique:
            if self.size < 40 * size:
                # This is fast when the buffer is still small
                idxs = utils.rng.permutation(self.size)[:size]
            else:
                # This is fast when the buffer gets larger
                idxs = np.unique(utils.rng.integers(self.size, size=(size,)))
                while idxs.size < size:
                    new_idxs = utils.rng.integers(self.size, size=(size - idxs.size,))
                    idxs = np.unique(np.concatenate([idxs, new_idxs]))
        else:
            idxs = utils.rng.integers(self.size, size=(size,))

        return {name: torch.from_numpy(t[idxs, :]) for name, t in self._data.items()}

    def data(self, name):
        return self._data[name][:self.size]


class Aggregator:

    def __init__(self, mode):
        self.n = 0
        self.data = None
        self.initial_data = {
            "min": np.nan,
            "max": np.nan,
            "last": np.nan,
            "sum": 0,
            "mean": 0
        }[mode]
        self.push_func = {
            "min": np.fmin,
            "max": np.fmax,
            "last": lambda _, val: val,
            "sum": np.add,
            "mean": lambda prev, val: (prev * self.n + val) / (self.n + 1)
        }[mode]
        self.reset()

    def reset(self):
        self.n = 0
        self.data = self.initial_data

    def push(self, val):
        if isinstance(val, torch.Tensor):
            val = val.detach().numpy().copy()
        if val is None or (np.isscalar(val) and np.isnan(val)):
            return
        self.data = self.push_func(self.data, val)

    @property
    def value(self):
        return self.data


class Method:
    @dataclass
    class Config(utils.BaseDataClass):
        """Configuration parameters as required by a specific method."""
        actor_gen: Optional[callable] = None
        n_actors: int = 0
        critic_gen: Optional[callable] = None
        n_critics: int = 0
        use_targets: bool = True  # Whether target networks are used by the method or not
        use_init_buf: bool = False  # Whether a replay buffer of initial states is used by the method or not

    @dataclass
    class Parameters(utils.BaseDataClass):
        """Hyperparameters that can be tuned by users of the method."""
        buffer_size: int = int(1e5)  # Size of the replay buffer storing the collected experience (and initial states)
        state_buffer_size: int = 0  # Size of the replay buffer of recently visited states
        batch_size: int = 64
        warm_up: int = 10 * batch_size  # Minimum number of experience samples to collect before training the models
        gamma: float = 0.99  # Discount factor
        tau: float = 0.001  # Weight used to update the target models
        optimizer: str = "adam"  # Optimizer to use, either "adam" or "sgd"
        eta_actor: float = 1e-3  # Learning rate actor
        eta_critic: float = 1e-3  # Learning rate critic
        stride_actor: int = 2  # Number of timesteps before each actor update (critic is updated every timestep, i.e. stride_critic=1)
        stride_targets: int = 2  # Number of timesteps before each target network update

    method_id = None
    __methods = {}

    def __init_subclass__(cls, **kwargs):
        method_id = kwargs.pop("method_id")
        assert method_id not in cls.__methods, "Subclass method ids should be unique"
        super().__init_subclass__(**kwargs)
        cls.__methods[method_id] = cls
        cls.method_id = method_id

    def __init__(self, obs_space, act_space, config=None, params=None):
        super().__init__()
        config = self.Config.create(config)
        assert config.n_critics > 0 or config.n_actors > 0
        params = self.Parameters.create(params)
        assert params.warm_up >= params.batch_size
        self.params = params

        # params are not included in the checkpoint as they are saved with the job file and cannot be pickled (can contain callbacks)
        self.checkpoint = {"models": {}, "model_targets": {}, "optimizers": {}}
        self.models: Dict[str, Tuple[Model, ...]] = dict()
        self.model_targets: Dict[str, Tuple[Model, ...]] = dict()
        self.optimizers: Dict[str, torch.optim.Optimizer] = dict()
        self.train_functions: Dict[str, Callable] = dict()
        self.train_strides: Dict[str, int] = dict()
        self._setup_models("critic", config.critic_gen, config.n_critics, config.use_targets,
                           self.params.optimizer, self.params.eta_critic, self.train_critics, 1)
        self._setup_models("actor", config.actor_gen, config.n_actors, config.use_targets,
                           self.params.optimizer, self.params.eta_actor, self.train_actors, self.params.stride_actor)

        self.buffers = self._setup_buffers(self.params.buffer_size, obs_space, act_space, config.use_init_buf, self.params.state_buffer_size)
        self.checkpoint["buffers"] = self.buffers
        self.stats: Dict[str, Aggregator] = dict()

        self._policy = None
        self._behavior_policy = None
        self.it = 0

    @classmethod
    def from_id(cls, method_id, obs_space, act_space, **method_kwargs):
        return cls.__methods[method_id](obs_space, act_space, **method_kwargs)

    def _setup_models(self, model_name, model_gen, n_models, use_targets, optim, eta, train_func, train_stride):
        models = tuple()
        model_targets = tuple()
        optimizer = None
        if n_models > 0:
            if callable(model_gen):
                model_gen = [model_gen]
            # Generate models
            models = tuple(mg() for mg in model_gen for _ in range(n_models))
            self.checkpoint["models"][model_name] = tuple(m.state_dict() for m in models)
            # Generate optimizer
            params = sum(([*m.parameters()] for m in models), [])
            if len(params) > 0:
                optimizer = {
                    "sgd": torch.optim.SGD,
                    "adam": torch.optim.Adam
                }[optim](params, lr=eta)
                self.checkpoint["optimizers"][model_name] = optimizer.state_dict()
            if use_targets:
                # Generate target models
                model_targets = tuple(mg() for mg in model_gen for _ in range(n_models))
                for m, mt in zip(models, model_targets):
                    # Initialize target model parameters to be equal to corresponding model parameters
                    mt.load_state_dict(m.state_dict())
                    # And disable gradients for target model parameters
                    for pt in mt.parameters():
                        pt.requires_grad_(False)
                # Store state dicts
                self.checkpoint["model_targets"][model_name] = tuple(mt.state_dict() for mt in model_targets)
        self.models[model_name] = models
        self.model_targets[model_name] = model_targets
        self.optimizers[model_name] = optimizer
        self.train_functions[model_name] = train_func
        self.train_strides[model_name] = train_stride

    def _save_models(self):
        # Models' state dicts are automatically updated when an optimization step is performed or another state_dict is loaded.
        # This is not the case for optimizers' state dicts, so we have to store it again in the checkpoint right before saving.
        for model_name in self.checkpoint["optimizers"].keys():
            self.checkpoint["optimizers"][model_name] = self.optimizers[model_name].state_dict()

    def _load_models(self, checkpoint):
        for model_name in self.models.keys():
            for i, m in enumerate(self.models[model_name]):
                m.load_state_dict(checkpoint["models"][model_name][i])
            for i, mt in enumerate(self.model_targets[model_name]):
                mt.load_state_dict(checkpoint["model_targets"][model_name][i])
            if self.optimizers[model_name] is not None:
                self.optimizers[model_name].load_state_dict(checkpoint["optimizers"][model_name])

    def update_target_models(self):
        for models, model_targets in zip(self.models.values(), self.model_targets.values()):
            for m, mt in zip(models, model_targets):
                for p, pt in zip(m.parameters(), mt.parameters()):
                    # pt = tau * p + (1 - tau) * pt
                    pt *= (1 - self.params.tau)
                    pt += self.params.tau * p
                mt.after_update()

    @staticmethod
    def _setup_buffers(buffer_size, obs_space, act_space, use_init_buf, state_buf_size):
        buffers = {}
        state_fields = Method._space_fields("s", obs_space)
        action_fields = Method._space_fields("a", act_space)
        buffers['exp'] = Buffer(buffer_size, [
            *[{**f, "name": f"{f['name']}0"} for f in state_fields],  # sc0, sd0
            *action_fields,  # ac, ad
            {"name": "r", "dim": (1,), "cont": True},
            *[{**f, "name": f"{f['name']}1"} for f in state_fields],  # sc1, sd1
            {"name": "t", "dim": (1,), "cont": True}
        ])
        if use_init_buf:
            buffers['init'] = Buffer(buffer_size, state_fields)
        if state_buf_size > 0:
            buffers['state'] = Buffer(state_buf_size, state_fields)
        return buffers

    @staticmethod
    def _space_fields(name, space):
        fields = []
        continuous_dim, discrete_dim = utils.dims_from_space(space)
        if continuous_dim is not None:
            fields.append({"name": f"{name}c", "dim": continuous_dim, "cont": True})
        if discrete_dim is not None:
            fields.append({"name": f"{name}d", "dim": discrete_dim, "cont": False})
        if len(fields) == 1:
            fields[0]["name"] = name
        return fields

    def save_checkpoint(self, path):
        self._save_models()
        torch.save(self.checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self._load_models(checkpoint)
        self.buffers = checkpoint["buffers"]
        # self.params = checkpoint["params"]

    def store_sample(self, name, *sample):
        self.buffers[name].store(sample)

    def train_step(self):
        buffer_size = min(buffer.size for buffer in self.buffers.values())
        if buffer_size >= self.params.warm_up:
            self.it += 1
            batches = {name: buffer.sample(self.params.batch_size) for name, buffer in self.buffers.items()}
            for model_name, train_func in self.train_functions.items():
                if self.models[model_name] and self.it % self.train_strides[model_name] == 0:
                    train_func(batches)
                    for model in self.models[model_name]:
                        model.after_update()
            if self.it % self.params.stride_targets == 0:
                with torch.no_grad():
                    self.update_target_models()

    def _setup_stats(self, cfg):
        for key, mode in cfg.items():
            self.stats[key] = Aggregator(mode)

    def reset_stats(self):
        for agg in self.stats.values():
            agg.reset()

    def fetch_stats(self):
        return {key: agg.value for key, agg in self.stats.items()}

    def train_actors(self, batches):
        raise NotImplementedError()

    def train_critics(self, batches):
        raise NotImplementedError()

    @property
    def actors(self):
        return self.models["actor"]

    @property
    def critics(self):
        return self.models["critic"]

    @property
    def actor(self):
        return self.actors[0] if self.actors else None

    @property
    def critic(self):
        return self.critics[0] if self.critics else None

    @property
    def actor_targets(self):
        return self.model_targets["actor"]

    @property
    def critic_targets(self):
        return self.model_targets["critic"]

    @property
    def actor_optimizer(self):
        return self.optimizers["actor"]

    @property
    def critic_optimizer(self):
        return self.optimizers["critic"]

    @property
    def policy(self):
        """Learned policy."""
        return self._policy

    @property
    def behavior_policy(self):
        """Behavior policy."""
        return self._behavior_policy


class DualCRL(Method, method_id="dualcrl"):
    MIN_PROB = 1e-6

    @dataclass
    class EntropyRegularizationParameters(utils.BaseDataClass):
        teacher: Optional[Policy] = None
        alpha: Union[float, utils.Schedule] = 0.1

    @dataclass
    class ValueConstraintParameters(utils.BaseDataClass):
        reward: callable
        lower_bound: float

    @dataclass
    class DensityConstraintParameters(utils.BaseDataClass):
        lower_bound: callable
        upper_bound: callable

    @dataclass
    class TransitionConstraintParameters(utils.BaseDataClass):
        cost: callable
        upper_bound: float

    @dataclass
    class Parameters(Method.Parameters):
        eta_visitation: float = 1e-3
        eta_modifiers: float = 1e-2
        stride_visitation: int = 1
        stride_modifiers: int = 10
        use_actor: bool = False
        use_targets: bool = False
        eps: Union[float, utils.Schedule] = 0.1
        hidden_dims: Tuple[int, ...] = (50,)
        eval_states: Optional[np.ndarray] = None
        eval_actions: Optional[np.ndarray] = None
        er: Optional['DualCRL.EntropyRegularizationParameters'] = None
        vc: Optional['DualCRL.ValueConstraintParameters'] = None
        db: Optional['DualCRL.DensityConstraintParameters'] = None
        pb: Optional['DualCRL.DensityConstraintParameters'] = None
        ab: Optional['DualCRL.DensityConstraintParameters'] = None
        tc: Optional['DualCRL.TransitionConstraintParameters'] = None

    def __init__(self, obs_space, act_space, params=None):
        self.cont = False
        Sc, Sd = utils.dims_from_space(obs_space)
        assert (Sc is None) ^ (Sd is None)
        if Sd is None:
            self.cont = True
        Ac, Ad = utils.dims_from_space(act_space)
        assert (Ac is None) ^ (Ad is None)
        assert Ad is None if self.cont else Ac is None
        params = self.Parameters.create(params)

        def actor_gen():
            return DeepActor(Sc, Ac, self.params.hidden_dims, act_space.high) if self.cont else TabularActor(Sd, Ad)
        def critic_gen():
            return DeepCritic(Sc, Ac, self.params.hidden_dims) if self.cont else TabularCritic(Sd, Ad)
        def visitation_gen():
            return KernelDensity(Sc, self.buffers['state']) if self.cont else TabularVisitationDensity(Sd)
        def get_modifiers_gen(state_only=False):
            if state_only:
                A = 0 if self.cont else 1
            else:
                A = Ac if self.cont else Ad
            if self.cont:
                return lambda: DeepRewardModifier(Sc, A, self.params.hidden_dims)
            else:
                return lambda: TabularRewardModifier(Sd, A)

        config = Method.Config(
            critic_gen=critic_gen,
            n_critics=1,
            use_targets=params.use_targets,
            use_init_buf=False,
        )
        if params.use_actor:
            config.actor_gen = actor_gen
            config.n_actors = 1

        super().__init__(obs_space, act_space, config=config, params=params)
        self._setup_stats({
            "critic_loss": "mean",
            "actor_loss": "mean",
            "visitation_loss": "mean",
            "eps": "last",
            "Q": "last",
            "a_logits": "last",
            "a_mean": "last",
            "a_std": "last",
            "d_logits": "last",
            "modifiers_vc_loss": "mean",
            "modifiers_db_loss": "mean",
            "modifiers_pb_loss": "mean",
            "modifiers_ab_loss": "mean",
            "modifiers_tc_loss": "mean",
            "alpha": "last",
            "vc_w": "last",
            "db_rl": "last",
            "db_ru": "last",
            "pb_rl": "last",
            "pb_ru": "last",
            "ab_rl": "last",
            "ab_ru": "last",
            "ab_rla": "last",
            "ab_rua": "last",
            "tc_r": "last",
            "debug": "last",
        })
        self._policy = ContinuousPolicy(self.actor) if self.cont else DiscretePolicy(actor=self.actor, critic=self.critic)
        self._behavior_policy = ContinuousPolicy(self.actor, eps=self.params.eps) if self.cont else DiscretePolicy(actor=self.actor, critic=self.critic, eps=self.params.eps)

        self._eval_states = self.params.eval_states
        self._eval_actions = self.params.eval_actions
        if self._eval_states is None:
            self._eval_states = torch.cartesian_prod(*[torch.linspace(low, high, 20) for (low, high) in zip(obs_space.low, obs_space.high)]) if self.cont else torch.arange(obs_space.n)
        if self._eval_actions is None:
            self._eval_actions = torch.cartesian_prod(*[torch.linspace(low, high, 20) for (low, high) in zip(act_space.low, act_space.high)]) if self.cont else torch.arange(act_space.n)
        self._eval_states = torch.as_tensor(self._eval_states).reshape((-1, *Sc) if self.cont else (-1, 1))
        self._eval_actions = torch.as_tensor(self._eval_actions).reshape((-1, *Ac) if self.cont else (-1, 1))
        self._eval_shape = (self._eval_states.size(0), self._eval_actions.size(0))
        self._eval_grid = tuple(t[idxs.flatten()] for (t, idxs) in zip([self._eval_states, self._eval_actions], torch.meshgrid([torch.arange(dim) for dim in self._eval_shape], indexing='ij')))

        n_visitation = 0
        if self.params.db is not None or self.params.pb is not None:
            # Setup the state visitation density model if visitation density bounds are specified
            n_visitation = 1
        self._setup_models("visitation", visitation_gen, n_visitation, False, self.params.optimizer, self.params.eta_visitation, self.train_visitation, self.params.stride_visitation)

        if self.params.er is not None and self.params.er.teacher is None:
            self.params.er.teacher = MaxEntropyTeacher()
        if self.params.vc is not None:
            modifiers_gen = lambda: TabularRewardModifier(1, 1)
            self._setup_models("vc_rm", modifiers_gen, 1, False, "adam", self.params.eta_modifiers, self.train_modifiers_vc, self.params.stride_modifiers)
            self.vc_w = self.models["vc_rm"][0].rewards[:, 0]
        if self.params.db is not None:
            self._setup_models("db_rm", get_modifiers_gen(True), 2, False, "adam", self.params.eta_modifiers, self.train_modifiers_db, self.params.stride_modifiers)
        if self.params.pb is not None:
            self._setup_models("pb_rm", get_modifiers_gen(False), 2, False, "adam", self.params.eta_modifiers, self.train_modifiers_pb, self.params.stride_modifiers)
        if self.params.ab is not None:
            modifiers_gen = [get_modifiers_gen(False), get_modifiers_gen(True)]
            # TODO: split up? The averaged models should probably be updated more frequently (stride=1)
            self._setup_models("ab_rm", modifiers_gen, 2, False, "adam", self.params.eta_modifiers, self.train_modifiers_ab, self.params.stride_modifiers)
        if self.params.tc is not None:
            self._setup_models("tc_rm", get_modifiers_gen(False), 1, False, "adam", self.params.eta_modifiers, self.train_modifiers_tc, self.params.stride_modifiers)

    @property
    def visitation(self):
        return self.models["visitation"][0] if self.models["visitation"] else None

    @property
    def visitation_optimizer(self):
        return self.optimizers["visitation"]

    def fetch_stats(self):
        with torch.no_grad():
            self.stats["eps"].push(float(self.params.eps))  # TODO: epsilon and alpha schedule are actually not controlled here, so would be better kept track of in run.py
            self.stats["Q"].push(self.critic(*self._eval_grid).reshape(self._eval_shape))
            if self.actor is not None:
                if self.cont:
                    dist = self.actor.distribution(self._eval_states)
                    self.stats["a_mean"].push(dist.loc)
                    self.stats["a_std"].push(dist.scale)
                else:
                    self.stats["a_logits"].push(self.actor.logits)
            if self.visitation is not None:
                self.stats["d_logits"].push(self.visitation(self._eval_states))
            # Reward modifiers
            if self.params.er is not None:
                self.stats["alpha"].push(float(self.params.er.alpha))
            if self.params.vc is not None:
                self.stats["vc_w"].push(self.vc_w)
            if self.params.db is not None:
                self.stats["db_rl"].push(self.models["db_rm"][0](self._eval_states))
                self.stats["db_ru"].push(self.models["db_rm"][1](self._eval_states))
            if self.params.pb is not None:
                self.stats["pb_rl"].push(self.models["pb_rm"][0](*self._eval_grid).reshape(self._eval_shape))
                self.stats["pb_ru"].push(self.models["pb_rm"][1](*self._eval_grid).reshape(self._eval_shape))
            if self.params.ab is not None:
                self.stats["ab_rl"].push(self.models["ab_rm"][0](*self._eval_grid).reshape(self._eval_shape))
                self.stats["ab_ru"].push(self.models["ab_rm"][1](*self._eval_grid).reshape(self._eval_shape))
                self.stats["ab_rla"].push(self.models["ab_rm"][2](self._eval_states))
                self.stats["ab_rua"].push(self.models["ab_rm"][3](self._eval_states))
            if self.params.tc is not None:
                self.stats["tc_r"].push(self.models["tc_rm"][0](*self._eval_grid).reshape(self._eval_shape))
        return super().fetch_stats()

    def adjusted_reward(self, r, s0, a, s1):
        ra = r
        if self.params.er is not None:
            ra -= float(self.params.er.alpha) * (self.policy.batch_log_prob(s0, a) - self.params.er.teacher.batch_log_prob(s0, a))
        if self.params.vc is not None:
            ra += self.vc_w * self.params.vc.reward(s0, a, s1)  # TODO: the result of the reward callback could also be stored in the replay buffer
        if self.params.db is not None:
            rl, ru = self.models['db_rm']
            ra += rl(s0) - ru(s0)
        if self.params.pb is not None:
            rl, ru = self.models['pb_rm']
            ra += rl(s0, a) - ru(s0, a)
        if self.params.ab is not None:
            rl, ru, rla, rua = self.models['ab_rm']
            ra += rl(s0, a) - rla(s0) - ru(s0, a) + rua(s0)
        if self.params.tc is not None:
            rc = self.models['tc_rm'][0]
            ra -= rc(s0, a) * self.params.tc.cost(s0, a, s1)
        return ra

    def train_actors(self, batches):
        self.train_actors_continuous(batches) if self.cont else self.train_actors_discrete(batches)

    def train_actors_discrete(self, batches):
        s, *_ = batches['exp'].values()
        self.actor_optimizer.zero_grad()
        # For discrete actions, the expectation over the actions can be calculated exactly
        with torch.no_grad():
            Q = self.critic(s)
        J = Q  # (B, A)
        log_prob = self.policy.batch_log_prob(s, None)  # or self.actor(s)
        if self.params.er is not None:
            J -= float(self.params.er.alpha) * (log_prob - self.params.er.teacher.batch_log_prob(s, None))
        loss = -torch.mean(torch.sum(J * log_prob.exp(), dim=1))
        self.stats["actor_loss"].push(loss)
        loss.backward()
        self.actor_optimizer.step()

    def train_actors_continuous(self, batches):
        s, *_ = batches['exp'].values()
        self.actor_optimizer.zero_grad()
        # For continuous actions, the expectation over the actions is approximated using samples that are passed through the critic
        a = self.policy.batch_forward(s)
        Q = self.critic(s, a)
        J = Q
        if self.params.er is not None:
            J -= float(self.params.er.alpha) * (self.policy.batch_log_prob(s, a) - self.params.er.teacher.batch_log_prob(s, a))
        loss = -torch.mean(J)
        self.stats["actor_loss"].push(loss)
        loss.backward()
        self.actor_optimizer.step()

    def train_critics(self, batches):
        s0, a, r, s1, t = batches['exp'].values()
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            ra = self.adjusted_reward(r, s0, a, s1)
            critic_targets = self.critic_targets if self.critic_targets else self.critics
            a1 = self.policy.batch_forward(s1)  # Uses actor when available, otherwise greedy w.r.t. critic
            Qts = [critic_target(s1, a1) for critic_target in critic_targets]
            # Qts = [torch.max(critic_target(s1), dim=1, keepdim=True)[0] for critic_target in critic_targets]  # DQN
            y = ra + (1 - t) * self.params.gamma * torch.min(torch.stack(Qts, dim=2), dim=2)[0]
        loss = sum(torch.mean((critic(s0, a) - y)**2) for critic in self.critics)
        self.stats["critic_loss"].push(loss)
        loss.backward()
        self.critic_optimizer.step()

    def train_visitation(self, batches):
        self.train_visitation_continuous(batches) if self.cont else self.train_visitation_discrete(batches)

    def train_visitation_discrete(self, batches):
        s0, *_ = batches['exp'].values()
        self.visitation_optimizer.zero_grad()
        # Maximize average log likelihood
        data_logits = self.visitation(s0)
        loss = -torch.mean(data_logits)
        self.stats["visitation_loss"].push(loss)
        loss.backward()
        self.visitation_optimizer.step()

    def train_visitation_continuous(self, batches):
        # For now we use KDE, this method can be used to implement the training of GMMs or normalizing flows
        pass

    def train_modifiers_vc(self, batches):
        s0, a, r, s1, *_ = batches['exp'].values()
        self.optimizers['vc_rm'].zero_grad()
        with torch.no_grad():
            rv = torch.as_tensor(self.params.vc.reward(s0, a, s1))
        loss = torch.sum(self.vc_w * (torch.mean(rv, 0) - (1 - self.params.gamma) * self.params.vc.lower_bound))
        self.stats['modifiers_vc_loss'].push(loss)
        loss.backward()
        self.optimizers['vc_rm'].step()

    def train_modifiers_db(self, batches):
        s, *_ = batches['exp'].values()
        rl, ru = self.models['db_rm']
        self.optimizers['db_rm'].zero_grad()
        with torch.no_grad():
            d = torch.maximum(torch.tensor(self.MIN_PROB), self.visitation(s).exp())
            # TODO: there is a memory/computing trade-off here. For now the bounds are (re)computed for every batch,
            #  alternatively, the bounds can be computed once when adding the experience and stored in the replay buffer
            dl = torch.as_tensor(self.params.db.lower_bound(s))
            du = torch.as_tensor(self.params.db.upper_bound(s))
            wl = torch.clamp(1 - dl / d, min=-1)  # Already upper bounded by 1
            wu = torch.clamp(du / d - 1, max=1)  # Already lower bounded by -1
        # loss = torch.mean(rl(s)*(1 - dl / d) + ru(s)*(du / d - 1))
        loss = torch.mean(rl(s)*wl + ru(s)*wu)  # Clamped factors improve stability
        # Alternative using uniform samples over the state space:
        # loss = torch.mean(rl(s) * (d - dl) + ru(s) * (du - d))
        self.stats['modifiers_db_loss'].push(loss)
        loss.backward()
        self.optimizers['db_rm'].step()

    def train_modifiers_pb(self, batches):
        s, a, *_ = batches['exp'].values()
        rl, ru = self.models['pb_rm']
        self.optimizers['pb_rm'].zero_grad()
        with torch.no_grad():
            p = torch.maximum(torch.tensor(self.MIN_PROB), self.visitation(s).exp() * self.policy.batch_log_prob(s, a).exp())
            pl = torch.as_tensor(self.params.pb.lower_bound(s, a))
            pu = torch.as_tensor(self.params.pb.upper_bound(s, a))
            wl = torch.clamp(1 - pl / p, min=-1)
            wu = torch.clamp(pu / p - 1, max=1)
        # loss = torch.mean(rl(s, a)*(1 - pl / p) + ru(s, a)*(pu / p - 1))
        loss = torch.mean(rl(s, a)*wl + ru(s, a)*wu)  # Clamped factors improve stability
        self.stats['modifiers_pb_loss'].push(loss)
        loss.backward()
        self.optimizers['pb_rm'].step()

    def train_modifiers_ab(self, batches):
        self.train_modifiers_ab_continuous(batches) if self.cont else self.train_modifiers_ab_discrete(batches)

    def train_modifiers_ab_continuous(self, batches):
        s, a, *_ = batches['exp'].values()
        rl, ru, rla, rua = self.models['ab_rm']
        self.optimizers['ab_rm'].zero_grad()
        with torch.no_grad():
            pi = torch.maximum(torch.tensor(self.MIN_PROB), self.policy.batch_log_prob(s, a).exp())
            pil = torch.as_tensor(self.params.ab.lower_bound(s, a))
            piu = torch.as_tensor(self.params.ab.upper_bound(s, a))
            wl = torch.clamp(1 - pil / pi, min=-1)
            wu = torch.clamp(piu / pi - 1, max=1)
        rlsa = rl(s, a)
        rusa = ru(s, a)
        # For continuous actions, the expectation over the actions is approximated using samples
        # loss = torch.mean(rlsa*(1 - pil / pi) + rusa*(piu / pi - 1))
        loss = torch.mean(rlsa*wl + rusa*wu)  # Clamped factors improve stability
        # Update rla and rua models:
        loss += torch.mean((rlsa.detach() * pil / pi - rla(s))**2 + (rusa.detach() * piu / pi - rua(s))**2)
        self.stats['modifiers_ab_loss'].push(loss)
        loss.backward()
        self.optimizers['ab_rm'].step()

    def train_modifiers_ab_discrete(self, batches):
        s, *_ = batches['exp'].values()
        a = None
        rl, ru, rla, rua = self.models['ab_rm']
        self.optimizers['ab_rm'].zero_grad()
        with torch.no_grad():
            pi = self.policy.batch_log_prob(s, a).exp()
            pil = torch.as_tensor(self.params.ab.lower_bound(s, torch.arange(self.critic.A).reshape((1, -1))))
            piu = torch.as_tensor(self.params.ab.upper_bound(s, torch.arange(self.critic.A).reshape((1, -1))))
        rlsa = rl(s, a)
        rusa = ru(s, a)
        # For discrete actions, the expectation (sum) over the actions can be calculated exactly
        loss = torch.mean(torch.sum(rlsa*(pi - pil) + rusa*(piu - pi), dim=1))
        # Update rla and rua models:
        loss += torch.mean((torch.sum(rlsa.detach() * pil, dim=1) - rla(s))**2 + (torch.sum(rusa.detach() * piu, dim=1) - rla(s))**2)
        self.stats['modifiers_ab_loss'].push(loss)
        loss.backward()
        self.optimizers['ab_rm'].step()

    def train_modifiers_tc(self, batches):
        s0, a, r, s1, *_ = batches['exp'].values()
        rc = self.models['tc_rm'][0]
        self.optimizers['tc_rm'].zero_grad()
        with torch.no_grad():
            c = torch.as_tensor(self.params.tc.upper_bound - self.params.tc.cost(s0, a, s1))
        loss = torch.mean(rc(s0, a)*c)
        self.stats['modifiers_tc_loss'].push(loss)
        loss.backward()
        self.optimizers['tc_rm'].step()


if __name__ == "__main__":
    # Test speed of taking unique samples from buffer
    import time
    import math


    def timeit(f, repeats, timeout=None):
        start = time.time()
        end = math.inf if timeout is None else start + timeout
        for _ in range(repeats):
            f()
            if time.time() > end:
                return math.inf
        return (time.time() - start) / repeats


    def sample_multi(N, k):
        return torch.multinomial(torch.ones(N), k, False)


    def sample_perm(N, k):
        return torch.randperm(N)[:k]


    def sample_top(N, k):
        return torch.topk(torch.rand(N), k)[1]


    def sample_redraw(N, k):
        idxs = torch.unique(torch.randint(N, (k,)))
        while idxs.numel() < k:
            new_idxs = torch.randint(N, (k - idxs.numel(),))
            idxs = torch.unique(torch.cat([idxs, new_idxs]))
        return idxs


    samplers = {"multi": sample_multi, "perm": sample_perm, "top": sample_top, "redraw": sample_redraw}
    Ns = [10 ** p for p in range(2, 7)]
    ks = [2 ** p for p in range(5, 11)]
    repeats = 10000
    timeout = 30

    ts = {name: math.nan for name in samplers.keys()}
    for N in sorted(Ns):
        print(f"Timings (in ms) for N={N}:")
        print(f"{'k':<4}" + "".join(f" \t {name:^6}" for name in samplers.keys()))
        print("-" * (4 + 12 * len(samplers)))
        for k in sorted(ks):
            if k > N:
                break
            for name, sampler in samplers.items():
                if math.isinf(ts[name]):
                    ts[name] = math.inf
                else:
                    ts[name] = timeit(lambda: sampler(N, k), repeats, timeout) * 1e3
            print(f"{k:<4}" + "".join(f" \t {ts[name]:6.3f}" for name in samplers.keys()))
        print()

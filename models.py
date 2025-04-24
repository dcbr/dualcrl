import gymnasium as gym
import scipy.stats
import torch

import utils


class Model(torch.nn.Module):
    """Generic model base class"""

    def __init__(self, state_dim, action_dim, continuous):
        super().__init__()
        self.state_dim = utils.make_tensor(state_dim)
        self.action_dim = utils.make_tensor(action_dim)
        self.continuous = continuous
        # Flattened state and action dimensions
        self.S = torch.prod(self.state_dim).item()
        self.A = torch.prod(self.action_dim).item()
        # Discrete state and action transforms
        self.sd_transform = utils.RavelTransform(self.state_dim)
        self.ad_transform = utils.RavelTransform(self.action_dim)

    @staticmethod
    def _create_table(size):
        return torch.nn.Parameter(torch.zeros(size))

    @staticmethod
    def _create_network(input_dim, output_dim, hidden_dims, hidden_activation='tanh', output_activation='none'):
        in_dims = [input_dim] + list(hidden_dims)
        out_dims = list(hidden_dims) + [output_dim]
        activations = {
            'relu': torch.nn.ReLU,
            'softplus': torch.nn.Softplus,
            'tanh': torch.nn.Tanh,
            'none': torch.nn.Identity,
        }
        layers = []
        for i, o in zip(in_dims, out_dims):
            layers.append(torch.nn.Linear(i, o))
            layers.append(activations[hidden_activation]())
        layers.pop(len(layers)-1)
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            *layers,
            activations[output_activation](),
        )

    def load_state_dict(self, state_dict, strict=True):
        # Allow lists or numpy arrays instead of tensors to load a model
        state_dict = {key: value if isinstance(value, torch.Tensor) else torch.tensor(value) for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict)

    def transform_state(self, s, *, inverse=False):
        if not self.continuous:
            return self.sd_transform.unravel(s) if inverse else self.sd_transform.ravel(s)
        return s

    def transform_action(self, a, *, inverse=False):
        if not self.continuous:
            return self.ad_transform.unravel(a) if inverse else self.ad_transform.ravel(a)
        return a

    def after_update(self):
        """Called after an optimizer step has updated this model."""
        pass


# region Tabular models (for discrete state and action spaces)
class TabularActor(Model):
    """Tabular actor."""

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, False)
        self.logits = self._create_table((self.S, self.A))
        self.after_update()

    def forward(self, s, a=None):
        """s: Bx*S long tensor ; a: Bx*A long tensor.
        Returns the log probability for the given s and given a (or each a if a is None)."""
        s = self.transform_state(s)
        # Normalize logits such that gradient knows dependencies between different logits of same state
        logits = self.logits - torch.logsumexp(self.logits, dim=1, keepdim=True)
        if a is None:
            # Equivalent to logits[s,:] but much faster (see pytorch/pytorch#41162)
            logits = torch.index_select(logits, dim=0, index=s)  # (B, A)
            return logits
        else:
            a = self.transform_action(a)
            # Alternative (doesn't seem faster): torch.gather(logits, dim=1, index=torch.unsqueeze(a, dim=1))
            return torch.unsqueeze(logits[s, a], dim=1)  # (B, 1)

    def distribution(self, s):
        """Get the Categorical probability distribution for the given s."""
        return torch.distributions.Categorical(logits=self(s))

    def sample(self, s, n=1):
        """Take samples for the given s"""
        a = self.distribution(s).sample((n,)).T  # Shape (B, n)
        a = self.transform_action(a, inverse=True)  # Shape (B, n, A)
        if n == 1:
            a = torch.squeeze(a, dim=1)  # Remove sample dimension
        return a

    def after_update(self):
        # Normalize logits
        with torch.no_grad():
            self.logits -= torch.logsumexp(self.logits, dim=1, keepdim=True)


class TabularCritic(Model):
    """Tabular critic."""

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, False)
        self.Q = self._create_table((self.S, self.A))

    def forward(self, s, a=None):
        """s: Bx*S long tensor ; a: Bx*A long tensor.
        Returns the value for the given s and given a (or each a if a is None)."""
        s = self.transform_state(s)
        if a is None:
            # Equivalent to Q[s,:] but much faster (see pytorch/pytorch#41162)
            Q = torch.index_select(self.Q, dim=0, index=s)  # (B, A)
            return Q
        else:
            a = self.transform_action(a)
            # Alternative (doesn't seem faster): torch.squeeze(torch.gather(Q, dim=1, index=torch.unsqueeze(a, dim=1)), dim=1)
            return torch.unsqueeze(self.Q[s, a], dim=1)  # (B, 1)


class TabularVisitationDensity(Model):
    """Tabular visitation density."""

    def __init__(self, state_dim):
        super().__init__(state_dim, 1, False)
        self.logits = self._create_table((self.S,))
        self.after_update()

    def forward(self, s):
        """s: Bx*S long tensor.
        Returns the log probability for the given s."""
        s = self.transform_state(s)
        # Normalize logits such that gradient knows dependencies between different logits
        logits = self.logits - torch.logsumexp(self.logits, dim=0, keepdim=True)
        return torch.unsqueeze(logits[s], dim=1)  # (B, 1)

    def distribution(self):
        """Get the Categorical probability distribution."""
        return torch.distributions.Categorical(logits=self.logits)

    def sample(self, n=1):
        """Take samples"""
        s = torch.distributions.Categorical(logits=self.logits).sample((n,))  # Shape (n,)
        s = self.transform_state(s)  # Shape (n, S)
        if n == 1:
            s = torch.squeeze(s, dim=0)  # Remove sample dimension
        return s

    def after_update(self):
        # Normalize logits
        with torch.no_grad():
            self.logits -= torch.logsumexp(self.logits, dim=0, keepdim=True)


class TabularRewardModifier(Model):
    """Tabular reward modifier."""

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, False)
        self.rewards = self._create_table((self.S, self.A))

    def forward(self, s, a=None):
        """s: Bx*S long tensor ; a: Bx*A long tensor.
        Returns the reward modification for the given s (and given a if it is not None)."""
        s = self.transform_state(s)
        if a is None:
            r = self.rewards[s]  # (B, A) but A should be 1
        else:
            a = self.transform_action(a)
            r = torch.unsqueeze(self.rewards[s, a], dim=1)  # (B, 1)
        return torch.maximum(torch.tensor(0), r)  # (B, 1)

    def after_update(self):
        with torch.no_grad():
            self.rewards[self.rewards < 0] = 0
# endregion


# region Neural models (for continuous state and action spaces)
class DeepActor(Model):
    """Deep Gaussian actor."""

    def __init__(self, state_dim, action_dim, hidden_dims, a_max):
        super().__init__(state_dim, action_dim, True)
        self.layers = self._create_network(self.S, 2*self.A, hidden_dims)
        self.a_max = torch.tensor(a_max)

    def forward(self, s, a=None, sigma_min=0, sigma_max=3):
        """s: Bx*S long tensor ; a: Bx*A long tensor.
        Returns the log probability for the given s and given a (or each a if a is None)."""
        if a is None:
            raise ValueError("`a` cannot be None")
        # s = self.transform_state(s)
        # a = self.transform_action(a)
        return self.distribution(s, sigma_min, sigma_max).log_prob(a)

    def distribution(self, s, sigma_min=0, sigma_max=3):
        """Get the Gaussian probability distribution for the given s."""
        # s = self.transform_state(s)
        params = self.layers(s)
        atanh_mu, log_sigma = params[:, :self.A], params[:, self.A:]
        mu = self.a_max * torch.tanh(atanh_mu)
        # mu = torch.clamp(mu, -self.a_max, self.a_max)
        sigma = torch.minimum(log_sigma.exp() + sigma_min, torch.tensor(sigma_max))
        return torch.distributions.Normal(mu, sigma)

    def sample(self, s, n=1, sigma_min=0, sigma_max=3):
        """Take samples for the given s"""
        a = self.distribution(s, sigma_min, sigma_max).rsample((n,))  # Shape (n, B, A)
        a = torch.swapaxes(a, 0, 1)  # Shape (B, n, A)
        # a = self.transform_action(a, inverse=True)
        if n == 1:
            a = torch.squeeze(a, dim=1)  # Remove sample dimension
        return a


class DeepCritic(Model):
    """Deep critic."""

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__(state_dim, action_dim, True)
        self.layers = self._create_network(self.S + self.A, 1, hidden_dims)

    def forward(self, s, a=None):
        if a is None:
            raise ValueError("`a` cannot be None")
        # s = self.transform_state(s)
        # a = self.transform_action(a)
        return self.layers(torch.cat((s, a), 1))  # (B, 1)


class KernelDensity(Model):
    """Kernel Visitation Density Estimator"""

    def __init__(self, state_dim, buffer):
        super().__init__(state_dim, 1, True)
        self.buffer = buffer

    def forward(self, s):
        """s: Bx*S long tensor.
        Returns the log probability for the given s."""
        # s = self.transform_state(s)
        return torch.tensor(self.distribution().logpdf(s.T)).unsqueeze(1)  # Shape (B, 1)

    def distribution(self):
        """Get the estimated probability distribution."""
        return scipy.stats.gaussian_kde(self.buffer.data('s').T)

    def sample(self, n=1):
        """Take samples"""
        s = torch.tensor(self.distribution().resample(n)).T  # Shape (n, S)
        # s = self.transform_state(s, inverse=True)
        if n == 1:
            s = torch.squeeze(s, dim=0)  # Remove sample dimension
        return s


class DeepRewardModifier(Model):
    """Deep reward modifier."""

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__(state_dim, action_dim, True)
        self.layers = self._create_network(self.S + self.A, 1, hidden_dims, output_activation="softplus")

    def forward(self, s, a=None):
        """s: Bx*S long tensor ; a: Bx*A long tensor.
        Returns the reward modification for the given s (and given a if it is not None)."""
        # s = self.transform_state(s)
        if a is None:
            r = self.layers(s)  # (B, 1)
        else:
            # a = self.transform_action(a)
            r = self.layers(torch.cat((s, a), 1))  # (B, 1)
        return r
#endregion


class Policy(Model):
    """Base policy class."""

    def __init__(self, model):
        super().__init__(model.state_dim, model.action_dim, model.continuous)

    @staticmethod
    def batch(x) -> torch.Tensor:
        # Convert input to tensor and add batch dimension
        return torch.atleast_2d(torch.tensor(x))  # (1, X)

    def unbatch(self, x: torch.Tensor):
        # Remove batch dimension
        x = torch.squeeze(x, dim=0)  # (X,)
        # Convert output to numpy or scalar (for discrete action spaces)
        return x.item() if x.numel() == 1 and not self.continuous else x.numpy()

    def forward(self, s):
        # These methods are used during rollouts, so no need for gradient calculations
        with torch.no_grad():
            s = self.batch(s)  # (1, S)
            a = self.batch_forward(s)  # (1, A)
            return self.unbatch(a)

    def log_prob(self, s, a):
        # These methods are used during rollouts, so no need for gradient calculations
        with torch.no_grad():
            s, a = self.batch(s), self.batch(a)  # (1, S) and (1, A)
            log_p = self.batch_log_prob(s, a)  # (1, 1)
            return self.unbatch(log_p)

    def batch_forward(self, s):
        # The batched methods are used during loss calculations, so gradient information should be kept track of here
        raise NotImplementedError("Subclasses should implement the batch forward method.")

    def batch_log_prob(self, s, a):
        # The batched methods are used during loss calculations, so gradient information should be kept track of here
        raise NotImplementedError("Subclasses should implement the batch prob method.")


class DiscretePolicy(Policy):
    """Epsilon-greedy policy derived from an actor or critic with a discrete action space."""

    def __init__(self, actor=None, critic=None, eps=0):
        assert actor is not None or critic is not None
        model = critic if actor is None else actor
        super().__init__(model)
        self.actor = actor
        self.critic = critic
        self.eps = eps

    def batch_forward(self, s):
        B = s.size(0)
        # Sample random actions
        ar = torch.randint(self.A, size=(B,))
        ar = self.ad_transform.unravel(ar)  # (B, A)
        # Sample model actions
        if self.actor is None:
            am = torch.argmax(self.critic(s), dim=1)  # (B,)
            am = self.ad_transform.unravel(am)  # (B, A)
        else:
            am = self.actor.sample(s)  # (B, A)
        # Multiplex random and model actions
        a = torch.where(torch.rand((B, 1)) < float(self.eps), ar, am)  # (B, A)
        return a

    def batch_log_prob(self, s, a):
        if self.actor is None:
            am = torch.argmax(self.critic(s), dim=1)  # (B,)
            if a is None:
                p = torch.nn.functional.one_hot(am, self.A)  # (B, A)
            else:
                a = self.ad_transform.ravel(a)  # (B,)
                p = torch.where(am == a, torch.tensor(1.0), torch.tensor(0.0))  # (B,)
                p = torch.unsqueeze(p, dim=1)  # (B, 1)
        else:
            p = self.actor(s, a).exp()  # (B, 1)
        # Take epsilon-greedy exploration into account
        log_pe = torch.log((1 - float(self.eps)) * p + float(self.eps) / self.A)
        return log_pe


class ContinuousPolicy(Policy):
    """Policy derived from an actor with a continuous action space."""

    def __init__(self, actor, eps=0):
        super().__init__(actor)
        self.actor = actor
        self.sigma_min = eps
        self.sigma_max = 3.0

    def batch_forward(self, s):
        # Sample model actions
        a = self.actor.sample(s, sigma_min=float(self.sigma_min), sigma_max=float(self.sigma_max))  # (B, A)
        return a

    def batch_log_prob(self, s, a):
        log_p = self.actor(s, a, sigma_min=float(self.sigma_min), sigma_max=float(self.sigma_max))  # (B, 1)
        return log_p


class MaxEntropyTeacher(Policy):

    def __init__(self):
        super().__init__(Model(0, 0, False))

    def batch_log_prob(self, s, a):
        return torch.zeros((s.size(0), 1))

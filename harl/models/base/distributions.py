"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from typing import Optional, Sequence, Union
from harl.utils.models_tools import init, get_init_method


def repeat_sample(sample_fn, available_action):
    available_action = available_action.bool()
    cnt = 0
    while cnt < 20:
        cnt += 1
        actions = sample_fn()
        valid = torch.gather(available_action, dim=-1, index=actions.unsqueeze(-1))==1
        if valid.all():
            break
    if cnt > 1:
        print("Sample multiple times:", cnt)
    return actions

class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""
    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def sample(self):
        if self.mask is not None:
            return repeat_sample(super().sample, self.mask).unsqueeze(-1)
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)
        self.available_actions = None

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions==0] = -1e10
        #     return MaskedCategorical(logits=x, mask=available_actions)
        return FixedCategorical(logits=x, mask=available_actions)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)


# https://pytorch.org/rl/_modules/torchrl/modules/distributions/discrete.html#MaskedCategorical
class MaskedCategorical(torch.distributions.Categorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to to masked items will be zeroed and the probability
            re-normalized along its last dimension.
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (float, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in the then mask tensor when
            sparse_mask == True, the padding_value will be ignored.

        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([2, 3, 0, 2, 2, 0, 2, 0, 2, 2])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> print(dist.log_prob(torch.ones_like(sample)))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedCategorical(probs=prob, mask=mask)
        >>> print(dist.log_prob(torch.arange(10)))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
    ) -> None:
        if not ((mask is None) ^ (indices is None)):
            raise ValueError(
                f"A ``mask`` or some ``indices`` must be provided for {type(self)}, but not both."
            )
        if mask is None:
            mask = indices
            sparse_mask = True
        else:
            sparse_mask = False

        mask = mask.bool()
        if probs is not None:
            if logits is not None:
                raise ValueError(
                    "Either `probs` or `logits` must be specified, but not both."
                )
            # unnormalized logits
            probs = probs.clone()
            probs[~mask] = 0
            probs = probs / probs.sum(-1, keepdim=True)
            logits = probs.log()
        logits = self._mask_logits(
            logits,
            mask,
            neg_inf=neg_inf,
            sparse_mask=sparse_mask,
            padding_value=padding_value,
        )
        self.neg_inf = neg_inf
        self._mask = mask
        self._sparse_mask = sparse_mask
        self._padding_value = padding_value
        super().__init__(logits=logits)

    def sample(
        self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()

        ret = super().sample(sample_shape)
        if not self._sparse_mask:
            return ret.unsqueeze(-1)

        size = ret.size()
        # Python 3.7 doesn't support math.prod
        # outer_dim = prod(sample_shape)
        # inner_dim = prod(self._mask.size()[:-1])
        outer_dim = torch.empty(sample_shape, device="meta").numel()
        inner_dim = self._mask.numel() // self._mask.size(-1)
        idx_3d = self._mask.expand(outer_dim, inner_dim, -1)
        ret = idx_3d.gather(dim=-1, index=ret.view(outer_dim, inner_dim, 1))
        ret =  ret.view(size)
        return ret.unsqueeze(-1)


    # # # TODO: Improve performance here.
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self._sparse_mask:
            return super().log_prob(value)

        idx_3d = self._mask.view(1, -1, self._num_events)
        val_3d = value.view(-1, idx_3d.size(1), 1)
        mask = idx_3d == val_3d
        idx = mask.int().argmax(dim=-1, keepdim=True)
        ret = super().log_prob(idx.view_as(value))
        # Fill masked values with neg_inf.
        ret = ret.view_as(val_3d)
        ret = ret.masked_fill(
            torch.logical_not(mask.any(dim=-1, keepdim=True)), self.neg_inf
        )
        ret = ret.resize_as(value)
        return ret
    
    def log_probs(self, actions):
        return (
            self
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )


    @staticmethod
    def _mask_logits(
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        neg_inf: float = float("-inf"),
        sparse_mask: bool = False,
        padding_value: Optional[int] = None,
    ) -> torch.Tensor:
        if mask is None:
            return logits

        if not sparse_mask:
            return logits.masked_fill(~mask, neg_inf)

        if padding_value is not None:
            padding_mask = mask == padding_value
            if padding_value != 0:
                # Avoid invalid indices in mask.
                mask = mask.masked_fill(padding_mask, 0)
        logits = logits.gather(dim=-1, index=mask)
        if padding_value is not None:
            logits.masked_fill_(padding_mask, neg_inf)
        return logits
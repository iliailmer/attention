# Muon optimizer
# implementation based on https://kellerjordan.github.io/posts/muon/

import torch
from torch.optim import Optimizer


def newtonschulz5(G, steps=5, eps=1e-7):
    # see listing in https://kellerjordan.github.io/posts/muon/
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G  # .bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=False, ortho_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ortho_steps=ortho_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)
                state = self.state[p]

                if len(state) == 0:
                    state["velocity"] = torch.zeros_like(p.data)  # B_t
                v = state["velocity"] = state["velocity"] * momentum + grad
                if nesterov:
                    update = grad + momentum * v
                else:
                    update = v
                if p.ndim == 2:
                    ns_ort = newtonschulz5(update, steps=group["ortho_steps"])
                    p.data.add_(ns_ort, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)

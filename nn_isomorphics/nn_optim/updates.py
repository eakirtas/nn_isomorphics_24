import torch as T


class M_ABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1, lr_out=1, g=None):
        p.addcmul_(p.abs(), grad.mul(lr_in).tanh(), value=-lr_out)

    @T.no_grad()
    def tanh_part(self, grad, lr_in=1):
        return grad.mul(lr_in).tanh()

    @T.no_grad()
    def update(self, p, grad, lr_out=1):
        p.addcmul_(p.abs(), grad, value=-lr_out)

    def __repr__(self):
        return "M_ABS"


class N_Clip(object):
    @T.no_grad()
    def __call__(self, p, grad, lr):
        if p.grad is not None:
            p.add_(grad, alpha=-lr).clamp_min_(0)

    def __repr__(self):
        return "N_CLIP"

"""Single-device gradient replay for a joint loss over microbatch outputs.

Gao et al. https://arxiv.org/abs/2101.06983 section 3.3; author reference
https://github.com/luyug/GradCache. Parameters must remain fixed during a call.
Forward must not update running statistics or other mutable model buffers.
Caller owns autocast, gradient zeroing, clipping and the single optimizer step.
CPU and explicitly supplied CUDA device RNGs are replayed, including dropout.
"""
import torch


def cached_backward(forward, batches, loss_fn, cuda_devices=()):
    if not batches:
        raise ValueError('An update must contain at least one microbatch')
    devices = list(cuda_devices)
    states, values = [], []
    with torch.no_grad():
        for batch in batches:
            states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]))
            value = forward(batch)
            if value.ndim != 2 or not torch.isfinite(value).all():
                raise ValueError('Expected finite [examples, features] output')
            values.append(value.detach())
    sizes = [len(value) for value in values]
    joint = torch.cat(values).detach().requires_grad_()
    loss = loss_fn(joint)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError('Expected finite scalar loss')
    gradient, = torch.autograd.grad(loss, joint)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite cached gradient')
    for batch, state, expected, grad in zip(batches, states, values, gradient.split(sizes)):
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(state[0])
            for device, rng in zip(devices, state[1]):
                torch.cuda.set_rng_state(rng, device)
            actual = forward(batch)
            # Do not silently train on gradients from a different stochastic pass.
            torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
            torch.autograd.backward(actual, grad)
    return loss.detach()

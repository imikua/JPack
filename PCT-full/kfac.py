import math
import jittor as jt
import jittor.nn as nn
import numpy as _np
from tools import AddBias


def compute_cov_a(a, classname, layer_info, fast_cnn):
    # Flatten to 2D if needed: (batch*..., features) for correct covariance computation
    if len(a.shape) > 2:
        a = a.reshape((-1, a.shape[-1]))
    batch_size = a.shape[0]
    if classname == 'Conv2d':
        raise NotImplementedError("Conv2d not used in this project")
    elif classname == 'AddBias':
        a = jt.ones((a.shape[0], 1))
    return a.transpose(0, 1) @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    # Flatten to 2D if needed (except AddBias which handles 3D itself)
    if classname != 'AddBias' and len(g.shape) > 2:
        g = g.reshape((-1, g.shape[-1]))
    batch_size = g.shape[0]
    if classname == 'Conv2d':
        raise NotImplementedError("Conv2d not used in this project")
    elif classname == 'AddBias':
        g = g.reshape(g.shape[0], g.shape[1], -1)
        g = g.sum(-1)
    g_ = g * batch_size
    return g_.transpose(0, 1) @ (g_ / g.shape[0])


def update_running_stat(aa, m_aa, momentum):
    """EMA update using numpy roundtrip to prevent Jittor graph chain accumulation."""
    aa_np = aa.data
    m_aa_np = m_aa.data
    if not isinstance(aa_np, _np.ndarray):
        aa_np = _np.array(aa_np)
    if not isinstance(m_aa_np, _np.ndarray):
        m_aa_np = _np.array(m_aa_np)
    result = momentum * m_aa_np + (1.0 - momentum) * aa_np
    m_aa.assign(jt.array(result.astype(_np.float32)))


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        # CRITICAL: Use .data (numpy) to detach bias from computation graph,
        # then create a fresh jt.array so AddBias gets a leaf variable.
        # This matches PyTorch's module.bias.data behavior.
        bias_data = module.bias.data
        if not isinstance(bias_data, _np.ndarray):
            bias_data = _np.array(bias_data)
        self.add_bias = AddBias(jt.array(bias_data.copy().astype(_np.float32)))
        self.module.bias = None

    def execute(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class KFACOptimizer(object):
    """KFAC Optimizer for Jittor.

    KEY DESIGN: Jittor does NOT support loss.backward() or reliable jt.grad()
    for deep computation graphs. The official API is optimizer.step(loss).

    To extract raw gradients for KFAC preconditioning, we use an internal SGD
    optimizer with lr=1.0, momentum=0.0:
      1) Save param values as numpy
      2) _sgd.step(loss)  →  p_new = p_old - 1.0 * grad
      3) grad = p_old - p_new
      4) Restore p_old
      5) Apply KFAC preconditioning on extracted grad
      6) Update params with momentum

    For Fisher stats, we still use jt.grad on module OUTPUTS (shorter graph,
    more reliable than gradients w.r.t. all parameters).
    """

    def __init__(self, model, lr=0.25, momentum=0.9, stat_decay=0.99,
                 kl_clip=0.001, damping=1e-2, weight_decay=0,
                 fast_cnn=False, Ts=1, Tf=10):

        def _should_skip_split_bias(child):
            # Empirically, Jittor's gradient propagation becomes unstable for the
            # final scalar-value head after wrapping `Linear(..., out=1)` into
            # `SplitBias`. Keep those heads as plain Linear+bias and update them
            # with first-order gradients instead of KFAC preconditioning.
            return (
                child.__class__.__name__ == 'Linear'
                and hasattr(child, 'weight')
                and len(child.weight.shape) == 2
                and int(child.weight.shape[0]) == 1
            )

        def split_bias(module):
            for mname, child in list(module.named_children()):
                if hasattr(child, 'bias') and child.bias is not None:
                    if _should_skip_split_bias(child):
                        continue
                    new_mod = SplitBias(child)
                    _replace_child(module, mname, new_mod)
                else:
                    split_bias(child)

        def _replace_child(parent, name, new_child):
            """Replace a child module in parent, handling both Sequential and regular modules."""
            if isinstance(parent, nn.Sequential):
                idx = int(name)
                # Jittor Sequential stores layers in a list attribute
                if hasattr(parent, 'layers') and isinstance(parent.layers, list):
                    parent.layers[idx] = new_child
                # Also set as named attribute
                try:
                    setattr(parent, name, new_child)
                except Exception:
                    pass
            else:
                setattr(parent, name, new_child)

        split_bias(model)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias', 'ScalarValueHead'}
        self.modules = []
        self.model = model

        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                if classname in ['Linear', 'Conv2d'] and module.bias is not None:
                    # Unsplit layers are intentionally excluded from KFAC
                    # tracking; they will still receive raw first-order updates
                    # in Phase 5 below.
                    continue
                self.modules.append(module)

        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self._inputs = {}
        self._outputs = {}
        self._velocity = {}

        self.momentum = momentum
        self.stat_decay = stat_decay
        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay
        self.fast_cnn = fast_cnn
        self.Ts = Ts
        self.Tf = Tf
        self.acc_stats = False
        self._capturing = False

        self._wrap_modules()
        print(f"[KFAC] Initialized: {len(self.modules)} modules tracked, "
              f"{sum(1 for _ in model.parameters())} total params")

    def _wrap_modules(self):
        """Wrap execute methods of target modules to capture inputs and outputs."""
        for m in self.modules:
            original_execute = m.execute
            kfac_ref = self

            def make_wrapped(mod, orig_exec):
                def wrapped_execute(*args, **kwargs):
                    output = orig_exec(*args, **kwargs)
                    if kfac_ref._capturing:
                        if len(args) > 0:
                            kfac_ref._inputs[mod] = args[0]
                        kfac_ref._outputs[mod] = output
                    return output
                return wrapped_execute

            m.execute = make_wrapped(m, original_execute)

    def zero_grad(self):
        """No-op, Jittor handles gradients automatically."""
        pass

    def _update_input_stats(self):
        """Update input covariance stats (m_aa) from saved inputs."""
        for m in self.modules:
            if m not in self._inputs:
                continue
            classname = m.__class__.__name__
            a = self._inputs[m].stop_grad()
            aa = compute_cov_a(a, classname, None, self.fast_cnn)
            if self.steps == 0:
                aa_np = aa.data
                if not isinstance(aa_np, _np.ndarray):
                    aa_np = _np.array(aa_np)
                self.m_aa[m] = jt.array(aa_np.copy().astype(_np.float32))
            else:
                update_running_stat(aa, self.m_aa[m], self.stat_decay)

    def _update_output_grad_stats(self, loss):
        """Update output gradient covariance stats (m_gg) via jt.grad on module outputs."""
        outputs = []
        output_modules = []
        for m in self.modules:
            if m in self._outputs:
                outputs.append(self._outputs[m])
                output_modules.append(m)
        if not outputs:
            return

        # jt.grad on module outputs (short graph) is reliable
        grad_outputs = jt.grad(loss, outputs, retain_graph=True)
        for m, g_out in zip(output_modules, grad_outputs):
            classname = m.__class__.__name__
            gg = compute_cov_g(g_out, classname, None, self.fast_cnn)
            if self.steps == 0:
                gg_np = gg.data
                if not isinstance(gg_np, _np.ndarray):
                    gg_np = _np.array(gg_np)
                self.m_gg[m] = jt.array(gg_np.copy().astype(_np.float32))
            else:
                update_running_stat(gg, self.m_gg[m], self.stat_decay)

    def _clear_captured(self):
        """Clear saved inputs/outputs to release computation graph references."""
        self._inputs = {}
        self._outputs = {}

    def backward_and_step(self, loss, retain_graph=False, max_grad_norm=None):
        """Combined backward + KFAC preconditioning + step.

        Uses optimizer.step(loss) to extract gradients (the official Jittor way).
        All preconditioning and parameter updates done in numpy to prevent graph accumulation.
        """

        if self.acc_stats:
            self._update_input_stats()
            self._update_output_grad_stats(loss)
            self._clear_captured()
            return

        # ---- Phase 1: Extract gradients with Jittor's native backward ----
        # Save current parameters as numpy for the later manual momentum update.
        all_params = list(self.model.parameters())
        saved_np = {}
        for p in all_params:
            pd = p.data
            if not isinstance(pd, _np.ndarray):
                pd = _np.array(pd)
            saved_np[id(p)] = pd.copy().astype(_np.float32)
        # Recreate a probe optimizer every step to avoid any stale internal
        # state from previous backward calls.
        probe_optim = nn.SGD(self.model.parameters(), lr=1.0, momentum=0.0)
        probe_optim.backward(loss)

        grad_np = {}
        nonzero = 0
        for p in all_params:
            try:
                g = p.opt_grad(probe_optim)
            except Exception:
                g = None

            if g is None:
                g_np = _np.zeros_like(saved_np[id(p)], dtype=_np.float32)
            else:
                gd = g.data
                if not isinstance(gd, _np.ndarray):
                    gd = _np.array(gd)
                g_np = gd.astype(_np.float32)

            grad_np[id(p)] = g_np
            if _np.abs(g_np).sum() > 1e-10:
                nonzero += 1

        jt.sync_all()

        # Gradient clipping
        if max_grad_norm is not None:
            total_norm = sum(float((g ** 2).sum()) for g in grad_np.values()) ** 0.5
            if total_norm > max_grad_norm:
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                for pid in grad_np:
                    grad_np[pid] = grad_np[pid] * clip_coef

        if self.steps % 10 == 0:
            total_grad_norm = sum(float((g ** 2).sum()) for g in grad_np.values()) ** 0.5
            print(f'[KFAC] step={self.steps} params={len(all_params)} '
                  f'nonzero_grads={nonzero}/{len(all_params)} '
                  f'grad_norm={total_grad_norm:.6f} '
                  f'm_aa={len(self.m_aa)} m_gg={len(self.m_gg)} Q_g={len(self.Q_g)}')

        # ---- Phase 2: weight decay ----
        if self.weight_decay > 0:
            for p in all_params:
                grad_np[id(p)] = grad_np[id(p)] + self.weight_decay * saved_np[id(p)]

        # ---- Phase 3: KFAC preconditioning (all numpy) ----
        updates_np = {}
        for m in self.modules:
            plist = list(m.parameters())
            if not plist:
                continue
            p = plist[0]
            g_np = grad_np.get(id(p))
            if g_np is None:
                continue

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0 and m in self.m_gg and m in self.m_aa:
                gg_data = self.m_gg[m].data
                aa_data = self.m_aa[m].data
                if not isinstance(gg_data, _np.ndarray):
                    gg_data = _np.array(gg_data)
                if not isinstance(aa_data, _np.ndarray):
                    aa_data = _np.array(aa_data)
                d_g_np, Q_g_np = _np.linalg.eigh(gg_data)
                d_a_np, Q_a_np = _np.linalg.eigh(aa_data)
                self.d_g[m] = (d_g_np * (d_g_np > 1e-6)).astype(_np.float32)
                self.Q_g[m] = Q_g_np.astype(_np.float32)
                self.d_a[m] = (d_a_np * (d_a_np > 1e-6)).astype(_np.float32)
                self.Q_a[m] = Q_a_np.astype(_np.float32)

            if m in self.Q_g and m in self.Q_a:
                v1 = self.Q_g[m].T @ g_np @ self.Q_a[m]
                v2 = v1 / (self.d_g[m].reshape(-1, 1) * self.d_a[m].reshape(1, -1) + la)
                v = (self.Q_g[m] @ v2 @ self.Q_a[m].T).reshape(g_np.shape).astype(_np.float32)
                updates_np[id(p)] = v
            else:
                updates_np[id(p)] = g_np

        # ---- Phase 4: KL clip ----
        vg_sum = 0.0
        for p in all_params:
            if id(p) in updates_np:
                v = updates_np[id(p)]
                g = grad_np.get(id(p), v)
                vg_sum += float((v * g * self.lr * self.lr).sum())
        nu = min(1.0, math.sqrt(self.kl_clip / max(vg_sum, 1e-10)))

        # ---- Phase 5: SGD with momentum + parameter update (all numpy) ----
        lr_actual = self.lr * (1.0 - self.momentum)
        for param_idx, p in enumerate(all_params):
            pid = id(p)
            fg = updates_np.get(pid, grad_np.get(pid, _np.zeros(p.shape, dtype=_np.float32))) * nu

            velocity_key = param_idx
            if velocity_key not in self._velocity or self._velocity[velocity_key].shape != fg.shape:
                self._velocity[velocity_key] = _np.zeros_like(fg)
            self._velocity[velocity_key] = self.momentum * self._velocity[velocity_key] + fg

            new_p = (saved_np[pid] - lr_actual * self._velocity[velocity_key]).astype(_np.float32)
            p.update(jt.array(new_p))

        self.steps += 1
        self._clear_captured()
        jt.gc()

    def step(self):
        """For compatibility - actual step happens in backward_and_step."""
        pass


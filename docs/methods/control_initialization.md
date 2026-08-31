# Control Initialization Strategies: `ControlInitMethod`

This document explains, precisely and exhaustively, what each member of
`ControlInitMethod` actually does — mathematically and in code — so you can
verify that `COLD`, `WARM`, and `RANDOMIZED` implement what you intend, the
same way `docs/methods/training_modes.md` does for the two training modes.
Every claim below is tied to a specific class or function you can inspect or
test directly, and the two documents are complementary: initialization
chooses *where each per-time-step refinement starts from*; the training mode
(`"end_to_end"` vs `"layerwise"`) chooses *how the parameters that refinement
uses get optimized*. The two axes are orthogonal — see §5.

Scope: `src/mbl/models/iterative/initializers.py` (`ControlInitializer`,
`ControlInitMethod`, `build_control_initializer`), its two consumers
`src/mbl/models/unfolded/base.py` (`UnfoldedController`) and
`src/mbl/models/analytic/iterative_gd.py`
(`AnalyticalIterativeGDController`), and the application-level wiring in
`src/mbl/applications/recipes/unfolded.py` and
`src/mbl/applications/studies/nb03_unfolding.py`.

---

## 1. What is being initialized, and when

A `ControlInitializer` proposes the starting point $u^{(0)}$ that an
`IterativeRefinement` then refines via $J$ gradient-descent steps (see
`docs/methods/training_modes.md` §1.1 for that inner loop). The abstract
contract (`ControlInitializer.__call__`,
`src/mbl/models/iterative/initializers.py`) is:

$$
u^{(0)}_t = \text{initializer}(t, y_t, u_{t-1}) ,
$$

where $u_{t-1}$ is the **previous time step's already-refined control**
(`None` at $t=0$). The critical fact to internalize before reading the
per-method sections below — easy to get wrong from the name alone — is:

> **The initializer is called at *every* time step $t = 0, \dots,
> \text{horizon}-1$ of a rollout, not just at $t=0$.** "Control
> initialization" means "how does each step's *inner* refinement get
> seeded", not "how is the trajectory's first control chosen."

This is visible directly in both call sites. Inside
`UnfoldedController.get_control_policy`
(`src/mbl/models/unfolded/base.py`):

```python
def policy(t: int, y: torch.Tensor) -> torch.Tensor:
    nonlocal prev_u
    u = self.config.control_initializer(t, y, prev_u)   # <- called every t
    u = self.config.iterative_refinement(t, y, u)
    prev_u = u.detach()   # zero-copy; breaks the autograd graph across time steps
    return u
```

and `policy(t, y_t)` is itself invoked once per time step by
`StateSpaceSystem.run` (`src/mbl/core/system/state_space_system.py`): "At each
step $t$: observes $y_t$, queries $u_t = \text{policy}(t, y_t)$, then
advances $x_{t+1}$." `prev_u` is a closure-local variable created fresh by
`get_control_policy()`, which itself is called once per `forward()` — so its
scope is **exactly one rollout**; nothing about it persists from one
training batch/epoch to the next (only the initializer object itself — and,
for `RANDOMIZED`, the RNG stream it owns — persists across the whole run;
see §4.3).

`prev_u = u.detach()` is equally load-bearing: the control fed forward as
next step's warm-start seed is **detached from the autograd graph**. Whatever
gradient flows to $u_t$'s own refinement never backpropagates through the
warm-start link into $u_{t-1}$'s refinement — the temporal chain WARM builds
(§3) is a *forward-pass-only* convenience, never a hidden extra path for
gradients to travel through time.

---

## 2. `ControlInitMethod.COLD` — the classical zero start

### 2.1 Math

$$
u^{(0)}_t = 0 \in \mathbb{R}^m \qquad \text{for every } t .
$$

Every time step's inner refinement starts from the same fixed origin,
completely independent of $t$ and of whatever control was realized at
$t - 1$.

### 2.2 Code

`ConstantInitializer(0.0, control_dim, dtype, device)` — a `torch.full`
fill, ignoring both `t` and `prev_u` entirely:

```python
def __call__(self, t, y, prev_u=None):
    return torch.full((y.shape[0], self.control_dim), self.constant, dtype=..., device=...)
```

`build_control_initializer` names this the historical default: "the
classical, and historically the only, deep-unfolding cold start" — every
inner refinement in the original formulation began from $u=0$ before deep
unfolding's outer-loop training was even introduced.

### 2.3 What to check

- With `COLD`, the realized $u^{(0)}_t$ must be **identically zero** at every $t$, every rollout, regardless of `prev_u` — instrument `ConstantInitializer.__call__` (or inspect the first refinement iterate) at $t>0$ and confirm it never depends on the previous step's outcome.
- `COLD` is deterministic and carries no RNG state: two rollouts with identical `(x0, w, v)` batches must produce byte-identical $u^{(0)}$ sequences regardless of how many prior epochs/phases have run.

---

## 3. `ControlInitMethod.WARM` — temporal warm-starting

### 3.1 Math

$$
u^{(0)}_t =
\begin{cases}
0 & t = 0 \\
u_{t-1} & t > 0
\end{cases}
$$

where $u_{t-1}$ is $t-1$'s **final, refined** control (after all $J$ inner
iterations), not $t-1$'s own seed. Intuitively: since the optimal LQR
control trajectory is typically smooth in $t$ (particularly under
process/measurement noise that is small relative to the trend), the
converged control from one instant is usually a much better starting guess
for the next instant's inner optimization than a fixed zero — fewer inner
iterations are then needed to reach the same accuracy, or the same $J$
iterations reach a better local optimum.

### 3.2 Code

`WarmStartInitializer` (`src/mbl/models/iterative/initializers.py`) wraps a
fallback initializer — always `ConstantInitializer(0.0, ...)`, per
`build_control_initializer` — used only at $t=0$:

```python
def __call__(self, t, y, prev_u=None):
    return prev_u if prev_u is not None else self.fallback(t, y, None)
```

The chaining itself is not this class's responsibility: `prev_u` is threaded
by the *caller* (`UnfoldedController.get_control_policy`'s `policy` closure,
§1) — `WarmStartInitializer` only decides what to do given whatever
`prev_u` it's handed at time $t$. This is why the SAME `WarmStartInitializer`
class works unmodified for the whole-horizon macro-sweep solver too (§4.2):
the temporal chaining logic lives once, at each caller's own policy closure,
and `WarmStartInitializer` is reused verbatim by both.

### 3.3 The stop-gradient subtlety (§1's `prev_u = u.detach()`)

Because `prev_u` is detached before being handed to the next step's
`WarmStartInitializer` call, **the warm-start link never becomes a gradient
path**. Concretely: $\partial \, u^{(0)}_t / \partial \theta$ where $\theta$
is a learned parameter used by step $t-1$'s refinement is *not* part of the
computation graph — only $u^{(0)}_t$'s *value* (a plain tensor, gradient-free
at that point) crosses the time-step boundary. Backprop through the unrolled
recursion (`docs/methods/training_modes.md` §1.3) therefore still trains
each time step's $J$-iteration refinement independently with respect to the
learned $\alpha^{(j)}$ (and $P$) — WARM changes *where* that per-step
optimization starts, never *what* gets differentiated.

### 3.4 What to check

- At $t=0$, `WARM` must behave identically to `COLD` (both fall back to $0$) — a run with `WARM` and one with `COLD` should produce identical first-time-step inner-iterate trajectories, all else equal.
- At $t>0$, the realized $u^{(0)}_t$ should equal the *previous* time step's **final** refined control (after `iterative_refinement`'s full $J$ iterations), not e.g. its own $u^{(0)}_{t-1}$ or an intermediate inner iterate — verify by comparing `prev_u` (as captured in the closure) against the last entry of `iterative_refinement`'s inner trajectory for step $t-1$.
- `WARM` never touches any learnable parameter or RNG state, so it is (like `COLD`) fully deterministic given `(x0, w, v)` — any observed run-to-run variation under `WARM` indicates noise/batch nondeterminism elsewhere, not this initializer.
- Confirm gradients genuinely do not cross the time-step boundary: e.g. zero out $\alpha^{(j)}$'s gradient contribution from every time step except one, and check that time step's loss gradient is unaffected by whether `WARM` or `COLD` was used for the *following* step (since that step's seed is detached, it cannot feed a gradient backward into an earlier step either).

---

## 4. `ControlInitMethod.RANDOMIZED` — seeded Gaussian proposals

### 4.1 Math

$$
u^{(0)}_t \sim \mathcal N\bigl(0, \; \texttt{random\_std}^2 I_m\bigr) \qquad \text{independently redrawn at every } t ,
$$

drawn from **one persistent, seeded stream** shared across the initializer's
entire lifetime (see §4.3) — never resampled per epoch from a fresh seed,
and never reusing the same draw twice. Both `t` and `prev_u` are ignored
entirely (§1's warning applies most sharply here: this is not "randomize the
trajectory's first control", it is "propose an independent random guess at
every single time step, every single rollout").

### 4.2 Code

`SamplerInitializer` wraps a `Distribution` (`src/mbl/models/samplers.py`):

```python
def __call__(self, t, y, prev_u=None):
    return torch.as_tensor(self.sampler(y.shape[0], self.control_dim), dtype=..., device=...)
```

`build_control_initializer` supplies a `TorchGaussianDistribution(std=random_std, seed=seed, dtype=dtype, device=device)`.
Its `__call__` draws on its **own internal `torch.Generator`**, seeded once
at construction (`torch.Generator().manual_seed(seed)`), and every call
**advances that same stream** — this is a persistent RNG object, not a
"reset to `seed` every draw" convenience:

```python
noise = torch.randn(shape, generator=self._generator, dtype=self.dtype, device=self._generator.device).to(self.device)
return noise * self.std + self.mean   # mean is fixed at 0.0
```

Two details worth internalizing precisely:

- **Device-native, then moved.** The draw happens on `self._generator`'s own device (CPU, unless a device-matched generator was explicitly supplied) and is moved to `self.device` only afterward — torch's RNG algorithm differs per device, so sampling directly on a non-CPU device would silently break the bit-for-bit-reproducible-for-a-given-seed contract the moment `self.device` differs from the generator's device. A residency change (CPU → CUDA) must never also be a numerics change.
- **`get_signature`** reports `{"type": "Gaussian", "seed": seed, "mean": 0.0, "std": random_std}` — the *seed*, not the generator's live (mutating) internal state. The seed is what makes a `RANDOMIZED` run's provenance reproducible and cacheable; the generator's moment-to-moment state is deliberately not signed (it isn't meaningful specification data — the seed alone determines the entire subsequent draw sequence).

### 4.3 The persistence trap — the thing most worth double-checking

`build_control_initializer` is called exactly **once** per controller build
(inside `build_unfolded_controller`), producing exactly **one**
`SamplerInitializer` (and therefore one `TorchGaussianDistribution`, one
`torch.Generator`) that lives for the controller's **entire lifetime** —
every epoch, every training batch, every phase of a layer-wise schedule
(`docs/methods/training_modes.md` §3), and every subsequent evaluation
rollout, all draw from the *same, ever-advancing* stream. Consequences worth
verifying against your actual intent:

- **The realized $u^{(0)}$ distribution's *individual draws* are never repeated.** Epoch 50's random proposal at $t=3$ is *not* the same draw as epoch 1's random proposal at $t=3$ — both are draws from the same distribution, at different points along one long, deterministic-given-`seed` sequence. If you intend to test sensitivity to *one fixed* random initial guess held constant across the whole run, `RANDOMIZED` as implemented does **not** give you that — it gives a reproducible *sequence* of different draws, never one fixed draw reused. (`ConstantInitializer` fixed at a nonzero constant, or a `SamplerInitializer` wrapping a `Distribution` you freeze/cache yourself, would be the right tool for that instead.)
- **Under a `LayerwiseTrainingPlan` schedule, later phases see different $u^{(0)}$ realizations than earlier ones** — purely because more calls have already advanced the stream by the time a later phase runs, not because of anything phase-specific. A warm-up phase trained early and one trained late are *not* being compared under matched random initial guesses.
- **Two different contenders built with the same `seed` are *not* secretly sharing draws.** Each `build_unfolded_controller` call constructs its own fresh `torch.Generator().manual_seed(seed)`. Same `seed` + same call shape/dtype/device means their **very first** draw is identical (a freshly-seeded generator's first sample depends only on the seed and the requested shape) — but every draw after that diverges as soon as the two controllers' call cadences differ (different horizons, different epoch counts, different phase structure). Do not rely on same-seed contenders staying draw-synchronized beyond their first call.
- This RNG stream is **entirely separate** from the shared evaluation/training batch sampler (`GaussianBatchSpec`, the Stochastic Fairness & Determinism doctrine's `(x0, w, v)` parity mechanism) — matching `initial_state`/noise realizations across contenders (batch fairness) says nothing about whether their `RANDOMIZED` control-initialization draws happen to align, and vice versa. Don't conflate the two when reasoning about what is and isn't held fixed across a comparison.

### 4.4 What to check

- Two runs built with the same `random_init_std`/`random_init_seed` (and identical shapes/dtype/device) must reproduce **bit-identical** $u^{(0)}$ sequences end to end — this is the determinism contract; any divergence is a bug (e.g. a stray extra sampler call, a device mismatch, or a generator being rebuilt mid-run).
- Confirm the draw is redrawn at *every* $t$ within a single rollout (not just $t=0$) by checking that $u^{(0)}_t$ for $t>0$ is statistically independent of $u^{(0)}_{t-1}$ (unlike `WARM`, which makes them equal by construction).
- Confirm the stream is *not* reset between epochs/phases: sample $u^{(0)}_0$ at the very start of two different epochs of the same run and check they differ (a bug that accidentally rebuilt the initializer, or reseeded per epoch, would make them identical).

---

## 5. The factory, and orthogonality with training mode

`build_control_initializer` (`src/mbl/models/iterative/initializers.py`) is the
single mapping site from the enum to a concrete `ControlInitializer`:

```python
def build_control_initializer(method, *, control_dim, dtype, device, random_std=1.0, seed=0):
    cold = ConstantInitializer(0.0, control_dim, dtype, device)
    if method is ControlInitMethod.COLD:       return cold
    if method is ControlInitMethod.WARM:       return WarmStartInitializer(cold)
    if method is ControlInitMethod.RANDOMIZED: return SamplerInitializer(TorchGaussianDistribution(std=random_std, seed=seed, ...), control_dim, dtype, device)
    raise ValueError(...)
```

`build_unfolded_controller` (`src/mbl/applications/recipes/unfolded.py`) calls
this **once**, inside `UnfoldedBuildSpec`'s construction path, regardless of
which recipe (`UnfoldedRecipe`/`"unfolded"`, i.e. `"end_to_end"`, or
`WarmStartUnfoldedRecipe`/`"unfolded_warmstart"`, i.e. `"layerwise"`) is
building the controller. **This is the structural reason initialization and
training mode are orthogonal**: `ControlInitMethod` selects *which
`ControlInitializer` object* the controller is built with, once, before any
training phase begins; the training mode then decides how the optimizer(s)
schedule updates to the parameters that `iterative_refinement` reads at
every step — the two never interact. Any of `COLD`/`WARM`/`RANDOMIZED`
combines freely with either `"end_to_end"` or `"layerwise"`.

### 5.1 Application-level surface (`NB03Config`)

Every contender's init method is independently selectable:

```python
NB03Config.standard_gd_init_method: ControlInitMethod = ControlInitMethod.COLD   # Standard-GD (fixed step, true P)
UnfoldedModelConfig.init_method:    ControlInitMethod = ControlInitMethod.COLD   # per learned model (Unfolded-alpha / Unfolded-alpha+P)
NB03Config.random_init_std:         float = 1.0    # shared by every contender using RANDOMIZED
```

`random_init_seed` is not a separate top-level field — every `RANDOMIZED`
contender is seeded from the *same* `cfg.seed` (see §4.3's third bullet for
what "same seed, independent generators" actually implies). All three
defaults are `ControlInitMethod.COLD`: unless you opt in, every NB03
contender starts every inner refinement from zero at every time step,
exactly as the "classical" deep-unfolding formulation does.

### 5.2 Provenance: signature folding

`_unfolded_recipe_signature`
(`src/mbl/applications/recipes/unfolded.py`) folds the control-init selection
into a recipe's signature **only when it is non-default**:

```python
if init_method is ControlInitMethod.COLD:
    return signature                       # unchanged — no init_method key at all
signature["init_method"] = init_method.value
if init_method is ControlInitMethod.RANDOMIZED:
    signature["random_init_std"] = random_init_std
    signature["random_init_seed"] = random_init_seed
```

This is deliberate, not an oversight: introducing this knob leaves every
pre-existing cold-start run's cache digest — and any golden-master
regression test pinned to it — byte-identical, while `WARM`/`RANDOMIZED`
correctly become distinct cache entries the moment they're selected. If you
switch a contender's `init_method` and its cached artifact is unexpectedly
*reused* rather than retrained, check first whether you switched *to*
`COLD` (which folds out of the signature) from something else, or whether
`random_init_std`/`seed` actually changed for a `RANDOMIZED` contender (a
`WARM`→`RANDOMIZED` switch changes the signature; a `random_init_std` edit
alone does too, but only while already on `RANDOMIZED`).

---

## 6. The second consumer: whole-horizon macro-sweep refinement

`ControlInitizer` is deliberately algorithm-agnostic
(`src/mbl/models/iterative/initializers.py`'s module docstring): besides the
per-time-step deep-unfolded controller (§1), it is reused verbatim by
`AnalyticalIterativeGDController.solve`
(`src/mbl/models/analytic/iterative_gd.py`), the whole-horizon signal-space
gradient-descent solver. The usage pattern differs in one structural way
worth being precise about:

```python
with torch.no_grad():
    prev_u: torch.Tensor | None = None
    def proposal_policy(t, y):
        nonlocal prev_u
        u = control_initializer(t, y, prev_u)
        prev_u = u.detach()
        return u
    X, _, U = rollout(proposal_policy)      # ONE rollout: builds the FULL U^(0), all time steps
    J = evaluate_cost(X, U)
    for i in range(1, max_iters + 1):
        result = sweep_strategy.sweep(i - 1, U, X, ctx)   # macro-sweeps NEVER call control_initializer again
        U, X, J = result.U_next, result.X_next, result.J_next
```

Here, `control_initializer` is invoked exactly **once per solve**, across
one rollout, to build the *entire* initial control sequence $U^{(0)} \in
\mathbb{R}^{T \times m}$ (chaining across time exactly as in §1, e.g. `WARM`
still means "seed step $t$ from step $t-1$'s value *within this one
rollout*"). Every subsequent macro-iteration ($i = 1, \dots,
\texttt{max\_iters}$, a Jacobi or Gauss-Seidel sweep via `sweep_strategy`)
refines that single $U^{(0)}$ directly — `control_initializer` plays no role
after the first rollout. This is the "whole-horizon macro-sweep" analogue of
what the deep-unfolded controller does per time step at micro-scale:
`ControlInitMethod` always answers "what does the very first proposal look
like, before any refinement has touched it", just at a different
granularity depending on which controller consumes it.

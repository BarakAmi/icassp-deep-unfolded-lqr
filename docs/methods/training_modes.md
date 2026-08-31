# Training Modes for Deep-Unfolded Controllers: `end_to_end` vs. `layerwise`

This document explains, precisely and exhaustively, what the two training
modes of a learned deep-unfolded LQR controller actually do — mathematically
and in code. It exists to let you verify that `"end_to_end"` and
`"layerwise"` implement what they are intended to implement, by tying every
claim to a specific class or function you can inspect or test directly.

Scope: `src/mbl/engine/training_plan.py` (`TrainingPlan`, `LayerwiseTrainingPlan`,
`OptimizerSpec`), `src/mbl/engine/strategy.py` (`GradientDescentStrategy`,
`LayerwiseGradientDescentStrategy`), `src/mbl/models/unfolded/layerwise.py`
(`ParameterActivation`, `LayerFreeze`, `LayerFreezeBuilder`), and the
application wiring in `src/mbl/applications/recipes/unfolded.py` and
`src/mbl/applications/studies/nb03_unfolding.py`.

---

## 1. The shared inner loop

Both training modes optimize the **same** learned objects through the
**same** per-time-step inner iteration; they differ only in *how* the outer
optimization over that iteration's parameters is scheduled. Establishing the
shared math first makes the difference between the two modes precise rather
than impressionistic.

### 1.1 The unrolled gradient-descent update

At time step $t$, starting from $u^{(0)} = 0$ (or another seed under a
non-cold `ControlInitMethod`), a deep-unfolded controller runs a fixed
number $J$ of gradient-descent iterations against the local cost-to-go:

$$
u^{(j+1)} = u^{(j)} - \alpha^{(j)} \, \nabla_u V\bigl(x_t, u^{(j)}\bigr),
\qquad j = 0, \dots, J-1,
$$

$$
\nabla_u V = M\, u + C^\top x_t, \qquad M = R + B^\top P B, \quad C = B^\top P A .
$$

This is implemented once, generically, by
`GradientDescentRefinement.refine_step` / `__call__`
(`src/mbl/models/iterative/refinement.py`):

```python
def refine_step(self, iteration_index, u, *args):
    grad = self.get_gradient(u, *args)
    u_next = self.apply_constraints(
        u - self.step_size.for_iteration(iteration_index) * grad, iteration_index
    )
    return u_next, grad
```

`get_gradient` computes exactly $u \mapsto u \, M + y \, C^\top$
(`LQRGDRefinement.get_gradient`, `src/mbl/models/unfolded/iterative_refinement.py`).
**This inner update is never autograd-differentiated on its own terms** — it
is the closed-form Riccati gradient, evaluated directly. What *is*
autograd-differentiated is the outer composition of $J$ such updates with
respect to the learned parameters that produce $\alpha^{(j)}$ (and,
optionally, $P$); see §1.3.

Two concrete refinements supply $(M, C)$:

| Class | $P$ | Learned quantity |
|---|---|---|
| `StepSizeRefinement` | the **true**, precomputed Riccati $P$ (static `M_stack`/`C_stack`, built once) | only $\alpha^{(j)}$ |
| `RiccatiRefinement` | a **learned** $P$, recomputed from the live parameter whenever it changes (cached per rollout, invalidated in `on_rollout_start`) | $\alpha^{(j)}$ **and** $P$ |

Both are built by the single shared constructor
`build_unfolded_controller` (`src/mbl/applications/recipes/unfolded.py`), which
branches on `UnfoldedKind`:

- `LEARNED_STEP_SIZE` → `StepSizeRefinement` (fixed true $P$).
- `LEARNED_STEP_SIZE_AND_MATRIX` → `RiccatiRefinement` (learned $P$).
- `FIXED` → `StepSizeRefinement` with `step_size.get_raw().requires_grad_(False)` (nothing learned; not covered further here — it never trains).

### 1.2 Reparameterizations (what "learned" actually means)

Neither learned quantity is optimized directly in its constrained form; both
are reparameterized so that unconstrained gradient steps on the raw
`nn.Parameter` can never leave the feasible set.

**Step size** (`StepSizeParameter`, `src/mbl/models/unfolded/parameters.py`):
a raw tensor $\rho \in \mathbb{R}^{J \times m}$ is squashed through a
sigmoid,

$$
\alpha^{(j)} = \sigma(\rho_j) \cdot \alpha_{\max} \in (0, \alpha_{\max})^m ,
$$

initialized so $\sigma(\rho_j) \cdot \alpha_{\max} \approx \alpha_{\text{init}}$
for every row (`StepSizeParameter.initialize`, via the logit of
`alpha_init / alpha_max`). `alpha_max` is a hard reparameterization bound —
gradient descent on $\rho$ can push $\alpha^{(j)}$ arbitrarily close to it
but never past it.

**Riccati-replacement matrix** (`RiccatiMatrixParameter`, same module): a raw
lower-triangular Cholesky factor $L \in \mathbb{R}^{n \times n}$ (initialized
to $L = I$, so $P = I$ initially) reparameterizes

$$
P = L L^\top ,
$$

which is positive semi-definite for *any* real $L$ by construction — the
optimizer can never produce an indefinite "cost-to-go matrix" no matter how
it moves $L$.

### 1.3 What "training" optimizes end to end

Whichever mode is used, one full rollout evaluates the composition of $J$
inner updates (§1.1) at every time step, producing a batch of trajectories
and a scalar cost. Backpropagating through that composition computes
$\partial \text{loss} / \partial \rho$ (and $\partial \text{loss} / \partial L$
when $P$ is learned) — this is the "deep unfolding" / learning-to-optimize
step: the *inner* update stays the closed-form Riccati kernel; the *outer*
loop that decides $\alpha^{(0)}, \dots, \alpha^{(J-1)}$ (and $P$) is standard
backprop. The two training modes differ only in **how the outer
optimization is scheduled across $\rho$'s $J$ rows** — never in the forward
math above.

---

## 2. Mode `"end_to_end"`

### 2.1 What it does

Every row of $\rho$ (and $L$, if present) is trainable from epoch 0, and a
**single** optimizer takes a joint gradient step over all of them, every
epoch, for the plan's full epoch budget. There is no freezing, no phases,
and no schedule — it is standard, flat backpropagation through the unrolled
recursion.

Per training step (`GradientDescentStrategy.step`,
`src/mbl/engine/strategy.py`):

$$
\theta \leftarrow \theta - \eta \, \widehat{\nabla}_\theta \, \frac{1}{|\mathcal B|}\sum_{b \in \mathcal B} \text{cost}_b(\theta),
\qquad \theta = (\rho,\, L) ,
$$

where the expectation is over one sampled training batch $\mathcal B$ (a
`BatchSampler` draw), $\widehat\nabla_\theta$ is the Adam/AdamW/SGD update
computed from that raw gradient (never the raw gradient itself), and this
repeats for `plan.epochs` epochs.

### 2.2 Configuration: `TrainingPlan` / `OptimizerSpec`

```python
TrainingPlan(
    optimizer=OptimizerSpec(name="adam", learning_rate=..., hyperparameters={...}),
    epochs=...,
    gradient_clip_norm=None,   # optional global grad-norm clip, applied to every param
    loss_reduction="mean",     # or "sum" — how the per-batch cost tensor reduces to a scalar
)
```

`OptimizerSpec.build(parameters)` is the **only** place an optimizer is ever
constructed from the spec (`src/mbl/engine/training_plan.py`), and
`TrainingPlan.training_config` derives the logged `TrainingConfig`'s
`learning_rate`/`optimizer_name`/`optimizer_kwargs` from that *same* spec —
so the optimizer actually executed and the one logged/signed can never
drift apart (this is deliberate: it closes what the codebase calls the "C4
provenance defect", a bug class where a different learning rate was logged
than the one actually run).

### 2.3 Code path

```
UnfoldedModelConfig(training_mode="end_to_end", plan=<TrainingPlan>)
  -> family = "unfolded"                                  (nb03_unfolding._learned_unfolded_contender)
  -> UnfoldedRecipe(plan=..., kind=..., ...)               (applications/recipes/unfolded.py)
  -> TrainableRecipe.build_engine (inherited, unmodified)  (applications/recipes/base.py)
       rollout = RolloutModel(controller)
       module  = controller.as_module()
       strategy = GradientDescentStrategy.from_plan(rollout, module, plan)
       Runner(phases=[TrainingPhase(strategy, epochs=plan.epochs, name=label)])
```

`GradientDescentStrategy.from_plan` builds the optimizer over
**`module.parameters()`** — every learnable tensor the controller registers
via `TrainableController.as_module()` — so nothing is excluded and nothing
needs masking: the optimizer's own parameter set already *is* "everything
trainable this run."

### 2.4 What to check to confirm this is happening

- Exactly **one** `TrainingPhase` is passed to `Runner` for an `"unfolded"` family run (`UnfoldedRecipe` never calls `LayerwiseTrainingPlan.compile`).
- `strategy.optimizer.param_groups` contains every row of `step_size` (and `riccati_matrix`, if present) from epoch 0 — none are excluded or gradient-masked.
- `unfolded_parameter_log_summaries`' `step_size (alpha)` narration (via `StructuredTrainingLogCallback`) should show **every row** changing epoch over epoch, from the very first logged epoch — a row that stays frozen at its initial value under `"end_to_end"` would indicate a bug, not the intended behavior.
- The persisted provenance (`EngineTrainedSynthesizer.synthesize`'s `provenance["training"]`) is `recipe.plan.get_signature()` — a flat `TrainingPlan` signature, not a `LayerwiseTrainingPlan` one.

---

## 3. Mode `"layerwise"` (greedy "warm-start" training)

### 3.1 What it does

Instead of one optimizer touching every row at once, training is split into
$J$ **warm-up phases** — phase $j$ trains (a subset of) row $j$ of $\rho$
while every other row stays exactly frozen at its already-converged value —
followed by an **optional trailing refinement phase** that unfreezes
everything (rows *and*, for the first time by default, $P$) for joint
fine-tuning. This is a *warm start* in the literal sense: phase $j+1$
inherits phase $j$'s converged rows as its initialization rather than
starting the whole $J$-row problem from scratch.

Schedule, in order:

$$
\underbrace{\text{phase}_0}_{\text{row } 0 \text{ only}} \to
\underbrace{\text{phase}_1}_{\text{row } 1 \text{ (or } 0..1\text{)}} \to \cdots \to
\underbrace{\text{phase}_{J-1}}_{\text{row } J-1 \text{ (or } 0..J-1\text{)}} \to
\underbrace{\text{refinement}}_{\text{all rows} + P \text{ (optional)}}
$$

### 3.2 Configuration: `LayerwiseTrainingPlan`

```python
LayerwiseTrainingPlan(
    optimizer=OptimizerSpec(...),        # every phase's OWN optimizer is built from this spec
    warmup_epochs_per_layer=...,         # epochs per per-layer phase
    refinement_epochs=0,                 # epochs of the trailing joint phase; 0 disables it
    train_matrix_from="refinement",      # "refinement" | "each" | "last_layer"
    activation="single",                 # "single" | "cumulative"
    gradient_clip_norm=None,
    loss_reduction="mean",
)
```

`LayerwiseTrainingPlan.compile(num_layers, has_matrix)`
(`src/mbl/engine/training_plan.py`) turns this into `num_layers` warm-up
`PhaseSpec`s plus (if `refinement_epochs > 0`) one trailing `PhaseSpec`:

```python
for layer in range(num_layers):
    rows = frozenset(range(layer + 1)) if activation == "cumulative" else frozenset({layer})
    train_matrix = has_matrix and (
        train_matrix_from == "each"
        or (train_matrix_from == "last_layer" and layer == num_layers - 1)
    )
    # -> PhaseSpec(ParameterActivation(rows, train_matrix), warmup_epochs_per_layer, f"warmup_layer_{layer}")
if refinement_epochs > 0:
    # -> PhaseSpec(ParameterActivation("all", has_matrix), refinement_epochs, "refinement")
```

**`activation`** controls which rows a warm-up phase activates:

- `"single"` (the classic greedy default): phase $j$ activates *only* row $j$; rows $0, \dots, j-1$ stay frozen at their phase-$j-1$ values, and rows $j+1, \dots, J-1$ have not trained yet at all.
- `"cumulative"`: phase $j$ activates rows $0, \dots, j$ — every row trained so far keeps training, not just the newest one.

**`train_matrix_from`** controls *when* the learned matrix $P$ first
becomes trainable, and this is the field most worth double-checking against
your intent, since a mismatched choice silently changes what gets learned:

| Value | $P$ trains during... |
|---|---|
| `"refinement"` (default) | **only** the trailing refinement phase. **If `refinement_epochs == 0`, $P$ never trains at all** — it stays at its initial value ($P = I$) for the entire run. This is a real, silent trap: the compiled formula for `train_matrix` is `has_matrix and (train_matrix_from == "each" or (train_matrix_from == "last_layer" and layer == num_layers - 1))`, which is unconditionally `False` for every warm-up phase when `train_matrix_from == "refinement"` — that value is *only* honored by the separate trailing-phase branch, which does not exist when `refinement_epochs == 0`. |
| `"each"` | every warm-up phase, from phase 0 onward — jointly with whichever step-size row(s) that phase activates. |
| `"last_layer"` | only the final warm-up phase ($j = J-1$) onward — one phase earlier than `"refinement"`'s default (which waits for the separate trailing phase). |

If your goal is "Unfolded-$\alpha$+P should always end up with a genuinely
trained $P$", verify `refinement_epochs > 0` whenever `train_matrix_from`
is left at its `"refinement"` default — this is exactly the kind of
configuration mismatch this document is meant to help you catch.

### 3.3 The freeze mechanism, and why it needs three parts

Freezing a subset of one parameter tensor's rows under a *stateful*
optimizer (Adam/AdamW) is more subtle than zeroing a gradient. Two failure
modes were verified empirically against the live optimizer before this
design was adopted (documented in `docs/planning/03_studies/nb03_deep_unfolded_lqr/deep_unfolding_benchmark_blueprint_v2.md`
§7.1):

1. **Weight-decay leak.** With gradient masking alone plus `weight_decay > 0`, both Adam and AdamW moved a "frozen" parameter anyway (`2.0 → 1.9` in one step in the reproduced case) — decay is applied regardless of the gradient's value.
2. **Momentum carryover.** Even at `weight_decay = 0`, a row that received a real gradient in an earlier phase kept moving for every subsequent step under a *shared* optimizer, purely from residual Adam moment state ($m$, $v$) — masking the gradient to zero does not reset momentum already accumulated from that row's non-zero history.

The freeze contract closes both, via three independent mechanisms that must
all hold together:

**(a) A fresh, phase-owned optimizer.**
`LayerwiseGradientDescentStrategy.__init__` calls
`optimizer_spec.build(module.parameters())` itself, at construction — it
never accepts an already-built optimizer from a caller. `build_engine`
constructs one `LayerwiseGradientDescentStrategy` per compiled `PhaseSpec`,
so **every phase's Adam state starts at zero**; momentum from phase $j$
cannot leak into phase $j+1$ because no optimizer object survives the phase
boundary.

**(b) Post-backward gradient masking, for the step size.**
`LayerFreezeBuilder.build` compiles a `ParameterActivation` into a
`LayerFreeze` against the controller's *live* parameter tensors:

```python
grad_masks[name] = self._step_size_mask(name, parameter, activation)  # bool mask, True on active rows
trainable[name] = True  # step_size's raw tensor stays requires_grad=True; masking does the real work
```

and `LayerwiseGradientDescentStrategy.step` applies it between `.backward()`
and the optimizer step:

```python
loss.backward()
for name, parameter in self.module.named_parameters():
    mask = self.freeze.grad_masks.get(name)
    if mask is not None and parameter.grad is not None:
        parameter.grad.mul_(mask)          # frozen rows -> exact zero gradient, every step
self.optimizer.step()
```

Combined with (a), a masked row's Adam moments never receive a nonzero
gradient at any point in that phase, so its update is exactly zero, every
step.

**(c) `weight_decay` is structurally forbidden.**
Both `LayerwiseTrainingPlan.__post_init__` and
`LayerwiseGradientDescentStrategy.__init__` independently reject a
nonzero `optimizer.hyperparameters["weight_decay"]`:

```python
if self.optimizer.hyperparameters.get("weight_decay", 0.0):
    raise ValueError("... forbids weight_decay ...")
```

— defense in depth (the plan's own guard, plus the strategy's own guard for
any caller that constructs the strategy directly), because masking alone
cannot stop decay from pulling a frozen row toward zero (failure mode 1
above). If you need regularization on the *active* entries only, it must be
mask-aware, applied outside torch's built-in `weight_decay`.

**The learned matrix $P$ freezes differently — a whole-tensor toggle, not a
mask**, because it has no row structure to freeze partially:

```python
def _apply_activation(self) -> None:
    for name, parameter in self.module.named_parameters():
        if name in self.freeze.trainable:
            parameter.requires_grad_(self.freeze.trainable[name])
```

called at the start of every `step()` (idempotent). A
`requires_grad=False` parameter's `.grad` stays `None` forever, and every
torch optimizer skips a `None`-gradient parameter *entirely* — it therefore
cannot be touched by weight decay either, unlike the masked step-size rows
(verified empirically in the same §7.1 investigation this design is built
against).

### 3.4 Code path

```
UnfoldedModelConfig(training_mode="layerwise", schedule=<LayerwiseTrainingPlan>)
  -> family = "unfolded_warmstart"                          (nb03_unfolding._learned_unfolded_contender)
  -> WarmStartUnfoldedRecipe(schedule=..., kind=..., ...)    (applications/recipes/unfolded.py)
  -> WarmStartUnfoldedRecipe.build_engine (overrides TrainableRecipe entirely; never calls super())
       phase_specs = schedule.compile(num_iterations, has_matrix=...)
       freezer = LayerFreezeBuilder()
       strategies = [
           LayerwiseGradientDescentStrategy(
               rollout, module, schedule.optimizer,
               freezer.build(spec.activation, controller.config.parameters),
               loss_reduction=schedule.resolve_loss_reduction(),
               gradient_clip_norm=schedule.gradient_clip_norm,
           )
           for spec in phase_specs                            # <- one fresh optimizer PER phase
       ]
       phases = [TrainingPhase(s, epochs=spec.epochs, name=spec.name) for s, spec in zip(strategies, phase_specs)]
       Runner(phases=phases)
```

`WarmStartUnfoldedRecipe` subclasses `ModelRecipe` directly, **not**
`TrainableRecipe`: `TrainableRecipe` assumes one flat `plan: TrainingPlan`
field its (inherited, unmodified) `build_engine` reads directly, and this
family's specification is a whole *schedule*, not a single plan — a
different shape, not a narrower one — so both `build_engine` and
`build_synthesizer` are implemented directly on `WarmStartUnfoldedRecipe`
(via the dedicated `WarmStartSynthesizer`, the schedule-aware sibling of
`EngineTrainedSynthesizer`). `Runner` itself needed no changes: it already
accepts any `Sequence[TrainingPhase]`, so a multi-phase schedule is simply a
longer `phases` list.

`gradient_clip_norm`, when set, is applied only to the parameters that are
*currently* `requires_grad=True`, computed **after** masking:

```python
trainable_params = [p for p in self.module.parameters() if p.requires_grad]
if trainable_params:
    torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)
```

so a frozen row's zeroed gradient can never inflate the clipped norm and
artificially shrink the active rows' step.

### 3.5 What to check to confirm this is happening

- `Runner` receives `num_layers` (+ 1 if `refinement_epochs > 0`) separate `TrainingPhase` objects for an `"unfolded_warmstart"` family run, each named `"warmup_layer_<j>"` or `"refinement"` — check the phase-transition log lines the `Runner` emits at INFO.
- Each phase's `LayerwiseGradientDescentStrategy.optimizer` is a **distinct object** (`id(strategies[j].optimizer) != id(strategies[j+1].optimizer)` for every $j$) — this is what makes claim (a) of §3.3 true; if a caller ever passed one shared optimizer across phases, momentum carryover would silently reappear.
- During warm-up phase $j$ (with `activation="single"`), `step_size.get_numpy()`'s row $j$ should change epoch over epoch while every row $\neq j$ stays *exactly* constant — not merely "slowly changing", but bit-for-bit frozen (masking gives an exact zero update, not an approximately-small one).
- With `activation="cumulative"`, rows $0, \dots, j$ should all continue changing during phase $j$, not just row $j$.
- `riccati_matrix.get_numpy()` should stay at its initial value ($P = I$) through every warm-up phase under `train_matrix_from="refinement"`, and only start changing once the `"refinement"` phase begins — if it changes earlier, `train_matrix_from` is not doing what you expect (see the §3.2 trap).
- The persisted provenance (`WarmStartSynthesizer.synthesize`'s `provenance["training"]`) is `recipe.schedule.get_signature()` — a `LayerwiseTrainingPlan` signature (with `warmup_epochs_per_layer`, `refinement_epochs`, `train_matrix_from`, `activation` all present), not a flat `TrainingPlan` one.
- Constructing a `LayerwiseTrainingPlan` or `LayerwiseGradientDescentStrategy` with `hyperparameters={"weight_decay": <nonzero>}` must raise `ValueError` immediately — if it doesn't, the freeze-contract guard has regressed.

---

## 4. Side-by-side comparison

| | `"end_to_end"` | `"layerwise"` |
|---|---|---|
| Family / recipe | `"unfolded"` / `UnfoldedRecipe` | `"unfolded_warmstart"` / `WarmStartUnfoldedRecipe` |
| Spec class | `TrainingPlan` | `LayerwiseTrainingPlan` |
| Strategy class | `GradientDescentStrategy` | `LayerwiseGradientDescentStrategy` |
| `Runner.phases` | exactly 1 | $J$ (+ 1 if `refinement_epochs > 0`) |
| Optimizers constructed | 1, over every learnable row from epoch 0 | 1 **per phase**, fresh, never shared |
| Row freezing | none — everything trains jointly throughout | gradient-masked per `ParameterActivation`, until a row's phase (or the refinement phase) arrives |
| $P$ (if learned) trains... | from epoch 0, jointly with every $\alpha^{(j)}$ | per `train_matrix_from` — possibly never, if misconfigured (§3.2) |
| `weight_decay` | permitted | **forbidden** (`ValueError` at construction) |
| Total epochs | `plan.epochs` | $J \times$ `warmup_epochs_per_layer` + `refinement_epochs` |
| Provenance signature | `TrainingPlan.get_signature()` | `LayerwiseTrainingPlan.get_signature()` (distinct cache identity — switching modes is never silently served from the other mode's cached run) |

Both regimes are available to **either** learned `UnfoldedKind`
(`LEARNED_STEP_SIZE` or `LEARNED_STEP_SIZE_AND_MATRIX`) — the layer-wise
machinery is kind-agnostic (`LayerwiseTrainingPlan.compile`'s `has_matrix`
flag simply skips matrix-activation bookkeeping for a model with no
learnable matrix). In the current NB03 study
(`src/mbl/applications/studies/nb03_unfolding.py`), Unfolded-$\alpha$ defaults to
`"end_to_end"` and Unfolded-$\alpha$+P defaults to `"layerwise"`
(`_default_unfolded_alpha` / `_default_unfolded_alpha_p`), but this is a
free per-model choice (`UnfoldedModelConfig.training_mode`), not a fixed
property of either learned kind — you may freely run either kind under
either mode to isolate whether an observed effect comes from *what* is
learned or from *how* it is scheduled.

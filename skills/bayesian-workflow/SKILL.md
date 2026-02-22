---
name: bayesian-workflow
description: >
  Bayesian modeling workflow and iterative model-building strategy. Use when planning a
  modeling approach, deciding how to build up from simple to complex models, choosing between
  model specifications, or discussing Bayesian workflow principles. Triggers on: Bayesian
  workflow, iterative modeling, model building strategy, model expansion, prior predictive
  simulation, fake data simulation, simulation-based calibration, model criticism, combining
  information from multiple sources, or when a user asks "how should I model this?"
---

# Bayesian Workflow

Principled, iterative model building based on Gelman, Vehtari & McElreath (2025) and Gelman et al. (2020). For PyMC API details, see the `pymc-modeling` skill. This skill governs *strategy*.

Real-world Bayesian modeling is not "specify a model, fit it, report results." It is an iterative process of building up models, checking them against data and domain knowledge, and expanding only when simpler models demonstrably fail.

## Required Workflow Steps

For EVERY modeling task, execute these steps in order. Do not skip steps.

### Step 1: Start with the Simplest Plausible Model

Write the simplest model that could address the question. Fit it. Understand it.

- For grouped data: start with complete pooling (single intercept), then no pooling, then partial pooling
- For regression: start with one or two key predictors, not all of them
- For time series: start with a simple trend or random walk before adding structure

**Why**: Complex models are understood in relation to simpler ones. Each expansion reveals what the added complexity buys. Computational problems are easier to diagnose in simple models.

Save `results.nc` immediately after your first successful sampling run. You can overwrite it later.

### Step 2: Prior Predictive Check

BEFORE calling `pm.sample()` on your first model, run a prior predictive check:

```python
with model:
    prior_pred = pm.sample_prior_predictive(draws=500)

# Check implied range on the OBSERVABLE scale
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior predictive range: [{prior_y.min():.1f}, {prior_y.max():.1f}]")
print(f"Prior predictive mean: {prior_y.mean():.1f}, std: {prior_y.std():.1f}")
```

Ask: does the generative model produce datasets that look like plausible datasets from this domain? Look for absurd implications (negative counts, probabilities outside [0,1], impossibly large effects). If the prior predictive range is unreasonable, adjust priors and re-check before fitting.

Prior predictive checks are especially important for complex models where interactions between priors on multiple parameters create unexpected behavior on the outcome scale.

### Step 3: Fit and Diagnose

After fitting, check convergence before interpreting results:

```python
# Check divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

# Summary with convergence diagnostics
print(az.summary(idata, var_names=[...]))
```

**Pass criteria** — all must hold before proceeding:
- Zero divergences (or < 0.1% and randomly scattered)
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 and ESS_tail > 400

If diagnostics fail, the problem is usually the model, not the sampler (the "folk theorem"). Reparameterize (non-centered for weak data, centered for strong data) or simplify before reaching for sampler tuning knobs.

### Step 4: Posterior Predictive Check

Run a posterior predictive check after EVERY model you fit:

```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Compare posterior predictions to observed data
az.plot_ppc(idata, kind="cumulative")
```

The goal is not to "accept" or "reject" the model. The goal is to find *specific, interpretable ways* the model fails, which tell you what to fix next. Look for:
- Systematic deviations in the tails or center
- Patterns the model should capture but doesn't (clustering, heteroscedasticity, nonlinearity)
- Focus on the quantities that matter for the question at hand

### Step 5: Expand and Compare

Add ONE piece of complexity at a time. Fit the expanded model, repeat Steps 3-4, then compare:

```python
# Compute log-likelihood if using nutpie
pm.compute_log_likelihood(idata_simple, model=model_simple)
pm.compute_log_likelihood(idata_complex, model=model_complex)

comparison = az.compare({
    "simple": idata_simple,
    "complex": idata_complex,
})
print(comparison[["rank", "elpd_loo", "d_loo", "dse", "weight"]])
```

**Decision rule**: If `d_loo < 2 * dse`, models are effectively equivalent — prefer the simpler one.

Fit the expanded model even when you believe the simpler one is sufficient. The comparison itself is informative:
- If the expansion doesn't improve fit, you've demonstrated the simpler model is adequate — report this
- If it does improve fit, you've identified important structure
- The *difference* between models tells you something about the data-generating process

### Step 6: Report the Full Workflow

Report the sequence of models, not just the final one. The modeling journey IS the analysis.

Structure your report as:
1. What simple model you started with and what it revealed
2. What prior predictive checks showed about your assumptions
3. What posterior predictive checks revealed about model fit at each stage
4. What expansions you tried, which helped, and which didn't
5. How model comparison (LOO) informed your choices
6. Final parameter estimates with uncertainty and interpretation
7. Sensitivity: how conclusions change under reasonable alternative specifications

## When to Simulate Fake Data First

For novel or complex model structures, simulate from your model with known parameter values before touching real data. Then fit to the simulated data and check parameter recovery:

1. Choose plausible parameter values (informed by domain knowledge)
2. Simulate a dataset from the generative model
3. Fit the model to simulated data
4. Check: are the true parameters recovered within posterior intervals?

This catches specification bugs, non-identifiability, and prior-likelihood conflicts before real-data messiness obscures the picture. It is most valuable for custom likelihoods, latent variable models, and models with complex indexing.

## Hierarchical Models and Partial Pooling

You rarely want complete pooling (ignoring group differences) or no pooling (ignoring similarities). Hierarchical structure lets the data determine how much to share across groups.

When building hierarchical models iteratively:
1. Start with complete pooling (single intercept)
2. Move to no pooling (group-specific intercepts) — compare to pooled
3. Add partial pooling (hierarchical intercepts) — compare to both
4. Allow slopes to vary by group if the question warrants it
5. Add group-level predictors to explain between-group variance

At each step, compare via LOO and check whether the added structure reduces between-group variance.

## References

- Gelman, Vehtari, and McElreath (2025). "Statistical Workflow." *Philosophical Transactions of the Royal Society A.*
- Gelman, Vehtari, Simpson, et al. (2020). "Bayesian Workflow." arXiv:2011.01808.
- Gabry, Simpson, Vehtari, Betancourt, and Gelman (2019). "Visualization in Bayesian workflow." *JRSS A.*

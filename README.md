[![Pipeline Status](https://github.com/sandialabs/bayesppr_matlab/actions/workflows/matlab.yml/badge.svg)](https://github.com/sandialabs/bayesppr_matlab/actions/workflows/matlab.yml)

![](logo.png)

# BayesPPR_matlab

A MATLAB implementation of Bayesian Projection Pursuit Regression (BayesPPR).

# References

Gavin Collins, Devin Francom, and Kellin Rumsey, “Bayesian Projection Pursuit Regression,” Submitted to Bayesian Analysis. 
Also on [arxiv](https://arxiv.org/abs/2210.09181) (arXiv:2210.09181 [stat.ME])

# Using BayesPPR inside a larger Gibbs sampler

`bppr` fits a fixed response from start to finish. When the response is itself a latent variable updated by another part of a sampler (data augmentation), use `bpprGibbs`. It keeps the chain's state between visits and advances it a few steps at a time against the current response:

```matlab
g = bpprGibbs(X, y0, n_keep, 'n_ridge_mean', 2);
for it = 1:n_iter
    % ... update the latent response y ...
    g.setResponse(y);   % keep the ridge functions, swap the response
    g.step(3);          % 3 RJMCMC updates, each followed by draws of s2 and coefs
    mu = g.fitted();    % regression function at the training inputs
    s2 = g.s2();
    if keep, g.save(); end
end
mod = g.model();        % bpprModel with the saved draws: predict(), plot()
```

Each step is the usual BayesPPR update: an RJ move on the ridge functions with the coefficients and s2 integrated out, followed by exact draws of s2 and the coefficients. That kernel leaves p(ridge functions, s2, coefs | y) invariant, so interleaving it with updates of y is a valid Gibbs sampler.

Options added for this use, available in both `bppr` and `bpprGibbs` (the defaults reproduce the original model exactly):

| option | default | meaning |
|---|---|---|
| `intercept` | `true` | include a constant basis function |
| `center_basis` | `false` | center each ridge basis column over the training inputs. With `intercept = false`, every posterior draw of the regression function then averages to exactly zero over the training inputs. |
| `shape_s2`, `scale_s2` | `0`, `0` | inverse-gamma prior on the error variance; `(0, 0)` is the original improper p(s2) ∝ 1/s2 |
| `s2_fixed` | `NaN` | hold the error variance at this value instead of sampling it |

# Scalar-Mixing Probes and Layer-Localization Metrics

This section describes the probing methodology we use to test whether a language
model computes linguistic structure in a shallow-to-deep progression across its
layers (the "classical NLP pipeline" of Tenney et al., 2019). We (i) train
*scalar-mixing* probes that learn where in the network a given task's information
lives, and (ii) summarize that localization with two complementary statistics —
the **center of gravity** of the learned mixing weights and the **expected layer**
of the cumulative differential scores.

## 1. Setup and notation

Let the encoder have $L$ transformer layers. For an input sentence it produces
$L+1$ hidden states per token, $\mathbf{h}^{(0)},\dots,\mathbf{h}^{(L)}\in\mathbb{R}^{d}$,
where $\mathbf{h}^{(0)}$ is the (contextless) embedding layer and $\mathbf{h}^{(L)}$
the final layer. Each probing example targets a token or one or two spans:

- **Single-token tasks** (POS, entities, constituents): the target token's
  representation, taken at its last subword.
- **Span / span-pair tasks** (dependencies, coreference, SRL, relations): each
  span is mean-pooled over its subword tokens; for two-span tasks the two pooled
  vectors are concatenated. We write $\mathbf{h}^{(l)}_i\in\mathbb{R}^{d}$ for the
  layer-$l$ representation of example $i$ (with $d$ doubled for span pairs).

A task $\tau$ has a label set $\mathcal{Y}_\tau$; the probe maps the pooled
representation to $\mathcal{Y}_\tau$.

## 2. Scalar-mixing probe

Following the ELMo scalar mix (Peters et al., 2018) as used in edge probing
(Tenney et al., 2019), a single probe is given access to *all* layers through a
learned convex combination. For example $i$,

$$
\mathbf{r}_i \;=\; \gamma \sum_{l=0}^{L} \alpha_l \,\mathrm{LN}\!\big(\mathbf{h}^{(l)}_i\big),
\qquad
\boldsymbol{\alpha} \;=\; \mathrm{softmax}(\mathbf{s}),
\tag{1}
$$

with learnable mixing logits $\mathbf{s}\in\mathbb{R}^{L+1}$ and a learnable scalar
$\gamma\in\mathbb{R}$. $\mathrm{LN}(\cdot)$ is a per-layer LayerNorm without affine
parameters, applied independently to each layer before mixing. This normalization
is essential: without it, layers with systematically larger activation norm
(typically the deeper layers in decoder-style models) dominate the sum regardless
of their task relevance, and $\boldsymbol{\alpha}$ fails to reflect *where the
information is*.

The mixed representation is passed to a probe head $f_\theta$, either **linear**,
$f_\theta(\mathbf{r})=\mathbf{W}\mathbf{r}+\mathbf{b}$, or a one-hidden-layer
**MLP** with ReLU and dropout. We report the linear head unless noted; it is the
more conservative choice and yields the cleaner localization signal.

**Optimization.** Probes are trained with cross-entropy and AdamW. The head
parameters use the base learning rate $\eta$ and weight decay; the mixing
parameters $(\mathbf{s},\gamma)$ use a larger learning rate $25\eta$ and **no
weight decay**. Because the head can already fit the task from a near-uniform
average of the (LayerNorm'd) layers, the mixing logits receive little gradient
pressure at the base rate and remain diffuse; the higher rate lets
$\boldsymbol{\alpha}$ concentrate on the layers that actually help. We use early
stopping on a validation split.

**Selectivity.** To distinguish task structure from lexical memorization we pair
every probe with a control task (Hewitt and Liang, 2019): each target-token type
is assigned a fixed random label, and a probe of identical architecture is trained
on those labels. We report **selectivity**, the task accuracy minus the control
accuracy.

## 3. Localization metric I: mixing-weight center of gravity

After training the full-model probe of Eq. (1), the softmax weights
$\boldsymbol{\alpha}$ form a distribution over layers. Its **center of gravity**,

$$
\mathrm{COG}(\tau) \;=\; \sum_{l=0}^{L} l\,\alpha_l ,
\tag{2}
$$

summarizes which layers the probe draws on. COG reflects both where task
information is present and where it is most linearly accessible; it tends to sit
deep, because later layers accumulate information in the residual stream and are
weighted even when they introduce nothing new. It is therefore a coarse localizer,
which we complement with the differential metric below.

## 4. Localization metric II: expected layer from cumulative scoring

To measure where information is *first introduced*, we train a sequence of
scalar-mixing probes $P^{(\ell)}$, $\ell=0,\dots,L$, each restricted to the layer
prefix $\{0,\dots,\ell\}$ (i.e. the mix of Eq. (1) over the first $\ell{+}1$
layers). Let $s(\ell)$ be the score of $P^{(\ell)}$. The per-layer **differential
score** is

$$
\Delta(\ell) \;=\; s(\ell) - s(\ell-1),\qquad \ell = 1,\dots,L,
\tag{3}
$$

and the **expected layer** is the differential-weighted mean layer index,

$$
\bar{E}(\tau)
\;=\;
\frac{\sum_{\ell=1}^{L} \ell\,\Delta(\ell)}{\sum_{\ell=1}^{L}\Delta(\ell)}
\;=\;
\frac{\sum_{\ell=1}^{L} \ell\,\big(s(\ell)-s(\ell-1)\big)}{\,s(L)-s(0)\,},
\tag{4}
$$

where the denominator telescopes to the total gain from the embedding probe to the
full probe. Intuitively, $\bar{E}$ places mass on the layer at which adding a layer
most improves the task, so simpler features that saturate early yield a shallow
$\bar{E}$ and features that require deeper composition yield a deep one. The
endpoints $s(0)$ and $s(L)$ are reported as the baseline ($\ell{=}0$) and
full-model ($\ell{=}L$) scores.

We use three estimator choices that matter in practice.

**(a) Score = accuracy (micro-$F_1$).** We take $s(\ell)$ to be accuracy, i.e.
micro-averaged $F_1$ for our single-label tasks. Macro-$F_1$ is far higher variance
on the imbalanced, many-class tasks (entities, SRL, coreference) and destabilizes
the differential; accuracy is smooth and is the comparable quantity to the
$F_1$ scores reported by Tenney et al.

**(b) Unclamped differential.** We compute Eq. (4) on the *raw* differential
without clamping $\Delta(\ell)$ to be non-negative. On a curve that saturates and
then fluctuates, the plateau's positive and negative increments cancel, so noise
contributes $\approx 0$ to both the numerator and the (fixed) denominator and the
expectation stays on the layers of genuine gain. Clamping negatives to zero keeps
only the upward jitter; because each increment is weighted by its layer index
$\ell$, that residual noise is pulled toward large $\ell$ and inflates $\bar{E}$,
which can even exceed $\mathrm{COG}$. We guard against a degenerate denominator by
leaving $\bar{E}$ undefined when the total gain $s(L)-s(0)$ is negligible.

**(c) Multi-seed averaging (Monte-Carlo cross-validation).** Each $P^{(\ell)}$ is
trained independently, so $s(\ell)$ carries optimization and sampling noise. We
average the score curve over $K$ runs that vary **both** the train/validation/test
split and the initialization,

$$
\bar{s}(\ell) \;=\; \frac{1}{K}\sum_{k=1}^{K} s_k(\ell),
$$

and evaluate Eqs. (3)–(4) on $\bar{s}$. Varying the split (not just the seed) is
necessary for small datasets, where the dominant noise source is *which examples
land in the held-out set* rather than training randomness.

**Reliability.** We flag tasks whose curve is noise-dominated with a
signal-to-noise ratio

$$
\mathrm{SNR}(\tau) \;=\; \frac{\bar{s}(L)-\bar{s}(0)}{\hat{\sigma}},
\qquad
\hat{\sigma} = \mathrm{std}\big(\{\bar{s}(\ell)-\bar{s}(\ell-1)\}_{\ell > L/2}\big),
$$

the total gain relative to the layer-to-layer standard deviation over the second
half of the network. An expected layer with $\mathrm{SNR}\lesssim 3$ is treated as
unreliable (the probe does not localize the task cleanly), rather than reported as
a point estimate.

## 5. Relation to Tenney et al. (2019)

Our two statistics are the mixing-weight center of gravity (their Eq. 2) and the
expected layer of the cumulative differential (their Eqs. 3–4). We deviate from
their setup in three ways, all of which we found necessary to obtain stable
estimates in our probing pipeline: (i) a simpler span representation
(mean-pooling and concatenation) and a linear/MLP scalar-mix head, rather than
learned self-attention span pooling with a two-layer MLP; (ii) accuracy /
micro-$F_1$ scoring of the cumulative curve; and (iii) multi-seed Monte-Carlo
averaging with an explicit reliability (SNR) criterion. We report $\bar{E}$ and
$\mathrm{COG}$ per task, ordered by $\bar{E}$, alongside the baseline and
full-model scores.

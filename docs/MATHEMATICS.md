# The Mathematics Implemented in NeuroSim Lab

Everything below is exactly what the backend computes (PyTorch + numpy), and
what the Math/Simulation/Gradients tabs visualise with real numbers.

## 1. The neuron

For inputs $x_1,\dots,x_n$, weights $w_1,\dots,w_n$ and bias $b$:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

**Vectorised** (per layer): $\mathbf{z} = W\mathbf{a}_{prev} + \mathbf{b}$,
$W \in \mathbb{R}^{m \times n}$ (one row per output neuron).

## 2. Activation

$a = f(z)$ — ReLU $\max(0,z)$, sigmoid $\sigma(z)=\frac{1}{1+e^{-z}}$, tanh,
GELU, softmax $\frac{e^{z_i}}{\sum_j e^{z_j}}$, or a custom safe formula.

## 3. Forward propagation

$$\mathbf{a}^{(0)} = \mathbf{x},\qquad \mathbf{a}^{(l)} = f^{(l)}\!\left(W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

The `/api/inspect/forward-trace` endpoint recomputes this layer by layer in
numpy from the learned parameters; the Math tab renders each weighted sum.

## 4. Loss functions

| Loss | Formula | Task |
|---|---|---|
| Cross-entropy | $\mathcal{L} = -\log p_{y}$ where $p = \mathrm{softmax}(\mathbf{z})$ (or pre-applied softmax probabilities clamped to $[10^{-8}, 1]$ — mathematically equivalent composition) | multi-class |
| Binary cross-entropy | $-\big[y\log \hat y + (1-y)\log(1-\hat y)\big]$ | binary (1 sigmoid output) |
| MSE | $\frac{1}{N}\sum (y - \hat y)^2$ | regression |
| MAE | $\frac{1}{N}\sum |y - \hat y|$ | regression |

Optional L2: $\mathcal{L} \mathrel{+}= \lambda \sum_\theta \theta^2$.

## 5. Gradients & backpropagation

Chain rule from the output backwards:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} = \Big(\big(W^{(l+1)}\big)^{\!\top} \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l+1)}}\Big) \odot f^{(l)\prime}(\mathbf{z}^{(l)}),\qquad \frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} \big(\mathbf{a}^{(l-1)}\big)^{\!\top}$$

Computed by PyTorch autograd (including through **custom activations**, which
are compiled into differentiable tensor ops — hence exact derivatives without
hand-written ones; the Activation Lab plots this derivative over $x \in [-5,5]$).

## 6. Gradient descent & weight update

$$w_{new} = w_{old} - \eta \, \frac{\partial \mathcal{L}}{\partial w}$$

Optimizers: **SGD** (plain), **Momentum** ($v \leftarrow \mu v - \eta g$, $w \leftarrow w + v$, $\mu=0.9$),
**Adam** (bias-corrected first/second moment estimates), **RMSprop**.
The `/api/inspect/backprop` endpoint performs one real forward+backward pass
and one SGD-style step so you can verify the formula digit by digit.

## 7. Training vocabulary

- **Batch** — a mini-subset (e.g. 16 samples) used for each update.
- **Epoch** — one full pass over the training set (≈ `n/batch_size` updates; the live counter shows both).
- **Train / validation / test splits** — stratified; scalers and imputers are fitted on train only; test is touched exactly once at the end.

## 8. Metrics

- Accuracy $= \frac{\text{correct}}{N}$; per-class precision/recall/F1 (macro-averaged);
  regression: MSE, MAE, $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$.

## 9. Failure modes the diagnoser looks for

- **Overfitting** — train metric ≫ validation metric. Fixes: smaller net, dropout, L2, more data.
- **Underfitting** — both metrics near the majority-class baseline / low $R^2$. Fixes: capacity, training time, features.
- **Vanishing gradients** — $\|\nabla_{W^{(1)}}\| \ll \|\nabla_{W^{(L)}}\|$; products of small derivatives decay exponentially with depth. Fixes: ReLU-family, He/Xavier init, shallower nets.
- **Exploding gradients** — huge $\|\nabla W\|$; unstable loss/overflow. Fixes: lower $\eta$, scaling, gentler activations.
- **Dead neurons** — ReLU units whose pre-activation is negative for the whole batch ⇒ gradient 0 forever (fraction measured from real batches).
- **Saturation** — sigmoid/tanh outputs pinned in the flat tails where $f'(z)\approx 0$.
- **Learning rate** — loss oscillation (too high) or ≈0 relative improvement per epoch (too low).

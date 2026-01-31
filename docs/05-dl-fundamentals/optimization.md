# Optimization (Neural Network Training Fundamentals)

Optimization is the part of training where you *systematically improve* a neural network’s parameters (weights and biases) so that its loss gets smaller on the training data. In this chapter, we’ll treat optimization as an iterative process first, then introduce the main tools (SGD, momentum, RMSProp, Adam, learning-rate decay, batch normalization) with both intuition and minimal Keras snippets.

![Optimization](https://i.imgur.com/M2Bdn9r.png)


## Optimization as a process

Think of training a Fashion-MNIST classifier as repeated, small edits to the model so it becomes less wrong over time.

**Running example (Fashion-MNIST):**

- Input: 28×28 grayscale clothing images.
- Output: one of 10 classes (T-shirt/top, trouser, …, ankle boot).
- Model: e.g., a small MLP that outputs class probabilities.

**The optimization loop (what happens repeatedly):**

1. **Forward pass:** compute predictions for a batch of images.
2. **Loss:** convert “how wrong” the predictions are into a single number.
3. **Backward pass:** compute gradients (how each parameter affects the loss).
4. **Update step:** change parameters using an optimizer.
5. **Repeat:** across many mini-batches (steps) and epochs (full passes through the dataset).

![Optimization Process](https://i.imgur.com/7w8gHCh.png)

**Parameters vs hyperparameters**

- **Parameters:** Learned values *inside* the model that get updated during training (via backpropagation + the optimizer). Examples: the weights and biases in Dense layers, and (in general) any trainable tensors that determine how inputs are transformed into outputs.

- **Hyperparameters:** Settings you choose *outside* the model’s learned weights that control the learning process or model capacity. Examples include the learning rate, batch size, optimizer choice (SGD/RMSProp/Adam), momentum/betas, learning-rate schedule, number of layers/units, regularization strength (L2/weight decay), dropout rate, and early-stopping patience.

A quick rule of thumb: if it’s learned automatically from data during training, it’s a **parameter**; if you set it before (or while) training to guide learning, it’s a **hyperparameter**.

> Practical habit: When training is bad, first ask “is the optimization process unstable or misconfigured?” before redesigning the model.

## Where basic gradient descent fails

“Basic gradient descent” usually means using a single global learning rate and applying gradient updates without any stabilization tricks. It can fail even when the model *could* learn, because the path to a good solution is difficult.

![Loss landscape with valleys, plateaus, saddle points](https://i.imgur.com/6iLBRL4.png)

**Common failure modes (and what you observe):**

- **Learning rate too high:** loss may explode, bounce wildly, or never settle.
- **Learning rate too low:** loss decreases painfully slowly; training feels “stuck.”
- **Ravines / ill-conditioning:** the loss changes steeply in one direction and slowly in another, causing zig-zagging rather than smooth progress.
- **Saddle points:** places where the gradient can be near zero but the point is *not* a good minimum (flat in some directions, curved in others).
- **Noisy gradients (real data):** updates based on small batches are “directionally correct on average” but noisy step-to-step.

**Why this happens (intuition)**

- Gradients give a local direction, not a global plan.
- A single learning rate is a blunt tool: some parameters need bigger steps, others need smaller steps.
- Noise can be helpful (it can shake you out of flat regions), but it can also make training unstable without smoothing.

## The optimization toolkit (intuition-first)

This section introduces the most common fixes as *ideas* (what problem they solve), independent of any specific framework.

### Normalize inputs (make gradients behave)

If inputs have wildly different scales, the loss surface becomes harder to navigate and updates can become inefficient or unstable. Normalizing inputs makes training smoother because parameter updates don’t have to fight inconsistent feature scales.

Two common choices:

- **Rescale pixels:** map [0, 255] → [0, 1].
- **Standardize features:** subtract mean and divide by standard deviation (common for tabular features and some intermediate representations).

### Batch, stochastic, and mini-batch gradient descent

- **Batch gradient descent:** uses the full dataset to compute one update (stable but expensive).
- **Stochastic gradient descent (SGD):** uses one example per update (fast but very noisy).
- **Mini-batch gradient descent:** uses a small batch (the standard default in deep learning because it’s efficient and reasonably stable).

![Gradient Descent Variations](https://i.imgur.com/NfeFkBK.png)

**All three methods are the same idea**: update model weights by following the loss gradient, repeatedly, until the model improves. The key difference is how many training examples you use to *estimate* the gradient for each update step—more examples gives a cleaner (less noisy) direction but costs more compute per step.

In practice, mini-batches are the default because they balance speed (vectorized computation) and stability (less noise than single-example updates).

| Method | Data per update | Updates per epoch (approx.) | Gradient noise | Speed per update | Typical use |
|---|---:|---:|---|---|---|
| Batch GD | All \(N\) examples | 1 | Low (smooth) | Slowest | Small datasets, debugging, very stable curves |
| SGD | 1 example | \(N\) | High (jittery) | Fastest | Rare as-is; conceptually important |
| Mini-batch GD | \(B\) examples (e.g., 32–256) | \(N/B\) | Medium | Fast (GPU-friendly) | Standard default for deep learning |

### Moving averages (a core building block)

A moving average is a simple way to “remember the recent past” of a quantity that jumps around from step to step (like gradients or losses). Instead of reacting fully to the newest value—which might be noisy—you blend it with the previous average, producing a smoother signal that leads to more stable updates.

**Exponential moving average (EMA)** uses a decay factor \(\beta\) (often 0.9, 0.99, etc.): higher \(\beta\) means longer memory and stronger smoothing, while lower \(\beta\) means the average adapts faster but is noisier. In optimization, EMAs show up directly in momentum (EMA of gradients), RMSProp (EMA of squared gradients), and Adam (both).

![Moving Averages](https://i.imgur.com/e7745TV.png)

Minimal implementation (conceptual):
```python
def ema_update(moving_avg, value, beta=0.9):
    # beta close to 1.0 = smoother but slower to react
    return beta * moving_avg + (1 - beta) * value
```

### Momentum (smooth direction)

Momentum adds “inertia” to gradient descent by maintaining a *velocity*—a running direction that accumulates past gradients—so each update is influenced by recent history, not just the current (noisy) mini-batch. This tends to reduce zig-zagging in narrow valleys (where gradients flip direction side-to-side) and helps the optimizer make faster progress along directions that stay consistently downhill.

Intuitively: plain gradient descent is like taking a new step based only on what you see right now; momentum is like rolling a ball that keeps moving in the same direction unless there’s strong evidence to turn. The key hyperparameter is `momentum` (often ~0.9): higher values smooth more (but can overshoot), lower values react faster (but keep more jitter).

![Momentum](https://i.imgur.com/1oQNhFQ.png)

**Minimal Keras snippet (SGD + momentum):** Keras exposes momentum directly on the SGD optimizer via the `momentum` argument. [keras](https://keras.io/api/optimizers/sgd/)

```python
from tensorflow import keras

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### RMSProp (adapt step sizes per parameter)

RMSProp is an *adaptive* optimizer: instead of using one global step size for every weight, it adjusts the effective step size **per parameter** based on recent gradient behavior. The core trick is to keep an exponential moving average of the **squared gradients**; weights that repeatedly see large gradients get their updates dampened, while weights with small gradients get relatively larger steps.  

Intuitively, this helps when the loss surface has uneven curvature (some directions are steep, others are flat): RMSProp automatically “steps carefully” in steep directions and “steps more boldly” in flat ones, which often makes training more stable than plain SGD with a single learning rate.

![RMSProp](https://i.imgur.com/uUCuoD7.png)

**Minimal Keras snippet:** RMSProp is available as `keras.optimizers.RMSprop(...)` with common knobs like `learning_rate`, `rho` (decay for the moving average), `momentum`, and `epsilon`. [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)

```python
from tensorflow import keras

model.compile(
    optimizer=keras.optimizers.RMSprop(
        learning_rate=1e-3,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7
    ),  # RMSprop parameters 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```
### Adam (momentum + adaptive scaling)

Adam is a practical “best of both worlds” optimizer: it keeps a momentum-like running direction (so updates don’t swing wildly from mini-batch noise) *and* it adapts step sizes per parameter (so weights that see consistently large gradients don’t take overly aggressive steps).  This combination makes training feel more “self-tuning” than plain SGD because the optimizer both smooths the direction and rescales the magnitude of updates automatically. [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

In everyday terms: momentum helps you move steadily in the right direction, while adaptive scaling helps you avoid taking the same-size step on parameters that live in very different gradient regimes.  Adam still has a learning rate, and it still matters—but Adam often works well with reasonable defaults, which is why it’s commonly used as a first optimizer choice. [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)


![Adam](https://i.imgur.com/vu4Xkv2.png)

**Minimal Keras snippet:** `keras.optimizers.Adam(...)` exposes `learning_rate`, `beta_1`, and `beta_2` (the EMA decay factors that control the “memory” of the optimizer). [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

```python
from tensorflow import keras

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999
    ),  # Adam parameters 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Learning rate decay (schedule smaller steps later)

Learning rate decay means you **intentionally change** the learning rate during training, usually starting larger to make rapid progress and then reducing it so training can “settle” into a good solution instead of bouncing around near the end. A simple way to think about it: early training is about finding the right region of the loss landscape, and late training is about careful fine-tuning.

In Keras, a common approach is to use a scheduler callback that updates the learning rate each epoch via a function `schedule(epoch, lr)` and applies the returned value. [keras](https://keras.io/api/callbacks/learning_rate_scheduler/)

```python
from tensorflow import keras

# Example: inverse time decay (simple, predictable)
def lr_schedule(epoch, lr):
    initial_lr = 1e-3
    decay_rate = 1.0
    return initial_lr / (1.0 + decay_rate * epoch)

lr_cb = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)  # schedule(epoch, lr) 

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train_oh,
    validation_data=(x_val, y_val_oh),
    epochs=20,
    batch_size=128,
    callbacks=[lr_cb]
)
```

### Batch normalization (stabilize activations)

Batch normalization (BN) normalizes a layer’s activations during training using statistics computed from the current mini-batch, which often makes optimization easier by keeping activation scales more stable as training progresses. [keras](https://keras.io/2/api/layers/normalization_layers/batch_normalization/)
It also keeps **moving averages** of the batch mean and variance (stored as `moving_mean` and `moving_var`) so that at inference time it can use these learned running estimates instead of batch statistics. [keras](https://keras.io/2/api/layers/normalization_layers/batch_normalization/)

For this stage of the handbook (before CNNs), you’ll mostly use BN in fully connected networks: a common pattern is `Dense → BatchNormalization → ReLU`. The Keras `Dense` docs also note that if a `Dense` layer is followed by `BatchNormalization`, it’s recommended to set `use_bias=False` since BN has its own offset term. [keras](https://keras.io/api/layers/core_layers/dense/)

```python
from tensorflow import keras

inputs = keras.Input(shape=(784,))  # e.g., flattened Fashion-MNIST

x = keras.layers.Dense(256, use_bias=False)(inputs)  # recommended when followed by BN 
x = keras.layers.BatchNormalization()(x)             # BN layer (tracks moving_mean/moving_var) 
x = keras.layers.ReLU()(x)

x = keras.layers.Dense(128, use_bias=False)(x)       # recommended when followed by BN 
x = keras.layers.BatchNormalization()(x)             # BN layer 
x = keras.layers.ReLU()(x)

outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

## Summary

| Technique | Core idea (what changes) | What it helps with | Common tradeoffs / gotchas | When to use | Keras “hook” (minimal) |
|---|---|---|---|---|---|
| Input normalization (rescale / standardize) | Make inputs have a consistent scale (e.g., pixels \(\to\) \([0,1]\), or zero-mean/unit-variance).  [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling) | Smoother optimization, less sensitivity to learning rate, faster convergence. | Must fit stats on train only when standardizing (`adapt`).  [keras](https://keras.io/api/layers/preprocessing_layers/numerical/normalization/) | Almost always (especially first model). | `keras.layers.Rescaling(1./255)`  [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling) or `keras.layers.Normalization(...); norm.adapt(x_train)`  [keras](https://keras.io/api/layers/preprocessing_layers/numerical/normalization/) |
| Batch Gradient Descent | Use **all** training data to compute each gradient update. | Very stable direction per update. | Slow/expensive per step; only ~1 update per epoch. | Small datasets, analysis/debugging. | Mostly controlled by how you feed data (full dataset per step), not a special optimizer. |
| Stochastic Gradient Descent (SGD) | Use **one** example per update. | Fast updates; noise can help exploration. | Very noisy; can be unstable without tuning/momentum. | Rare “as-is”; mainly conceptual baseline. | `keras.optimizers.SGD(...)`  [keras](https://keras.io/api/optimizers/sgd/) |
| Mini-batch Gradient Descent | Use a **batch** of size \(B\) per update. | Efficient + reasonably stable; standard default. | Batch size affects noise, speed, memory. | Default choice for deep learning training loops. | `model.fit(..., batch_size=B)` (optimizer can be SGD/Adam/etc.). |
| Moving average (EMA) | Smooth a noisy signal by mixing new value with past average. | Stabilizes direction/magnitude estimates; foundation for momentum/RMSProp/Adam. | Higher decay = smoother but slower to react. | Anytime you want stability (esp. noisy gradients). | Implemented inside optimizers (momentum/RMSProp/Adam). |
| Momentum (SGD + velocity) | Keep a “velocity” (EMA of gradients) so updates follow a consistent direction. | Reduces zig-zag in narrow valleys; speeds progress along consistent downhill directions. | Can overshoot if learning rate is high or momentum too large. | When SGD is jittery/slow; common with vision models. | `keras.optimizers.SGD(learning_rate=..., momentum=...)`  [keras](https://keras.io/api/optimizers/sgd/) |
| RMSProp (adaptive scaling) | Maintain EMA of squared gradients; scale updates per-parameter. | Handles uneven curvature; stabilizes training with noisy gradients. | Still needs LR tuning; `rho/epsilon` matter sometimes. | Good default when plain SGD struggles; also common for RNN-style training historically. | `keras.optimizers.RMSprop(learning_rate=..., rho=..., momentum=..., epsilon=...)`  [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop) |
| Adam (momentum + adaptive scaling) | Combine momentum-like direction (1st moment) + RMSProp-like scaling (2nd moment).  [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) | Strong default; works well across many problems with minimal tuning. | Learning rate still matters; can generalize worse than SGD in some setups (problem-dependent). | First-choice optimizer for many projects and quick baselines. | `keras.optimizers.Adam(learning_rate=..., beta_1=..., beta_2=...)`  [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) |
| Learning rate decay (scheduling) | Start with larger LR, then reduce over time (epoch-based schedule). | Faster early learning + more stable fine-tuning later. | Bad schedules can slow learning or freeze progress too early. | Once training works, use to improve final metrics/stability. | `keras.callbacks.LearningRateScheduler(schedule)` with `schedule(epoch, lr)`  [keras](https://keras.io/api/callbacks/learning_rate_scheduler/) |
| Batch normalization (BN) | Normalize activations during training using batch stats; maintain moving averages for inference.  [keras](https://keras.io/2/api/layers/normalization_layers/batch_normalization/) | More stable optimization; often tolerates higher learning rates; can speed training. | Different behavior train vs inference; placement matters. BN tracks `moving_mean`/`moving_var`.  [keras](https://keras.io/2/api/layers/normalization_layers/batch_normalization/) | Deeper MLPs; when training is sensitive/unstable; common in many architectures. | `keras.layers.BatchNormalization()`  [keras](https://keras.io/2/api/layers/normalization_layers/batch_normalization/) (often used as `Dense(use_bias=False) -> BN -> activation`).  [keras](https://keras.io/api/layers/core_layers/dense/) |

## Practical workflow

Use this as a repeatable checklist when optimization is the bottleneck.

1. **Normalize inputs first** (rescale images or use a normalization layer).
2. **Start with a reliable optimizer** (Adam is a common default), and keep other knobs fixed.
3. **Pick a reasonable batch size** (e.g., 64–256) and train for a small number of epochs to validate the setup.
4. **Watch training + validation curves**, not just final accuracy.
5. If training is unstable:
   - Lower learning rate.
   - Add batch normalization (especially in deeper networks).
   - Try momentum (SGD+momentum) or RMSProp if needed.
   - Add learning rate decay once the model is learning but improvements slow down.

### Common mistakes (and fixes)
- **Mistake: No input normalization.** Fix: rescale/standardize inputs early in the pipeline.
- **Mistake: Learning rate too high.** Fix: reduce LR by 10× and retry before changing architecture.
- **Mistake: Confusing optimizer problems with model capacity.** Fix: stabilize optimization first, then adjust model size.
- **Mistake: Changing many knobs at once.** Fix: one change per run; keep short notes.

### Quick glossary
- **Hyperparameter:** a setting you choose (learning rate, batch size, optimizer, momentum).
- **Saddle point:** a flat-ish point that is not a true minimum; gradients can be small but you’re not “done.”
- **SGD:** stochastic gradient descent (often implemented as mini-batch updates in practice).
- **Mini-batch gradient descent:** update using a batch (typical default).
- **Moving average:** a smoothed estimate of a noisy signal across steps.
- **Momentum:** uses a moving average of gradients to smooth direction.
- **RMSProp:** uses a moving average of squared gradients to adapt step sizes.
- **Adam:** combines momentum-like direction + RMSProp-like scaling.
- **Learning rate decay:** decreases learning rate over epochs/steps to fine-tune.
- **Batch normalization:** normalizes activations during training and uses moving averages for inference. 

> Optimize like a scientist: keep the data clean, change one knob at a time, and trust the process more than your guesses.
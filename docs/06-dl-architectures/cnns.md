# Convolutional Neural Networks 

Convolutional Neural Networks (CNNs) are neural networks built for images. They use convolution layers to spot small patterns (like edges or corners) and often use pooling layers to shrink the image representation so the model is faster and needs fewer weights.

The big idea: find simple patterns first, then combine them into more meaningful features as you go deeper in the network.

![CNN](https://i.imgur.com/uRMwLxE.png)


## What they solve (why CNNs work)

In images, nearby pixels usually belong together (a line, a corner, a texture), and the same kind of pattern can show up anywhere in the picture. 

A convolution layer takes one small “pattern detector” (a filter) and slides it across the whole image, so it can spot that pattern no matter where it appears, without needing separate weights for every location.

> Rule of thumb: Convolution finds useful patterns; downsampling (pooling or using a bigger stride) makes feature maps smaller, which speeds things up and makes the model less sensitive to small shifts.


## Layers: convolution and pooling

### Intuition first (what these layers *do*)

A CNN is usually built from repeating “blocks”: **convolution → (optional) pooling/downsampling**, and then a small classifier at the end (often Dense layers).  

Convolution layers learn to detect small patterns (like edges) and turn them into feature maps, while pooling (or strided convolution) makes those maps smaller so the next layers are faster and focus on the most important signals.

Typical flow:
`Conv2D (+ activation)` → `Conv2D` → `MaxPooling2D` → repeat → `Flatten`/`GlobalAveragePooling2D` → `Dense`


### Convolutional layer (Conv2D) — what it is
In Keras, `Conv2D` creates a convolution kernel and applies it across the input’s height and width to produce output feature maps.   
Conceptually: a small filter slides over the image; at each location it computes a score, and that score becomes one pixel in an output feature map. 

**Key knobs you choose**

- `filters`: how many feature maps you want (more filters = more pattern types). 
- `kernel_size`: the window size (commonly `(3, 3)`). 
- `strides`: step size of the slide (bigger stride = smaller output). 
- `padding`: `"valid"` (no padding) or `"same"` (keep size when `strides=1`). 

**Shapes (channels_last)**

- Input: `(batch, height, width, channels)`
- Output: `(batch, new_height, new_width, filters)` 



![Gif](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20260214%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20260214T120004Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=946ed3824176011062336c47aa7ea980d3d29d8075d560ae3924a31d82b23f3a)

### Pooling layer (MaxPooling2D) — what it is
`MaxPooling2D` downsamples by taking the maximum value in each window, independently for each channel, sliding the window using `strides`.   

Pooling keeps the number of channels the same and mainly reduces height/width (so compute drops). 

**Key knobs you choose**

- `pool_size`: window size (commonly `(2, 2)`). 
- `strides`: step size (often equals `pool_size`). 
- `padding`: `"valid"` or `"same"`. 

**Shape idea (channels_last)**

- Input: `(batch, height, width, channels)`
- Output: `(batch, pooled_height, pooled_width, channels)` 


### How to build these blocks in Keras 

**1) One convolution layer**
```python
from tensorflow.keras import layers

x = layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu",
)(x)
```

**2) Add max pooling (common downsampling)**
```python
x = layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="valid",
)(x)
```

**3) A small CNN you can extend**
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),   # channels_last

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10)  # logits for 10 classes
])
```

> Rule of thumb: `Conv2D` grows “what you know” (features/channels), and pooling/stride shrinks “where you know it” (height/width).



## Forward propagation (conv + pool)

Forward propagation is just the “data flow” through your CNN: you start with an input image (or a batch of images), apply convolutions to create feature maps, apply activations (e.g., ReLU), and sometimes downsample with pooling to make the next layers cheaper.

### Convolution forward pass (what happens)
At a high level, a convolution layer scans the input with each filter and produces one output feature map per filter (so `filters=64` means 64 output channels).  
The important practical detail during the forward pass is **shape tracking**: stride and padding decide how fast spatial size shrinks, while the number of filters decides how many channels you produce.

**Shape checklist (channels_last)**

- Input: `(batch, H, W, Cin)`
- Output after conv: `(batch, H2, W2, Cout)` where `Cout = filters`
- With `padding="same"` and `strides=1`, Keras keeps spatial size (so `H2 = H`, `W2 = W`). 
- With `padding="valid"`, spatial size usually gets smaller because you don’t pad borders. 

### Pooling forward pass (max pooling)

Pooling does not create new channels; it only reduces height/width by summarizing small windows (for max pooling: keep the maximum).  
Keras defines `MaxPooling2D` as downsampling by taking the maximum value over a window for each channel and shifting the window by `strides`. 

**Output size (quick reference)**
Keras provides the output-size formulas for max pooling:

- For `padding="valid"`: `out = floor((in - pool_size) / strides) + 1` 
- For `padding="same"`: `out = floor((in - 1) / strides) + 1` 

### What learners should do in practice

- Write down the tensor shape after every layer (or call `model.summary()` early and often).
- Decide where you want to downsample (after every 1–2 conv layers is common), and confirm the spatial sizes match your plan.
- If your model becomes too slow or too big, the first knobs to adjust are: earlier downsampling (pooling or larger stride) and fewer filters.

### Tiny Keras snippet: watch shapes during the forward pass
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
])

model.summary()
```

### Optional debugging trick: inspect intermediate outputs
```python
debug_model = tf.keras.Model(model.input, [layer.output for layer in model.layers])

x = tf.random.uniform((1, 128, 128, 3))
outs = debug_model(x)

for i, o in enumerate(outs):
    print(i, o.shape)
```


## Back propagation (conv + pool)

Backpropagation is how the network learns.  
After a forward pass computes a loss (how wrong the prediction was), backprop sends a “blame signal” (the gradient) backward to answer two questions:

- Which weights should change?
- In which direction, and by how much?

Think of it as: **forward pass builds the answer, backward pass tells every layer how to improve it**.

### 1) Convolution backprop (what gets updated)

A convolution layer has three main things gradients flow through:

**A) Bias gradients**

- Each output channel has (usually) one bias value.
- The bias gradient is basically: “add up the gradient values for that channel across all spatial positions (and across the batch).”

**B) Filter (kernel) gradients**

- The same filter is used at many positions during the forward pass.
- So during backprop, the filter’s gradient is the **sum of contributions from every position where it was applied**.
- Intuition: if a filter helped reduce the loss in many places, its update will be larger.

**C) Input gradients**

- Each output value came from a small input patch.
- During backprop, the gradient from an output value is **sent back to the pixels in that patch**, scaled by the filter weights.
- Result: pixels that influenced many outputs (because many windows covered them) collect more gradient.

> Learner mental model: convolution backprop is still “sliding-window math,” just running in reverse to compute updates.


### 2) Max pooling backprop (only the winner gets the gradient)

Max pooling keeps the **largest** value in each window.  
So in the backward pass:

- The gradient goes **only** to the element that was the maximum in that window.
- Every other element in that window gets **0** gradient.

Practical detail: implementations keep track of the “winning index” (argmax) from the forward pass so they know where to send the gradient in the backward pass.

> Why this matters: max pooling gradients are **sparse** (most positions get zero), which can make learning depend heavily on the strongest activations.


### 3) Average pooling backprop (everyone shares)

Average pooling outputs the mean of a window.  
So in the backward pass:

- The incoming gradient is **split evenly** across all elements in the window.
- If the window has 4 elements (2×2), each gets about one quarter of the gradient (per channel).

> Compared to max pooling, average pooling gradients are **dense** (many positions receive some gradient).


> Rule of thumb: Convolution spreads learning across many locations (shared filters); max pooling routes learning through “winners only.”




## Build a CNN in TensorFlow/Keras (end-to-end)

A clean beginner-friendly CNN is: a small convolutional “feature extractor” (Conv2D + MaxPooling2D stacks) followed by a classifier head (Flatten + Dense). TensorFlow’s CNN tutorial uses a Sequential model for CIFAR-10 with three `Conv2D` layers, two `MaxPooling2D` layers, then `Flatten` and `Dense` layers. [tensorflow](https://www.tensorflow.org/tutorials/images/cnn)

**Model code (template you can reuse)**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),  # channels_last

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10)  # logits for 10 classes
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.summary()
```

**Training loop (minimal)**
```python
# Example: x_train shape (N, 32, 32, 3), y_train shape (N,)
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64
)
```


### Practical best practices (things that prevent bugs)

- **Track shapes on purpose.** Decide one data format and stick to it (`channels_last` is the common default), because `Conv2D` expects different tensor layouts depending on `data_format`, and this is a very common source of shape mistakes.   
  Practical habit: run `model.summary()` whenever you add/remove a layer, and make sure the height/width are shrinking exactly when you planned.

- **Use padding intentionally (especially early layers).** If you want to keep the same height/width while you extract early features, use `padding="same"` with `strides=1` (Keras notes this preserves spatial size in that case).   
  Then downsample *deliberately* (either with pooling or with a larger stride) so you control the speed/accuracy trade-off instead of shrinking “by accident.”

- **Know how pooling changes size.** With pooling, `"valid"` typically shrinks more because there is no padding, while `"same"` keeps output sizes larger; Keras provides explicit output-shape formulas for both modes.   
  Practical habit: before training, write the planned sizes (e.g., 128→64→32→16) next to your pooling/stride choices and verify they match the model summary.



### Quick glossary

- **Conv2D:** A layer that learns a set of small filters (kernels) and slides them across the input image (height × width) to produce output feature maps.   
  In Keras, it returns `activation(conv2d(inputs, kernel) + bias)` (activation and bias are optional). 

- **Kernel / filter:** The small grid of learnable numbers used by `Conv2D`. One filter produces **one** output channel (one feature map). 

- **Feature map:** The output produced by applying one filter across the image; it shows where that filter “fires strongly” in different locations.

- **Stride:** How many pixels the filter (or pooling window) moves each step. Bigger stride usually means a smaller output (more downsampling). 

- **Padding:** What you do at the borders before sliding the window.  
  `padding="valid"` means no padding, while `padding="same"` pads evenly; for `Conv2D`, Keras notes that with `padding="same"` and `strides=1`, the output has the same height/width as the input. 

- **MaxPooling2D:** A layer that downsamples by taking the **maximum** value inside each window (per channel), moving the window by `strides`. 


> Everything should be made as simple as possible, but not simpler.
>
> — Albert Einstein 

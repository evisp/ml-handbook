# Data Augmentation

Data augmentation means creating *new training examples* from your existing ones by applying small, realistic changes (for images: flips, rotations, crops, brightness changes).  

You are not changing the task; you are teaching the model that “this is still the same class even if it looks a bit different.”

![Augmentation examples grid](https://i.imgur.com/JWV32it.png)

## When to use it (and when not to)

Use data augmentation when your model learns the training set well but does worse on new images (a common sign of overfitting).  

It is also useful when your dataset is small, or when real-world photos vary a lot in lighting, position, zoom, and background.

Avoid (or limit) augmentation when a transform can change the label.  
Example: flipping a “left arrow” to the right, or rotating digits like 6/9, can silently create wrong labels.

## Benefits (what you gain)

- Better generalization: the model sees more variety during training, so it is less surprised at test time.
- Less dependence on “accidents” in the dataset (exact camera angle, exact brightness, exact framing).
- A cheaper way to get more useful training signal than collecting and labeling lots of new images.

> Rule of thumb: Augment the kind of variation you expect in real life, and keep labels correct.

## Common augmentation methods

You can think of augmentation as “safe changes” that keep the label the same but make the training images more varied.  
A good strategy is to pick **2–4** augmentations that match real-world variation for your dataset, then tune how strong they are.

![Data Augmentation](https://i.imgur.com/br4CYRC.jpeg)

### 1) Geometry (position and shape)

Use these when objects can appear in different places, sizes, or orientations.

- Flips (horizontal/vertical): useful when left vs right does *not* change the class.
- Small rotations: helps when the camera is slightly tilted.
- Translations (shifts): teaches the model that the object can be a bit off-center.
- Zoom / random crop: teaches robustness to framing (object closer/farther, partially cropped).

### 2) Color and lighting

Use these when the same object might be photographed under different light conditions.

- Brightness: darker / brighter photos (indoors vs outdoors). 
- Contrast: washed-out vs sharp images. 
- Saturation / hue (RGB images): different color temperature, camera settings, or filters. 

Tip: For tasks where color is *the* clue (e.g., ripeness by color), keep these changes small.

### 3) Noise and compression (camera + social media effects)

Use these when images come from phones, messaging apps, or the web.

- JPEG quality changes: simulates compression artifacts. 
- Small noise / blur (if available in your pipeline): simulates sensor noise or motion blur.

### 4) “Cut and mix” methods (advanced, very task-dependent)

These are strong regularizers, but you should test carefully because they can change what the model focuses on.

- Cutout / random erasing: hide small patches so the model can’t rely on one tiny detail.
- Mixup: blend two images and also blend their labels.
- CutMix: cut a patch from one image and paste it into another, then mix labels by area.

> Rule of thumb: Start with geometry + a little lighting, get a baseline, then add stronger methods only if you need them.


## Practical: Data augmentation in TensorFlow (two clean ways)

TensorFlow provides **image augmentation layers** like `RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomContrast`, and notes these layers apply random transforms and are only active during training.   

TensorFlow also explains two common placements: inside the model (often best on GPU for images) or in the `tf.data` pipeline (CPU, asynchronous).

### Option A: Augmentation inside the model (simple, portable)

This is the easiest setup: you put augmentation layers **as part of the model**, right after the input (and usually after `Rescaling`).  

TensorFlow points out two big reasons to do this: you can export a truly end-to-end model (more portable), and it helps reduce *training/serving skew* because the same preprocessing logic is packaged with the model. 

#### When to choose this option (use cases)
Choose “inside the model” when:

- You want a single SavedModel that “just works” on raw images (no separate preprocessing script to remember). 
- You train on GPU and want image preprocessing/augmentation to benefit from running on device (TensorFlow notes this is often the best option for image preprocessing and augmentation layers when training on a GPU). 
- You are teaching/learning and want fewer moving parts (no custom `tf.data.map` function at first).

Avoid this option (or reconsider) when:

- You are training on a TPU: TensorFlow recommends placing preprocessing layers in the `tf.data` pipeline on TPU (with a couple of exceptions like `Normalization`/`Rescaling`). 
- Your augmentation needs very custom logic that is easier to write with `tf.image` functions.

#### Key behavior learners should remember

Image augmentation layers are **random** and TensorFlow notes they are **only active during training** (similar to how `Dropout` behaves).   
So: training sees varied images; validation/test sees the original images (no random changes).

#### Minimal Keras example (good default)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])  # Active only during training.[1]

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 255),  
    data_augmentation,

    # ... your CNN backbone ...
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(10)
])
```


### Option B: Augmentation in a `tf.data` pipeline (fast input pipelines)

This option applies augmentation **before** the data reaches the model: you transform the dataset with `Dataset.map(...)` and keep training fast with `prefetch(...)`. 

TensorFlow shows this pattern so that input work (decode, resize, augment) can run efficiently in parallel with model training. 

#### When to choose this option (use cases)

Choose augmentation in `tf.data` when:

- You want a **high-throughput input pipeline** (CPU prepares batches while the GPU/accelerator trains). 
- You train on **TPU**: TensorFlow recommends placing preprocessing layers in the `tf.data` pipeline on TPU (with a couple of exceptions like `Normalization`/`Rescaling`). 
- Your training data comes from files (JPEG/PNG), and your pipeline already does decode/resize/shuffle/batch—augmentation fits naturally there. 

Avoid (or simplify) this option when:

- You want the easiest “single exported model” story; putting augmentation inside the model can be more portable. 
- You’re new and your pipeline is already complicated; start inside the model, then move to `tf.data` once everything works.

#### Key idea learners should remember

Augmentation should be applied to the **training dataset only**; validation/test should measure performance on clean (non-augmented) inputs.  

With preprocessing layers, you can control training behavior explicitly (e.g., calling them with `training=True` inside your `map` function). 

#### Recommended pipeline pattern
A clean, common order is:
`shuffle -> map(augment) -> batch -> prefetch`

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])  # These layers apply random transforms meant for training. 

def augment(x, y):
    x = tf.cast(x, tf.float32)
    x = data_augmentation(x, training=True)  # force augmentation on training batches
    return x, y

# Training pipeline
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(64)
train_ds = train_ds.prefetch(AUTOTUNE)  # common performance pattern. 

# Validation pipeline (no augmentation)
val_ds = val_ds.batch(64).prefetch(AUTOTUNE)
```


### Practical: Basic ops with `tf.image` (manual but very flexible)

TensorFlow’s `tf.image` module includes functions like `random_flip_left_right`, `random_brightness`, `random_contrast`, `random_crop`, and `random_jpeg_quality`. 

It also provides stateless random versions (e.g., `stateless_random_flip_left_right`, `stateless_random_brightness`) when you want deterministic results given a seed. 

```python
import tensorflow as tf

def augment_with_tf_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # keep values in a stable dtype.

    image = tf.image.random_flip_left_right(image)  # horizontal flip. 
    image = tf.image.random_brightness(image, max_delta=0.1)  # lighting change.
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # contrast change. 

    # Example crop: output size must match your model input.
    image = tf.image.random_crop(image, size=(224, 224, 3))  # random crop. 

    return image, label
```

Tips that help you avoid mistakes:

- Always confirm your final image shape matches the model input shape (especially after crops).
- Keep augmentations “small” at first; if training becomes unstable, reduce rotation/zoom/brightness ranges.
- Apply augmentation only to the training set (not validation/test), otherwise you measure the wrong thing.


> REMEMBER: Augmentation is not about making images look “cool”, it’s about teaching the model the right kind of variation.
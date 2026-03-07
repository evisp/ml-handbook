# Transfer Learning with Keras Applications

## Introduction

Building AI models from scratch takes massive amounts of data and computer power—think millions of labeled photos and weeks of training time. Transfer learning solves this by letting you start with a model already trained on a huge general dataset (like everyday objects), then quickly adapt it to *your* specific problem with far less data and time.

This page breaks down the key ideas in plain language, shows how to "freeze" model parts to keep what works, and gives a complete Keras example using the Stanford Dogs dataset.

![Transfer Learning](https://miro.medium.com/v2/resize:fit:1400/1*JGtSZIhGT5-VHf3wvq_s_w.png)

## What is Transfer Learning?

**The core idea**: Take a model that's already smart at a broad task (like recognizing 1000 everyday objects from ImageNet), then reuse its knowledge for *your* specific problem.

> **Real-world motivation**: Training AI from zero takes **millions** of labeled photos + months of computer time. Transfer learning cuts this to **thousands** of photos + **hours**.

![Transfer Learning](https://towardsdatascience.com/wp-content/uploads/2018/11/1TIMA09tVqZe7tA6DckoP6g.png)

**How it works**:

- **Step 1**: Start with a pre-trained model (e.g., ResNet trained on ImageNet's 14M images)
- **Step 2**: Replace the final layer with one matching *your* classes (e.g., 120 dog breeds)
- **Step 3**: Train only the new layer first (backbone stays frozen)
- **Step 4**: Optionally "fine-tune" more layers for even better results

**Why it beats training from scratch**:

- **Faster**: Hours vs. weeks
- **Less data**: 1K photos vs. 1M+
- **Better results**: Leverages "general vision skills" (edges, shapes, textures) already learned
- **Less overfitting**: Pre-trained weights act as built-in regularization

## What is Fine-Tuning?

**The next step after feature extraction**: First you trained only the new top layers while keeping the backbone frozen. Fine-tuning now carefully updates some backbone layers too, adapting the model more precisely to your specific task.

**Why use it**: Feature extraction gives solid results quickly. Fine-tuning typically improves accuracy by 5-15% but requires more data and careful tuning.

**How it works** (two phases):

- **Phase 1**: Train new head only, backbone frozen
- **Phase 2** (fine-tuning): 
  - Unfreeze top 20-50 layers of backbone (early layers stay frozen)
  - Use 10x smaller learning rate (1e-5 vs 1e-4)
  - Train longer while monitoring validation closely

**Key differences**:

| Approach | Backbone | Learning Rate | Data Needed | Expected Accuracy |
|----------|----------|---------------|-------------|------------------|
| Feature Extraction | Frozen | Normal (1e-4) | 1K+ | 75-80% |
| Fine-tuning | Top layers trainable | Very low (1e-5) | 10K+ | 85-90% |

**When to fine-tune**:

- You have 10K+ training examples
- Your data differs significantly from ImageNet
- Feature extraction performance has plateaued
- Skip fine-tuning with fewer than 1K examples (high overfitting risk)



## Frozen Layers

**Simple concept**: A "frozen" layer cannot update its weights during training (`trainable=False`), but it still processes data normally through the forward pass.

**Why freeze layers**:

- Early layers already learned universal features (edges, corners, textures) from massive datasets like ImageNet
- Freezing prevents overwriting these valuable "building blocks" 
- Dramatically reduces trainable parameters (25M → 100K), speeds training 10x
- Prevents overfitting when working with limited target data

**Layer roles by position**:

- **Early layers** (first 20-50%): Universal features → **always freeze**
- **Middle layers**: Medium-level patterns → **freeze for small data, unfreeze for fine-tuning**
- **Late layers** (last 20-30%): Task-specific features → **unfreeze during fine-tuning**

**Freezing strategy by data size**:

| Phase | Frozen Layers | Trainable Layers | Best For |
|-------|---------------|------------------|----------|
| Feature Extraction | All backbone layers | New head only | <5K examples |
| Light Fine-tuning | First 80% of backbone | Top 20% + head | 5K-20K examples |
| Heavy Fine-tuning | First 50% of backbone | Top 50% + head | 20K+ examples |

**Key benefit**: Control exactly which parts of the model learn from your data while protecting what already works well.


## Keras Applications Workflow

![Phases](https://i.imgur.com/ugD3SCk.png)

**Step 1: Pick your base model**
Choose a proven architecture like ResNet50V2 or EfficientNetB0, already trained on ImageNet's 14 million images. Set `include_top=False` so you can add your own classification layers.

**Step 2: Freeze the backbone completely**
Mark the entire base model as non-trainable. This keeps its 25 million pre-trained weights fixed while you train only your new layers. Training speed increases dramatically.

**Step 3: Add your classification head**
Stack these layers on top:
- Global Average Pooling (reduces spatial dimensions)
- Dense layer with ReLU activation  
- Dropout for regularization
- Final Dense layer matching your number of classes

**Step 4: Two-phase training strategy**
- **Phase 1** (Feature Extraction): Train only new head at normal learning rate (1e-4). Expect good baseline accuracy fast.
- **Phase 2** (Fine-tuning): Unfreeze top 20-30 layers of backbone, drop learning rate to 1e-5, train longer for maximum accuracy.


### Choose Base Model

**Start with proven models already trained on 14 million ImageNet images**. Keras provides battle-tested architectures like ResNet50V2, EfficientNetB0, or MobileNetV3—pick based on your compute budget and accuracy needs.

**Key settings**:

- `weights='imagenet'` loads pre-trained weights
- `include_top=False` removes the original 1000-class output layer  
- `input_shape=(224, 224, 3)` standard size for most models

**Quick choice guide**:

- **ResNet50V2**: Great balance of speed + accuracy (25M params)
- **EfficientNetB0**: Newer, slightly better than ResNet (5M params)  
- **MobileNetV3**: Fastest for phones/edge devices (2M params)

This single line gets you 90% of a production model:

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
```

### Load Dataset

**Stanford Dogs Dataset**: 20,000+ labeled photos across 120 dog breeds—perfect for transfer learning practice. Much smaller than ImageNet (14M images) but challenging enough to show real improvements from fine-tuning.

**Why this dataset works well**:

- Realistic size: ~12K training images, ~8.5K test images
- 120 fine-grained classes (not just "dog" vs "cat")
- Available instantly through TensorFlow Datasets
- Tests model's ability to learn subtle visual differences between breeds

**Dataset structure**:

- Images already cropped to dogs (bounding boxes provided)
- Labels are integers 0-119 mapping to specific breeds
- Variable image sizes get resized to 224×224 for the base model

**Loading gives you**:

- Training split for learning breed patterns
- Test split for honest accuracy measurement  
- Class count (120) to size your final layer perfectly

**Next step**: Preprocess images with model-specific normalization so the pre-trained weights work correctly.


```python
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=False,
    with_info=True,
    shuffle_files=True
)
NUM_CLASSES = ds_info.features['label'].num_classes
```

### Data Preprocessing Pipeline

**Pre-trained models expect exact input format**. Wrong preprocessing kills performance.

**Three must-do steps**:

- Resize all images to 224×224 pixels (model's expected input size)
- Apply model-specific normalization (ResNetV2: RGB channels shifted to [-1,+1] range) 
- Batch into groups of 32 + prefetch for GPU speed

**Why this matters**: ResNet was trained on precisely normalized ImageNet images. Your Stanford Dogs photos must match exactly.

**Light augmentation for dog breeds**:

- Random horizontal flips
- Small rotations (±10°)
- Brightness tweaks

**Result**: Data pipeline that works perfectly with pre-trained weights.

```python
def preprocess(example):
    image = tf.cast(example['image'], tf.float32) / 255.0
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet_v2.preprocess_input(tf.cast(image * 255, tf.float32))
    return image, example['label']

train_ds = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
```

### Build & Train Feature Extractor

**Connect your frozen base model to new classification layers**. Only these new layers learn from Stanford Dogs data.

**The simple head structure**:

- Base model outputs 7×7×2048 feature maps
- Global Average Pooling → single 2048-number vector per image
- Dense(512, relu) → learns dog breed patterns  
- Dropout(0.2) → prevents overfitting to 120 breeds
- Dense(120, softmax) → breed probabilities

**Phase 1 training setup**:

- Optimizer: Adam at normal learning rate (1e-4)
- Loss: `sparse_categorical_crossentropy` (perfect for 0-119 labels)
- Metric: accuracy (top-1 breed prediction)

**What happens during training**:

- Backbone stays 100% frozen (25M weights unchanged)
- Only ~100K new parameters learn (200x fewer than full model)
- Expect 75-85% test accuracy after 5-10 epochs (1-2 hours on GPU)
- Training curves show fast improvement, then plateau

**Key insight**: You're teaching the model "what makes a Labrador different from a Golden Retriever" using ImageNet's vision foundation.

```python
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

### Fine-Tune Model

**Phase 2: Improve beyond the plateau**. Feature extraction got you to 75-85% accuracy. Fine-tuning squeezes out extra 5-15% by adapting backbone layers to dog breeds.

**The careful unfreezing process**:

- Keep first 70-80% of layers frozen (universal edges/textures)
- Unfreeze only top 20-30 layers (high-level features closest to original classifier)
- Drop learning rate 10x lower (1e-5 instead of 1e-4)

**Why this works**:

- Top layers were originally tuned for ImageNet's 1000 classes
- They adapt well to 120 dog breeds (similar "object recognition" task)
- Early layers stay frozen to preserve fundamental vision skills

**Training dynamics**:

- Much slower weight updates prevent destroying pre-trained knowledge
- Validation accuracy climbs gradually over 5-10 more epochs
- Final accuracy: 85-92% on Stanford Dogs test set
- Watch for overfitting—use early stopping if validation stalls

**Trainable parameters jump**: ~100K → 2-5M, but still 5-10x fewer than training from scratch.

**Result**: State-of-the-art breed recognition using just consumer GPU and afternoon of training.

```python
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5/10),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

history2 = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

## Tips & Pitfalls

**Always verify**: Use `model.summary()` to confirm trainable params (100K → 2-5M after unfreezing).

**Must-have callbacks**: `EarlyStopping(patience=5)`, `ReduceLROnPlateau(factor=0.2)`, `ModelCheckpoint('best.h5')`.

**Stanford Dogs expectations**: 75-85% after feature extraction, 85-92% after fine-tuning.

**Avoid these traps**:

- Wrong `preprocess_input()` → 20% accuracy drop
- Same LR for fine-tuning → destroys pre-trained weights  
- Unfreezing too many layers → instant overfitting
- No validation monitoring → training forever


> "Transfer learning turns weeks of training into hours, and millions of images into thousands. It's how production AI gets built."  
> — Every ML engineer, everywhere

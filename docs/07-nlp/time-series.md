# Time Series Forecasting: Learning the Logic of Time

![Motivation](https://i.imgur.com/pXC41dh.png)

### A Practical Guide Using Bitcoin (BTC) & TensorFlow

In standard machine learning, we treat data like individual photographs, *isolated moments in time*. In **Time Series Forecasting**, we treat data like a movie. The order of the frames is what creates the meaning. To predict what happens next, we have to understand the "story" the data has told so far.


## 1. The Core Intuition: What's in a Signal?

At its heart, forecasting is the art of **Decomposition**. This means taking a complex, messy signal and breaking it down into its simpler, fundamental ingredients. 

When we analyze any sequence of data, we assume it isn't just a random walk; instead, it is a "story" composed of different layers of logic. Some layers are predictable and stable, while others are chaotic and sudden. By separating these layers, we can filter out the confusion and focus on the patterns that actually repeat.

### The Three Layers of the "Story"

To predict the future, we must isolate these three components:

1.  **The Trend ($T_t$):** This represents the long-term momentum or the "underlying mood" of the data. It ignores daily drama and focuses on whether the value is generally growing or shrinking over a span of years.
2.  **The Seasonality ($S_t$):** These are the predictable "heartbeats"—patterns that repeat at fixed, regular intervals (daily, weekly, or yearly). This is the rhythmic pulse of the data that happens regardless of the long-term trend.
3.  **The Noise/Residuals ($R_t$):** This is the "chaos" or the leftover signal. It represents completely unpredictable events that don't follow a pattern and cannot be easily modeled.

![Bitcoin](https://i.imgur.com/TtCrsX3.png)

### Understanding Bitcoin through Decomposition

* **Trend:** Despite massive crashes, Bitcoin's multi-year history shows a clear upward growth curve as global adoption increases.
* **Seasonality:** Bitcoin often exhibits "weekend dips" where activity softens when professional markets are closed, followed by "Monday recoveries."
* **Noise:** A sudden social media post from a tech mogul or a surprise government ban in a specific country creates a sharp, unpredictable spike or drop that doesn't fit the usual pattern.


> **The Professional Logic:** We do not try to predict the **Noise**. Our goal is to capture the **Trend** and **Seasonality** so well that the noise doesn't distract the model from the truth.


## 2. The Golden Rule: Consistent Patterns (Stationarity)

A neural network learns by finding "rules." But if the rules of your data keep changing, the model gets confused. This is the problem of **Non-Stationarity**.

To build a reliable model, we must ensure the "rules" of the data remain consistent over time. 

>> **Stationarity** refers to data where statistical properties—like the average value and the degree of fluctuation—stay constant regardless of when you measure them. When data is 

>> **Non-Stationary**, these properties shift over time, making it nearly impossible for a model to generalize because the "logic" it learned from the past no longer applies to the future.

#### **Non-Stationary (The Moving Target):**
- Data that shows a clear trend or changing volatility over time.
- *Example:* **Bitcoin Price.** The average price in 2013 is fundamentally different from 2026, meaning the model can't use the same "scale" for both.

#### **Stationary (The Stable Signal):** 
- Data that fluctuates around a constant mean with consistent variance.
- *Example:* **Daily Price Returns.** Instead of looking at the total price, you look at the *percentage change* from day to day. This "centers" the data, making the patterns recognizable across different years.

* **The Problem:** In 2015, Bitcoin moved in steps of $10. In 2026, it moves in steps of $1,000. If you train a model on 2015 data, it won't understand the "math" of 2026 because the scale is totally different.
* **The Fix:** We often look at **Price Changes** (differences) rather than raw prices. By looking at how much the price changed since yesterday, we keep the data in a consistent range that the model can understand.

![Stationary](https://i.imgur.com/YsncSFD.png)

## 3. Preparing the Data: The Sliding Window

Neural networks cannot process a single, infinite line of data; they require a structured format of **Questions** (Inputs) and **Answers** (Targets). To transform a continuous timeline into a dataset the model can learn from, we use a technique called the **Sliding Window**.

Think of this as a moving "lookback" frame that scans across your history. By defining a fixed window size, you "slice" your history into hundreds or thousands of individual examples. Each window represents one specific lesson for the model: "Given these past $N$ days, the result was this $1$ future day."

### Transforming Time into Samples
As the window moves forward one step at a time, it creates overlapping pairs that preserve the chronological order:

* **Window 1:**
    * **Input (The Question):** Prices from Day 1 to Day 7.
    * **Target (The Answer):** Price on Day 8.
* **Window 2:**
    * **Input (The Question):** Prices from Day 2 to Day 8.
    * **Target (The Answer):** Price on Day 9.



This process effectively turns a single long chain of events into a "textbook" of examples, allowing the model to see how different patterns in the past consistently lead to specific outcomes in the future.


![Sliding Window](https://i.imgur.com/GiRaXV0.png)

## 4. The Engineering: Scaling and Splitting

Before feeding timeseries (e.g., Bitcoin) data into a model, we have to do two things to keep the math stable:

1.  **Scaling:** Computers hate big, varied numbers. We "squash" our prices so they all fall between 0 and 1. This prevents the model's math from "breaking" (exploding) during training.
2.  **The "No-Cheating" Split:** In normal AI, you can shuffle your data. **Never shuffle time series data.** Your training data must always come from the past, and your testing data must always come from the future. If you shuffle, the model effectively "sees" the future during its training.


## 5. Building the Data Factory (`tf.data`)

When working with years of data, loading everything into your computer's memory at once is inefficient and often impossible. Instead of a "static pile" of data, we need a **"Conveyor Belt."** In TensorFlow, we use the `tf.data` API to build a pipeline that prepares, slices, and feeds data to the model in real-time.

>> This "Data Factory" ensures that while the model is learning from one batch of data, the next batch is already being prepared behind the scenes.

This keeps the process fast and prevents the computer from "choking" on large datasets.

### The Pipeline Steps

To create this conveyor belt, we follow a specific sequence of logic:

1.  **Stream Creation:** We turn our raw list of numbers into a continuous stream.
2.  **Windowing:** We instruct the factory to group that stream into our defined window sizes (e.g., 30 days of history + 1 day for the target).
3.  **Formatting:** We separate each window into the "Question" (the past) and the "Answer" (the future).
4.  **Batching & Prefetching:** We group several windows together into "batches" and use `prefetch` so the next batch is always ready the millisecond the model needs it.



### The Implementation

```python
import tensorflow as tf

def create_time_series_pipeline(data, window_size, batch_size):
    # 1. Turn the raw array into a continuous stream of data
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # 2. Slice the stream into windows (Size + 1 to include the target)
    # shift=1 ensures we move the window forward by one day each time
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # 3. Flatten the windows and split into (Input Features, Target Label)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))
    
    # 4. Group into batches and 'prefetch' to keep the training process fast
    # Prefetching allows the CPU to prepare data while the GPU is training
    return dataset.batch(batch_size).prefetch(1)
```

* **In the Bitcoin Context:** Even if we have minute-by-minute data covering several years, this factory allows us to train on that massive history without crashing our system. It handles the "slicing" of the BTC timeline into thousands of training lessons automatically.



## 6. The Architectures: LSTM vs. GRU

Standard neural networks process data as if they are seeing it for the first time, every time. For sequences like time series, this is a fatal flaw—if a model "forgets" what happened five minutes ago, it cannot understand the context of what is happening now. To solve this, we use **Recurrent Neural Networks (RNNs)**. 

Unlike basic models, these have "internal loops" that allow information to persist. They don't just see the current data point; they see the current point *relative* to the memory of the past.



### Choosing Your "Memory Engine"

When building a forecaster, you generally choose between two specialized architectures designed to handle long sequences without "losing the thread."

#### 1. LSTM (Long Short-Term Memory)
The LSTM is the "heavyweight champion" of sequential data. Its secret weapon is the **Cell State**—a long-term memory highway that runs through the entire sequence. Specialized "gates" act as filters, deciding exactly which information is important enough to keep and which is irrelevant noise that should be forgotten.

* **The Strength:** It is incredibly good at maintaining a "long-term" memory over hundreds of time steps.
* **In the Bitcoin Context:** An LSTM can remember that a major "Halving" event occurred weeks ago or that a specific price floor was established last month, even while it is busy processing the frantic price ticks of the last ten minutes.

#### 2. GRU (Gated Recurrent Unit)
The GRU is a more modern, streamlined version of the LSTM. It simplifies the math by merging the memory states into a single vector. It uses fewer "gates," which means it has fewer parameters to train.

* **The Strength:** Because it is simpler, it is much faster to train and requires less computing power. It often performs just as well as an LSTM on smaller or less complex datasets.
* **In the Bitcoin Context:** If you are building a bot that needs to make split-second decisions on high-frequency data (like 1-minute candles), a GRU is often the better choice because of its speed and efficiency.

> **Professional Takeaway:** Start with a **GRU** to get a baseline quickly. If your data is very long and complex and the GRU isn't catching the deeper patterns, upgrade to an **LSTM**.

## 7. The Reality Check: The Baseline Test

The most important question you must ask after training your model is: **Is this network actually intelligent, or is it just a "fancy mirror"?** In time series, there is a common trap where a model appears to have high accuracy, but in reality, it has only learned to copy the most recent data point and present it as the next day's prediction. This is known as a **Persistence Model** or a **Naive Forecast**.

### The "Lazy" Competitor
A Naive Forecast assumes that tomorrow will be exactly like today ($y_{t+1} = y_t$). 

* **The Challenge:** In volatile markets like Bitcoin, the price of "Tomorrow" is often very close to the price of "Today." Because of this, a "lazy" model that just repeats the last price can actually achieve a very low error rate. 
* **The Mission:** Your sophisticated LSTM or GRU must prove its worth by outperforming this lazy guess. If your model's error (MAE or RMSE) is higher than or equal to the Naive Forecast, your model has failed to learn the "story"—it is simply lagging behind the current price.

### Why This Matters
Practicioners often get excited seeing a model's prediction line follow the actual price line closely. However, if you shift that prediction line one day to the left and it perfectly matches the real price, your model isn't forecasting; it’s just repeating.

* **In the Bitcoin Context:** If Bitcoin is at $60,000 today and your model predicts $60,050 for tomorrow, check if it’s actually identifying a "Monday Recovery" pattern or if it's just sticking close to the $60,000 mark because it's afraid to deviate. 

> **Professional Takeaway:** Always calculate your **Baseline Error** first. It is the minimum "score" your AI must beat to be considered useful. If you can't beat a "lazy" guess, the model is not ready for the real world.


## Glossary: Grouped by Concept

To make these terms easier to navigate, we can group them into the three main stages of the project: **Data Physics**, **Data Preparation**, and **Model Intelligence**.

### 1. Data Physics (The Nature of the Signal)
These terms describe the raw data before we touch it.
* **Stationarity:** A property where the "rules" of the data (like average and volatility) stay constant over time. Stationary data is much easier for neural networks to model.
* **Trend ($T_t$):** The long-term direction of the data (upward or downward growth).
* **Seasonality ($S_t$):** Predictable "heartbeats" or patterns that repeat at fixed intervals (like weekend dips).
* **Residual/Noise ($R_t$):** The "chaos" or unpredictable events that don't follow any pattern.

### 2. Data Preparation (The Factory)
These terms describe how we format and "clean" the data for the model.
* **Normalization (Scaling):** Squashing raw values (like Bitcoin's $70,000 price) into a small range (0 to 1) to keep the model's math stable.
* **Lookback (Window Size):** The specific number of past steps the model is allowed to "see" to make a prediction.
* **Horizon:** How far into the future we are predicting (e.g., a "1-day horizon").
* **Data Leakage:** A critical error where future information accidentally "leaks" into the training set, usually caused by shuffling time series data.
* **Batch Size:** The number of training windows the "Data Factory" feeds the model at one single time.

### 3. Model Intelligence (The Brain)
These terms describe how the model learns and how we judge its success.
* **Features & Labels:** The "Inputs" (what the model sees) and the "Target" (the answer it is trying to guess).
* **LSTM & GRU:** Specialized neural network architectures with "internal memory" designed to remember sequences.
* **Naive Forecast (The Baseline):** A "lazy" prediction that simply guesses tomorrow will be the same as today. Your model must beat this to be useful.
* **MAE (Mean Absolute Error):** A metric that tells you, on average, how many dollars your prediction was away from the actual price.
* **Vanishing Gradient:** A math problem where models "forget" long-term patterns; LSTMs were built specifically to fix this.


> **Final Note:** A model is only as good as the patterns it finds. If Bitcoin's "story" changes tomorrow because of a global event, your model must be updated with that new information to stay relevant.

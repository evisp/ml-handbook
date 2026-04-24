# Sequential & NLP Models

**Understand the flow of data over time.** Learn to build systems that process information in order—whether you're forecasting the stock market or creating a chatbot that remembers the start of a conversation.

![Motivation](https://i.imgur.com/XoQpFkD.png)


## [RNNs and LSTMs](./rnns.md)
**Give your models a memory**

Standard AI looks at data points in isolation. **Recurrent Neural Networks (RNNs)** use loops to remember what happened just a moment ago, making them ideal for text and sensor data.

**What you'll learn:**

* **Hidden States:** How models pass "memories" from one step to the next.
* **The Vanishing Gradient:** Why basic models struggle with long sentences.
* **LSTMs & GRUs:** Specialized tools that learn to "forget" useless info and focus on what matters.


## [Time Series Forecasting](./time-series.md)
**Predict the future using the past**

Move beyond simple guesses. Use deep learning to analyze patterns in time-stamped data, like energy demand or market shifts, to predict what comes next.

**What you'll learn:**

* **Windowing:** How to slice long histories into bite-sized training pieces.
* **Seasonality:** Identifying repeating cycles (like holiday shopping surges).
* **Multi-step Prediction:** Forecasting an entire week ahead instead of just tomorrow.


## [Understanding NLP](./nlp-overview.md)
**See the big picture before writing a single line of code**

Before building NLP systems, you need to understand how they think. Learn how raw human language travels through a pipeline and gets transformed into something a machine can reason about.

**What you'll learn:**

* **The NLP Pipeline:** The full journey from raw text to model output.
* **Encodings vs. Embeddings:** Why one is a lookup table and the other carries meaning.
* **The Semantic Gap:** How machines learn that "King - Man + Woman = Queen."


## [Word Embeddings](./embeddings.md)
**Turn language into math**

Computers don't read words; they read numbers. Learn how tools like **Word2Vec** turn words into coordinates in a digital map so the computer understands that "King" and "Queen" are related.

**What you'll learn:**

* **Dense Embeddings:** Why points in a mathematical space represent meaning better than simple lists.
* **Semantic Similarity:** Calculating how "close" two words are in meaning.
* **Word Relationships:** The logic that allows a model to understand that "King - Man + Woman = Queen."


## [Evaluation Metrics for NLP](./metrics.md)
**Measure if your model actually "gets it"**

You can't just use "Percent Correct" for language. If a model translates a sentence perfectly but uses a different synonym, is it wrong? Learn the industry standards for grading AI text.

**What you'll learn:**

* **BLEU Score:** The standard for checking translation quality.
* **ROUGE:** How to measure if a summary captured the main points.
* **Perplexity:** A metric that calculates how "confused" a model is by new text.


## [Question Answering Bots](./qa-bots.md)
**Find the needle in the haystack**

Combine memory and word meanings to build bots that can read a 50-page manual and instantly find the specific answer a user is looking for.

**What you'll learn:**

* **Context Windows:** Helping the AI stay focused on the relevant text.
* **Span Prediction:** Teaching the model to highlight the exact start and end of an answer.
* **Search & Extract:** The logic behind modern AI assistants and document search tools.


> **Language is a journey, not a snapshot.** By understanding memory and word relationships, you transform raw text into the intelligence that powers the modern world.
# Word Representations & Embeddings: Teaching Machines What Words Mean

![Embeddings Motivation](https://i.imgur.com/HegwaMh.png)

### Not All Numbers Are Equal

On the previous page, we established the core problem: computers don't read words, they read numbers. And we saw that the *way* you convert words into numbers has enormous consequences. A bad representation makes your model blind. A good one makes it intelligent.

This page walks through the complete journey of word representations — from the simplest possible approach to the most sophisticated — using the same three sentences throughout so you can see exactly what each technique gains, and what it still gets wrong.

**Our running example — three sentences we'll use for every technique:**

> *"The cat sat on the mat."*
>
> *"The dog sat on the floor."*
>
> *"Paris is the capital of France."*

These sentences are simple enough to fit in a diagram, but varied enough to expose every strength and weakness we'll discuss.


## 1. The Spectrum of Representations

Before diving into any single technique, it helps to see all of them on one map. Every method in this page is an answer to the limitations of the one before it.

```
Simpler ◄──────────────────────────────────────────────────────► More Powerful

 Bag of     TF-IDF     N-grams     CBOW /      GloVe     fastText     ELMo
  Words                            Skip-gram
                                  (Word2Vec)

 No order   Weighted   Local       Learned      Global    Handles      Context-
 No meaning No meaning order only  meaning      context   rare words   aware
```

Each step solves one specific problem while introducing a new one. By the end of this page, you'll understand exactly why each transition happened — and which tool to reach for in practice.

![Representation Spectrum](https://i.imgur.com/fGS3RSS.png)


## 2. Bag of Words: The Simplest Possible Start

**Bag of Words (BoW)** is the most naive way to represent text numerically. The idea is almost embarrassingly simple: ignore word order entirely, count how many times each word appears, and use those counts as your representation.

The name says it all — you take your sentence, throw all the words into a bag, and shake it. The order is gone. All that remains is *which* words appeared and *how often*.

### Building the Vocabulary

The first step is to collect every unique word across all your sentences. This is the **vocabulary** — the complete list of words the model knows.

From our three sentences, the vocabulary is:

```
["the", "cat", "sat", "on", "mat", "dog", "floor", "paris", "is", "capital", "of", "france"]
```

*(12 unique words after lowercasing)*

### Building the Matrix

Each sentence becomes a vector — one number per vocabulary word, representing the count of that word in the sentence.

| Sentence | the | cat | sat | on | mat | dog | floor | paris | is | capital | of | france |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| "The cat sat on the mat" | **2** | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| "The dog sat on the floor" | **2** | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| "Paris is the capital of France" | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |

### What BoW Gets Right

It works. It is fast to compute, easy to understand, and on simple tasks like spam detection, where the *presence* of words like "prize" and "winner" is enough, it performs surprisingly well.

### What BoW Gets Wrong

The moment you look at the matrix, the problem is obvious. The sentences *"The cat sat on the mat"* and *"The mat sat on the cat"* would produce **identical vectors**. The order of the words, the thing that carries most of the meaning, is completely invisible.

Additionally, the word `"the"` appears in almost every sentence and dominates the counts, drowning out the words that actually matter.

> **The Professional Logic:** BoW is a useful baseline and nothing more. If a BoW model solves your problem, you're done. If it doesn't, the next technique gives you a smarter way to count.

![Bag of Words Matrix](https://i.imgur.com/2SadTWn.png)


## 3. TF-IDF: Not All Words Matter Equally

**TF-IDF (Term Frequency–Inverse Document Frequency)** keeps the core idea of BoW — representing text as word counts — but fixes its biggest flaw: the assumption that all words are equally important.

The insight is simple: a word that appears in *every* sentence tells you nothing about what makes any specific sentence unique. The word `"the"` is useless for distinguishing meaning. The word `"Paris"`, on the other hand, only appears in one sentence — which means it is highly informative about that sentence specifically.

TF-IDF rewards words that are **frequent in a specific sentence** but **rare across all sentences**.

### The Two Components

**Term Frequency (TF)** measures how often a word appears in a specific sentence. It is the same count as BoW, sometimes normalized by sentence length.

**Inverse Document Frequency (IDF)** measures how rare a word is across the entire collection of sentences. Words that appear in every sentence get a low IDF score. Words that appear in only one sentence get a high IDF score.

The final score for each word in each sentence is:

$$\text{TF-IDF}(word, sentence) = TF \times IDF$$

### Applied to Our Example

After applying TF-IDF to our three sentences, the weights shift dramatically:

| Word | Sentence 1 | Sentence 2 | Sentence 3 |
|---|---|---|---|
| `"the"` | Low | Low | Low |
| `"sat"` | Medium | Medium | 0 |
| `"cat"` | **High** | 0 | 0 |
| `"paris"` | 0 | 0 | **High** |
| `"france"` | 0 | 0 | **High** |

The word `"the"` is penalized because it appears everywhere. The word `"paris"` is rewarded because it appears only once and is therefore highly distinctive.

### What TF-IDF Gets Right

It is significantly smarter than BoW. For tasks like **document search** and **keyword extraction**, TF-IDF is still used in production systems today.

### What TF-IDF Gets Wrong

It is still a bag. Order is still completely ignored. And critically, `"cat"` and `"dog"` still have no mathematical relationship — the model has no idea that both are animals. The meaning of words is still invisible.

> **Professional Takeaway:** TF-IDF is your go-to when you need a fast, interpretable, no-training-required baseline for text classification or search. It regularly outperforms more complex models when your dataset is small.

![TF-IDF Weight Comparison](https://i.imgur.com/k9JsJ66.png)


## 4. N-grams: Capturing Local Word Order

Both BoW and TF-IDF treat sentences as unordered collections of individual words. **N-grams** introduce the first hint of order by looking at *sequences* of consecutive words rather than individual ones.

An **n-gram** is a contiguous sequence of `n` words. When `n=2`, we call them **bigrams**. When `n=3`, **trigrams**.

### From Our Example

**Unigrams (n=1)** — the standard BoW approach:
```
"cat", "sat", "on", "the", "mat"
```

**Bigrams (n=2)** — pairs of consecutive words:
```
"the cat", "cat sat", "sat on", "on the", "the mat"
```

**Trigrams (n=3)** — triples of consecutive words:
```
"the cat sat", "cat sat on", "sat on the", "on the mat"
```

Notice immediately that bigrams capture something BoW cannot: `"cat sat"` and `"sat cat"` are now *different features*. Word order, at least locally, is preserved.

### What N-grams Get Right

They are a simple but effective way to capture local context. In tasks like **language detection** and **spam filtering**, character-level n-grams are still state-of-the-art baselines. They require no training and are fast to compute.

### What N-grams Get Wrong

The vocabulary explodes. With 12 unique words in our example, unigrams give us 12 features. Bigrams give us up to 144. Trigrams up to 1,728. In a real corpus of 50,000 words, the feature space becomes completely unmanageable.

More importantly, n-grams still produce **sparse vectors** — mostly zeros — and still have no understanding of *meaning*. The bigram `"cat sat"` and `"dog sat"` are completely unrelated features, even though they describe essentially the same event with different animals.

> **The Professional Logic:** N-grams fix the *order* problem but don't fix the *meaning* problem. We need a fundamentally different approach — one that learns representations from data rather than constructing them by hand.

![N-gram Extraction Diagram](https://i.imgur.com/5rflK91.png)


## 5. Prediction-Based Embeddings: CBOW and Skip-gram

Everything we've built so far has been **hand-crafted**: we count words, we weight them, we chain them together. But what if instead of designing the representation ourselves, we *trained a model to learn it*?

This is the core idea behind **Word2Vec** — and its two training strategies: **CBOW** and **Skip-gram**.

Both are built on the same foundational principle, known as the **Distributional Hypothesis**:

>> *"A word is known by the company it keeps."* Words that consistently appear in the same contexts tend to have similar meanings. `"cat"` and `"dog"` both appear near words like `"sat"`, `"the"`, and `"on"` — so the model learns to place them close together in the embedding space.

![A word is known by the company it keeps](https://i.imgur.com/8al1y2F.png)

Neither CBOW nor Skip-gram is trying to solve a language task directly. They are solving a **prediction task** as a pretext — and the embeddings that emerge as a byproduct of that training are what we actually want.

### The Context Window

Both approaches use the concept of a **context window** — a fixed number of words to the left and right of a target word that define its "neighborhood."

Using our sentence *"The cat sat on the mat"* with a window of size 2:

```
Context: ["The", "cat", "on", "the"]   →   Target: "sat"
```

The four surrounding words are the context. The center word is the target.

![CBOW vs Skip-gram Diagram](https://i.imgur.com/jKcG0qF.png)

### CBOW: Predicting the Center from the Crowd

**Continuous Bag of Words (CBOW)** asks: *"Given the surrounding words, what is the missing center word?"*

Think of it as a **fill-in-the-blank** game:

> *"The ___ sat on the mat."*
> The model sees `["The", "sat", "on", "the"]` and must predict `"cat"`.

The model learns by adjusting the embedding of every word until its predictions improve. Words that are often surrounded by the same context words naturally end up with similar embeddings.

**In our running example:**

- Input: `["the", "sat", "on", "the"]` (context around `"cat"`)
- Task: Predict → `"cat"`
- Side effect: The model also trains on `["the", "sat", "on", "the"]` around `"dog"` in sentence 2
- Result: `"cat"` and `"dog"` end up with similar embeddings because they share the same neighborhood

**The Strength:** CBOW is fast and efficient. Because it averages the context, it handles frequent words well. It is ideal for large datasets.

### Skip-gram: Predicting the Crowd from the Center

**Skip-gram** flips the task: *"Given the center word, predict the surrounding words."*

Instead of one prediction per window, it generates **multiple training pairs** — one for each context word:

> Center word: `"sat"`
> Predictions: → `"the"`, → `"cat"`, → `"on"`, → `"the"`

**In our running example:**

- Input: `"sat"`
- Task: Predict → `"the"`, `"cat"`, `"on"`, `"the"` (sentence 1) AND `"the"`, `"dog"`, `"on"`, `"the"` (sentence 2)
- Result: `"sat"` learns to be associated with animals, prepositions, and articles — a rich, nuanced embedding

**The Strength:** Skip-gram produces higher quality embeddings for rare words. Because it generates more training pairs per sentence, it squeezes more signal from less data. It is the preferred choice for smaller datasets.

> **Professional Takeaway:** **CBOW is faster, Skip-gram is smarter.** Use CBOW when training speed matters and your dataset is large. Use Skip-gram when your vocabulary has many rare or specialized words and embedding quality matters more than training time.



## 6. Negative Sampling: Teaching by Contrast

There is a hidden engineering problem inside CBOW and Skip-gram. Every time the model makes a prediction, it computes a score for *every word in the vocabulary* — which might be 100,000 words. Updating all of those weights for every single training step is computationally brutal.

**Negative Sampling** solves this by changing the question. Instead of asking *"which word fits here?"* across all vocabulary, it asks a simpler binary question: *"does this word-context pair make sense?"*

### Positive and Negative Pairs

From our sentences, we naturally get **positive pairs** — word-context combinations that actually appear together:

```
Positive: ("sat", "cat")  ✓  — these appear near each other in Sentence 1
Positive: ("sat", "dog")  ✓  — these appear near each other in Sentence 2
```

**Negative pairs** are randomly sampled word combinations that do *not* appear together and are almost certainly meaningless:

```
Negative: ("sat", "france")  ✗  — these never appear together
Negative: ("sat", "capital")  ✗  — these never appear together
```

The model trains by learning to **score positive pairs high** and **score negative pairs low**. Instead of updating 100,000 weights per step, it updates only a handful — the positive pair plus a small number of randomly sampled negatives (typically 5–20).

### Why This Works

The model learns *contrast*. It doesn't just learn that `"cat"` fits near `"sat"` — it learns that `"france"` doesn't. This contrastive signal is surprisingly rich and produces excellent embeddings at a fraction of the computational cost.

> **The Professional Logic:** Negative sampling is what makes Word2Vec trainable on billions of words on a standard machine. Without it, the same training would require a server farm. Good engineering is what turns good theory into practical tools.

![Negative Sampling Diagram](https://i.imgur.com/vqZAXkK.png)


## 7. The Big Four: Word2Vec, GloVe, fastText, and ELMo

CBOW, Skip-gram, and Negative Sampling are the *engine*. The following four models are the *vehicles* built around that engine — each one extending the core idea to fix a specific remaining limitation.

![The Big Four Models](https://i.imgur.com/faT3yqR.png)

### Word2Vec
**The problem it solved:** BoW, TF-IDF, and n-grams produce no meaningful geometry. `"cat"` and `"dog"` are unrelated numbers. Word2Vec was the breakthrough that proved learned embeddings could encode *semantic relationships* — the famous `king - man + woman = queen` result.

**How it works:** Trains either CBOW or Skip-gram on a large corpus. The resulting embedding matrix maps every word to a dense vector where similar words cluster together.

**Its remaining limitation:** Each word gets exactly *one* embedding, regardless of context. The word `"bank"` gets one fixed vector — even though `"river bank"` and `"savings bank"` mean entirely different things.


### GloVe (Global Vectors for Word Representation)
**The problem it solved:** Word2Vec learns from local context windows only — it sees words in their immediate neighborhood but misses global statistical patterns across the entire corpus.

**How it works:** Instead of predicting words, GloVe directly factorizes a **co-occurrence matrix** — a table that counts how often every pair of words appears together across the entire dataset. By capturing global co-occurrence statistics, it builds embeddings that reflect corpus-wide patterns, not just local windows.

**In our running example:** GloVe would notice that `"cat"` and `"dog"` co-occur with the exact same set of words across the whole corpus — giving them very similar embeddings — while Word2Vec might miss some of this if those patterns are spread across distant sentences.

**Its remaining limitation:** Still one fixed vector per word. The polysemy problem (`"bank"`) remains unsolved.


### fastText
**The problem it solved:** Both Word2Vec and GloVe are helpless against **out-of-vocabulary (OOV) words** — words they've never seen during training. Typos, rare names, and domain-specific jargon simply get no representation.

**How it works:** Instead of treating each word as an atomic unit, fastText breaks words into **character n-grams** (subword pieces). The word `"capital"` becomes `["cap", "api", "pit", "ita", "tal"]`. The final word embedding is the sum of its subword embeddings.

**The power of this:** Even if fastText has never seen the word `"Holbertonian"`, it can construct a reasonable embedding from pieces like `"Holbert"`, `"bert"`, `"onian"` that it *has* seen. It handles morphologically rich languages and technical vocabulary far better than its predecessors.

**Its remaining limitation:** Still produces one static embedding per word. Context still doesn't change the meaning.


### ELMo (Embeddings from Language Models)
**The problem it solved:** Every model so far assigns a single, fixed vector to each word. But meaning is fundamentally contextual. The word `"sat"` in *"The cat sat on the mat"* and the name *"SAT exam"* should have different representations.

**How it works:** ELMo trains a deep bidirectional language model that reads the entire sentence in *both directions* before producing an embedding. The embedding for each word is not fixed — it is **dynamically computed** based on the full sentence context.

**In our running example:**

- `"sat"` in *"The cat sat on the mat"* → embedding reflects a physical action
- `"capital"` in *"Paris is the capital of France"* → embedding reflects a political concept
- If `"capital"` appeared in *"I need capital to start a business"* → embedding would shift toward finance

**Its remaining limitation:** ELMo is computationally expensive. It was also quickly superseded by **Transformers** (like BERT), which solve the same context problem with greater efficiency and power.

### Side-by-Side Comparison

| Model | Captures Meaning | Handles OOV | Context-Aware | Speed |
|---|---|---|---|---|
| Word2Vec | Yes | No | No | Fast |
| GloVe | Yes | No | No | Fast |
| fastText | Yes | Yes | No | Fast |
| ELMo | Yes | Yes | Yes | Slow |

> **Professional Takeaway:** In practice, **fastText** is the best default static embedding — it handles OOV gracefully and trains quickly. Use **ELMo** or a Transformer-based model when context-sensitivity is critical. For most projects, start with fastText and upgrade only if performance demands it.


## 8. Reality Check: When Embeddings Fail

Embeddings are powerful, but they are not magic. Before deploying them, you need to understand three concrete ways they can quietly fail.

### Failure 1: Out-of-Vocabulary Words (OOV)

Word2Vec and GloVe assign no representation to words they've never seen. In production systems this happens constantly — user typos, new product names, emerging slang, technical jargon. A model that silently ignores 10% of its input is a dangerous model.

> *In our example:* If a new sentence arrives saying *"The kitten sat on the mat"*, and `"kitten"` wasn't in the training vocabulary, Word2Vec simply has no vector for it — even though it's semantically almost identical to `"cat"`.

**The fix:** Use fastText for its subword representations, or a Transformer model with a subword tokenizer.

### Failure 2: Bias in the Training Data

Embeddings learn from human-generated text — and human text is full of bias. Studies have shown that classic embeddings associate certain professions with specific genders, and certain names with positive or negative sentiment, purely because those associations were frequent in the training corpus.

> *A real example:* The analogy `"man : doctor :: woman : nurse"` was produced by Word2Vec trained on news articles — reflecting the gender distribution of those articles, not the reality of medicine.

**The fix:** Be aware of it. Audit your embeddings before deployment. Consider debiasing techniques or training on more balanced corpora.

### Failure 3: Domain Shift

An embedding trained on Wikipedia has no understanding of medical terminology, legal language, or financial jargon. `"discharge"` in a hospital context means something entirely different from `"discharge"` in a military or electrical context. A general-purpose embedding will get this wrong.

> *In practice:* A customer service bot for a bank trained on general-purpose GloVe embeddings may fail to distinguish `"account balance"` from `"balance beam"` in edge cases where context is ambiguous.

**The fix:** Fine-tune your embeddings on domain-specific text, or use a domain-specific pre-trained model.

> **Professional Takeaway:** Always interrogate your embeddings before trusting them. Probe them with analogies, check for OOV rates on your specific dataset, and audit for bias. A fast sanity check before training saves weeks of debugging after deployment.


## Glossary

### 1. Counting-Based Methods

* **Bag of Words (BoW):** A representation that counts word occurrences in a document, ignoring order entirely.
* **TF-IDF:** A weighting scheme that rewards words frequent in one document but rare across all documents.
* **Term Frequency (TF):** How often a word appears in a specific document.
* **Inverse Document Frequency (IDF):** A penalty applied to words that appear in many documents, reducing the weight of common words.
* **N-gram:** A contiguous sequence of `n` words. Bigrams capture pairs; trigrams capture triples.
* **Sparse Vector:** A vector that is mostly zeros. BoW, TF-IDF, and n-gram vectors are sparse.

### 2. Prediction-Based Embeddings

* **Distributional Hypothesis:** The principle that words appearing in similar contexts have similar meanings. The theoretical foundation of all learned embeddings.
* **Context Window:** The fixed number of words to the left and right of a target word used to define its neighborhood during training.
* **CBOW (Continuous Bag of Words):** Predicts the center word from its surrounding context. Fast and efficient on large datasets.
* **Skip-gram:** Predicts surrounding context words from a center word. Better quality embeddings for rare words.
* **Negative Sampling:** A training technique that replaces the expensive "predict from all vocabulary" step with a binary "does this pair make sense?" question, using randomly sampled incorrect pairs as contrast.
* **Dense Vector:** A vector where most values are non-zero real numbers. Learned embeddings are dense.

### 3. Embedding Models

* **Word2Vec:** The landmark model that proved learned embeddings encode semantic relationships. Trains via CBOW or Skip-gram.
* **GloVe:** Extends Word2Vec by incorporating global co-occurrence statistics across the entire corpus, not just local windows.
* **fastText:** Extends GloVe by decomposing words into character n-grams, allowing it to construct embeddings for unseen words.
* **ELMo:** Produces context-sensitive embeddings by reading the full sentence bidirectionally. The word `"bank"` gets a different vector in different sentences.
* **Co-occurrence Matrix:** A table counting how often every pair of words appears together across a corpus. The input to GloVe.
* **Subword Tokenization:** Breaking words into smaller character-level pieces. Used by fastText and modern Transformers to handle OOV words.

### 4. Failure Modes

* **Out-of-Vocabulary (OOV):** A word the model has no embedding for, because it was never seen during training.
* **Domain Shift:** The drop in performance when a model trained on one type of text is applied to a different domain.
* **Embedding Bias:** Learned associations in an embedding space that reflect historical biases present in the training corpus.
* **Polysemy:** The property of a word having multiple distinct meanings (e.g., `"bank"`). Static embeddings fail to handle this; context-aware models like ELMo address it.


> **What's Next:** You now understand how to represent language as mathematics. The next step is learning how to *evaluate* whether your NLP model is actually doing its job — because accuracy alone tells you almost nothing about language quality. That's exactly what the Evaluation Metrics page covers.

# Evaluation Metrics for NLP: How Do You Grade a Sentence?

![Evaluation Metrics Motivation](https://i.imgur.com/WCjWJfh.png)

### Getting the Right Answer is Easy to Measure. Getting the Right Sentence is Not.

In standard machine learning, grading a model is straightforward. It predicted 
`1`, the answer was `1` — correct. It predicted `0`, the answer was `1` — wrong. 
Accuracy, precision, recall. Clean, mathematical, unambiguous.

Language doesn't work like that.

Consider a model that translates *"Students must submit their work on time"* as 
*"Learners are required to hand in assignments punctually."* Every word is 
different. The meaning is identical. A standard accuracy metric would score 
this as completely wrong.

This is the central challenge of NLP evaluation: **two sentences can mean the 
same thing with zero word overlap, and two sentences can share every word while 
meaning completely different things.** We need metrics that are sensitive to 
meaning, not just surface form.

This page introduces the three most important NLP evaluation metrics — BLEU, 
ROUGE, and Perplexity — and shows exactly when and why to use each one.

## 1. Why NLP Needs Its Own Metrics

Before introducing the metrics, we need to understand the three fundamental 
questions that NLP evaluation must answer — because each metric answers a 
different one.

> *"Is this translation faithful to the original?"* → **BLEU**
>
> *"Does this summary capture the key points?"* → **ROUGE**
>
> *"How confident is this model about the language it generates?"* → **Perplexity**

![Metrics](https://i.imgur.com/wdrclRY.png)

The most important professional insight on this page is this: **there is no 
universal NLP metric.** The task always comes first. Using BLEU to evaluate 
a summarizer, or ROUGE to evaluate a language model, produces meaningless 
numbers. The metric must match the task.

### The Running Example

We will use a single source sentence to illustrate these evaluation metrics.

>> **Source sentence:**
>> *"Students who miss more than three consecutive days without notifying 
>> staff may be subject to academic review."*

From this one sentence, we will derive three tasks:

| Task | What We Produce | Metric |
|---|---|---|
| **Translation** | A French translation of the source | BLEU |
| **Summarization** | A one-sentence summary of the source | ROUGE |
| **Generation** | A language model completes a related sentence | Perplexity |

The same source. Three tasks. Three metrics. Each one motivated by the 
specific challenge of evaluating that task.


## 2. Applications and Their Metrics

Different NLP tasks require different evaluation strategies. Here is the 
landscape before we go deep on any single metric.

### Translation

**Task:** Convert text from one language to another while preserving meaning.

**Metric:** BLEU — measures how closely the machine translation matches one 
or more human reference translations at the n-gram level.

**Example systems:** Google Translate, DeepL, multilingual customer support bots.

### Summarization

**Task:** Compress a long document into a shorter one that retains the key information.

**Metric:** ROUGE — measures how much of the human-written reference summary 
appears in the machine-generated summary.

**Example systems:** News summarizers, document abstractors, meeting note generators.

### Language Modeling & Text Generation
**Task:** Generate fluent, coherent text — completions, responses, continuations.

**Metric:** Perplexity — measures how confidently and accurately a model 
predicts the next word in a sequence.

**Example systems:** Autocomplete, chatbots, code generation tools.

### Text Classification & Question Answering
**Task:** Assign labels to text, or extract specific answers from a passage.

**Metric:** Standard metrics apply here — **Accuracy**, **F1 Score**, 
**Exact Match (EM)**. These tasks have clear correct answers, so standard 
classification metrics are appropriate.

> **The Professional Logic:** Always ask *"what does success look like for 
> this task?"* before choosing a metric. A metric that doesn't reflect your 
> definition of success will mislead you — even if the numbers look impressive.

![NLP Tasks and Metrics Map](https://i.imgur.com/JJNbzg3.png)


## 3. BLEU Score: Measuring Translation Quality

**BLEU (Bilingual Evaluation Understudy)** is the standard metric for 
evaluating machine translation. It answers the question:

>> *"How much of the machine-generated output matches a human reference translation?"*

The core idea is **precision of n-grams**: how many of the word sequences 
in the machine output also appear in the human reference?

### Setting Up the Example

**Source (English):**

> *"Students who miss more than three consecutive days without notifying 
> staff may be subject to academic review."*

**Human Reference Translation (French):**

> *"Les étudiants qui manquent plus de trois jours consécutifs sans 
> notifier le personnel peuvent faire l'objet d'une révision académique."*

**Machine Translation A (Good):**

> *"Les étudiants qui manquent plus de trois jours consécutifs sans 
> informer le personnel peuvent être soumis à une révision académique."*

**Machine Translation B (Bad):**

> *"Les étudiants académique jours personnel consécutifs révision 
> manquent notifier."*

Translation B contains all the right words; but in a random, meaningless order.
A good metric must reward A and penalize B.

### How BLEU Works: N-gram Precision

BLEU computes **precision at multiple n-gram levels** and combines them.

**Unigram precision (n=1):** What fraction of words in the output appear 
in the reference?

For Translation A, nearly every word matches the reference. For Translation B, 
the words match but the order is completely wrong — which unigram precision 
alone cannot detect.

**Bigram precision (n=2):** What fraction of consecutive word pairs in the 
output appear in the reference?

Here Translation B collapses. Word pairs like `"étudiants académique"` never 
appear in the reference. Bigrams begin to capture order.

**The BLEU formula combines precision across n-gram levels:**

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:

- $p_n$ is the **modified precision** for n-grams of size $n$
- $w_n$ is the weight for each n-gram level, typically $w_n = \frac{1}{N}$
- $N$ is the maximum n-gram order, typically $N = 4$
- $BP$ is the **Brevity Penalty** (explained below)

The modified precision for n-grams of order $n$ is:

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min\left(\text{count}(\text{n-gram}, \hat{y}),\ \text{count}(\text{n-gram}, y)\right)}{\sum_{\text{n-gram} \in \hat{y}} \text{count}(\text{n-gram}, \hat{y})}$$

Where $\hat{y}$ is the machine output and $y$ is the human reference. The 
$\min$ prevents the model from cheating by repeating a single correct word 
many times.

### The Brevity Penalty

Without a length penalty, a model could achieve perfect precision by outputting 
a single word — if that word appears in the reference, the precision is 1.0. 
BLEU prevents this with the **Brevity Penalty (BP)**:

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

Where $c$ is the length of the candidate output and $r$ is the length of 
the reference. If the output is shorter than the reference, the score is 
penalized exponentially.

**BLEU scores range from 0 to 1.** In practice, a BLEU score above 0.4 
is considered good for machine translation. Human translations against 
each other typically score around 0.6–0.7 — because even humans rarely 
translate identically.

### What BLEU Gets Right

It is fast, reproducible, and language-independent. It correlates well 
with human judgment on translation quality at the corpus level. For this 
reason it has been the standard benchmark for translation systems for over 
two decades.

### What BLEU Gets Wrong

BLEU is precision-focused — it measures whether the output is faithful 
to the reference. It does not measure whether the output is **complete**. 
A translation that captures only half the sentence but does so perfectly 
can score deceptively high.

It also has no concept of synonyms. Our good translation used `"informer"` 
where the reference used `"notifier"` — both are correct French translations 
of `"notifying"`. BLEU counts this as a mismatch.

> **Professional Takeaway:** BLEU is a useful proxy, not a truth. Always 
> report BLEU at multiple n-gram levels and always combine it with human 
> evaluation on a sample of outputs before trusting the number.

![BLEU N-gram Matching Diagram](https://i.imgur.com/bmrQ5TS.png)


## 4. ROUGE Score: Measuring Summary Quality

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is the standard 
metric for evaluating text summarization. It answers a fundamentally different 
question from BLEU:

>> *"How much of the human reference summary appears in the machine-generated summary?"*

Where BLEU asks *"is the output precise?"*, ROUGE asks *"is the output complete?"* 
This is the difference between **precision** (what you said was right) and 
**recall** (did you say everything important).

### Setting Up the Example

**Source (full sentence):**

> *"Students who miss more than three consecutive days without notifying 
> staff may be subject to academic review."*

**Human Reference Summary:**

> *"Missing three or more days without notifying staff risks academic review."*

**Machine Summary A (Good — abstractive):**

> *"Unexcused absences of three or more consecutive days may trigger an 
> academic review process."*

**Machine Summary B (Bad — incomplete):**

> *"Students may miss days."*

**Machine Summary C (Lazy — extractive copy):**

> *"Students who miss more than three consecutive days without notifying 
> staff may be subject to academic review."*

Summary C is the full original sentence. It scores perfectly by any n-gram 
metric — but it is not a summary. A good metric should reward A and penalize 
both B and C.

### ROUGE-1: Unigram Recall

**ROUGE-1** measures the overlap of individual words between the machine 
summary and the human reference.

$$\text{ROUGE-1} = \frac{\text{Number of overlapping unigrams}}{\text{Total unigrams in reference}}$$

For Summary A, words like `"days"`, `"staff"`, `"academic"`, `"review"` all 
appear in the reference. The recall is high even though the exact phrasing differs.

For Summary B, only `"students"` and `"days"` match — very low recall.

### ROUGE-2: Bigram Recall

**ROUGE-2** raises the bar by requiring consecutive word pairs to match. 
This penalizes summaries that get the right words but in the wrong order 
or context.

$$\text{ROUGE-2} = \frac{\text{Number of overlapping bigrams}}{\text{Total bigrams in reference}}$$

Summary A scores lower on ROUGE-2 than ROUGE-1 because it paraphrases — 
the bigrams are different even though the meaning is the same. This is 
a known weakness of ROUGE: it penalizes good abstractive summaries.

### ROUGE-L: Longest Common Subsequence

**ROUGE-L** is more flexible. Instead of requiring exact n-gram matches, 
it finds the **Longest Common Subsequence (LCS)** — the longest sequence 
of words that appears in both the reference and the output, in the same 
order but not necessarily consecutively.

$$\text{ROUGE-L} = \frac{|\text{LCS}(\hat{y},\ y)|}{|y|}$$

Where $|\text{LCS}(\hat{y}, y)|$ is the length of the longest common 
subsequence and $|y|$ is the length of the reference.

ROUGE-L handles paraphrasing better than ROUGE-2 because it allows gaps 
between matching words. Summary A scores well on ROUGE-L even though its 
exact phrasing differs from the reference.

### The BLEU–ROUGE Distinction

| | BLEU | ROUGE |
|---|---|---|
| **Primary focus** | Precision | Recall |
| **Question asked** | Is the output faithful? | Is the output complete? |
| **Penalizes** | Hallucination (saying wrong things) | Omission (missing key points) |
| **Best for** | Translation | Summarization |

In practice, a good summarization system should score well on **both** 
ROUGE (recall — it covers the key points) and a modified BLEU (precision — 
it doesn't add hallucinated content). Using only one misleads you.

> **Professional Takeaway:** ROUGE rewards summaries that cover the reference 
> content. But a summary that copies the source verbatim scores perfectly on 
> ROUGE while being completely useless. Always inspect high-scoring outputs 
> manually — the number alone is not enough.

![ROUGE Recall vs BLEU Precision Diagram](https://i.imgur.com/HKJGLix.png)


## 5. Perplexity: Measuring Model Confidence

**Perplexity** is a completely different kind of metric. BLEU and ROUGE 
compare a machine output to a human reference. Perplexity has no reference. 
Instead, it measures something internal to the model itself:

>> *"How surprised is this model by the text it sees?"*

A language model assigns a probability to every possible next word. A model 
that has learned the patterns of English well should assign high probability 
to natural, grammatical continuations — and low probability to random or 
incoherent ones. Perplexity quantifies this confidence.

### The Intuition

Imagine a language model is asked to predict the next word in:

> *"Students who miss more than three consecutive days without notifying 
> staff may be subject to academic ___."*

A well-trained model knows that `"review"` is a highly probable next word 
in this context. It assigns a high probability to `"review"` and low 
probabilities to random words like `"banana"` or `"seventeen"`.

Now imagine the model is asked to predict the next word in:

> *"Academic notifying staff three students days miss consecutive ___."*

This is the same words in a random order. A good model should be very 
uncertain — no word is obviously the right continuation. Its probability 
distribution is flat and confused.

**Perplexity measures this confusion.** Low perplexity = confident = the 
text follows patterns the model understands. High perplexity = confused = 
the text is unexpected or incoherent.

### The Formula

Given a sequence of $N$ words $w_1, w_2, \ldots, w_N$, perplexity is 
defined as:

$$\text{PP}(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}}$$

Which can be written using the chain rule as:

$$\text{PP}(W) = \left(\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1, \ldots, w_{i-1})}\right)^{\frac{1}{N}}$$

The intuition behind the formula: perplexity is the **geometric mean of the 
inverse probabilities** of each word in the sequence. If the model assigns 
high probability to every word, the inverse probabilities are small, and 
perplexity is low. If the model is consistently surprised, probabilities are 
low, inverse probabilities are large, and perplexity is high.

### Applied to the Example

| Text | Model Confidence | Perplexity |
|---|---|---|
| *"Students who miss three consecutive days may face academic review"* | High — natural and expected | Low (~20–50) |
| *"Students who miss three consecutive days may face academic banana"* | Medium — natural until the last word | Medium (~200–500) |
| *"Academic notifying staff three students days miss consecutive"* | Low — no natural pattern | High (~1000+) |

A language model trained on institutional text would assign very high 
probability to the first sentence — it matches patterns seen many times 
during training. The nonsense reordering breaks every pattern the model 
learned, producing high perplexity.

### What Perplexity Measures — and What It Doesn't

Perplexity measures **fluency**, not **correctness**. A model can generate 
a perfectly grammatical, stylistically natural sentence that is factually 
wrong — and it will score low perplexity. The sentence *"Students who pass 
all their projects are subject to immediate expulsion"* is fluent, 
institutional-sounding, and completely false. A language model may assign 
it low perplexity regardless.

This is why perplexity is used to evaluate **language models** during 
training — it tells you whether the model is learning the statistical 
patterns of the language — but not to evaluate factual accuracy or 
task-specific quality.

> **Professional Takeaway:** Use perplexity to compare two versions of 
> the same model (lower is better) or to monitor training progress. 
> Never use it as the sole metric for a production system — a fluent 
> liar scores just as well as a fluent truth-teller.

![Perplexity Confidence Scale](https://i.imgur.com/i9PUbSC.png)


## 6. Choosing the Right Metric

With three metrics in hand, the practical question is: which one do you 
reach for, and when?

The answer always starts with the task. Here is the decision map:

| Task | Primary Metric | Secondary Metric | What to Watch For |
|---|---|---|---|
| Machine Translation | BLEU | Human evaluation | Synonym mismatches, fluency |
| Summarization | ROUGE-L | ROUGE-1, ROUGE-2 | Verbatim copying, omission |
| Language Modeling | Perplexity | Human evaluation | Fluent nonsense |
| Text Classification | F1 Score | Accuracy | Class imbalance |
| Question Answering | Exact Match | F1 Score | Partial answers |
| Dialogue / Chat | Human evaluation | Perplexity | Coherence, relevance |

### The Two-Metric Rule

A consistent professional practice is to always use **at least two metrics** 
for any NLP system:

- One metric that rewards **precision** (did you say the right things?)
- One metric that rewards **recall** (did you say everything important?)

For translation: BLEU (precision) + a recall-aware metric or human evaluation.

For summarization: ROUGE (recall) + a precision check to catch hallucinations.

For generation: Perplexity (fluency) + human evaluation for factual accuracy.

No single metric captures everything. The combination is what gives you 
an honest picture of your system's behavior.

### When Human Evaluation is Non-Negotiable

All three metrics share a fundamental limitation: they measure surface form, 
not meaning. There are specific situations where automatic metrics will 
mislead you and human evaluation is the only honest option:

- **Creative or open-ended generation** — where many valid outputs exist 
  and none of them match the reference
- **Cross-lingual evaluation** — where the reference translation reflects 
  one valid style and the system uses another equally valid one
- **High-stakes applications** — medical, legal, or safety-critical text 
  where a fluent wrong answer is worse than no answer

> **Professional Takeaway:** Automatic metrics are fast, cheap, and 
> reproducible — use them always. Human evaluation is slow, expensive, 
> and irreplaceable — use it before shipping anything to production.


## 7. Reality Check: When Metrics Lie

Every metric in this page has a known failure mode. Knowing them is what 
separates a practitioner from someone who just runs the numbers.

### BLEU Lies When Order Doesn't Matter

BLEU gives a high score to an output that uses exactly the right words 
in a slightly wrong order. In many language pairs, word order varies 
legitimately. A German-to-English translator that produces a grammatically 
correct but differently ordered sentence gets penalized unfairly.

> *In our example:* A model that translates the source as *"Academic 
> review may follow for students missing three or more consecutive days 
> without staff notification"* is correct and natural — but scores low 
> BLEU against a reference with a different structure.

### ROUGE Lies When Copying Is Easy

A system that returns the full original source document as its summary 
achieves a perfect ROUGE score. Verbatim extraction is trivially rewarded. 
This is why good summarization evaluation always checks for the **compression 
ratio** alongside ROUGE — a system that compresses nothing has learned nothing.

> *In our example:* Summary C — the full original sentence — would score 
> near-perfect on all ROUGE variants despite being completely useless as a summary.

### Perplexity Lies When Fluency and Truth Diverge

A language model fine-tuned on formal institutional text will assign low 
perplexity to any institutional-sounding sentence — regardless of whether 
it is true. The sentence *"Students are required to submit three projects 
per day"* is grammatically natural and contextually plausible. It would 
score low perplexity even though it is factually wrong.

> *The deeper problem:* As language models become more fluent, perplexity 
> scores improve while factual reliability does not necessarily follow. 
> Perplexity tracks fluency. Truth requires a different kind of evaluation 
> entirely — one the field is still actively developing.

> **Professional Takeaway:** Never trust a single metric in isolation. 
> Report multiple metrics, inspect your outputs manually, and be honest 
> about what your evaluation setup can and cannot detect. A number is a 
> proxy for quality — it is never a guarantee of it.


## Glossary

### 1. Core Concepts

* **Evaluation Metric:** A quantitative measure used to assess the quality 
  of a model's output relative to a reference or standard.
* **Reference:** A human-generated output used as the gold standard for 
  comparison. Most NLP metrics compare machine output to one or more references.
* **Precision:** The fraction of the model's output that is correct. 
  Answers the question: *"Of what the model said, how much was right?"*
* **Recall:** The fraction of the reference content that appears in the 
  model's output. Answers the question: *"Of what should have been said, 
  how much was covered?"*
* **F1 Score:** The harmonic mean of precision and recall. Balances both 
  concerns into a single number.

### 2. BLEU

* **BLEU (Bilingual Evaluation Understudy):** A precision-focused metric 
  that measures n-gram overlap between machine output and human reference. 
  Standard for translation evaluation.
* **Modified Precision:** A version of n-gram precision that clips counts 
  to prevent a model from cheating by repeating a single correct word.
* **Brevity Penalty (BP):** A multiplicative penalty applied when the 
  machine output is shorter than the reference, preventing artificially 
  high precision from very short outputs.
* **N-gram Overlap:** The number of n-word sequences shared between 
  two texts. Higher overlap suggests greater similarity.

### 3. ROUGE

* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** A 
  recall-focused metric that measures how much of the reference appears 
  in the machine output. Standard for summarization evaluation.
* **ROUGE-1:** Measures unigram (single word) recall between output and reference.
* **ROUGE-2:** Measures bigram (two-word sequence) recall between output and reference.
* **ROUGE-L:** Measures recall based on the Longest Common Subsequence — 
  more flexible than bigram matching, handles paraphrasing better.
* **Longest Common Subsequence (LCS):** The longest sequence of words 
  appearing in the same order in both texts, not necessarily consecutively.
* **Compression Ratio:** The ratio of summary length to source length. 
  Used alongside ROUGE to detect extractive copying.

### 4. Perplexity

* **Perplexity:** A measure of how surprised a language model is by a 
  given text. Low perplexity means the text matches patterns the model 
  learned; high perplexity means the text is unexpected or incoherent.
* **Language Model:** A model that assigns probabilities to sequences 
  of words. Used for text generation, autocomplete, and speech recognition.
* **Fluency:** The property of text that sounds natural and grammatically 
  correct to a native speaker. Perplexity measures fluency, not correctness.
* **Chain Rule of Probability:** The decomposition of the probability of 
  a sequence into a product of conditional probabilities: 
  $P(w_1, \ldots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_1, \ldots, w_{i-1})$

### 5. Evaluation Strategy

* **Automatic Evaluation:** Using a mathematical metric to assess quality 
  without human involvement. Fast and reproducible but limited to surface form.
* **Human Evaluation:** Having human annotators assess quality directly. 
  Slow and expensive but the only reliable method for open-ended tasks.
* **Exact Match (EM):** A strict metric that awards a score only when the 
  model output exactly matches the reference. Used in question answering.
* **Hallucination:** Content generated by a model that is fluent and 
  confident but factually wrong. Not detected by BLEU, ROUGE, or Perplexity.

> **What's Next:** You now know how to measure whether an NLP system is 
> doing its job. The final step is building one. The next page shows how 
> to combine everything from this module — embeddings, similarity search, 
> and evaluation — into a working Question Answering bot that can read 
> the Holberton knowledge base and answer real student questions.
# NLP Projects

## Project: Semantic Job-CV Matcher

### Can a machine understand if you are the right person for the job?

Every time someone applies for a job online, an automated system reads their 
CV and decides — before any human does — whether it is worth passing on. 
These systems are called **Applicant Tracking Systems (ATS)**.

In this project, you will build one from scratch.

## The Mission

Build a system that takes a **CV or candidate profile** as input and returns 
a **ranked list of job postings** ordered by semantic relevance, from the 
best match to the worst. You will implement this twice: once using TF-IDF 
and once using word embeddings, then compare the results and explain the 
differences.

## Dataset

You have two options. Choose one as a group.

### Option A — Kaggle Dataset (Recommended for speed)

**Dataset:** [Job Recommendation Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

This dataset contains thousands of real job postings with titles, descriptions, 
required skills, and industries. It is clean, well-structured, and ready to use.

**How to use it:**

- Use the job descriptions as your **knowledge base** (the documents to search)
- Write or adapt a short **candidate profile** as your query (the CV)
- Your system ranks all job postings by relevance to the candidate profile

### Option B — Build Your Own (Recommended for depth)

Scrape or manually collect **50–100 real job postings** from LinkedIn, 
Indeed, or any job board in a domain you care about (tech, data science, 
design, etc.). Write **3–5 candidate profiles** of different skill levels 
and backgrounds.

**Why this is harder and better:** You will face real-world messiness — 
inconsistent formatting, missing fields, multilingual postings, and you 
will have to make design decisions that a pre-cleaned dataset doesn't force.

## Key Steps

### Step 1: Data Loading & Exploration
- Load the dataset and inspect its structure
- Understand the distribution of job categories, lengths, and vocabulary
- Identify which fields are most informative: title, description, required skills?
- Decide which fields to use and document your reasoning

### Step 2: Preprocessing Pipeline
- Lowercase, remove punctuation, tokenize
- Remove stop words — but think carefully: is `"not"` a stop word in a job 
  description? Is `"senior"`? Document every decision you make
- Analyze vocabulary size before and after preprocessing

### Step 3: TF-IDF Matching
- Build a TF-IDF matrix over all job postings
- Represent the candidate profile as a TF-IDF vector
- Compute cosine similarity between the profile and every job posting
- Return the top 10 matches and inspect them manually — do they make sense?

### Step 4: Embedding-Based Matching
- Represent every job posting as a document vector (average of word vectors)
- Use either a pre-trained model (GloVe or fastText via gensim) or train 
  Word2Vec on your job description corpus
- Represent the candidate profile the same way
- Compute cosine similarity and return the top 10 matches

### Step 5: Comparison & Analysis
- For the same candidate profile, compare the top 10 results from TF-IDF 
  vs embeddings side by side
- Find at least **2 cases where they agree** and **2 cases where they disagree**
- Explain *why* they disagree — what does each method see that the other misses?

### Step 6: Visualization
Your project must include at least **two visualizations**:

1. **TF-IDF keyword heatmap** — top TF-IDF keywords for the top 5 matched 
   job postings, shown as a heatmap. Do the keywords align with the candidate profile?
2. **Embedding space plot** — a t-SNE or PCA scatter plot of job postings 
   color-coded by category, with the candidate profile plotted as a distinct 
   marker. Does the candidate land in the right neighborhood?

### Step 7: Failure Analysis
This is the most important step. Find cases where your system fails:

- A job that should match but ranks low — why?
- A job that ranks high but is clearly wrong — why?
- What would you need to fix this? (Better preprocessing? A larger embedding 
  model? A different similarity metric?)

## Deliverables

| Deliverable | Description |
|---|---|
| **Notebook** | A single clean Colab notebook with markdown explanations between every code section |
| **Presentation** | 10-minute live demo + 5-minute explanation of design choices |
| **Reflection** | One section at the end of the notebook answering: what does your system get right, what does it get wrong, and what would the next version look like? |


## Evaluation Criteria

You are **not** evaluated on accuracy alone. A system that ranks poorly but 
whose authors understand exactly *why* and *what to fix* will score higher 
than a system that ranks well by accident.

| Criterion | Weight | What We Look For |
|---|---|---|
| **Pipeline quality** | 25% | Is the preprocessing thoughtful and well-documented? Are design decisions justified? |
| **Method comparison** | 25% | Is the TF-IDF vs embedding comparison honest and insightful? |
| **Visualizations** | 20% | Are the plots clean, readable, and informative? Do they support the narrative? |
| **Failure analysis** | 20% | Can the group identify and explain specific failure cases? |
| **Presentation** | 10% | Can every group member explain any part of the system? |


## Tips & Warnings

- **Do not shuffle your data** for any reason — there is no train/test split 
  here, but order matters for reproducibility
- **Document every decision** — why did you choose fastText over GloVe? 
  Why did you keep or remove certain stop words? These decisions are your 
  intellectual contribution
- **Start with TF-IDF** — get it working completely before touching embeddings. 
  A working TF-IDF system is worth more than a broken embedding system
- **The candidate profile is a design choice** — experiment with different 
  levels of detail. Does a one-sentence profile perform differently from a 
  full paragraph?
- **Beware of domain mismatch** — if you use a pre-trained GloVe model trained 
  on Wikipedia, it may not understand technical job-specific vocabulary. 
  fastText handles this better. Test both and compare.


## Glossary

* **ATS (Applicant Tracking System):** Automated software used by employers 
  to filter and rank job applications before human review.
* **Query:** The input to a search or retrieval system — in this case, the 
  candidate profile.
* **Knowledge Base:** The collection of documents being searched — in this 
  case, the job postings.
* **Cosine Similarity:** A measure of how similar two vectors are, regardless 
  of their magnitude. Scores range from 0 (completely different) to 1 (identical).
* **Domain Mismatch:** The drop in performance when an embedding model trained 
  on one type of text (e.g. Wikipedia) is applied to a different domain 
  (e.g. technical job descriptions).
* **Ranking:** Ordering a list of documents by their relevance score relative 
  to a query.



## Project: News Article Clustering & Topic Detection


### Can a machine organize the news — without being told what any article is about?

Every major news platform — Google News, Apple News, Reuters — groups articles 
by topic automatically. No human reads each article and assigns it a category. 
Instead, an algorithm finds structure in the text itself, grouping articles that 
talk about similar things into clusters — without ever being given a label.

This is **unsupervised learning applied to language**, and it is one of the 
most practically useful things you can build with the techniques from the last 
two notebooks.

In this project, you will build a system that takes a collection of raw news 
articles and automatically discovers the topics hiding inside them.


## The Mission

Build a pipeline that:

1. Ingests a collection of raw news articles
2. Represents each article as a vector using TF-IDF and then word embeddings
3. Groups them into clusters using KMeans
4. Labels and evaluates those clusters — do they make human sense?
5. Visualizes the topic landscape of the news in 2D

You will do this twice — once with TF-IDF and once with embeddings — and 
compare not just the clusters but what each method reveals about the data.

## Dataset

You have two options. Choose one as a group.

### Option A — Kaggle Dataset (Recommended for speed)

**Dataset:** [BBC News Classification](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification)

This dataset contains **2,225 BBC news articles** from 2004–2005, pre-labeled 
across 5 categories: business, entertainment, politics, sport, and technology. 

**Important:** The labels exist in the dataset but you must **not use them 
during clustering**. They are only used at the end to evaluate whether your 
clusters recovered the real categories.

**Alternative:** [All the News](https://www.kaggle.com/datasets/snapcrack/all-the-news) — 
a larger, more challenging dataset of 200,000+ articles from 15 publications. 
Use a random sample of 500–1000 articles if you choose this one.

### Option B — Build Your Own (Recommended for depth)

Collect **100–200 articles** from 4–6 topics of your choice using an RSS feed, 
a news API (NewsAPI has a free tier), or manual collection. Suggested topics 
that produce clean clusters: **AI & technology, climate change, sports, 
politics, health, entertainment**.

**Why this is harder and better:** You choose the topics, which means you 
have a hypothesis about what the clusters should look like — and you can 
directly test whether the algorithm agrees with you.

## Key Steps

### Step 1: Data Loading & Exploration
- Load the articles and inspect their structure
- What is the average article length? What is the vocabulary size?
- What are the most frequent words overall — before and after stop word removal?
- If using BBC: what is the distribution across the 5 categories? 
  Is it balanced?

### Step 2: Preprocessing Pipeline
- Lowercase, remove punctuation, tokenize, remove stop words
- Consider: should you remove numbers? What about named entities like 
  `"Boris Johnson"` or `"Premier League"`? Document your decision
- Compare vocabulary size before and after preprocessing
- Identify the top 20 TF-IDF keywords across the entire corpus — 
  do they hint at the topics hiding in the data?

### Step 3: TF-IDF Clustering
- Build a TF-IDF matrix over all articles
- Apply **KMeans clustering** — experiment with different values of K
- For each cluster, extract the **top 10 TF-IDF keywords** — these become 
  the cluster's "label"
- Manually inspect 3–5 articles from each cluster — do they belong together?
- If using BBC: compare your clusters to the ground truth categories using 
  the **Adjusted Rand Index (ARI)**

### Step 4: Embedding-Based Clustering
- Represent every article as a document vector (average word vectors)
- Use pre-trained GloVe or fastText — training Word2Vec on 2,000 articles 
  is feasible but pre-trained models will give richer representations
- Apply KMeans with the same K as Step 3
- Extract the most central words per cluster (closest to the cluster centroid)
- Compare to your TF-IDF clusters — which method produces more coherent groupings?

### Step 5: Finding the Right K
This is one of the most important and underappreciated steps in clustering.

- Plot the **Elbow Curve**: run KMeans for K = 2 to 15, record the 
  inertia (within-cluster sum of squares) for each K, and plot it
- Identify the "elbow" — the point where adding more clusters stops 
  improving the result significantly
- Does the optimal K match the number of real categories in your dataset?

### Step 6: Visualization
Your project must include at least **three visualizations**:

1. **Elbow Curve** — inertia vs K, clearly annotated with the chosen K
2. **t-SNE Cluster Plot (TF-IDF)** — all articles as dots in 2D, 
   color-coded by TF-IDF cluster. Annotate a handful of representative 
   article titles
3. **t-SNE Cluster Plot (Embeddings)** — same layout but color-coded 
   by embedding cluster. Place the two plots side by side for direct comparison

**Bonus visualization:** If using BBC, add a confusion matrix comparing 
your clusters to the ground truth categories — which topics did the 
algorithm confuse with each other, and why?

### Step 7: Failure Analysis
- Find an article that landed in the wrong cluster — why did the algorithm 
  get confused? What words pulled it in the wrong direction?
- Find two clusters that overlap significantly in the t-SNE plot — 
  what do they have in common?
- Which method — TF-IDF or embeddings — produced more meaningful clusters? 
  Support your answer with specific examples, not just scores.


## Deliverables

| Deliverable | Description |
|---|---|
| **Notebook** | A single clean Colab notebook with markdown explanations between every code section |
| **Presentation** | 10-minute live demo + 5-minute explanation of design choices |
| **Reflection** | One section at the end of the notebook answering: what topics did your system discover, what did it confuse, and what would the next version look like? |


## Evaluation Criteria

You are **not** evaluated on cluster purity alone. A group that produces 
impure clusters but can explain exactly which articles were misclassified 
and why will score higher than a group with perfect clusters and no 
understanding of how they got there.

| Criterion | Weight | What We Look For |
|---|---|---|
| **Pipeline quality** | 25% | Is preprocessing thoughtful? Are design decisions on stop words, vocabulary, and field selection justified? |
| **Cluster analysis** | 25% | Are the clusters inspected manually and labeled meaningfully? Is the elbow curve used to justify K? |
| **Visualizations** | 20% | Are the t-SNE plots readable and informative? Is the comparison between methods clear? |
| **Failure analysis** | 20% | Can the group identify specific misclassified articles and explain why the algorithm failed? |
| **Presentation** | 10% | Can every group member explain any part of the system? |


## Tips & Warnings

- **Do not peek at the labels** during development — it will bias every 
  decision you make. Treat them as a sealed envelope opened only during 
  evaluation
- **KMeans is sensitive to initialization** — always set `random_state=42` 
  and run with `n_init=10` so results are stable and reproducible
- **The elbow is often ambiguous** — if you can't find a clear elbow, 
  that is a valid finding. Report it honestly and justify your choice of K
- **TF-IDF often beats embeddings on short texts** — if your articles are 
  short (under 100 words), don't be surprised if TF-IDF clusters are cleaner. 
  Explain why
- **Named entities dominate clusters** — `"Manchester United"` will pull 
  all sports articles together even if the article is really about finance. 
  Consider whether to keep or remove named entities and test both
- **t-SNE is not deterministic** — run it with a fixed `random_state` and 
  never compare two t-SNE plots that used different seeds


## Glossary

* **Clustering:** Grouping data points into clusters based on similarity, 
  without using any labels. An unsupervised learning technique.
* **KMeans:** An algorithm that partitions data into K clusters by 
  minimizing the distance between each point and its cluster center (centroid).
* **Centroid:** The average point of all documents in a cluster. 
  The "center of gravity" of the cluster in vector space.
* **Inertia:** The sum of squared distances from each point to its 
  cluster centroid. Lower inertia means tighter clusters.
* **Elbow Curve:** A plot of inertia vs K used to find the optimal 
  number of clusters — the point where adding more clusters stops 
  helping significantly.
* **Adjusted Rand Index (ARI):** A metric that measures how well your 
  clusters match a ground truth labeling, adjusted for chance. 
  Scores range from -1 to 1; higher is better.
* **Ground Truth:** The real labels that exist in the dataset but are 
  withheld during clustering and used only for evaluation.
* **t-SNE:** A dimensionality reduction technique that projects 
  high-dimensional vectors into 2D while preserving local cluster structure. 
  Used for visualization only — distances between clusters are not meaningful.

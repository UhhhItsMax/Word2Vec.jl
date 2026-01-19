```@meta
CurrentModule = Word2Vec
```

# Word2Vec.jl

Lightweight Julia implementation of **Word2Vec CBOW** with **ConEc embeddings** (global+local context) and **t-SNE visualization**.

# Getting Started

## 1. Installation

```julia
using Pkg
Pkg.add(url="https://github.com/UhhhItsMax/Word2Vec.jl")
```

## 2. Train CBOW Model

**Input**: Text file (`corpus.txt`), one sentence per line.

```julia
using Word2Vec

# Load as sentences
sentences = read_corpus_sentences("corpus.txt")

model = train_cbow(sentences; 
    dim=100,     # embedding size
    window=5,    # context window
    epochs=5,    # training passes
    lr=0.05,     # learning rate
    min_count=2, # ignore rare words
    seed=42,     # reproducible
    verbose=true # show loss
)
```


Example Corpus Text File:
```
The quick brown fox jumps over the lazy dog.
Julia is a high-performance language for technical computing.
Word embeddings capture semantic relationships between words.
CBOW predicts target words from context windows.
Machine learning models require clean tokenized input data.
Neural networks learn hierarchical feature representations.
t-SNE visualization reveals clusters in high-dimensional embeddings.
Context matrices store word co-occurrence statistics.
Global context provides broad distributional semantics.
Local context adapts embeddings to specific documents.
Analogy tasks test vector arithmetic like king - man + woman ≈ queen.
Cosine similarity measures embedding angular distance.
Vocabulary filtering removes rare words below min_count.
Learning rate controls gradient descent step size.
Epochs iterate multiple passes over training data.

the quick brown fox jumps over lazy dog.
word embeddings capture semantic relationships between words.
cbow predicts target words from context windows.
machine learning models require clean tokenized input data.
neural networks learn hierarchical feature representations.
t-sne visualization reveals clusters high dimensional embeddings.
context matrices store word co occurrence statistics.
global context provides broad distributional semantics.
local context adapts embeddings specific documents.
analogy tasks test vector arithmetic like king man woman queen.
cosine similarity measures embedding angular distance.
vocabulary filtering removes rare words below min_count.
learning rate controls gradient descent step size.
epochs iterate multiple passes training data.
```

## 3. Query Embeddings

```julia
# Single word vector
v = get_embedding(model, "embedding")

# Similarity
similarity(model, "vocabulary", "words")

# Classic analogies
 analogy(model, "machine", "learning", "neural")
```

## 4. Visualize (t-SNE)

```julia
words = read_wordlist("words.txt")  # one word per line
p = plot_tsne(model; words=words, normalize=true, annotate=true)
```


Example Words Text File:
```
neural
network
embedding
vector
context
window
cbow
t-sne
cluster
```
## 5. ConEc Embeddings

```julia
# Build global context (one-time)
cm = ConEcModel(model, "global_corpus.txt"; a=0.6)

# Embed any local document
local_embs = conec_embeddings_for_file(cm, "local_corpus.txt")
local_embs["embedding"]
```
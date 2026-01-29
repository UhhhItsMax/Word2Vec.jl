```@meta
CurrentModule = Word2Vec
```

# Word2Vec.jl

Lightweight Julia implementation of **Word2Vec CBOW** with **ConEc embeddings** (global+local context) and **t-SNE visualization**.

---

## Overview

#### Word2Vec
Word2Vec is a family of neural embedding models that map words to vectors text corpora. Words that appear in similar contexts obtain similar vector representations. These embeddings capture semantic and syntactic regularities and enable linear operations such as similarity search and analogical reasoning.

#### CBOW
CBOW (Continuous Bag-of-Words) is a Word2Vec training architecture. It learns word embeddings by predicting a target word from the average of its surrounding context words. The model ignores word order within the context window but is computationally efficient and performs well for frequent words. In CBOW, the input embedding matrix serves as the learned word representation space.

#### ConEc
ConEc (Contextualized Embeddings with Global Context) extends Word2Vec by incorporating global corpus-level information in addition to the local context window. For each word, a global context vector is derived from its co-occurrence statistics across the entire corpus and combined with the local CBOW context. A mixing parameter controls the contribution of global versus local information, improving robustness and representation quality, especially for rare or ambiguous words.

Together, CBOW and ConEc provide a simple but expressive framework for learning word embeddings that capture both local syntactic structure and global semantic regularities.

## Getting Started

### 1. Installation

```julia
using Pkg
Pkg.add(url="https://github.com/UhhhItsMax/Word2Vec.jl")
using Word2Vec
```

### 2. Tests

```julia
Pkg.test("Word2Vec")
```

## What’s inside

- Train Word2Vec CBOW models
- Load and save Word2Vec-compatible embeddings
- Compute ConEc embeddings for new documents
- Visualize embeddings with t-SNE
- Evaluation and Benchmarking of Word2Vec and ConEc models

See the sidebar for detailed guides.

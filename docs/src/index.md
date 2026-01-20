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
using Word2Vec
```

## 2. Tests

```julia
Pkg.test("Word2Vec")
```

# What’s inside

- Train Word2Vec CBOW models
- Load and save Word2Vec-compatible embeddings
- Compute ConEc embeddings for new documents
- Visualize embeddings with t-SNE
- Evaluation and Benchmarking of Word2Vec and ConEc models

See the sidebar for detailed guides.

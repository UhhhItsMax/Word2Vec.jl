## Getting started

This package provides tools to learn and use word embeddings with Word2Vec and its Context Encoder (ConEc) extension. It focuses on training CBOW‑style Word2Vec models, constructing sparse context representations, and applying them to small text datasets for evaluation and visualization.

### General idea

The core idea is to represent each word by a real‑valued vector (an embedding) such that words appearing in similar contexts have similar vectors.
Word2Vec learns these embeddings from text using neural models like Continuous Bag of Words (CBOW), which predicts a target word from its surrounding context, while ConEc extends this by encoding richer, sparse context information to refine the learned representations.

### Installation and loading

First, activate your project and add the package.

```julia
using Pkg

Pkg.activate("Word2Vec")

include("src/Word2Vec.jl")
using .Word2Vec
```

### Loading pre‑trained embeddings

You can load a standard Word2Vec file in either text or Gensim‑style binary format. The format is detected automatically.

```julia
emb_map = load_word2vec("test/data/word2vec.bin")    # or "word2vec.txt"
```

`emb_map` is a `Dict{String, Vector{Float64}}` mapping each word to its embedding vector.

### Building a Word2Vec model

To work with embeddings efficiently, convert the dictionary into a `Word2VecModel`.

```julia
model = load_pretrained_model("test/data/word2vec.txt")
```

`Word2VecModel` stores all embeddings in a dense matrix, together with the vocabulary, a word‑to‑index map, and precomputed vector norms. This is useful for later similarity or nearest‑neighbor queries.

### Getting embeddings for words

Once a model is loaded, you can access the embedding of a word as a view into the underlying matrix.

```julia
v_human = get_embedding(model, "human")

println(size(v_human))  # (embedding_dim,)
```

The returned object behaves like a dense vector and can be used with standard Julia linear‑algebra operations, such as dot products or norms.
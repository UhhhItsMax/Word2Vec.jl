# Word2Vec.jl

A lightweight Julia implementation of **Word2Vec (CBOW)** with utilities for evaluation, **ConEc** embeddings (global+local context), and quick **t-SNE visualization**.

**Creator:** Maximilian Hans ([@UhhhItsMax](https://github.com/UhhhItsMax)) — hans.maximilian@icloud.com

**Contributors:**
- Paul Mathias Nelde ([@designationna](https://github.com/designationna)) — paul.nelde@fu-berlin.de
- Mika Paul Merten ([@42Strike](https://github.com/42Strike)) - merten@campus.tu-berlin.de

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://UhhhItsMax.github.io/Word2Vec.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://UhhhItsMax.github.io/Word2Vec.jl/dev/)
[![Build Status](https://github.com/UhhhItsMax/Word2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UhhhItsMax/Word2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UhhhItsMax/Word2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/UhhhItsMax/Word2Vec.jl)

---

## Installation

This package is currently installed directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/UhhhItsMax/Word2Vec.jl")
using Word2Vec
```

For development:

```julia
using Pkg
Pkg.develop(url="https://github.com/UhhhItsMax/Word2Vec.jl")
using Word2Vec
```

---

## Getting Started

### 1) Loading and saving Word2Vec models

Word2Vec.jl supports loading and saving Word2Vec-compatible text and binary models.

#### Loading a model

```julia
model = load_word2vec("vectors.txt")
model = load_word2vec("vectors.bin")
```
- Text format: word val1 val2 ...
- Binary format: Gensim-style hybrid format (Float32 vectors)
- Binary vectors are automatically converted to Float64

#### Saving a model

```julia
save_word2vec(model, "model.txt"; format = :text)
save_word2vec(model, "model.bin"; format = :binary)
```

- Supported Formats:
    - :text — human-readable, compatible with Gensim
    - :binary — compact binary Word2Vec format

### 2) Train a CBOW model on a text corpus

Assumption: your corpus is a `.txt` where **each line is treated as a sentence**.

```julia
sentences = read_corpus_sentences("corpus.txt")

model = train_cbow(sentences)
```

> If you want to ignore sentence boundaries (flat token stream), use:
> `tokens = read_corpus_tokens("corpus.txt")`

### 3) Query embeddings and model evaluation

```julia
# single word embedding vector
v = get_embedding(model, "virtue")

# similarity
similarity(model, "virtue", "reason")

# analogies (classic word2vec demo)
analogy(model, "king", "man", "woman")
```

---

### 4) ConEc embeddings (global + local context)

ConEc creates embeddings from a mixture of a **global context matrix** and a **local document context**.

```julia
# Assume you already have a trained Word2Vec model `model`
# Build ConEc using a global corpus
cm = ConEcModel(model, "corpus.txt"; window_size=5, min_count=2, a=0.6)

# Compute ConEc embeddings for a local document
embs = conec_embeddings_for_file(cm, "local_doc.txt"; window_size=5, min_count=1)

# embs is Dict{String,Vector{Float64}}
embs["virtue"]
```

---

### 5) Tests

Run tests (with coverage):

```julia
Pkg.test("Word2Vec")
```

---

### 6) More Information

For more information, visit our [Documentation](https://uhhhitsmax.github.io/Word2Vec.jl/stable/)


---

## License

MIT (see [`LICENSE`](./LICENSE)).

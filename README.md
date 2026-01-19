# Word2Vec.jl

A lightweight Julia implementation of **Word2Vec (CBOW)** with utilities for evaluation, **ConEc** embeddings (global+local context), and quick **t-SNE visualization**.

**Creator:** Maximilian Hans ([@UhhhItsMax](https://github.com/UhhhItsMax)) — hans.maximilian@icloud.com

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
```

For development:

```julia
using Pkg
Pkg.develop(url="https://github.com/UhhhItsMax/Word2Vec.jl")
```

---

## Getting Started

### 1) Train a CBOW model on a text corpus

Assumption: your corpus is a `.txt` where **each line is treated as a sentence**.
We provide helpers that clean lines (lowercasing, punctuation stripping, ignoring dashed separators).

```julia
using Word2Vec

# Read as sentences (Vector{Vector{String}}), preserving sentence boundaries
sentences = read_corpus_sentences("corpus.txt")

# Train CBOW (example hyperparameters)
model = train_cbow(
    sentences;
    dim=100,
    window=5,
    epochs=5,
    min_count=2,
)
```

> If you want to ignore sentence boundaries (flat token stream), use:
> `tokens = read_corpus_tokens("corpus.txt")`

### 2) Query embeddings

```julia
using Word2Vec

# single word embedding vector
v = get_embedding(model, "virtue")

# similarity
similarity(model, "virtue", "reason")

# analogies (classic word2vec demo)
analogy(model, "king", "man", "woman"; topk=10)
```

---

## Visualize embeddings with t-SNE

### Plot a subset of words from a file

If you have a word list (one word per line, `#` comments allowed):

```julia
using Word2Vec

words = read_wordlist("words_big.txt")

p = plot_tsne(
    model;
    words=words,
    normalize=true,
    perplexity=30,
    max_iter=1000,
    annotate=false,
)

# Save (Plots.jl API)
using Plots
savefig(p, "tsne.png")
```

Notes:
- `plot_tsne` currently supports `dims=2` (scatter plot).
- If you set `annotate=true`, labels are drawn next to points (best for small lists).

---

## ConEc embeddings (global + local context)

ConEc creates embeddings from a mixture of a **global context matrix** and a **local document context**.

```julia
using Word2Vec

# Assume you already have a trained Word2Vec model `model`
# Build ConEc using a global corpus
cm = ConEcModel(model, "corpus.txt"; window_size=5, min_count=2, a=0.6)

# Compute ConEc embeddings for a local document
embs = conec_embeddings_for_file(cm, "local_doc.txt"; window_size=5, min_count=1)

# embs is Dict{String,Vector{Float64}}
embs["virtue"]
```

---

## REPL / Terminal one-liners

### Train + t-SNE plot + save image (from terminal)
corpus.txt and words_big.txt aren't provided here, general structure is:
for corpus.txt one sentence per line
for words_big.txt one word per line

```bash
julia --project -e '
using Word2Vec, Plots

sentences = read_corpus_sentences("corpus.txt")
model = train_cbow(sentences; dim=100, window=5, epochs=5, min_count=2)

words = read_wordlist("words_big.txt")
p = plot_tsne(model; words=words, normalize=true, perplexity=30, max_iter=1000)

savefig(p, "tsne.png")
println("Saved: tsne.png")
'
```

---

## Data formats

### `corpus.txt`
- one sentence per line
- dashed separators like `---` / `------` are ignored by default
- punctuation is stripped by default (non-alphanumeric → spaces)

### `words_big.txt`
- one word per line
- empty lines are ignored
- lines starting with `#` are ignored (comments)

---

## Development

Run tests (with coverage):

```bash
julia --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.test(coverage=true)'
```

---

## License

MIT (see `LICENSE`).

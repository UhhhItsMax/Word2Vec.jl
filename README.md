# Word2Vec.jl

**Creator:** Maximilian Hans ([@UhhhItsMax](https://github.com/UhhhItsMax)) — hans.maximilian@icloud.com

**Contributors:**
- Paul Mathias Nelde ([@designationna](https://github.com/designationna)) — paul.nelde@fu-berlin.de
- Mika Paul Merten ([@42Strike](https://github.com/42Strike)) - merten@campus.tu-berlin.de

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://UhhhItsMax.github.io/Word2Vec.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://UhhhItsMax.github.io/Word2Vec.jl/dev/)
[![Build Status](https://github.com/UhhhItsMax/Word2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UhhhItsMax/Word2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UhhhItsMax/Word2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/UhhhItsMax/Word2Vec.jl)

---
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

## Installation

This package can be installed directly from GitHub:

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

This guide gives a minimal end-to-end overview of how to obtain a Word2Vec model, train one yourself, and use it for querying, evaluation, and ConEc embeddings.

### How to get a Word2Vec model

There are two common ways to obtain a model:

1. Load a pre-trained Word2Vec model
2. Manually set embeddings

---

#### 1. Load a pre-trained Word2Vec model

Word2Vec.jl supports Word2Vec-compatible **text** and **binary** formats.


```julia
model = load_word2vec("vectors.txt")
model = load_word2vec("vectors.bin")
```

* Text format: `word val1 val2 ...`
* Binary format: Gensim-style Word2Vec binary (Float32 vectors)
* Binary vectors are automatically converted to `Float64`

#### 2. Manually set embeddings

```julia
vocab = ["king", "queen", "man", "woman"]
emb = randn(5, 4)
model = Word2VecModel(vocab, emb)
```

To learn more, please refer to our guides on [Word2VecModel](./docs/src/model.md) as well as [Model I/O](./docs/src/io.md).

---

### Training a CBOW model from a corpus

You can train a CBOW model directly from text. (e.g. the [Project Gutenberg](https://www.gutenberg.org/cache/epub/1661/pg1661.txt))

Assumption: the corpus is a `.txt` file where **each line is treated as a sentence**.

```julia
sentences = read_corpus_sentences("corpus.txt")

# or equivalently
sentences = [["the", "quick", "brown", "fox"],
             ["fox", "jumps", "over", "the", "lazy", "dog"]]

model = train_cbow(sentences)
```

If sentence boundaries are not relevant and you want a flat token stream:

```julia
tokens = read_corpus_tokens("corpus.txt")
```

preprocessing expectations:

* One sentence per line
* Tokens separated by whitespace
* Lowercasing and cleanup of special characters performed beforehand

More in-depth information can be found on our guide to [CBOW Training](./docs/src/cbow.md).

---

### Querying embeddings and basic evaluation

Once a model is available, embeddings can be queried and evaluated directly.

```julia
# retrieve a single word vector
v = get_embedding(model, "virtue")

# cosine similarity
similarity(model, "virtue", "reason")

# analogy queries
analogy(model, "king", "man", "woman")
```

These operations rely on cosine similarity in the embedding space and use precomputed vector norms for efficiency.

For more about benchmarking strategies, please see [Word2Vec Model Evaluation & Benchmarking](./docs/src/w2v_bench_ev.md) as well as [ConEc Embedding Benchmarking](./docs/src/conec_bench_ev.md).

---

### ConEc embeddings (global + local context)

ConEc combines **global corpus-level co-occurrence information** with **local document context**.

First, build a ConEc model from an existing Word2Vec model and a global corpus:

```julia
cm = ConEcModel(
    model,
    "corpus.txt";
    window_size = 5,
    min_count = 2,
    a = 0.6,
)
```

Then compute contextualized embeddings for a local document:

```julia
embs = conec_embeddings_for_file(
    cm,
    "local_doc.txt";
    window_size = 5,
    min_count = 1,
)
```

The result is a `Dict{String, Vector{Float64}}` mapping words to ConEc embeddings specific to the document.

Please refer to [ConEc Embeddings](./docs/src/conec.md) for more information.

---

### Visualization

Embeddings can be projected to two dimensions for inspection using t-SNE:

```julia
Word2Vec.plot_tsne(model; words = ["virtue", "reason", "justice"])
```

This is intended for qualitative analysis and debugging, not evaluation.

Details for plotting can be found at our [Visualization](./docs/src/visualization.md) page.

---

## More Information

For more information, visit our [Documentation](https://uhhhitsmax.github.io/Word2Vec.jl/stable/)


---

## License

MIT (see [`LICENSE`](./LICENSE)).

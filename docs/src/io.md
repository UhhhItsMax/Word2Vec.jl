```@meta
CurrentModule = Word2Vec
```

# Model I/O

This section documents loading and saving Word2Vec-compatible embedding models.

Word2Vec.jl supports both text and binary formats and automatically
detects the correct format when loading.

## Loading embeddings

Use `load_word2vec(path::String)` to load embeddings from a text or binary Word2Vec file.

### Supported formats

We currently support text and binary formating for loading models. The formats are ditected automatically.

#### Text format

- one word per line:
word val1 val2 val3 ...
- Optional numeric header lines (e.g. vocab_size dim) are ignored.
- Commonly used by tools such as Gensim.

#### Binary format

- ASCII header: vocab_size dim
    - ASCII header: vocab_size dim
    - ASCII words
    - Binary Float32 vectors
- Automatically converted to Float64 internally.

### Examples

```julia
model = load_word2vec(“word2vec.txt”) # text format
model = load_word2vec(“word2vec.bin) # binary format
```

## Saving embeddings

Use `save_word2vec(model::Word2VecModel, path::AbstractString; format=:text|:binary)` to save embeddings to a file in **text** or **binary** format.

### Supported formats

We currently support text and binary formating for saving models.

- :text (default)
    - Human-readable
    - Easy to inspect and debug
    - Larger file size
- :binary
    - Compact and fast to load
    - Compatible with the original Word2Vec C implementation
    - Uses Float32 storage on disk

### Examples

```julia
save_word2vec(model, “model.txt”; format = :text)
save_word2vec(model, “model.bin”; format = :binary)
```

## Consistency

Saving and reloading a model preserves the vocabulary order and embedding values (up to floating-point precision):

```julia
using Test
save_word2vec(model, “tmp.bin”; format = :binary)
model2 = load_word2vec(“tmp.bin”)

@test model2.vocab == model.vocab          # vocab order is identical
@test model2.embeddings ≈ model.embeddings # element-wise approximate equality
```

## Notes and Limitations

- Large pretrained models may require substantial RAM.
- Binary models are always converted to Float64 internally for consistency.
- Invalid formats passed to save_word2vec throw an ArgumentError.


```@meta
CurrentModule = Word2Vec
```

# Word2Vec Model

Word2VecModel is the in-memory representation of a trained Word2Vec embedding. It stores the vocabulary, embeddings, precomputed vector norms, and a mapping from words to column indices.

```@docs
Word2VecModel
```

## Construction

You can construct a Word2VecModel from:

### Separate vocab and embedding matrix:

```@example
using Word2Vec # hide
vocab = ["king", "queen", "man", "woman"]
emb = randn(5, 4)
model = Word2VecModel(vocab, emb)
```

### From a dictionary of word => vector mappings:

```@example
using Word2Vec # hide
emb_map = Dict(
"king" => [0.1, 0.2, 0.3, 0.4, 0.5],
"queen" => [0.2, 0.1, 0.4, 0.3, 0.0]
)
model = from_dict_data(emb_map)
```

### From loading a model

See [Model I/O](./io.md)

## Accessing embeddings

### Embedding vector

```@example
using Word2Vec # hide
vocab = ["king", "queen", "man", "woman"] # hide
emb = randn(5, 4) # hide
model = Word2VecModel(vocab, emb) # hide
vec = get_embedding(model, "king")
```

### Norm of embedding vector

```@example
using Word2Vec # hide
vocab = ["king", "queen", "man", "woman"] # hide
emb = randn(5, 4) # hide
model = Word2VecModel(vocab, emb) # hide
norm = get_embedding_norm(model, "queen")
```

## Notes

- Norms are precomputed to speed up cosine similarity and analogy computations.
- Construction will throw an error if any embedding vector has zero norm or if the number of columns in embeddings does not match the vocabulary size.
- All embedding vectors are stored as Float64 internally, even if provided as Float32.

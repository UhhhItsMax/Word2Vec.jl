# CBOW Training

The core of Word2Vec.jl is the **Continuous Bag-of-Words (CBOW)** algorithm, which trains a `Word2VecModel` by predicting target words from averaged context vectors. This serves as the foundation for querying embeddings and extensions like ConEc.

CBOW uses full softmax with cross-entropy loss, optimized for small corpora. Training produces input embeddings (`W_in`) suitable for downstream tasks.

## Quick Training Example

```julia
using Word2Vec, Random

# Example corpus (sentences or flat tokens)
sentences = [["the", "quick", "brown", "fox"], ["fox", "jumps", "over", "the", "lazy", "dog"]]

# Train CBOW model
model = train_cbow(sentences; dim=50, window=5, epochs=10, lr=0.05, seed=42, verbose=true)
```

## Key Concepts

- **Context Window**: Symmetric around target (e.g., `window=2` uses up to 4 neighbors).
- **Vocabulary Filtering**: `min_count` discards rare words.
- **Embeddings**: Final `model.embeddings` are input matrix `W_in` (dim × |vocab|).
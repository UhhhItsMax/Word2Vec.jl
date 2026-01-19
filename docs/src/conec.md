# ConEc Embeddings

**ConEc** extends Word2Vec CBOW models by combining **global context** (precomputed co-occurrences from a large corpus) with **local document context** to produce richer embeddings. Requires a trained `Word2VecModel` from `train_cbow`.

The method is inspired by [this paper](https://arxiv.org/abs/1706.02496): for each word $w$, compute context vector $c_w = \alpha \cdot c_w^{\text{global}} + (1-\alpha) \cdot c_w^{\text{local}}$, then $y_w = W_0 \cdot c_w$ where $W_0$ is the CBOW embedding matrix.

## Quick Example

```julia
using Word2Vec

# Step 1: Train CBOW base model (see CBOW Training)
model = train_cbow(sentences; dim=100)

# Step 2: Build ConEc with global corpus
cm = ConEcModel(model, "global_corpus.txt"; a=0.6)

# Step 3: Embed local document
embs = conec_embeddings_for_file(cm, "local_corpus.txt")
embs["embedding"]  # Vector{Float64} ConEc embedding
```

## Workflow Details

1. **Global Setup**: `ConEcModel` precomputes `SparseContextMatrix` from large corpus once.
2. **Local Computation**: For each new doc, build local `SparseContextMatrix` on-the-fly.
3. **Blending**: $\alpha=0.6$ defaults to 60% global + 40% local; OOV (out of vocabulary) uses local fallback.

## Relation to CBOW

ConEc *requires* a CBOW-trained `Word2VecModel`—global context aligns to its vocabulary/indices. No independent training needed.
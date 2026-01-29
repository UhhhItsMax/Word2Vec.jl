# ConEc Embeddings

**ConEc** extends Word2Vec CBOW embeddings by combining **global context** (from a large corpus) with **local document context** to produce richer, context-aware embeddings.  

The idea is: for each word \(w\),

c_w = α * c_w^global + (1 - α) * c_w^local  

y_w = W0 * c_w  

where W0 is the CBOW input embedding matrix and α ∈ [0,1] balances global vs local context.

Reference: [Co-occurrence Enhanced Word Embeddings (ConEc)](https://arxiv.org/abs/1706.02496)

---

## ConEcModel

```@docs
ConEcModel
```

---

## Example

```julia
# Step 1: Train a CBOW base model
model = train_cbow(sentences; dim=100)

# Step 2: Build ConEc model with global corpus
cm = ConEcModel(model, "global_corpus.txt"; a=0.6)

# Step 3: Compute ConEc embeddings for a local document
embs = conec_embeddings_for_file(cm, "local_corpus.txt"; window_size=5)

# Access embedding for a word
vec = embs["word1"]  # Vector{Float64} corresponding to "word1"
```

## Notes

- α controls blending: default 0.6 means 60% global + 40% local.
- Out-of-vocabulary words in global corpus fallback to local context only.
- If no tokens survive min_count in the local corpus, an empty dictionary is returned.
- Each returned vector has the same dimensionality as the original Word2Vec embeddings.
- Use ConEc embeddings for downstream tasks like similarity queries, analogies, or document-level representation.

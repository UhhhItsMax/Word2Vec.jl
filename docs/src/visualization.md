# Visualization

Visualize high-dimensional Word2Vec embeddings using **t-SNE** dimensionality reduction. `plot_tsne` produces interactive 2D scatter plots via [Plots.jl](https://docs.juliaplots.org/latest/), with optional word labels.

Supports subsetting to specific words (e.g., from `words_big.txt`) for clearer plots on large vocabs.

## Quick Example

```julia
# After training CBOW model
words = read_wordlist("words_big.txt")  # one word per line
p = plot_tsne(model; words=words, normalize=true, annotate=true)
```

## Parameters Explained

| Parameter | Purpose | Recommended |
|-----------|---------|-------------|
| `perplexity=30` | t-SNE balance (local/global structure) | 5-50 |
| `max_iter=1000` | Convergence iterations | 500-5000 |
| `normalize=true` | L2-normalize embeddings first | Improves separation |
| `annotate=true` | Show word labels | Small word lists only |
| `reduce_dims=50` | PCA pre-reduction (if `dim > 100`) | Speeds up large models |
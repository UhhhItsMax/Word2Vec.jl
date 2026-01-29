# Visualization

Visualize high-dimensional Word2Vec embeddings in 2D using **t-SNE** with [Plots.jl](https://docs.juliaplots.org/latest/).  
`plot_tsne` supports optional word subsetting for large vocabularies and annotation of labels.

```@docs
Word2Vec.plot_tsne
```

## Example

```julia
using Plots
words = read_wordlist("words_big.txt")

p = plot_tsne(
    model;
    words=words,
    normalize=true,
    perplexity=30,
    max_iter=1000,
    annotate=false,
)
# display plot
display(p)

# Save plot
savefig(p, "tsne_plot.png")
```

## Notes

- dims=2 is currently required (scatter plot in 2D only).
- normalize=true L2-normalizes embeddings, improving separation in the plot.
- annotate=true shows word labels; recommended for small word subsets.
- reduce_dims performs PCA pre-reduction if embedding dimension is high (speeds up t-SNE).
- perplexity controls the balance between local vs global structure (typical range: 5–50).
- max_iter sets the number of t-SNE iterations (default 1000; increase for larger corpora).
- markersize and other keyword arguments are forwarded to Plots.scatter for customization.
- Works best with a limited number of words for readability; for large vocabularies, use a filtered subset.


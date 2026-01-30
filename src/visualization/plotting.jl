"""
    plot_tsne(model::Word2VecModel; dims=2, words=nothing, normalize=false, seed=42,
              reduce_dims=50, max_iter=1000, perplexity=30,
              annotate=false, markersize=4, kwargs...)

Compute a t-SNE projection of word embeddings from `model` and return a 2D scatter plot.

# Arguments
- `model::Word2VecModel`: The Word2Vec model whose embeddings will be projected.

# Keyword arguments
- `dims::Int=2`: Target number of dimensions (currently must be 2).
- `words`: Optional subset of words to plot (`nothing` = all words).
- `normalize::Bool=false`: If true, normalize embeddings before projection.
- `seed::Int=42`: Random seed for reproducibility.
- `reduce_dims::Int=50`: Dimensionality reduction prior to t-SNE (e.g., via PCA).
- `max_iter::Int=1000`: Maximum number of t-SNE iterations.
- `perplexity::Int=30`: t-SNE perplexity parameter.
- `annotate::Bool=false`: If true, add word labels next to points.
- `markersize::Real=4`: Size of scatter plot markers.
- `kwargs...`: Additional keyword arguments passed to `Plots.scatter`.

# Returns
- `Plots.Plot`: Scatter plot of t-SNE projected word embeddings.

# Notes
- Uses [`tsne_embeddings`](@ref) internally.
- Only 2D projections are supported for plotting.
- Labels are only drawn if `annotate=true`.
"""
function plot_tsne(
        model::Word2VecModel;
        dims::Int = 2,
        words = nothing,
        normalize::Bool = false,
        seed::Int = 42,
        reduce_dims::Int = 50,
        max_iter::Int = 1000,
        perplexity::Int = 30,
        annotate::Bool = false,
        markersize::Real = 4,
        kwargs...
    )

    if dims != 2
        throw(ArgumentError("plot_tsne currently supports dims=2 only (got dims=$dims)."))
    end

    Y, labels = tsne_embeddings(
        model;
        dims = dims,
        words = words,
        normalize = normalize,
        seed = seed,
        reduce_dims = reduce_dims,
        max_iter = max_iter,
        perplexity = perplexity,
    )

    p = scatter(Y[:, 1], Y[:, 2]; legend = false, markersize = markersize, kwargs...)

    if annotate
        for i in eachindex(labels)
            annotate!(p, Y[i, 1], Y[i, 2], labels[i])
        end
    end

    return p
end


"""
    plot_tsne(model::Word2VecModel; dims=2, words=nothing, normalize=false, seed=42,
              reduce_dims=50, max_iter=1000, perplexity=30,
              annotate=false, markersize=4, kwargs...) -> plot

Compute a t-SNE projection (via [`tsne_embeddings`](@ref)) and return a scatter plot.

# Keyword arguments
- `annotate=false`: If `true`, add word labels next to each point.
- `markersize=4`: Marker size for scatter points.
- `kwargs...`: forwarded to `Plots.scatter` / `Plots.scatter!`.

# Returns
- A `Plots.Plot` object.
"""
function plot_tsne(
    model::Word2VecModel;
    dims::Int=2,
    words=nothing,
    normalize::Bool=false,
    seed::Int=42,
    reduce_dims::Int=50,
    max_iter::Int=1000,
    perplexity::Int=30,
    annotate::Bool=false,
    markersize::Real=4,
    kwargs...
)

    if dims != 2
        throw(ArgumentError("plot_tsne currently supports dims=2 only (got dims=$dims)."))
    end

    Y, labels = tsne_embeddings(
        model;
        dims=dims,
        words=words,
        normalize=normalize,
        seed=seed,
        reduce_dims=reduce_dims,
        max_iter=max_iter,
        perplexity=perplexity,
    )

    p = scatter(Y[:, 1], Y[:, 2]; legend=false, markersize=markersize, kwargs...)

    if annotate
        for i in eachindex(labels)
            annotate!(p, Y[i, 1], Y[i, 2], labels[i])
        end
    end

    return p
end
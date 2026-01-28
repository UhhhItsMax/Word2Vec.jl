
# i get warning, so i wrap the func here 
const _TSNE_METRIC = SqEuclidean()
_tsne_dist(a, b) = evaluate(_TSNE_METRIC, a, b)

"""
    embedding_points(model::Word2VecModel; words=nothing, normalize=false) -> (X, labels)

Extract embedding vectors from `model` in a format suitable for dimensionality reduction.

THe model stores embeddings as `(dim, |vocab|)`. This function returns points as rows:
`X` has shape `(n_points, dim)` and `labels[i]` corresponds to row `i` in `X`.

# Keyword arguments
- `words=nothing`: If `nothing`, use the full vocabulary. If a vector of strings, only keep words
  that exist in the model (`unknown` words are skipped).
- `normalize=false`: If `true`, L2-normalize each embedding vector before returning.

# Returns
- `X::Matrix{Float64}`: points-as-rows matrix, size `(n_points, dim)`
- `labels::Vector{String}`: labels for each row

# Notes
- If `words` is provided and none are in the vocab, `X` will have 0 rows and `labels` is empty.
"""
function embedding_points(model::Word2VecModel; words=nothing, normalize::Bool=false)
    if words === nothing
        labels = collect(model.vocab)
        X = permutedims(model.embeddings) # (V, dim)
    else
        idxs = Int[]
        labels = String[]
        for w in words
            if haskey(model.word_to_index, w)
                push!(idxs, model.word_to_index[w])
                push!(labels, String(w))
            end
        end
        X = permutedims(model.embeddings[:, idxs]) # (n, dim)
    end

    X = Matrix{Float64}(X)
    if normalize
        l2normalize_rows!(X)
    end
    return X, labels
end


"""
    tsne_embeddings(model::Word2VecModel;
                    dims=2, words=nothing, normalize=false,
                    seed=42, reduce_dims=50, max_iter=1000, perplexity=30, kwargs...) -> (Y, labels)

Compute a t-SNE embedding of the model's word vectors.

`TSne.tsne` expects points as rows and uses the API:
`tsne(X, ndims, reduce_dims, max_iter, perplexity; kwargs...)`.

# Keyword arguments
- `dims::Int=2`: Output dimensionality.
- `words=nothing`: Optional subset of words to embed.
- `normalize=false`: If `true`, L2-normalize word vectors before running t-SNE.
- `seed::Int=42`: RNG seed for reproducibility.
- `reduce_dims::Int=50`: Initial reduction dimension used inside TSne.jl (clamped to `size(X,2)`).
- `max_iter::Int=1000`: Number of optimization iterations.
- `perplexity::Int=30`: t-SNE perplexity hyperparameter.
- `kwargs...`: forwarded to `TSne.tsne`.

# Returns
- `Y::Matrix{Float64}`: size `(n_points, dims)`
- `labels::Vector{String}`: corresponding word labels
"""
function tsne_embeddings(
    model::Word2VecModel;
    dims::Int=2,
    words=nothing,
    normalize::Bool=false,
    seed::Int=42,
    reduce_dims::Int=50,
    max_iter::Int=1000,
    perplexity::Int=30,
    kwargs...
)
    X, labels = embedding_points(model; words=words, normalize=normalize)

    if size(X, 1) == 0
        return Matrix{Float64}(undef, 0, dims), labels
    end


    seed!(seed)
    rd = min(reduce_dims, size(X, 2))
    Y = tsne(
        X, dims, rd, max_iter, perplexity; 
        distance=_tsne_dist,
        progress=false, 
        kwargs...
    )
    return Matrix{Float64}(Y), labels
end
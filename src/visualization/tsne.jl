
const _TSNE_METRIC = SqEuclidean()


"""
    _tsne_dist(a, b)

Compute the squared Euclidean distance between vectors `a` and `b`.

# Arguments
- `a::AbstractVector{<:Real}`: First vector.
- `b::AbstractVector{<:Real}`: Second vector.

# Returns
- `Float64`: Squared Euclidean distance between `a` and `b`.

# Notes
- Uses `evaluate(_TSNE_METRIC, a, b)` internally, where `_TSNE_METRIC` is a constant `SqEuclidean()` metric.
- Intended for use as a helper function in t-SNE computations.
"""
_tsne_dist(a, b) = evaluate(_TSNE_METRIC, a, b)


"""
    embedding_points(model::Word2VecModel; words=nothing, normalize=false)

Return a points-as-rows matrix of embeddings suitable for dimensionality reduction.

# Arguments
- `model::Word2VecModel`: The trained Word2Vec model.

# Keyword Arguments
- `words=nothing`: If `nothing`, use all words in the vocabulary. Otherwise, provide a vector of strings; only existing words are included.
- `normalize=false`: If `true`, L2-normalize each row of the returned matrix.

# Returns
- `X::Matrix{Float64}`: Embedding vectors as rows `(n_points × dim)`.
- `labels::Vector{String}`: Corresponding words for each row.

# Notes
- Rows correspond to the selected words; `labels[i]` matches `X[i, :]`.
- If no provided words exist in the vocabulary, returns an empty matrix and empty label vector.
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
                    seed=42, reduce_dims=50, max_iter=1000, perplexity=30, kwargs...)

Compute a t-SNE projection of word embeddings from a `Word2VecModel`.

# Arguments
- `model::Word2VecModel`: The trained Word2Vec model.

# Keyword Arguments
- `dims::Int=2`: Number of output dimensions.
- `words=nothing`: Optional subset of words to embed. If `nothing`, all words are used.
- `normalize=false`: If `true`, L2-normalize embeddings before t-SNE.
- `seed::Int=42`: Random seed for reproducibility.
- `reduce_dims::Int=50`: Initial PCA reduction dimension (clamped to embedding dimension).
- `max_iter::Int=1000`: Maximum number of t-SNE optimization iterations.
- `perplexity::Int=30`: t-SNE perplexity hyperparameter.
- `kwargs...`: Additional keyword arguments passed to `TSne.tsne`.

# Returns
- `Y::Matrix{Float64}`: Row-wise t-SNE coordinates `(n_points × dims)`.
- `labels::Vector{String}`: Corresponding word labels for each row.

# Notes
- Uses `_tsne_dist` (squared Euclidean) as the distance metric.
- Returns an empty `(0 × dims)` matrix if no words are found in the model.
- Useful for visualization of embeddings in 2D or 3D.
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
"""
    cosine_similarity(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real})

Compute the **cosine similarity** between two vectors.

# Arguments
- `v1::AbstractVector{<:Real}` — First vector.
- `v2::AbstractVector{<:Real}` — Second vector.

# Keyword Arguments
- `n1::Union{<:Real, Nothing}` — precomputed norm of v1 or nothing.
- `n2::Union{<:Real, Nothing}` — precomputed norm of v2 or nothing.

# Returns
- `Float64` — Cosine similarity score in the range `[-1.0, 1.0]`.

# Throws
- `ArgumentError` — If either `v1` or `v2` is a zero vector.

# Notes
- Cosine similarity is defined as `dot(v1, v2) / (norm(v1) * norm(v2))`.
- Works with any numeric vector type (`Float32`, `Float64`, `Int`, etc.), but output is always `Float64`.
"""
function cosine_similarity(
        v1::AbstractVector{T}, v2::AbstractVector{T};
        n1::Union{T, Nothing} = nothing, n2::Union{T, Nothing} = nothing
    ) where {T <: Real}
    n1 = n1 === nothing ? norm(v1) : n1
    n2 = n2 === nothing ? norm(v2) : n2
    isapprox(n1, 0.0) && throw(ArgumentError("v1 is a zero vector, cosine similarity undefined"))
    isapprox(n2, 0.0) && throw(ArgumentError("v2 is a zero vector, cosine similarity undefined"))
    return dot(v1, v2) / (n1 * n2)
end


"""
    similarity(model::Word2VecModel, w1::AbstractString, w2::AbstractString)

Compute the **cosine similarity** between two words in a Word2Vec model.

# Arguments
- `model::Word2VecModel` — A trained Word2Vec model containing `vocab` and `embeddings`.
- `w1::AbstractString` — First word.
- `w2::AbstractString` — Second word.

# Returns
- `Float64` — Cosine similarity between the embedding vectors of `w1` and `w2`.  
  Range: [-1.0, 1.0].

# Notes
- Cosine similarity is computed as `dot(v1, v2) / (norm(v1) * norm(v2))`.
- Both words must exist in the model's vocabulary.
- Embedding vectors must be non-zero; otherwise similarity is undefined.

# Throws
- `KeyError` — If either `w1` or `w2` is not in the model vocabulary.
"""
function similarity(model::Word2VecModel, w1::AbstractString, w2::AbstractString)
    # Check words exist
    haskey(model.word_to_index, w1) ||
        throw(KeyError("Word '$w1' not found in Word2Vec model"))
    haskey(model.word_to_index, w2) ||
        throw(KeyError("Word '$w2' not found in Word2Vec model"))

    w1 = convert(String, w1)
    w2 = convert(String, w2)

    v1 = get_embedding(model, w1)
    v2 = get_embedding(model, w2)
    n1 = get_embedding_norm(model, w1)
    n2 = get_embedding_norm(model, w2)

    return cosine_similarity(v1, v2; n1 = n1, n2 = n2)
end


"""
    analogy(model::Word2VecModel, a::AbstractString, b::AbstractString, c::AbstractString; topk::Int = 5)

Solve word analogies using a trained Word2Vec model.

Given words `a`, `b`, `c`, this function finds words `x` such that:
a : b ≈ c : x
using vector arithmetic: `target = b - a + c`.

# Arguments
- `model::Word2VecModel` — A trained Word2Vec model containing `vocab` and `embeddings`.
- `a::AbstractString` — First word in the analogy (`a : b`).
- `b::AbstractString` — Second word in the analogy (`a : b`).
- `c::AbstractString` — Third word in the analogy (`c : ?`).

# Keyword Arguments
- `topk::Int=5` — Number of top candidates to return.

# Returns
- `Vector{String}` — List of the top `topk` words whose embeddings best satisfy the analogy.

# Notes
- Cosine similarity is used to rank candidate words.
- Input words are excluded from the returned results.
- Analogy is computed as `target = b - a + c`.
- Embedding vectors must be non-zero; otherwise similarity is undefined.

# Throws
- `KeyError` — If any of `a`, `b`, or `c` is not in the model vocabulary.
"""
function analogy(model::Word2VecModel, a::AbstractString, b::AbstractString, c::AbstractString; topk::Int = 5)
    # Safety checks
    for w in (a, b, c)
        haskey(model.word_to_index, w) ||
            throw(KeyError("Word '$w' not found in Word2Vec model"))
    end

    va = get_embedding(model, convert(String, a))
    vb = get_embedding(model, convert(String, b))
    vc = get_embedding(model, convert(String, c))

    target = vb - va + vc
    target_norm = norm(target)

    sims = Vector{Float64}(undef, length(model.vocab))

    @inbounds for (i, word) in enumerate(model.vocab)
        # exclude input words by setting value to -Inf
        sims[i] = word in (a, b, c) ? -Inf : cosine_similarity(target, @view model.embeddings[:, i]; n1 = target_norm, n2 = model.vector_norms[i])
    end

    return model.vocab[sortperm(sims, rev = true)[1:topk]]
end

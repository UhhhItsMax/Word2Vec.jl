"""
    Word2VecEvaluation

A Julia file providing evaluation utilities for Word2Vec models.

# Features
- Computes **cosine similarity** between words (`similarity`) or arbitrary vectors (`cosine_similarity`).
- Performs **word analogy queries** (`analogy`) using vector arithmetic.
- Safety checks ensure words exist in the model vocabulary.
- Handles zero vectors gracefully, raising informative errors.

# Included Functions
- `cosine_similarity(v1, v2)` — Computes cosine similarity between two numeric vectors.
- `similarity(model, w1, w2)` — Computes cosine similarity between two words in a Word2Vec model.
- `analogy(model, a, b, c; topk)` — Solves analogies of the form `a : b ≈ c : ?`.

# Notes
- All functions use `Float64` output for similarity scores.
- Analogy computation excludes the input words from the results.
- Designed for **in-memory Word2VecModel instances**.

# Dependencies
- Base Julia
- `LinearAlgebra` (for `norm` and `dot`)
"""
using LinearAlgebra

export similarity, analogy

"""
    cosine_similarity(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real}) :: Float64

Computes the **cosine similarity** between two vectors.

# Arguments
- `v1::AbstractVector{<:Real}`: First vector.
- `v2::AbstractVector{<:Real}`: Second vector.

# Returns
- `Float64`: Cosine similarity score in the range `[-1.0, 1.0]`.

# Throws
- `ArgumentError`: If either `v1` or `v2` is a zero vector.

# Notes
- Cosine similarity is defined as `dot(v1, v2) / (norm(v1) * norm(v2))`.
- Works with any numeric vector type (`Float32`, `Float64`, `Int`, etc.), but output is always `Float64`.
"""
function cosine_similarity(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real}) :: Float64
    n1 = norm(v1)
    n2 = norm(v2)
    n1 == 0.0 && throw(ArgumentError("v1 is a zero vector, cosine similarity undefined"))
    n2 == 0.0 && throw(ArgumentError("v2 is a zero vector, cosine similarity undefined"))
    return dot(v1, v2) / (n1 * n2)
end

"""
    similarity(model::Word2VecModel, w1::AbstractString, w2::AbstractString) :: Float64

Compute the **cosine similarity** between two words in a Word2Vec model.

# Arguments
- `model::Word2VecModel`: A trained Word2Vec model containing `vocab` and `embeddings`.
- `w1::AbstractString`: First word.
- `w2::AbstractString`: Second word.

# Returns
- `Float64`: Cosine similarity between the embedding vectors of `w1` and `w2`.  
  Range: [-1.0, 1.0].

# Notes
- Cosine similarity is computed as `dot(v1, v2) / (norm(v1) * norm(v2))`.
- Both words must exist in the model's vocabulary.
- Embedding vectors must be non-zero; otherwise similarity is undefined.

# Throws
- `KeyError`: If either `w1` or `w2` is not in the model vocabulary.
- `ArgumentError`: If either word has a zero vector embedding.
"""
function similarity(model::Word2VecModel, w1::AbstractString, w2::AbstractString) :: Float64
    # Check words exist
    haskey(model.word_to_index, w1) ||
        throw(KeyError("Word '$w1' not found in Word2Vec model"))
    haskey(model.word_to_index, w2) ||
        throw(KeyError("Word '$w2' not found in Word2Vec model"))

    # Get indices
    i1 = model.word_to_index[w1]
    i2 = model.word_to_index[w2]

    # Get vectors
    v1 = model.embeddings[:, i1]
    v2 = model.embeddings[:, i2]

    # Check for zero vectors
    norm(v1) == 0.0 && throw(ArgumentError("Embedding for '$w1' is a zero vector, similarity undefined"))
    norm(v2) == 0.0 && throw(ArgumentError("Embedding for '$w2' is a zero vector, similarity undefined"))

    # Return cosine similarity
    return cosine_similarity(v1, v2)
end


"""
    analogy(model::Word2VecModel, a::AbstractString, b::AbstractString, c::AbstractString; topk::Int = 5) :: Vector{String}

Solve word analogies using a trained Word2Vec model.

Given words `a`, `b`, `c`, this function finds words `x` such that:
a : b ≈ c : x
using vector arithmetic: `target = b - a + c`.

# Arguments
- `model::Word2VecModel`: A trained Word2Vec model containing `vocab` and `embeddings`.
- `a::AbstractString`: First word in the analogy (`a : b`).
- `b::AbstractString`: Second word in the analogy (`a : b`).
- `c::AbstractString`: Third word in the analogy (`c : ?`).

# Keyword Arguments
- `topk::Int=5`: Number of top candidates to return.

# Returns
- `Vector{String}`: List of the top `topk` words whose embeddings best satisfy the analogy.

# Notes
- Cosine similarity is used to rank candidate words.
- Input words are excluded from the returned results.
- Analogy is computed as `target = b - a + c`.
- Embedding vectors must be non-zero; otherwise similarity is undefined.

# Throws
- `KeyError`: If any of `a`, `b`, or `c` is not in the model vocabulary.
- `ArgumentError`: If any of the embeddings involved in the analogy are zero vectors.
"""
function analogy(model::Word2VecModel, a::AbstractString, b::AbstractString, c::AbstractString; topk::Int = 5)
    # Safety checks
    for w in (a, b, c)
        haskey(model.word_to_index, w) ||
            throw(KeyError("Word '$w' not found in Word2Vec model"))
    end

    ia = model.word_to_index[a]
    ib = model.word_to_index[b]
    ic = model.word_to_index[c]

    va = model.embeddings[:, ia]
    vb = model.embeddings[:, ib]
    vc = model.embeddings[:, ic]

    for vec in (va, vb, vc)
        norm(vec) == 0.0 && throw(ArgumentError("Input word vector is zero, cannot compute analogy"))
    end

    target = vb - va + vc

    sims = Vector{Float64}(undef, length(model.vocab))

    @inbounds for i in eachindex(model.vocab)
        sims[i] = cosine_similarity(target, model.embeddings[:, i])
    end

    # Optionally exclude input words
    for i in (ia, ib, ic)
        sims[i] = -Inf
    end

    return model.vocab[sortperm(sims, rev = true)[1:topk]]
end
"""
    Word2VecModel

Unified in-memory representation for Word2Vec embeddings.

# Fields
- `vocab::Vector{String}` — List of words in the vocabulary.
- `embeddings::Matrix{Float64}` — Embedding matrix of size `(dim, vocab_size)`; each column corresponds to a word vector.
- `vector_norms::Vector{Float64}` — Precomputed norms of each embedding vector (used for fast cosine similarity).
- `word_to_index::Dict{String, Int}` — Mapping from word strings to their column indices in `embeddings`.

# Notes
- Typically created via `load_word2vec` or by wrapping preloaded embeddings.
- `vector_norms` is optional for computation but recommended for efficiency.
- Provides the base structure for computing similarity, analogies, and ConEc embeddings.
"""
struct Word2VecModel{T <: Real}
    vocab::Vector{String}
    embeddings::Matrix{T}
    vector_norms::Vector{T}
    word_to_index::Dict{String, Int}
end


"""
    Word2VecModel(vocab::Vector{String}, embeddings::Matrix{T}) where {T<:Real}

Construct a `Word2VecModel` from a vocabulary and embedding matrix.

# Arguments
- `vocab::Vector{String}` — List of words in the vocabulary.
- `embeddings::Matrix{T}` — Embedding matrix of size `(dim, vocab_size)`; each column corresponds to a word vector.

# Returns
- `Word2VecModel{T}` — Struct containing the vocabulary, embeddings, precomputed vector norms, and a word-to-index mapping.

# Notes
- Automatically computes the norm of each embedding vector for efficient cosine similarity.
- Throws `ArgumentError` if:
  - Number of columns in `embeddings` does not match `length(vocab)`.
  - Any embedding vector has zero norm.
- Builds `word_to_index` mapping from words to column indices in `embeddings`.
"""
function Word2VecModel(vocab::Vector{String}, embeddings::Matrix{T}) where {T <: Real}
    size(embeddings, 2) == length(vocab) || throw(ArgumentError("embeddings must have one column per vocab entry"))

    word_to_index = Dict(word => idx for (idx, word) in enumerate(vocab))
    vector_norms = Vector{T}(undef, size(embeddings, 2))

    @inbounds for (j, col) in enumerate(eachcol(embeddings))
        n = norm(col)
        iszero(n) && throw(ArgumentError("embedding vector has zero norm for word $(vocab[j])"))
        vector_norms[j] = convert(T, n)
    end

    return Word2VecModel{T}(vocab, embeddings, vector_norms, word_to_index)
end


"""
    get_embedding(model::Word2VecModel, word::AbstractString)

Return a **view** of the embedding vector for a specific word.

# Arguments
- `model::Word2VecModel` — The Word2Vec model containing embeddings.
- `word::AbstractString` — Word whose embedding is requested.

# Returns
- `SubArray{T,1}` — A view of the embedding vector (length = embedding dimension).

# Notes
- Does **not** copy the vector; modifying the view will modify the underlying embeddings.
- Throws `KeyError` if `word` is not in `model.vocab`.
- Useful for fast access to individual embeddings without memory overhead.
"""
get_embedding(model::Word2VecModel, word::String) = @view model.embeddings[:, model.word_to_index[word]]


"""
    get_embedding_norm(model::Word2VecModel, word::AbstractString)

Return the **precomputed norm** of the embedding vector for a specific word.

# Arguments
- `model::Word2VecModel` — The Word2Vec model containing embeddings.
- `word::AbstractString` — Word whose embedding norm is requested.

# Returns
- `Float64` — Euclidean norm of the word’s embedding vector.

# Notes
- Uses the precomputed `vector_norms` stored in the model for efficiency.
- Throws `KeyError` if `word` is not in `model.vocab`.
- Useful for similarity computations without recomputing vector norms.
"""
get_embedding_norm(model::Word2VecModel, word::String) = model.vector_norms[model.word_to_index[word]]


"""
    from_dict_data(embeddings_map::Dict{String, Vector{T}})

Construct a `Word2VecModel` from a dictionary mapping words to embedding vectors.

# Arguments
- `embeddings_map::Dict{String, Vector{T}}` — Dictionary where keys are words and values are embedding vectors.

# Returns
- `Word2VecModel{T}` — A model containing:
    - `vocab` — Words in the dictionary.
    - `embeddings` — Matrix of vectors (columns correspond to words).
    - `vector_norms` — Precomputed Euclidean norms of each embedding.
    - `word_to_index` — Mapping from words to column indices.

# Notes
- All vectors must have the same dimensionality.
- Automatically converts vectors to type `T`.
- Useful for constructing a Word2Vec model from in-memory data rather than files.
"""
function from_dict_data(embeddings_map::Dict{String, Vector{T}}) where {T <: AbstractFloat}
    words = collect(keys(embeddings_map))
    dim = length(first(values(embeddings_map)))

    M = Array{T}(undef, dim, length(words))

    for (i, w) in enumerate(words)
        M[:, i] = convert.(T, embeddings_map[w])
    end

    return Word2VecModel(words, M)
end

"""
    SparseContextMatrix{T<:Real}

Store a sparse word context matrix for co-occurrence-based embeddings.

# Fields
- `mat::SparseMatrixCSC{T, Int}` — Sparse matrix of size `(V × V)`, where `V` is the vocabulary size. Entry `(i,j)` counts (or stores a normalized value of) how often word `i` co-occurs with word `j`.
- `vocab::Vector{String}` — Vocabulary tokens corresponding to the rows and columns of `mat`.
- `token_to_id::Dict{String, Int}` — Maps token strings to their column/row indices in `mat`.

# Notes
- Used to construct global or local context matrices for ConEc or other co-occurrence-based models.
- Typically built from a corpus using a sliding window approach.
- Supports any numeric type `T<:Real` for matrix entries.
"""
struct SparseContextMatrix{T <: Real}
    mat::SparseMatrixCSC{T, Int}
    vocab::Vector{String}
    token_to_id::Dict{String, Int}
end


"""
    SparseContextMatrix(path::AbstractString; window_size::Int=5, min_count::Int=1)

Construct a sparse word context matrix directly from a text corpus file.

# Arguments
- `path::AbstractString` — Path to a text file containing the corpus.
- `window_size::Int=5` — Symmetric context window size for counting co-occurrences.
- `min_count::Int=1` — Minimum token frequency; tokens occurring fewer times are ignored.

# Returns
- `SparseContextMatrix{Float64}` — Sparse context matrix with normalized co-occurrence counts, the vocabulary vector, and a token-to-index dictionary.

# Notes
- Tokens are normalized (lowercased and stripped of non-alphanumeric characters).
- The matrix `mat` has size `(V × V)`, where `V` is the number of tokens surviving `min_count` filtering.
- Enables quick creation of a `SparseContextMatrix` from raw text without manually computing co-occurrences.
"""
function SparseContextMatrix(
        path::AbstractString;
        window_size::Int = 5,
        min_count::Int = 1,
    )
    token_counts = get_occurence_counts(path)
    vocab, token_to_id = filter_vocabulary(token_counts, min_count)
    token_coocs = get_co_occurence_counts(path, token_to_id, window_size)
    normalized_coocs = normalize_coocs(token_coocs, token_counts, token_to_id)
    mat = dict_to_sparse(normalized_coocs, length(vocab))

    return SparseContextMatrix(mat, vocab, token_to_id)
end


"""
    save_sparse_context_matrix(path, scm)

Serialize and save a `SparseContextMatrix` to disk.

# Arguments
- `path::AbstractString` — File path where the serialized matrix will be saved.
- `scm::SparseContextMatrix` — Sparse context matrix to serialize and save.

# Returns
- `nothing`

# Notes
- Uses Julia's built-in `serialize` function.
- Can be loaded later with `load_sparse_context_matrix`.
- Overwrites the file at `path` if it already exists.
"""
function save_sparse_context_matrix(path::AbstractString, scm::SparseContextMatrix)
    open(path, "w") do io
        serialize(io, scm)
    end
    return nothing
end


"""
    load_sparse_context_matrix(path)

Load a previously saved `SparseContextMatrix` from disk.

# Arguments
- `path::AbstractString` — File path of the serialized `SparseContextMatrix` (created with `save_sparse_context_matrix`).

# Returns
- `SparseContextMatrix` — Deserialized sparse context matrix.

# Notes
- Uses Julia's built-in `deserialize` function.
- Raises an error if the file does not exist or is not a valid `SparseContextMatrix`.
- Commonly used to reload global or local context matrices for ConEc or other co-occurrence-based models.
"""
function load_sparse_context_matrix(path::AbstractString)
    return open(path, "r") do io
        return deserialize(io)
    end
end


"""
    normalize_coocs(token_coocs, token_counts, token_to_id)

Normalize raw co-occurrence counts for a corpus.

# Arguments
- `token_coocs::Dict{Tuple{Int, Int}, Int}` — Maps `(cooc_token_id, target_token_id)` → raw co-occurrence count.
- `token_counts::Dict{String, Int}` — Total occurrence counts of each token in the corpus.
- `token_to_id::Dict{String, Int}` — Mapping from token strings to their integer indices used in `token_coocs`.

# Returns
- `Dict{Tuple{Int,Int},Float64}` — Dictionary of the same shape as `token_coocs`, where each co-occurrence count 
  is normalized by dividing by the total count of the target token.

# Notes
- This normalization corresponds to computing `p(cooc | target)` for each pair.
- Typically used when building a `SparseContextMatrix` for ConEc or other co-occurrence-based embeddings.
"""
function normalize_coocs(
        token_coocs::Dict{Tuple{Int, Int}, Int},
        token_counts::Dict{String, Int},
        token_to_id::Dict{String, Int},
    )
    vocab_size = length(token_to_id)

    inv_target_counts = Vector{Float64}(undef, vocab_size)
    for (tok, id) in token_to_id
        inv_target_counts[id] = 1.0 / token_counts[tok]
    end

    normalized_coocs = Dict{Tuple{Int, Int}, Float64}()
    sizehint!(normalized_coocs, length(token_coocs))

    for ((cooc, target), v) in token_coocs
        normalized_coocs[(cooc, target)] = v * inv_target_counts[target]
    end

    return normalized_coocs
end


"""
    normalize_token(token::AbstractString)

Normalize a token string for consistent text processing.

# Arguments
- `token::AbstractString` — Input token to normalize.

# Returns
- `String` — Normalized token, converted to lowercase and with leading 
  and trailing non-alphanumeric characters removed.

# Notes
- Useful when building vocabularies, counting occurrences, or constructing
  context matrices to ensure consistent token representation.
"""
normalize_token(token::AbstractString)::String = replace(token, r"^\W+|\W+$" => "") |> lowercase


"""
    get_occurence_counts(path::AbstractString)

Count the occurrences of each token in a text file.

# Arguments
- `path::AbstractString` — Path to the text file containing the corpus.

# Returns
- `Dict{String, Int}` — Dictionary mapping each normalized token to its 
  frequency count in the file.

# Notes
- Tokens are normalized using `normalize_token` (lowercased and stripped of
  leading/trailing non-alphanumeric characters).
- Empty tokens are ignored.
"""
function get_occurence_counts(path::AbstractString)
    token_counts = Dict{String, Int}()
    open(path, "r") do io
        for line in eachline(io)
            for token in split(line)
                token = normalize_token(token)
                isempty(token) && continue
                get!(token_counts, token, 0)
                token_counts[token] += 1
            end
        end
    end
    return token_counts
end


"""
    filter_vocabulary(token_counts::Dict{String, Int}, min_count::Int)

Filter tokens by minimum occurrence and construct a token-to-index mapping.

# Arguments
- `token_counts::Dict{String, Int}` — Dictionary mapping tokens to their frequency counts.
- `min_count::Int` — Minimum frequency a token must have to be included in the vocabulary. Must be ≥ 1.

# Returns
- `vocab::Vector{String}` — Sorted vector of tokens that meet the minimum count.
- `token_to_id::Dict{String, Int}` — Mapping from each token to its index in `vocab` (1-based).

# Throws
- `ArgumentError` if `min_count < 1`.
- `ArgumentError` if the resulting vocabulary is empty.
"""
function filter_vocabulary(token_counts::Dict{String, Int}, min_count::Int)
    min_count < 1 && throw(ArgumentError("min_count must be ≥ 1"))

    vocab = sort([tok for (tok, count) in token_counts if count >= min_count])
    isempty(vocab) && throw(ArgumentError("Vocabulary is empty after min_count filtering"))

    token_to_id = Dict(tok => idx for (idx, tok) in enumerate(vocab))
    return vocab, token_to_id
end


"""
    get_co_occurence_counts(path::AbstractString, token_to_id::Dict{String, Int}, window_size::Int)

Count token co-occurrences in a text file using a symmetric sliding window.

# Arguments
- `path::AbstractString` — Path to the text file containing the corpus.
- `token_to_id::Dict{String, Int}` — Mapping from tokens to integer indices.
- `window_size::Int` — Size of the symmetric context window (≥ 1).

# Returns
- `Dict{Tuple{Int, Int}, Int}` — Dictionary where keys are `(context_id, target_id)` 
  pairs and values are counts of how often `context_id` occurs in the window of `target_id`.

# Notes
- The buffer ignores the first few tokens until it is full.
- Token normalization is applied (lowercase, trim non-alphanumeric characters).

# Throws
- `ArgumentError` if `window_size < 1`.
"""
function get_co_occurence_counts(
        path::AbstractString,
        token_to_id::Dict{String, Int},
        window_size::Int
    )
    window_size ≥ 1 || throw(ArgumentError("window_size must be ≥ 1"))

    token_coocs = Dict{Tuple{Int, Int}, Int}()
    token_buf = CircularBuffer{Int}(2 * window_size + 1)

    open(path, "r") do io
        for line in eachline(io)
            for token in split(line)
                token = normalize_token(token)
                isempty(token) && continue
                id = get(token_to_id, token, nothing)
                id === nothing && continue

                push!(token_buf, id)

                # Only count when the buffer is full (ignore first few tokens)
                !isfull(token_buf) && continue
                target = token_buf[window_size + 1]

                for context_id in token_buf
                    context_id == target && continue
                    token_coocs[(context_id, target)] = get(token_coocs, (context_id, target), 0) + 1
                end
            end
        end
    end

    return token_coocs
end


"""
    dict_to_sparse(coocs::Dict{Tuple{Int, Int}, T}, n::Int) where {T}

Convert a dictionary of co-occurrence counts into a sparse square matrix of size `n × n`.

# Arguments
- `coocs::Dict{Tuple{Int, Int}, T}` — Dictionary mapping `(row_index, col_index)` pairs to values.
- `n::Int` — Size of the square matrix.

# Returns
- `SparseMatrixCSC{T, Int}` — Sparse matrix with entries from `coocs`. Entries not specified in `coocs` are zeros.

# Throws
- `ArgumentError` if any row or column index in `coocs` is greater than `n`.
"""
function dict_to_sparse(coocs::Dict{Tuple{Int, Int}, T}, n::Int) where {T}
    nnz = length(coocs)
    rows = Vector{Int}(undef, nnz)
    cols = Vector{Int}(undef, nnz)
    vals = Vector{T}(undef, nnz)

    for (k, ((r, c), v)) in enumerate(coocs)
        r > n && throw(ArgumentError("Row index $r > n=$n"))
        c > n && throw(ArgumentError("Column index $c > n=$n"))
        rows[k] = r
        cols[k] = c
        vals[k] = v
    end

    return sparse(rows, cols, vals, n, n)

end

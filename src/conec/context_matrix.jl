using Serialization: serialize, deserialize
using SparseArrays: SparseMatrixCSC, sparse
using ..CircularBuffers: CircularBuffer, isfull

export SparseContextMatrix, save_sparse_context_matrix, load_sparse_context_matrix


"""
    SparseContextMatrix{T<:Real}

Represents a **sparse word context matrix** for co-occurrence-based embeddings.

# Fields
- `mat::SparseMatrixCSC{T, Int}`: The sparse context matrix of size `(V × V)`, 
  where `V` is the vocabulary size. Entry `(i,j)` typically counts (or stores a normalized value of) how often word `i` co-occurs with word `j`.
- `vocab::Vector{String}`: Vector of vocabulary tokens corresponding to the rows/columns of `mat`.
- `token_to_id::Dict{String, Int}`: Mapping from token strings to their column/row indices in `mat`.

# Notes
- Used for constructing global or local context matrices in ConEc or other co-occurrence-based models.
- Typically built from a corpus using a sliding window approach.
- Supports any numeric type `T<:Real` for the matrix entries.
"""
struct SparseContextMatrix{T<:Real}
    mat::SparseMatrixCSC{T, Int}
    vocab::Vector{String}
    token_to_id::Dict{String, Int}
end



"""
    SparseContextMatrix(path::AbstractString; window_size::Int=5, min_count::Int=1) -> SparseContextMatrix{Float64}

Build a **sparse word context matrix** directly from a text corpus file.

# Arguments
- `path::AbstractString`: Path to a text file containing the corpus.
- `window_size::Int=5`: Size of the symmetric context window for counting co-occurrences.
- `min_count::Int=1`: Minimum token frequency; tokens with fewer occurrences are ignored.

# Returns
- `SparseContextMatrix{Float64}`: A sparse context matrix with normalized co-occurrence counts, the vocabulary vector, and a token-to-index dictionary.

# Notes
- Token normalization is applied: lowercase and trim non-alphanumeric characters.
- The matrix `mat` is of size `(V × V)` where `V` is the number of tokens surviving `min_count` filtering.
- This constructor allows quick creation of a `SparseContextMatrix` from raw text without manually computing co-occurrences.
"""
function SparseContextMatrix(
    path::AbstractString;
    window_size::Int = 5,
    min_count::Int = 1,
)::SparseContextMatrix{Float64}
    token_counts = get_occurence_counts(path)
    vocab, token_to_id = filter_vocabulary(token_counts, min_count)
    token_coocs = get_co_occurence_counts(path, token_to_id, window_size)
    normalized_coocs = normalize_coocs(token_coocs, token_counts, token_to_id)
    mat = dict_to_sparse(normalized_coocs, length(vocab))

    return SparseContextMatrix(mat, vocab, token_to_id)
end


"""
    save_sparse_context_matrix(path::AbstractString, scm::SparseContextMatrix)

Serialize a `SparseContextMatrix` to disk.
"""
function save_sparse_context_matrix(path::AbstractString, scm::SparseContextMatrix)
    open(path, "w") do io
        serialize(io, scm)
    end
    return nothing
end


"""
    load_sparse_context_matrix(path::AbstractString)::SparseContextMatrix

Deserialize a `SparseContextMatrix` from disk.
"""
function load_sparse_context_matrix(path::AbstractString)::SparseContextMatrix
    open(path, "r") do io
        return deserialize(io)
    end
end


"""
    normalize_coocs(token_coocs::Dict{Tuple{Int, Int}, Int}, token_counts::Dict{String, Int},
                    token_to_id::Dict{String, Int}, vocab_size::Int)::Dict{Tuple{Int, Int}, Float64}

Normalize each co-occurrence by 1/count(target).
"""
function normalize_coocs(
    token_coocs::Dict{Tuple{Int, Int}, Int},
    token_counts::Dict{String, Int},
    token_to_id::Dict{String, Int},
)::Dict{Tuple{Int, Int}, Float64}
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
    normalize_token(token::String)

Trim non-alphanumeric chars from both ends and lowercase the token.
"""
normalize_token(token::AbstractString)::String = replace(token, r"^\W+|\W+$" => "") |> lowercase


"""
    get_occurence_counts(path::AbstractString)::Dict{String, Int}

Count token occurrences in a text file.
"""
function get_occurence_counts(path::AbstractString)::Dict{String, Int}
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


function filter_vocabulary(token_counts::Dict{String, Int}, min_count::Int)::Tuple{Vector{String}, Dict{String, Int}}
    min_count < 1 && throw(ArgumentError("min_count must be ≥ 1"))

    vocab = sort([tok for (tok, count) in token_counts if count >= min_count])
    isempty(vocab) && throw(ArgumentError("Vocabulary is empty after min_count filtering"))

    token_to_id = Dict(tok => idx for (idx, tok) in enumerate(vocab))
    return vocab, token_to_id
end


"""
    get_co_occurence_counts(path::AbstractString, token_to_id::Dict{String, Int}, window_size::Int)::Dict{Tuple{Int, Int}, Int}

Count token co-occurrences using a sliding window.
"""
function get_co_occurence_counts(
    path::AbstractString, 
    token_to_id::Dict{String, Int}, 
    window_size::Int
)::Dict{Tuple{Int, Int}, Int}
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

Convert co-occurrence counts into a sparse matrix.
"""
function dict_to_sparse(coocs::Dict{Tuple{Int, Int}, T}, n::Int)::SparseMatrixCSC{T, Int} where {T}
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

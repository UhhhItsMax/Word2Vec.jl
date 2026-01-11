module sparse_context_matrix

using SparseArrays: SparseMatrixCSC, sparse
using ..CircularBuffers: CircularBuffer, isfull

export SparseContextMatrix, build_context_matrix_from_file

struct SparseContextMatrix{T<:Real}
    mat::SparseMatrixCSC{T, Int}
    vocab::Vector{String}
    token_to_id::Dict{String, Int}
end


"""
    build_context_matrix_from_file(path::AbstractString; window_size::Int = 5, min_count::Int = 1)

Build a sparse context matrix from a text file.
"""
function build_context_matrix_from_file(
    path::AbstractString;
    window_size::Int = 5,
    min_count::Int = 1
)::SparseContextMatrix{Int}
    token_counts = get_occurence_counts(path)
    vocab, token_to_id = filter_vocabulary(token_counts, min_count)
    token_coocs = get_co_occurence_counts(path, token_to_id, window_size)
    mat = dict_to_sparse(token_coocs, length(vocab))

    return SparseContextMatrix(mat, vocab, token_to_id)
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
        for line in readlines(io)
            for token in split(line)
                token = normalize_token(token)
                isempty(token) && continue
                token_counts[token] = get(token_counts, token , 0) + 1
            end
        end
    end
    return token_counts
end


"""
    filter_vocabulary(token_counts::Dict{String, Int}, min_count::Int)

Filter tokens by minimum count and return `(vocab, token_to_id)`.
"""
function filter_vocabulary(token_counts::Dict{String, Int}, min_count::Int)::Tuple{Vector{String}, Dict{String, Int}}
    vocab = [tok for (tok, count) in token_counts if count >= min_count]
    token_to_id = Dict(tok => idx for (idx, tok) in enumerate(vocab))

    return vocab, token_to_id
end


"""
    get_co_occurence_counts(path::AbstractString, token_to_id::Dict{String, Int}, window_size::Int)::Dict{Tuple{Int, Int}, Int}

Count token co-occurrences using a sliding window.
"""
function get_co_occurence_counts(path::AbstractString, token_to_id::Dict{String, Int}, window_size::Int)::Dict{Tuple{Int, Int}, Int}
    token_coocs = Dict{Tuple{Int, Int}, Int}()
    token_buf = CircularBuffer{Int}(2 * window_size + 1)
    open(path, "r") do io
        for line in eachline(io)
            for token in split(line)
                token=normalize_token(token)
                isempty(token) && continue
                token_id = get(token_to_id, token, nothing)
                token_id === nothing && continue
                push!(token_buf, token_id)

                # add co-occurences for target token (middle element of buffer) and all other buffer elements
                # only count if buffer is currently full, i.e. ignore first few initial tokens in data
                !isfull(token_buf) && continue
                target = token_buf[window_size + 1]
                for cooc in token_buf
                    cooc == target && continue
                    token_coocs[(cooc, target)] = get(token_coocs, (cooc, target), 0) + 1 
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
    vals = Vector{Int}(undef, nnz)

    k=1
    for ((r, c), v) in coocs
        rows[k] = r
        cols[k] = c
        vals[k] = v
        k += 1
    end

    return sparse(rows, cols, vals, n, n) 

end

end # module

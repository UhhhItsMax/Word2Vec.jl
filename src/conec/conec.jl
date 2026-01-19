include("context_matrix.jl")
using SparseArrays: SparseMatrixCSC, sparse, spzeros, nzrange, rowvals, nonzeros, nnz
using ..sparse_context_matrix: SparseContextMatrix, build_context_matrix_from_file

export ConEcModel, build_conec_global, conec_embeddings_for_file

"""
    ConEcModel

Holds precomputed global context information for ConEc.

Fields:
- w2v::Word2VecModel                    trained word2vec model
- global_cm::SparseContextMatrix{Float64}  global context matrix (normalized)
- a::Float64                             weight on global context vs local context
"""
struct ConEcModel
    w2v::Word2VecModel
    global_cm::SparseContextMatrix{Float64}
    a::Float64
end


"""
Precompute the global context matrix for ConEc using the same
SparseContextMatrix code on the full training corpus.

The resulting `ConEcModel` can be reused to compute ConEc embeddings
for arbitrary local documents.
"""
function build_conec_global(
    w2v::Word2VecModel,
    global_corpus_path::AbstractString;
    window_size::Int = 5,
    min_count::Int = 1,
    a::Float64 = 0.6,
)::ConEcModel
    global_cm = build_context_matrix_from_file(
        global_corpus_path;
        window_size = window_size,
        min_count   = min_count,
    )
    return ConEcModel(w2v, global_cm, a)
end


# --------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------

"""
Extract a (Vx1) sparse context vector for `word` from a
SparseContextMatrix and map its indices into the word2vec
vocabulary index space (1..V).
"""
function _context_vector_for_word(
    word::String,
    scm::SparseContextMatrix{Float64},
    w2v_word_to_idx::Dict{String,Int},
    V::Int,
)::SparseMatrixCSC{Float64,Int}
    col_idx = get(scm.token_to_id, word, nothing)
    col_idx === nothing && return spzeros(Float64, V, 1)

    # scm.mat is (vocab_scm × vocab_scm); take the column for `word`
    col = scm.mat[:, col_idx]

    rows_w2v = Int[]
    vals     = Float64[]

    # iterate over nonzeros of this single column
    for k in nzrange(col, 1)
        r = rowvals(col)[k]
        v = nonzeros(col)[k]

        tok = scm.vocab[r]
        idx_w2v = get(w2v_word_to_idx, tok, 0)
        idx_w2v == 0 && continue

        push!(rows_w2v, idx_w2v)
        push!(vals, v)
    end

    if isempty(rows_w2v)
        return spzeros(Float64, V, 1)
    else
        # column index is always 1 here
        return sparse(rows_w2v, ones(Int, length(rows_w2v)),
                      vals, V, 1)
    end
end


# --------------------------------------------------------------------
# main API
# --------------------------------------------------------------------

"""
Compute ConEc embeddings for all words occurring in `local_path`.

- Uses the precomputed global context matrix stored in `model.global_cm`.
- Builds a local context matrix on the fly from `local_path` using the
  same SparseContextMatrix pipeline.
- For words present globally, uses a weighted combination

    c_w = a * c_w^global + (1-a) * c_w^local

  and then computes

    y_w = W0 * c_w

  where `W0` is the word2vec embedding matrix (`model.w2v.embeddings`).

- For words that never occurred in the global corpus (no global
  context), falls back to a = 0 (purely local context).
"""
function conec_embeddings_for_file(
    model::ConEcModel,
    local_path::AbstractString;
    window_size::Int = 5,
    min_count::Int = 1,
)::Dict{String,Vector{Float64}}
    w2v = model.w2v
    a   = model.a

    # 1) build local context matrix from the given document
    local_cm = build_context_matrix_from_file(
        local_path;
        window_size = window_size,
        min_count   = min_count,
    )

    # 2) setup vocabulary / index mappings
    V = length(w2v.vocab)
    E = w2v.embeddings                   # (dim × V)
    w2v_word_to_idx = w2v.word_to_index

    # 3) for each word in the local doc, compute its ConEc embedding
    conec = Dict{String,Vector{Float64}}()

    for word in local_cm.vocab
        # local context in w2v vocab space
        c_local = _context_vector_for_word(word, local_cm, w2v_word_to_idx, V)

        # global context in w2v vocab space
        c_global = _context_vector_for_word(word, model.global_cm, w2v_word_to_idx, V)

        # check if global context exists for this word
        has_global = nnz(c_global) > 0
        a_eff = has_global ? a : 0.0

        # if pure-global (a=1.0) but no global context, fall back to local
        if a_eff == 1.0 && !has_global
            a_eff = 0.0
        end

        # combined context vector: (V × 1)
        c = a_eff .* c_global .+ (1.0 - a_eff) .* c_local

        # skip only if both local and global contexts are zero
        nnz(c) == 0 && continue

        # ConEc embedding: y_w = W0 * c   (dim × 1)
        y = E * c
        conec[word] = vec(y)
    end

    return conec
end

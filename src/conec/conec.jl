"""
    ConEcModel

Store all information required to compute ConEc embeddings using a combination of global and local contexts.

# Fields
- `w2v::Word2VecModel` — Trained Word2Vec model whose input embeddings (`W0`) are used to map context vectors to word embeddings.
- `global_cm::SparseContextMatrix{Float64}` — Precomputed global context matrix capturing co-occurrence statistics of the vocabulary.
- `a::Float64` — Weight controlling the mixture of global and local context. The final context vector for a word `w` is computed as:
  `c_w = a * c_w^global + (1-a) * c_w^local`.

# Notes
- Typically created via an inner constructor that takes a `Word2VecModel` and a corpus path.
- Use `conec_embeddings_for_file` to compute ConEc embeddings for new documents.
- The `global_cm` can be reused across multiple local corpora to avoid recomputation.
"""
struct ConEcModel
    w2v::Word2VecModel
    global_cm::SparseContextMatrix
    a::Float64

    """
        ConEcModel(w2v::Word2VecModel, global_corpus_path::AbstractString;
                    window_size::Int=5, min_count::Int=1, a::Float64=0.6)

    Construct a `ConEcModel` by precomputing a global context matrix from a text corpus.

    # Arguments
    - `w2v::Word2VecModel` — Trained Word2Vec model whose embeddings will be used.
    - `global_corpus_path::AbstractString` — Path to a text corpus for computing global co-occurrence statistics.
    - `window_size::Int=5` — Symmetric context window size for counting co-occurrences.
    - `min_count::Int=1` — Minimum token frequency; tokens occurring fewer times are ignored.
    - `a::Float64=0.6` — Weight between global and local context vectors (`0 ≤ a ≤ 1`) when computing ConEc embeddings.

    # Returns
    - `ConEcModel` — Model containing the Word2Vec embeddings, precomputed global context matrix, and `a` parameter.

    # Notes
    - Automatically builds `global_cm::SparseContextMatrix` from the provided corpus.
    - Use `conec_embeddings_for_file` to compute ConEc embeddings for arbitrary local documents.
    """
    function ConEcModel(
            w2v::Word2VecModel, global_corpus_path::AbstractString;
            window_size::Int = 5,
            min_count::Int = 1,
            a::Float64 = 0.6
        )
        global_cm = SparseContextMatrix(
            global_corpus_path;
            window_size = window_size,
            min_count = min_count
        )
        return new(w2v, global_cm, a)
    end
end


"""
    _context_vector_for_word(word, scm, w2v_word_to_idx, V)

Compute a sparse context vector for a single `word` in the Word2Vec vocabulary space.

# Arguments
- `word::String` — Target word for which the context vector is extracted.
- `scm::SparseContextMatrix{Float64}` — Sparse context matrix (global or local) with co-occurrence counts.
- `w2v_word_to_idx::Dict{String,Int}` — Maps words to their indices in the Word2Vec embedding matrix.
- `V::Int` — Vocabulary size of the Word2Vec model (number of rows/columns in embedding matrix).

# Returns
- `SparseMatrixCSC{Float64,Int}` — Sparse `(V × 1)` vector representing the context of `word` in Word2Vec index space.  
  Returns `spzeros(V,1)` if `word` is not present in `scm` or has no nonzero co-occurrences.

# Notes
- Only words present in both `scm` and `w2v_word_to_idx` contribute nonzero entries.
- Suitable for computing ConEc embeddings: `y_w = W0 * c_w`, where `W0` is the Word2Vec embedding matrix.
- Intended for internal use only.
"""
function _context_vector_for_word(
        word::String,
        scm::SparseContextMatrix{T},
        w2v_word_to_idx::Dict{String, Int},
        V::Int,
    ) where {T <: Real}
    col_idx = get(scm.token_to_id, word, nothing)
    col_idx === nothing && return spzeros(T, V, 1)

    col = scm.mat[:, col_idx]
    nnz_col = nnz(col)
    nnz_col == 0 && return spzeros(T, V, 1)

    rows_w2v = Vector{Int}(undef, nnz_col)
    vals = Vector{T}(undef, nnz_col)
    count = 0

    @inbounds for k in nzrange(col, 1)
        r = rowvals(col)[k]
        v = nonzeros(col)[k]

        idx_w2v = get(w2v_word_to_idx, scm.vocab[r], 0)
        idx_w2v == 0 && continue

        count += 1
        rows_w2v[count] = idx_w2v
        vals[count] = v
    end

    count == 0 && return spzeros(T, V, 1)

    return sparse(rows_w2v[1:count], ones(Int, count), vals[1:count], V, 1)
end


"""
    conec_embeddings_for_file(model, local_path; window_size=5, min_count=1)

Compute ConEc embeddings for all words in a local corpus file.

# Arguments
- `model::ConEcModel` — ConEc model containing a pretrained Word2Vec model and a global context matrix.
- `local_path::AbstractString` — Path to a text file containing the local corpus.
- `window_size::Int=5` — Symmetric context window size for building the local context matrix.
- `min_count::Int=1` — Minimum token frequency; words occurring fewer times are ignored.

# Returns
- `Dict{String, Vector{Float64}}` — Maps words in the local corpus to their ConEc embeddings.  
  Words not present in either local or global context are omitted.

# Notes
- For each word, the context vector is computed as:

      c_w = a * c_w^global + (1 - a) * c_w^local

  where `a = model.a`. If the word has no global context, `c_w = c_w^local`.
- The final embedding is computed via:

      y_w = W0 * c_w

  where `W0` is the Word2Vec embedding matrix.
- Returns an empty dictionary if no tokens in the local corpus survive `min_count`.
"""
function conec_embeddings_for_file(
        model::ConEcModel,
        local_path::AbstractString;
        window_size::Int = 5,
        min_count::Int = 1,
    )
    # build local context
    local_cm = SparseContextMatrix(local_path; window_size = window_size, min_count = min_count)
    isempty(local_cm.vocab) && return Dict{String, Vector{Float64}}()

    V = length(model.w2v.vocab)
    E = model.w2v.embeddings
    w2v_word_to_idx = model.w2v.word_to_index
    a = model.a

    conec = Dict{String, Vector{Float64}}()

    for word in local_cm.vocab
        c_local = _context_vector_for_word(word, local_cm, w2v_word_to_idx, V)
        c_global = _context_vector_for_word(word, model.global_cm, w2v_word_to_idx, V)

        c = nnz(c_global) > 0 ? a .* c_global .+ (1 - a) .* c_local : c_local
        nnz(c) == 0 && continue

        conec[word] = vec(E * c)
    end

    return conec
end

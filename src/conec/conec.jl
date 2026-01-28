
"""
    ConEcModel

Holds all information needed to compute **ConEc embeddings** using a combination of global and local contexts.

# Fields
- `w2v::Word2VecModel`  
  A trained Word2Vec model whose input embeddings (`W0`) are used to map context vectors to word embeddings.

- `global_cm::SparseContextMatrix{Float64}`  
  The precomputed global context matrix derived from a corpus. This captures global co-occurrence statistics of the vocabulary.

- `a::Float64`  
  Weight parameter controlling the mixture of global and local context for ConEc embeddings.  
  For a word (w), its final context vector is computed as  
  `c_w = a * c_w^global + (1-a) * c_w^local`.

# Notes
- The struct is typically created using the inner constructor that takes a `Word2VecModel` and a corpus path.  
- ConEc embeddings for new documents can be computed with `conec_embeddings_for_file`.
- `global_cm` can be reused for multiple local corpora, avoiding recomputation.
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
    - `w2v::Word2VecModel`: A trained Word2Vec model whose embeddings will be used.
    - `global_corpus_path::AbstractString`: Path to a text corpus used to compute global co-occurrence statistics.
    - `window_size::Int=5`: Size of the symmetric context window for counting co-occurrences.
    - `min_count::Int=1`: Minimum frequency threshold; tokens occurring fewer times are ignored.
    - `a::Float64=0.6`: Weight between global and local context vectors when computing ConEc embeddings (`0 ≤ a ≤ 1`).

    # Returns
    - `ConEcModel`: A model containing the Word2Vec embeddings, precomputed global context matrix, and the `a` weighting parameter.

    # Notes
    - The constructor automatically builds the `global_cm::SparseContextMatrix` from the provided corpus.
    - ConEc embeddings for arbitrary local documents can later be computed with `conec_embeddings_for_file`.
    """
    function ConEcModel(w2v::Word2VecModel, global_corpus_path::AbstractString;
                        window_size::Int = 5,
                        min_count::Int = 1,
                        a::Float64 = 0.6)
        global_cm = SparseContextMatrix(global_corpus_path;
                                        window_size = window_size,
                                        min_count   = min_count)
        new(w2v, global_cm, a)
    end
end


"""
    _context_vector_for_word(word, scm, w2v_word_to_idx, V) -> SparseMatrixCSC{Float64,Int}

Compute a sparse context vector for a single `word` in the Word2Vec vocabulary space.

# Arguments
- `word::String`  
  The target word for which the context vector is extracted.

- `scm::SparseContextMatrix{Float64}`  
  The sparse context matrix (global or local) from which co-occurrence counts are taken.

- `w2v_word_to_idx::Dict{String,Int}`  
  Mapping from word strings to their indices in the Word2Vec embedding matrix.

- `V::Int`  
  Vocabulary size of the Word2Vec model (number of columns/rows in embedding matrix).

# Returns
- `SparseMatrixCSC{Float64,Int}`  
  A sparse `(V × 1)` vector representing the context of `word` in Word2Vec index space.  
  If `word` is not in `scm` or has no nonzero co-occurrences, returns `spzeros(V,1)`.

# Notes
- Only words that exist both in `scm` and in `w2v_word_to_idx` contribute nonzero entries.  
- The returned vector is suitable for computing ConEc embeddings:  
  `y_w = W0 * c_w`, where `W0` is the Word2Vec embedding matrix.
- This function is intended for internal use.
"""
function _context_vector_for_word(
    word::String,
    scm::SparseContextMatrix{T},
    w2v_word_to_idx::Dict{String,Int},
    V::Int,
) where {T<:Real}
    col_idx = get(scm.token_to_id, word, nothing)
    col_idx === nothing && return spzeros(T, V, 1)
    
    col = scm.mat[:, col_idx]
    nnz_col = nnz(col)
    nnz_col == 0 && return spzeros(T, V, 1)

    rows_w2v = Vector{Int}(undef, nnz_col)
    vals     = Vector{T}(undef, nnz_col)
    count = 0

    @inbounds for k in nzrange(col, 1)
        r = rowvals(col)[k]
        v = nonzeros(col)[k]

        idx_w2v = get(w2v_word_to_idx, scm.vocab[r], 0)
        idx_w2v == 0 && continue

        count += 1
        rows_w2v[count] = idx_w2v
        vals[count]     = v
    end

    count == 0 && return spzeros(T, V, 1)

    return sparse(rows_w2v[1:count], ones(Int, count), vals[1:count], V, 1)
end


"""
    conec_embeddings_for_file(model, local_path; window_size=5, min_count=1) -> Dict{String, Vector{Float64}}

Compute ConEc embeddings for all words occurring in a local corpus file.

# Arguments
- `model::ConEcModel`  
  A ConEc model containing a pre-trained Word2Vec model and a global context matrix.

- `local_path::AbstractString`  
  Path to a text file containing the local corpus for which embeddings are computed.

- `window_size::Int=5`  
  Symmetric context window size used to build the local context matrix.

- `min_count::Int=1`  
  Minimum frequency threshold; tokens with fewer occurrences in the local corpus are ignored.

# Returns
- `Dict{String, Vector{Float64}}`  
  Mapping from words in the local corpus to their ConEc embeddings (vectors of the same dimension as the Word2Vec embeddings).  
  Words that do not occur in either the local or global context are omitted.

# Notes
- For each word, the context vector is computed as:

      c_w = a * c_w^global + (1 - a) * c_w^local

  where `a` is the weight of the global context stored in `model.a`. If the word has no global context, `c_w = c_w^local`.

- The final embedding is computed via matrix multiplication with the Word2Vec embedding matrix `W0`:

      y_w = W0 * c_w

- Returns an empty dictionary if the local corpus has no tokens surviving `min_count`.
"""
function conec_embeddings_for_file(
    model::ConEcModel,
    local_path::AbstractString;
    window_size::Int = 5,
    min_count::Int = 1,
)
    # build local context
    local_cm = SparseContextMatrix(local_path; window_size=window_size, min_count=min_count)
    isempty(local_cm.vocab) && return Dict{String, Vector{Float64}}()

    V = length(model.w2v.vocab)
    E = model.w2v.embeddings                  
    w2v_word_to_idx = model.w2v.word_to_index
    a = model.a

    conec = Dict{String, Vector{Float64}}()

    for word in local_cm.vocab
        c_local  = _context_vector_for_word(word, local_cm, w2v_word_to_idx, V)
        c_global = _context_vector_for_word(word, model.global_cm, w2v_word_to_idx, V)

        c = nnz(c_global) > 0 ? a .* c_global .+ (1-a) .* c_local : c_local
        nnz(c) == 0 && continue

        conec[word] = vec(E * c)
    end

    return conec
end

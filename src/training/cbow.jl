export train_cbow

using Word2Vec
using Random
using LinearAlgebra: mul!


"""
    train_cbow(corpus; dim=50, window=2, epochs=5, lr=0.05, min_count=1, seed=42, verbose=false) -> Word2VecModel

Train a Word2Vec model using the **CBOW (Continuous Bag-of-Words)** objective.

CBOW predicts each *target* word from the average of its surrounding *context* word vectors
within a symmetric window. This implementation uses a full softmax with cross-entropy loss
and is intended for **toy/small corpora** (time per update is `O(|V|)`).

# Arguments
- `corpus`: Training text, provided as either
  - `Vector{<:AbstractString}`: a flat list of tokens, e.g. `split("the quick brown fox")`, or
  - `Vector{Vector{<:AbstractString}}`: a list of tokenized sentences.

# Keyword arguments
- `dim::Int=50`: Embedding dimensionality.
- `window::Int=2`: Context window size on each side of the target word.
- `epochs::Int=5`: Number of passes over the corpus.
- `lr::Float64=0.05`: Learning rate for SGD updates.
- `min_count::Int=1`: Discard words with frequency `< min_count`.
- `seed::Int=42`: RNG seed for reproducible training.
- `verbose::Bool=false`: If `true`, print average loss per epoch.

# Returns
A `Word2VecModel` whose `embeddings` matrix stores the learned **input** word vectors `W_in`
with shape `(dim, |vocab|)`.

# Notes
- This implementation uses **full softmax** (no negative sampling), so it is best suited for
  small vocabularies and demonstration purposes.
- Tokens filtered out by `min_count` are ignored during training.

# Throws
- ArgumentError("corpus is empty") if the corpus contains no tokens.
- ArgumentError("vocab is empty after min_count filtering") if all tokens are filtered out.
- ArgumentError if corpus is not a token vector or a vector of tokenized sentences.
"""
function train_cbow(
    corpus;
    dim::Int = 50,
    window::Int = 2,
    epochs::Int = 5,
    lr::Float64 = 0.05,
    min_count::Int = 1,
    seed::Int = 42,
    verbose::Bool = false,
)
    tokens = _flatten_corpus(corpus)
    isempty(tokens) && throw(ArgumentError("corpus is empty"))

    vocab, word_to_idx, idx_tokens = _build_vocab_and_encode(tokens; min_count=min_count)
    V = length(vocab)
    V == 0 && throw(ArgumentError("vocab is empty after min_count filtering"))

    rng = MersenneTwister(seed)

    # Params
    W_in  = 0.01 .* randn(rng, dim, V)
    W_out = 0.01 .* randn(rng, V, dim)

    # Buffers
    h      = zeros(Float64, dim)
    scores = zeros(Float64, V)
    probs  = zeros(Float64, V)
    grad_h = zeros(Float64, dim)

    n = length(idx_tokens)

    for ep in 1:epochs
        total_loss = 0.0
        count = 0

        order = collect(1:n)
        Random.shuffle!(rng, order)

        for pos in order
            target = idx_tokens[pos]
            target == 0 && continue

            ctx = _context_indices(idx_tokens, pos, window)
            isempty(ctx) && continue

            # h = mean(W_in[:, ctx])
            fill!(h, 0.0)
            @inbounds for j in ctx
                @views h .+= W_in[:, j]
            end
            h ./= length(ctx)

            # scores = W_out * h
            mul!(scores, W_out, h)

            _softmax!(probs, scores)

            # cross entropy loss
            total_loss += -log(probs[target] + 1e-12)
            count += 1

            # dscores = probs - onehot(target)  (reuse probs as dscores)
            @inbounds probs[target] -= 1.0

            # Update W_out: W_out[v,:] -= lr * dscores[v] * h
            @inbounds for v in 1:V
                dv = probs[v]
                if dv != 0.0
                    @views W_out[v, :] .-= lr .* (dv .* h)
                end
            end

            # grad_h = W_out' * dscores
            fill!(grad_h, 0.0)
            @inbounds for v in 1:V
                dv = probs[v]
                if dv != 0.0
                    @views grad_h .+= dv .* W_out[v, :]
                end
            end

            # Update context word vectors in W_in
            scale = lr / length(ctx)
            @inbounds for j in ctx
                @views W_in[:, j] .-= scale .* grad_h
            end
        end

        if verbose && count > 0
            println("epoch $ep/$epochs  avg_loss = $(total_loss / count)")
        end
    end

    return Word2VecModel(vocab, W_in)
end


"""
    _flatten_corpus(corpus) -> Vector{String}

Internal helper: normalize `corpus` into a single flat vector of `String` tokens.

Accepted inputs:
- `Vector{<:AbstractString}`: returned as `Vector{String}` via `String.(corpus)`.
- `Vector{Vector{<:AbstractString}}`: each sentence is converted and concatenated into one token stream.

This exists to support common inputs such as `split("...")`, which returns `Vector{SubString{String}}`.

# Returns
- A flat `Vector{String}` containing all tokens in order.

# Throws
- ArgumentError if corpus is neither a token vector nor a vector of tokenized sentences.
- ArgumentError if any element in the sentence list is not AbstractVector{<:AbstractString}.
"""
function _flatten_corpus(corpus)
    if corpus isa AbstractVector{<:AbstractString}
        return String.(corpus)
    elseif corpus isa AbstractVector
        out = String[]
        for sent in corpus
            sent isa AbstractVector{<:AbstractString} ||
                throw(ArgumentError("corpus must be Vector{String} or Vector{Vector{String}}"))
            append!(out, String.(sent))
        end
        return out
    else
        throw(ArgumentError("corpus must be Vector{String} or Vector{Vector{String}}"))
    end
end


"""
    _build_vocab_and_encode(tokens; min_count=1) -> (vocab, word_to_idx, idx_tokens)

Internal helper: build a vocabulary from a flat token stream and encode tokens as integer indices.

# Arguments
- `tokens::Vector{String}`: Flat token list.

# Keyword arguments
- `min_count::Int=1`: Only keep tokens with frequency `>= min_count`.

# Returns
- `vocab::Vector{String}`:
  Vocabulary list, sorted deterministically by decreasing frequency and then lexicographically.
- `word_to_idx::Dict{String,Int}`:
  Mapping from token to its 1-based index in `vocab`.
- `idx_tokens::Vector{Int}`:
  Encoded token stream. Tokens not in `vocab` are encoded as `0`.

# Notes
- This function does not throw if `vocab` becomes empty; callers may check `isempty(vocab)`.
"""
function _build_vocab_and_encode(tokens::Vector{String}; min_count::Int)
    counts = Dict{String, Int}()
    for w in tokens
        counts[w] = get(counts, w, 0) + 1
    end

    kept = [(w, c) for (w, c) in counts if c >= min_count]
    sort!(kept, by = x -> (-x[2], x[1])) 

    vocab = [wc[1] for wc in kept]
    word_to_idx = Dict{String, Int}(w => i for (i, w) in enumerate(vocab))

    idx_tokens = Vector{Int}(undef, length(tokens))
    for (i, w) in enumerate(tokens)
        idx_tokens[i] = get(word_to_idx, w, 0)
    end

    return vocab, word_to_idx, idx_tokens
end

"""
    _context_indices(idx_tokens, pos, window) -> Vector{Int}

Internal helper: collect the encoded context word indices around `pos`.

Context indices are taken from the range `[pos-window, pos+window]`, excluding `pos`.
Any token encoded as `0` (filtered/unknown) is skipped.

# Arguments
- `idx_tokens::Vector{Int}`: Encoded token stream (0 indicates filtered/unknown).
- `pos::Int`: Target position (1-based).
- `window::Int`: Window size on each side.

# Returns
- A `Vector{Int}` of context vocabulary indices (positive integers). May be empty.
"""
function _context_indices(idx_tokens::Vector{Int}, pos::Int, window::Int)
    n = length(idx_tokens)
    lo = max(1, pos - window)
    hi = min(n, pos + window)

    ctx = Int[]
    for i in lo:hi
        i == pos && continue
        wi = idx_tokens[i]
        wi == 0 && continue
        push!(ctx, wi)
    end
    return ctx
end

"""
    _softmax!(out, x) -> out

Internal helper: compute a numerically stable softmax of `x` into `out` in-place.

Uses the standard stabilization trick:
`softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))`

# Arguments
- `out::Vector{Float64}`: Output buffer (same length as `x`).
- `x::Vector{Float64}`: Input logits / scores.

# Returns
- The mutated `out` vector (for convenience).

# Notes
- Adds a small epsilon to the denominator to avoid division by zero in degenerate cases.
- Intended for inner-loop use to reduce allocations.
"""
function _softmax!(out::Vector{Float64}, x::Vector{Float64})
    m = maximum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        v = exp(x[i] - m)
        out[i] = v
        s += v
    end
    invs = 1.0 / (s + 1e-12)
    @inbounds for i in eachindex(out)
        out[i] *= invs
    end
    return out
end
"""
    train_cbow(corpus; dim=50, window=2, epochs=5, lr=0.05, min_count=1, seed=42, verbose=false)

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
        shuffle!(rng, order)

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
    _flatten_corpus(corpus::AbstractVector{<:AbstractString})

Convert a flat vector of tokens into a `Vector{String}`.

# Arguments
- `corpus::AbstractVector{<:AbstractString}`: A vector of tokens, which may be `SubString{String}` or any subtype of `AbstractString`.

# Returns
- `Vector{String}`: A new vector containing all tokens as `String` objects.

# Notes
- This method is used internally to ensure consistent string types before training or processing.
- Does not modify the original vector.
"""
_flatten_corpus(corpus::AbstractVector{<:AbstractString}) = String.(corpus)


"""
    _flatten_corpus(corpus::AbstractVector{<:AbstractVector{<:AbstractString}})

Flatten a vector of tokenized sentences into a single `Vector{String}`.

# Arguments
- `corpus::AbstractVector{<:AbstractVector{<:AbstractString}}`  
  A vector of sentences, where each sentence is a vector of tokens (e.g., `Vector{SubString{String}}`).

# Returns
- `Vector{String}`: A flat vector containing all tokens from all sentences in order, converted to `String`.

# Notes
- This is used internally to normalize input corpora before training.
- Preallocates the output vector for efficiency.
"""
function _flatten_corpus(corpus::AbstractVector{<:AbstractVector{<:AbstractString}})
    total_len = sum(length(sent) for sent in corpus)
    out = Vector{String}(undef, total_len)
    pos = 1
    for sent in corpus
        for tok in sent
            out[pos] = String(tok)
            pos += 1
        end
    end
    return out
end


"""
    _flatten_corpus(corpus)

Fallback method that throws an error for invalid corpus inputs.

# Arguments
- `corpus`: The input provided to `_flatten_corpus`.

# Throws
- `ArgumentError` if `corpus` is not a `Vector{String}` or a `Vector{Vector{String}}`.

# Notes
- Serves as a catch-all to ensure only supported corpus formats are processed.
"""
_flatten_corpus(corpus) = throw(
    ArgumentError("corpus must be Vector{String} or Vector{Vector{String}}")
)


"""
    _build_vocab_and_encode(tokens; min_count=1)

Internal helper: construct a vocabulary from a flat token vector and encode each token as an integer index.

# Arguments
- `tokens::Vector{String}`: Flat vector of tokens from a corpus.

# Keyword Arguments
- `min_count::Int=1`: Minimum frequency threshold; tokens with counts below this are excluded.

# Returns
- `vocab::Vector{String}`: Sorted list of tokens by decreasing frequency, then alphabetically.
- `word_to_idx::Dict{String,Int}`: Mapping from each token to its 1-based index in `vocab`.
- `idx_tokens::Vector{Int}`: Integer-encoded version of `tokens`, with tokens not in `vocab` encoded as `0`.

# Notes
- Does not throw if no tokens survive `min_count`; callers should check `isempty(vocab)`.
- Useful for converting text corpora into integer indices for Word2Vec training.
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
    _context_indices(idx_tokens, pos, window)

Internal helper: retrieve the indices of context words around a target position.

# Arguments
- `idx_tokens::Vector{Int}`: Encoded token sequence (0 for unknown/filtered tokens).
- `pos::Int`: 1-based index of the target token.
- `window::Int`: Symmetric window size around `pos`.

# Returns
- `Vector{Int}`: List of context token indices (positive integers), excluding the target.
  May be empty if no valid context tokens are present.
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
    _softmax!(out, x)

Internal helper: compute a numerically stable softmax of `x` in-place into `out`.

# Arguments
- `out::Vector{Float64}`: Preallocated output buffer (same length as `x`).
- `x::Vector{Float64}`: Input scores or logits.

# Returns
- `out::Vector{Float64}`: The softmax-normalized probabilities (same buffer, mutated in-place).

# Notes
- Implements `softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))` to improve numerical stability.
- Adds a small epsilon to the denominator to avoid division by zero.
- Designed for inner-loop performance to minimize allocations.
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
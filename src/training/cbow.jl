export train_cbow

using Random
using LinearAlgebra: mul!

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
    tokens =  (corpus)
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
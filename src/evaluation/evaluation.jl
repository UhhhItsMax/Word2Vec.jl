using LinearAlgebra

export similarity, analogy

function cosine_similarity(v1, v2)
    dot(v1, v2) / (norm(v1) * norm(v2))
end

function similarity(model::Word2VecModel, w1::AbstractString, w2::AbstractString)
    haskey(model.word_to_index, w1) ||
        throw(KeyError("Word '$w1' not found in Word2Vec model"))
    haskey(model.word_to_index, w2) ||
        throw(KeyError("Word '$w2' not found in Word2Vec model"))

    i1 = model.word_to_index[w1]
    i2 = model.word_to_index[w2]

    return cosine_similarity(
        model.embeddings[:, i1],
        model.embeddings[:, i2],
    )
end

function analogy(model::Word2VecModel, a::AbstractString, b::AbstractString, c::AbstractString; topk::Int = 5)
    # Safety checks
    for w in (a, b, c)
        haskey(model.word_to_index, w) ||
            throw(KeyError("Word '$w' not found in Word2Vec model"))
    end

    ia = model.word_to_index[a]
    ib = model.word_to_index[b]
    ic = model.word_to_index[c]

    va = model.embeddings[:, ia]
    vb = model.embeddings[:, ib]
    vc = model.embeddings[:, ic]

    # Analogy vector: b - a + c
    target = vb - va + vc

    sims = Vector{Float64}(undef, length(model.vocab))

    @inbounds for i in eachindex(model.vocab)
        sims[i] = cosine_similarity(target, model.embeddings[:, i])
    end

    # Optionally exclude input words
    for i in (ia, ib, ic)
        sims[i] = -Inf
    end

    return model.vocab[sortperm(sims, rev = true)[1:topk]]
end
using LinearAlgebra

function cosine_similarity(v1, v2)
    dot(v1, v2) / (norm(v1) * norm(v2))
end

function similarity(model, w1, w2)
    i1 = findfirst(==(w1), model.vocab)
    i2 = findfirst(==(w2), model.vocab)
    cosine_similarity(model.embeddings[:, i1], model.embeddings[:, i2])
end

function analogy(model, a, b, c; topk=5)
    va = model.embeddings[:, findfirst(==(a), model.vocab)]
    vb = model.embeddings[:, findfirst(==(b), model.vocab)]
    vc = model.embeddings[:, findfirst(==(c), model.vocab)]

    target = vb - va + vc

    sims = [
        cosine_similarity(target, model.embeddings[:, i])
        for i in eachindex(model.vocab)
    ]

    sortperm(sims, rev=true)[1:topk] .|> i -> model.vocab[i]
end
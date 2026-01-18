using BenchmarkTools
using LinearAlgebra

# ------------------------------------------------------------
# Training benchmark
# ------------------------------------------------------------

"""
    benchmark_training(corpus; kwargs...)

Benchmark CBOW training time.

Returns a `BenchmarkTools.Trial`.
"""
function benchmark_training(corpus; kwargs...)
    @benchmark train_cbow($corpus; $(kwargs...))
end

# ------------------------------------------------------------
# Similarity sanity checks
# ------------------------------------------------------------

"""
    similarity_check(model, pairs)

Compute cosine similarity for word pairs.

# Arguments
- `pairs`: Vector of `(word1, word2)` tuples

# Returns
- `Dict{Tuple{String,String}, Float64}`
"""
function similarity_check(model, pairs)
    results = Dict{Tuple{String,String}, Float64}()
    for (w1, w2) in pairs
        results[(w1, w2)] = similarity(model, w1, w2)
    end
    results
end

# ------------------------------------------------------------
# Analogy evaluation
# ------------------------------------------------------------

"""
    analogy_accuracy(model, analogies; topk=5)

Evaluate analogy accuracy.

Each analogy is `(a, b, c, expected)` meaning:
`a : b :: c : expected`.
"""
function analogy_accuracy(model, analogies; topk=5)
    correct = 0
    for (a, b, c, expected) in analogies
        preds = analogy(model, a, b, c; topk=topk)
        expected in preds && (correct += 1)
    end
    correct / length(analogies)
end

# ------------------------------------------------------------
# CONEC comparison
# ------------------------------------------------------------

"""
    compare_conec(model, analogies)

Compare analogy accuracy before and after CONEC enrichment.
"""
function compare_conec(model, analogies)
    base = analogy_accuracy(model, analogies)
    conec_model = conec_embedding(model)
    conec = analogy_accuracy(conec_model, analogies)

    (
        base = base,
        conec = conec,
        improvement = conec - base
    )
end
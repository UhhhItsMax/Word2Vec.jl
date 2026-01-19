using BenchmarkTools
using Plots
using Word2Vec

export benchmark_cbow_for_dim, benchmark_cbow_for_epochs, benchmark_cbow_for_window, benchmark_model_quality


function _plot_benchmark(results::Dict{<:Integer, T}, x_axis::AbstractString) where {T}
    xs = sort(collect(keys(results)))

    ys = map(xs) do x
        v = results[x]
        v isa BenchmarkTools.Trial ? minimum(v).time / 1e6 : v
    end

    plot(
        xs,
        ys;
        xlabel = x_axis,
        ylabel = "Time (ms)",
        title = "CBOW benchmark: time vs $x_axis",
        marker = :circle,
        lw = 2,
        legend = false,
    )
end


"""
    benchmark_cbow(corpus;
                   dim=50, window=2, epochs=5,
                   lr=0.05, min_count=1, seed=42)

Benchmark a single CBOW training configuration.

Returns a `BenchmarkTools.Trial`.
"""
function _benchmark_cbow(
    corpus;
    dim::Int = 50,
    window::Int = 2,
    epochs::Int = 5,
    lr::Float64 = 0.05,
    min_count::Int = 1,
    seed::Int = 42,
)
    @benchmark train_cbow(
        $corpus;
        dim = $dim,
        window = $window,
        epochs = $epochs,
        lr = $lr,
        min_count = $min_count,
        seed = $seed,
        verbose = false,
    )
end


"""
    benchmark_cbow_for_epochs(corpus, epochs_list; kwargs...)

Benchmark CBOW training for multiple epoch values.

Returns a `Dict{Int, BenchmarkTools.Trial}` mapping
`epochs => trial`.
"""
function _benchmark_cbow_param(
    corpus,
    values::AbstractVector{<:Int},
    param::Symbol;
    dim::Int = 50,
    window::Int = 2,
    epochs::Int = 5,
    lr::Float64 = 0.05,
    min_count::Int = 1,
    seed::Int = 42,
)
    results = Dict{Int, BenchmarkTools.Trial}()

    for v in values
        @info "Benchmarking CBOW ($param = $v)"

        kwargs = (
            dim = dim,
            window = window,
            epochs = epochs,
            lr = lr,
            min_count = min_count,
            seed = seed,
        )

        # override exactly one parameter
        kwargs = merge(kwargs, NamedTuple{(param,)}((v,)))

        results[v] = _benchmark_cbow(corpus; kwargs...)
    end

    display(_plot_benchmark(results, String(param)))
    return results
end

benchmark_cbow_for_epochs(
    corpus,
    epochs_list::AbstractVector{<:Int} = [1, 2, 5, 10];
    kwargs...
) = _benchmark_cbow_param(corpus, epochs_list, :epochs; kwargs...)


benchmark_cbow_for_dim(
    corpus,
    dims::AbstractVector{<:Int} = [1, 10, 25, 50];
    kwargs...
) = _benchmark_cbow_param(corpus, dims, :dim; kwargs...)


benchmark_cbow_for_window(
    corpus,
    windows::AbstractVector{<:Int} = [1, 2, 5];
    kwargs...
) = _benchmark_cbow_param(corpus, windows, :window; kwargs...)


"""
    SimilarityTest(w1, w2; higher_than = [])

A qualitative similarity test.

Checks that the similarity between `w1` and `w2` is higher than the similarity
between `w1` and each word in `higher_than`.
"""
struct SimilarityTest
    w1::String
    w2::String
    higher_than::Vector{String}
end


"""
    AnalogyTest(a, b, c, expected)

Analogy test of the form `a : b ≈ c : ?`.

`expected` may contain multiple valid answers.
"""
struct AnalogyTest
    a::String
    b::String
    c::String
    expected::Vector{String}
end


"""
    evaluate_model_quality(
        model::Word2VecModel;
        similarity_tests = SimilarityTest[],
        analogy_tests = AnalogyTest[],
        topk::Int = 5,
    )

Evaluate the qualitative performance of a Word2Vec model.

Returns a NamedTuple with pass rates and raw results.
"""
function benchmark_model_quality(
    model::Word2VecModel;
    similarity_tests::Vector{SimilarityTest} = SimilarityTest[],
    analogy_tests::Vector{AnalogyTest} = AnalogyTest[],
    topk::Int = 5,
)
    sim_pass = 0
    sim_total = length(similarity_tests)

    sim_results = Vector{Bool}()

    for t in similarity_tests
        try
            s_ref = similarity(model, t.w1, t.w2)
            passed = true

            for w in t.higher_than
                s_cmp = similarity(model, t.w1, w)
                if s_ref <= s_cmp
                    passed = false
                    break
                end
            end

            push!(sim_results, passed)
            sim_pass += passed ? 1 : 0
        catch
            push!(sim_results, false)
        end
    end

    ana_pass = 0
    ana_total = length(analogy_tests)

    ana_results = Vector{Bool}()

    for t in analogy_tests
        try
            preds = analogy(model, t.a, t.b, t.c; topk = topk)
            passed = any(p in t.expected for p in preds)

            push!(ana_results, passed)
            ana_pass += passed ? 1 : 0
        catch
            push!(ana_results, false)
        end
    end

    return (
        similarity = (
            passed = sim_pass,
            total = sim_total,
            accuracy = sim_total == 0 ? NaN : sim_pass / sim_total,
            results = sim_results,
        ),
        analogy = (
            passed = ana_pass,
            total = ana_total,
            accuracy = ana_total == 0 ? NaN : ana_pass / ana_total,
            results = ana_results,
        ),
    )
end

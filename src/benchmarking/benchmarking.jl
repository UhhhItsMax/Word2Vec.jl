
using BenchmarkTools
using Plots

export benchmark_cbow_for_dim, benchmark_cbow_for_epochs, benchmark_cbow_for_window, benchmark_model_quality


"""
    _plot_benchmark(results::Dict{<:Integer, T}, x_axis::AbstractString) where {T}

Plot benchmarking results for CBOW training time.

# Arguments
- `results::Dict{<:Integer, T}`: Mapping from an integer parameter (e.g., embedding dimension, window size, or number of epochs) to either:
    - a numeric value representing the time in milliseconds, or
    - a `BenchmarkTools.Trial` object, in which case the minimum runtime is extracted.
- `x_axis::AbstractString`: Label for the x-axis (e.g., `"dimension"`, `"window size"`, `"epochs"`).

# Behavior
- Converts `results` to two sorted vectors `(xs, ys)` where `xs` are the sorted keys and `ys` are the corresponding times in milliseconds.
- Plots `ys` versus `xs` using a line with circular markers.
- Labels the x-axis as `x_axis`, y-axis as `"Time (ms)"`, and adds a title `"CBOW benchmark: time vs x_axis"`.

# Returns
- A `Plots.Plot` object.
"""
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
    _benchmark_cbow(corpus; dim=50, window=2, epochs=5, lr=0.05, min_count=1, seed=42)

Benchmark a single CBOW training run using `BenchmarkTools`.

# Arguments
- `corpus`: Path to the training text corpus or any data structure accepted by `train_cbow`.
- `dim::Int=50`: Embedding dimensionality.
- `window::Int=2`: Context window size.
- `epochs::Int=5`: Number of training epochs.
- `lr::Float64=0.05`: Learning rate.
- `min_count::Int=1`: Minimum token frequency to include in the vocabulary.
- `seed::Int=42`: Random seed for reproducibility.

# Returns
- `BenchmarkTools.Trial`: Object containing timing information for the CBOW training run.

# Notes
- Uses the `@benchmark` macro to capture execution time.
- Wraps `train_cbow` with the given parameters and disables verbose output.
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
    _benchmark_cbow_param(corpus, values::AbstractVector{<:Int}, param::Symbol;
                          dim=50, window=2, epochs=5, lr=0.05, min_count=1, seed=42)

Benchmark CBOW training over a range of integer values for a single hyperparameter.

# Arguments
- `corpus`: Path to the training text corpus or any input accepted by `train_cbow`.
- `values::AbstractVector{<:Int}`: The integer values to test for the parameter.
- `param::Symbol`: The name of the parameter to vary (e.g., `:epochs`, `:dim`, `:window`).

# Keyword Arguments
- `dim::Int=50`: Embedding dimensionality.
- `window::Int=2`: Context window size.
- `epochs::Int=5`: Number of training epochs.
- `lr::Float64=0.05`: Learning rate.
- `min_count::Int=1`: Minimum token frequency.
- `seed::Int=42`: Random seed for reproducibility.

# Returns
- `Dict{Int, BenchmarkTools.Trial}`: Mapping each tested value to its corresponding benchmark trial.

# Notes
- Overrides only the parameter specified by `param`; other CBOW parameters remain at their defaults.
- Automatically plots a benchmark figure with `results` versus the tested parameter.
- Useful for systematically testing CBOW performance across different hyperparameter settings.
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


"""
    benchmark_cbow_for_epochs(corpus, epochs_list::AbstractVector{<:Int}=[1,2,5,10]; kwargs...)

Benchmark CBOW training across multiple epoch values.

# Arguments
- `corpus`: Path to the training text corpus or any input accepted by `train_cbow`.
- `epochs_list::AbstractVector{<:Int}=[1,2,5,10]`: The epoch counts to test.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `dim`, `window`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, BenchmarkTools.Trial}`: Mapping `epochs => trial`.

# Notes
- Plots training time versus epochs automatically.
- Useful for evaluating how CBOW training time scales with number of epochs.
"""
benchmark_cbow_for_epochs(
    corpus,
    epochs_list::AbstractVector{<:Int} = [1, 2, 5, 10];
    kwargs...
) = _benchmark_cbow_param(corpus, epochs_list, :epochs; kwargs...)


"""
    benchmark_cbow_for_dim(corpus, dims::AbstractVector{<:Int}=[1,10,25,50]; kwargs...)

Benchmark CBOW training across multiple embedding dimensionalities.

# Arguments
- `corpus`: Path to the training text corpus or any input accepted by `train_cbow`.
- `dims::AbstractVector{<:Int}=[1,10,25,50]`: The embedding dimensions to test.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `window`, `epochs`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, BenchmarkTools.Trial}`: Mapping `dim => trial`.

# Notes
- Plots training time versus embedding dimension automatically.
- Useful for evaluating how CBOW training time scales with embedding size.
"""
benchmark_cbow_for_dim(
    corpus,
    dims::AbstractVector{<:Int} = [1, 10, 25, 50];
    kwargs...
) = _benchmark_cbow_param(corpus, dims, :dim; kwargs...)


"""
    benchmark_cbow_for_window(corpus, windows::AbstractVector{<:Int}=[1,2,5]; kwargs...)

Benchmark CBOW training across multiple context window sizes.

# Arguments
- `corpus`: Path to the training text corpus or any input accepted by `train_cbow`.
- `windows::AbstractVector{<:Int}=[1,2,5]`: The context window sizes to test.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `dim`, `epochs`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, BenchmarkTools.Trial}`: Mapping `window => trial`.

# Notes
- Plots training time versus context window size automatically.
- Useful for evaluating how CBOW training time scales with the size of the context window.
"""
benchmark_cbow_for_window(
    corpus,
    windows::AbstractVector{<:Int} = [1, 2, 5];
    kwargs...
) = _benchmark_cbow_param(corpus, windows, :window; kwargs...)


"""
    SimilarityTest

Represents a **qualitative similarity test** between words.

# Fields
- `w1::String` — The reference word.
- `w2::String` — The target word that should be more similar to `w1`.
- `higher_than::Vector{String}` — A list of words that `w2` should be more similar to than.

# Notes
- Used in embedding evaluation to assert that `w1` is closer to `w2` than to any word in `higher_than`.
- Typically combined with a function that computes word similarities and verifies the test.
"""
struct SimilarityTest
    w1::String
    w2::String
    higher_than::Vector{String}
end


"""
    AnalogyTest

Represents a **word analogy test** of the form `a : b ≈ c : ?`.

# Fields
- `a::String` — The first word in the analogy pair.
- `b::String` — The second word in the analogy pair.
- `c::String` — The reference word for which we want to find the analogous word.
- `expected::Vector{String}` — One or more valid answers that correctly complete the analogy.

# Notes
- Used to evaluate word embeddings by testing whether vector arithmetic captures semantic or syntactic relationships.
- Example: if `a = "king"`, `b = "queen"`, `c = "man"`, then `expected = ["woman"]`.
"""
struct AnalogyTest
    a::String
    b::String
    c::String
    expected::Vector{String}
end


"""
    benchmark_model_quality(
        model::Word2VecModel;
        similarity_tests = SimilarityTest[],
        analogy_tests = AnalogyTest[],
        topk::Int = 5,
    ) -> NamedTuple

Evaluate the **qualitative performance** of a Word2Vec model using similarity and analogy tests.

# Arguments
- `model::Word2VecModel` — The trained Word2Vec model to evaluate.
- `similarity_tests::Vector{SimilarityTest}` — A list of pairwise similarity tests. Each test checks that the similarity of `w1` and `w2` is higher than a set of other words.
- `analogy_tests::Vector{AnalogyTest}` — A list of analogy tests of the form `a : b ≈ c : ?`. Each test may have multiple valid expected answers.
- `topk::Int=5` — Number of top predictions to consider for analogy tests.

# Returns
A `NamedTuple` with two fields: `:similarity` and `:analogy`, each containing:
- `passed::Int` — Number of tests passed.
- `total::Int` — Total number of tests.
- `accuracy::Float64` — Fraction of tests passed.
- `results::Vector{Bool}` — Individual pass/fail results for each test.

# Notes
- Any errors during evaluation are counted as failures.
- Useful for quick, qualitative checks of embedding quality.
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

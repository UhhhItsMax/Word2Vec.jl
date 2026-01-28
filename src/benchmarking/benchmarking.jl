

"""
    _plot_benchmark(results::Dict{<:Integer, T}, x_axis::AbstractString; mode::Symbol=:cbow) where {T}

Plot benchmarking results for CBOW or ConEc training times.

The function converts a dictionary of results into sorted vectors of keys and corresponding times,
then plots the times versus the keys using a line plot with circular markers. The plot title and
y-axis label are determined by the `mode`.

# Arguments
- `results::Dict{<:Integer, T}`: Maps an integer parameter (e.g., embedding dimension, window size, 
  or number of epochs) to either:
    - a numeric value representing time in milliseconds, or
    - a `Trial` object, from which the minimum runtime is extracted.
- `x_axis::AbstractString`: Label for the x-axis (e.g., `"dimension"`, `"window size"`, `"epochs"`).
- `mode::Symbol = :cbow`: Determines plot labels and title. Use `:cbow` for CBOW benchmarks, 
  `:conec` for ConEc benchmarks.

# Returns
- `Plots.plot`: A plot of benchmarking times versus the x-axis parameter.
"""
function _plot_benchmark(results::Dict{<:Integer, T}, x_axis::AbstractString; mode::Symbol = :cbow) where {T}
    xs = sort(collect(keys(results)))

    ys = map(xs) do x
        v = results[x]
        v isa Trial ? minimum(v).time / 1e6 : v
    end

    title_text = mode == :cbow ? "CBOW benchmark: time vs $x_axis" :
                 mode == :conec ? "ConEc benchmark: time vs $x_axis" :
                 "Benchmark: time vs $x_axis"

    plot(
        xs,
        ys;
        xlabel = x_axis,
        ylabel = "Time (ms)",
        title = title_text,
        marker = :circle,
        lw = 2,
        legend = false,
    )
end


"""
    _benchmark_cbow(corpus; dim=50, window=2, epochs=5, lr=0.05, min_count=1, seed=42)

Benchmark a single CBOW training run using `BenchmarkTools`.

Run `train_cbow` on the given `corpus` with specified hyperparameters and capture timing
information using the `@benchmark` macro.

# Arguments
- `corpus`: Training corpus, either as a file path or any structure accepted by `train_cbow`.
- `dim::Int=50`: Dimensionality of the embeddings.
- `window::Int=2`: Context window size.
- `epochs::Int=5`: Number of training epochs.
- `lr::Float64=0.05`: Learning rate.
- `min_count::Int=1`: Minimum token frequency to include in the vocabulary.
- `seed::Int=42`: Random seed for reproducibility.

# Returns
- `Trial`: BenchmarkTools `Trial` object containing execution time statistics.

# Notes
- Disables verbose output during training.
- Uses the `@benchmark` macro to measure execution time accurately.
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

Benchmark CBOW training across a range of integer values for a single hyperparameter.

Run `_benchmark_cbow` repeatedly for each value in `values`, overriding only the specified
`param` while keeping other hyperparameters at their default or given values. Displays a
plot of benchmark times versus the tested parameter.

# Arguments
- `corpus`: Training corpus, either as a file path or any input accepted by `train_cbow`.
- `values::AbstractVector{<:Int}`: Integer values to test for the chosen hyperparameter.
- `param::Symbol`: Name of the parameter to vary (e.g., `:epochs`, `:dim`, `:window`).

# Keyword Arguments
- `dim::Int=50`: Embedding dimensionality.
- `window::Int=2`: Context window size.
- `epochs::Int=5`: Number of training epochs.
- `lr::Float64=0.05`: Learning rate.
- `min_count::Int=1`: Minimum token frequency for vocabulary inclusion.
- `seed::Int=42`: Random seed for reproducibility.

# Returns
- `Dict{Int, Trial}`: Maps each tested value to its corresponding benchmark `Trial`.

# Notes
- Only the specified `param` is overridden; all other parameters remain unchanged.
- Automatically generates a plot of results versus the tested parameter.
- Useful for systematically evaluating CBOW performance across different hyperparameter values.
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
    results = Dict{Int, Trial}()

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

Benchmark CBOW training over a range of epoch values.

Run `_benchmark_cbow_param` to measure CBOW training times for each value in `epochs_list`,
while forwarding all other keyword arguments to `train_cbow`. Automatically plots training
time versus epochs.

# Arguments
- `corpus`: Training corpus, either as a file path or any input accepted by `train_cbow`.
- `epochs_list::AbstractVector{<:Int}=[1,2,5,10]`: Epoch counts to benchmark.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `dim`, `window`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, Trial}`: Maps each epoch count to its corresponding benchmark `Trial`.

# Notes
- Automatically generates a plot of training time versus epochs.
- Useful for assessing how CBOW training time scales with the number of epochs.
"""
benchmark_cbow_for_epochs(
    corpus,
    epochs_list::AbstractVector{<:Int} = [1, 2, 5, 10];
    kwargs...
) = _benchmark_cbow_param(corpus, epochs_list, :epochs; kwargs...)


"""
    benchmark_cbow_for_dim(corpus, dims::AbstractVector{<:Int}=[1,10,25,50]; kwargs...)

Benchmark CBOW training over a range of embedding dimensionalities.

Run `_benchmark_cbow_param` to measure CBOW training times for each value in `dims`,
while forwarding all other keyword arguments to `train_cbow`. Automatically plots training
time versus embedding dimension.

# Arguments
- `corpus`: Training corpus, either as a file path or any input accepted by `train_cbow`.
- `dims::AbstractVector{<:Int}=[1,10,25,50]`: Embedding dimensions to benchmark.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `window`, `epochs`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, Trial}`: Maps each embedding dimension to its corresponding benchmark `Trial`.

# Notes
- Automatically generates a plot of training time versus embedding dimension.
- Useful for assessing how CBOW training time scales with embedding size.
"""
benchmark_cbow_for_dim(
    corpus,
    dims::AbstractVector{<:Int} = [1, 10, 25, 50];
    kwargs...
) = _benchmark_cbow_param(corpus, dims, :dim; kwargs...)


"""
    benchmark_cbow_for_window(corpus, windows::AbstractVector{<:Int}=[1,2,5]; kwargs...)

Benchmark CBOW training over a range of context window sizes.

Run `_benchmark_cbow_param` to measure CBOW training times for each value in `windows`,
while forwarding all other keyword arguments to `train_cbow`. Automatically plots training
time versus context window size.

# Arguments
- `corpus`: Training corpus, either as a file path or any input accepted by `train_cbow`.
- `windows::AbstractVector{<:Int}=[1,2,5]`: Context window sizes to benchmark.

# Keyword Arguments
- All keyword arguments are forwarded to `train_cbow` (e.g., `dim`, `epochs`, `lr`, `min_count`, `seed`).

# Returns
- `Dict{Int, Trial}`: Maps each window size to its corresponding benchmark `Trial`.

# Notes
- Automatically generates a plot of training time versus context window size.
- Useful for assessing how CBOW training time scales with the size of the context window.
"""
benchmark_cbow_for_window(
    corpus,
    windows::AbstractVector{<:Int} = [1, 2, 5];
    kwargs...
) = _benchmark_cbow_param(corpus, windows, :window; kwargs...)


"""
    SimilarityTest

Represent a qualitative similarity test between words.

Each test asserts that the reference word `w1` is more similar to the target word `w2`
than to any word listed in `higher_than`.

# Fields
- `w1::String` — Reference word.
- `w2::String` — Target word expected to be more similar to `w1`.
- `higher_than::Vector{String}` — Words that `w2` should be more similar to than `w1`.

# Notes
- Used in embedding evaluation to verify relative word similarities.
- Typically paired with a function that computes word similarities and checks the test.
"""
struct SimilarityTest
    w1::String
    w2::String
    higher_than::Vector{String}
end


"""
    AnalogyTest

Represent a word analogy test of the form `a : b ≈ c : ?`.

Each test asserts that the relationship between `a` and `b` is analogous to the relationship
between `c` and one of the words in `expected`.

# Fields
- `a::String` — First word in the analogy pair.
- `b::String` — Second word in the analogy pair.
- `c::String` — Reference word for which the analogous word is sought.
- `expected::Vector{String}` — One or more valid answers that correctly complete the analogy.

# Notes
- Used to evaluate word embeddings by checking whether vector arithmetic captures semantic or syntactic relationships.
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
    )

Evaluate the qualitative performance of a Word2Vec model using similarity and analogy tests.

# Arguments
- `model::Word2VecModel` — Trained Word2Vec model to evaluate.
- `similarity_tests::Vector{SimilarityTest}` — List of pairwise similarity tests. Each test checks that `w1` is more similar to `w2` than to a set of other words.
- `analogy_tests::Vector{AnalogyTest}` — List of analogy tests of the form `a : b ≈ c : ?`. Each test may have multiple valid expected answers.
- `topk::Int=5` — Number of top predictions to consider for analogy tests.

# Returns
A `NamedTuple` with two fields, `:similarity` and `:analogy`, each containing:
- `passed::Int` — Number of tests passed.
- `total::Int` — Total number of tests.
- `accuracy::Float64` — Fraction of tests passed (`NaN` if no tests).
- `results::Vector{Bool}` — Individual pass/fail results for each test.

# Notes
- Any errors during evaluation are counted as failures.
- Provides a quick, qualitative check of embedding quality using user-defined similarity and analogy tests.
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


"""
    benchmark_conec_for_window(model::ConEcModel, local_path::String, windows::AbstractVector{<:Int}=[1,2,5])

Benchmark ConEc embedding computation across a range of context window sizes.

Run `conec_embeddings_for_file` for each value in `windows` and record computation times.
Automatically plots computation time versus window size.

# Arguments
- `model::ConEcModel` — ConEc model containing a pretrained CBOW model and global context.
- `local_path::String` — Path to the local corpus file.
- `windows::AbstractVector{<:Int}=[1,2,5]` — Context window sizes to benchmark.

# Returns
- `Dict{Int, Trial}` — Maps each window size to its corresponding benchmark `Trial`.
- Displays a plot of computation time versus window size.
"""
function benchmark_conec_for_window(model::ConEcModel, local_path::String, windows::AbstractVector{<:Int}=[1,2,5])
    results = Dict{Int, Trial}()

    for w in windows
        @info "Benchmarking ConEc (window = $w)"
        results[w] = @benchmark conec_embeddings_for_file($model, $local_path; window_size=$w)
    end

    display(_plot_benchmark(results, "window size"; mode=:conec))
    return results
end


"""
    benchmark_conec_for_local_corpus_size(model::ConEcModel, local_paths::Vector{String})

Benchmark ConEc embedding computation across multiple local corpora of varying sizes.

Run `conec_embeddings_for_file` for each file in `local_paths` and record computation times.

# Arguments
- `model::ConEcModel` — ConEc model with pretrained CBOW and global context.
- `local_paths::Vector{String}` — Paths to local corpus files to benchmark.

# Returns
- `Dict{String, Trial}` — Maps each local corpus file to its corresponding benchmark `Trial`.
"""
function benchmark_conec_for_local_corpus_size(model::ConEcModel, local_paths::Vector{String})
    results = Dict{String, Trial}()

    for path in local_paths
        @info "Benchmarking ConEc on corpus: $path"
        results[path] = @benchmark conec_embeddings_for_file($model, $path)
    end

    return results
end


"""
    benchmark_conec_for_dim(models::Vector{ConEcModel}, local_path::String, dims::Vector{Int}=[50,100,200])

Benchmark ConEc embedding computation for CBOW models with varying embedding dimensions.

Run `conec_embeddings_for_file` for each model in `models` and record computation times.
Automatically plots computation time versus embedding dimension.

# Arguments
- `models::Vector{ConEcModel}` — List of ConEc models trained with different CBOW dimensions.
- `local_path::String` — Path to the local corpus file.
- `dims::Vector{Int}=[50,100,200]` — Embedding dimensions corresponding to each model.

# Returns
- `Dict{Int, Trial}` — Maps each embedding dimension to its corresponding benchmark `Trial`.
- Displays a plot of computation time versus embedding dimension.
"""
function benchmark_conec_for_dim(models::Vector{ConEcModel}, local_path::String, dims::Vector{Int}=[50,100,200])
    results = Dict{Int, Trial}()

    for (model, dim) in zip(models, dims)
        @info "Benchmarking ConEc (dim = $dim)"
        results[dim] = @benchmark conec_embeddings_for_file($model, $local_path)
    end

    display(_plot_benchmark(results, "embedding dimension"; mode=:conec))
    return results
end
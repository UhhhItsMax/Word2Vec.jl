using BenchmarkTools
using Plots
using Word2Vec


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
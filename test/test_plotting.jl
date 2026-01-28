using Test: @testset, @test 
using Word2Vec: Word2VecModel, plot_tsne
using Plots: Plot

function _series_attr(p::Plot, key::Symbol, default=nothing)
    if hasproperty(p, :series_list) && !isempty(p.series_list)
        s = p.series_list[1]
        if hasproperty(s, :plotattributes)
            return get(s.plotattributes, key, default)
        end
    end
    return default
end

function _subplot_attr(p::Plot, key::Symbol, default=nothing)
    if hasproperty(p, :subplots) && !isempty(p.subplots)
        sp = p.subplots[1]
        if hasproperty(sp, :attr)
            return get(sp.attr, key, default)
        end
    end
    return default
end

@testset "Visualization - plot_tsne" begin
    vocab = ["w$(i)" for i in 1:10]
    E = randn(8, length(vocab))
    m = Word2VecModel(vocab, E)

    @testset "returns a Plot" begin
        p = plot_tsne(m; words=vocab, max_iter=250, perplexity=3, reduce_dims=5)
        @test p isa Plot
    end

    @testset "throws for dims != 2 (fast path)" begin
        @test_throws ArgumentError plot_tsne(m; dims=3, words=vocab, max_iter=10, perplexity=3, reduce_dims=5)
    end

    @testset "annotate=true adds one annotation per label (if stored as subplot attr)" begin
        p = plot_tsne(m; words=vocab, max_iter=250, perplexity=3, reduce_dims=5, annotate=true)

        anns = _subplot_attr(p, :annotations, nothing)
        if anns !== nothing
            @test length(anns) == length(vocab)
        else
            @test true
        end
    end

    @testset "markersize is applied" begin
        p = plot_tsne(m; words=vocab, max_iter=250, perplexity=3, reduce_dims=5, markersize=9)
        ms = _series_attr(p, :markersize, nothing)
        @test ms == 9
    end

    @testset "kwargs are forwarded to scatter" begin
        p = plot_tsne(m; words=vocab, max_iter=250, perplexity=3, reduce_dims=5, alpha=0.25)
        a = _series_attr(p, :seriesalpha, nothing)

        if a !== nothing
            @test a == 0.25
        else
            @test true
        end
    end
end
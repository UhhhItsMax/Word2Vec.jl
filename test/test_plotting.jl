using Test
using Word2Vec
using Plots

@testset "Visualization - plot_tsne" begin
    vocab = ["a","b","c","d","e","f","g","h"]
    E = randn(5, length(vocab))
    m = Word2VecModel(vocab, E)

    @testset "returns a Plots.Plot" begin
        words = vocab[1:6]
        p = plot_tsne(m; words=words, seed=1, max_iter=200, perplexity=2)
        @test p isa Plots.Plot
    end

    @testset "throws for dims != 2 (fast path)" begin
        @test_throws ArgumentError plot_tsne(m; dims=3, words=vocab[1:5])
    end

    @testset "annotate=true adds one annotation per label" begin
        words = vocab[1:5]
        p = plot_tsne(m; words=words, annotate=true, seed=1, max_iter=200, perplexity=2)

        anns = p.subplots[1].annotations
        @test length(anns) == length(words)
    end

    @testset "markersize is applied" begin
        words = vocab[1:5]
        p = plot_tsne(m; words=words, markersize=9, seed=1, max_iter=200, perplexity=2)

        ms = p.series_list[1].plotattributes[:markersize]
        @test ms == 9
    end

    @testset "kwargs are forwarded to scatter" begin
        p = plot_tsne(m; words=vocab[1:5], title="My TSNE", seed=1, max_iter=200, perplexity=2)
        @test p.subplots[1].attr[:title] == "My TSNE"
    end
end
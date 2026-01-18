using Test
using Word2Vec

@testset "Visualization - plot_tsne" begin
    vocab = ["a","b","c"]
    E = randn(5, 3)
    m = Word2VecModel(vocab, E)

    p = plot_tsne(m; words=vocab, max_iter=250, perplexity=5)
    @test p !== nothing
end
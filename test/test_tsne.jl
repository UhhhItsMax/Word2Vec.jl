using Test
using Word2Vec

@testset "Visualization - embedding_points" begin
    vocab = ["a", "b", "c"]
    E = [1.0  0.0  0.1;
         0.0  2.0  0.2]
    m = Word2VecModel(vocab, E)

    X, labels = embedding_points(m)
    @test size(X) == (3, 2)
    @test labels == vocab

    X2, labels2 = embedding_points(m; words=["b", "x", "a"])
    @test labels2 == ["b", "a"]
    @test size(X2) == (2, 2)

    X3, labels3 = embedding_points(m; words=["doesnotexist"])
    @test isempty(labels3)
    @test size(X3) == (0, 2)

    Xn, _ = embedding_points(m; normalize=true)
    @test isapprox(norm(Xn[1, :]), 1.0; atol=1e-12)
    @test isapprox(norm(Xn[2, :]), 1.0; atol=1e-12)
    @test isapprox(norm(Xn[3, :]), 1.0; atol=1e-12)
end

@testset "Visualization - tsne_embeddings" begin
    vocab = ["a", "b", "c"]
    E = randn(5, 3)
    m = Word2VecModel(vocab, E)

    Y, labels = tsne_embeddings(m; dims=2, max_iter=250, perplexity=5, seed=1)
    @test labels == vocab
    @test size(Y) == (3, 2)
    @test all(isfinite, Y)

end
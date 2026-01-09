using Test
using Word2Vec

@testset "CBOW" begin
    tokens = split("the quick brown fox jumps over the lazy dog the fox is quick")
    m = train_cbow(tokens; dim=10, window=2, epochs=2, lr=0.05, seed=1)

    @test m isa Word2VecModel
    @test length(m.vocab) > 0
    @test size(m.embeddings) == (10, length(m.vocab))
    @test all(isfinite, m.embeddings)

    @test haskey(m.word_to_index, "fox")
    v = get_embedding(m, "fox")
    @test size(v) == (10,)

    m1 = train_cbow(tokens; dim=8, window=2, epochs=2, lr=0.05, seed=123)
    m2 = train_cbow(tokens; dim=8, window=2, epochs=2, lr=0.05, seed=123)
    @test m1.vocab == m2.vocab
    @test isapprox(m1.embeddings, m2.embeddings; atol=1e-12, rtol=0)

    sents = [split("hello world hello"), split("world of julia")]
    m3 = train_cbow(sents; dim=6, window=1, epochs=1, lr=0.05, seed=7)
    @test m3 isa Word2VecModel
    @test size(m3.embeddings) == (6, length(m3.vocab))

    redirect_stdout(devnull) do
        train_cbow(tokens; dim=5, window=1, epochs=1, lr=0.05, seed=1, verbose=true)
    end
    @test true

    m4 = train_cbow(["solo"]; dim=4, window=2, epochs=1, lr=0.05, seed=1)
    @test m4 isa Word2VecModel
    @test size(m4.embeddings) == (4, length(m4.vocab))

    toks2 = ["a","a","b"]
    m5 = train_cbow(toks2; dim=4, window=1, epochs=1, lr=0.05, seed=1, min_count=2)
    @test "a" in m5.vocab
    @test !("b" in m5.vocab)

    @test_throws ArgumentError train_cbow(String[])
    @test_throws ArgumentError train_cbow(tokens; min_count=10^9)
end
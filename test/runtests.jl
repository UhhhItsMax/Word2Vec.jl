using Word2Vec
using Test

@testset "Word2Vec.jl" begin

end

@testset "pretrained loader" begin
    path = joinpath(@__DIR__, "data", "small_model.txt")
    emb = Word2Vec.load_word2vec(path)
    model = Word2Vec.from_pretrained(emb)

    @test length(model.vocab) == 3          # vocab_size
    @test size(model.embeddings, 1) == 5   # vector_size / embedding dimension

    @testset "king embedding first element" begin
        idx = findfirst(==("king"), model.vocab)
        @test isapprox(model.embeddings[1, idx], 0.1; atol=1e-6)
    end

    @testset "queen vector length" begin
        idx = findfirst(==("queen"), model.vocab)
        @test size(model.embeddings, 1) == 5  # embedding dimension
    end
end

@testset "pretrained gensim loader txt file" begin
    path = joinpath(@__DIR__, "data", "word2vec.txt")
    emb = Word2Vec.load_word2vec(path)
    
    @test length(keys(emb)) > 0  # basic check: embeddings are non-empty

    model = Word2Vec.from_pretrained(emb)

    # check dimensions
    first_word = first(model.vocab)
    idx = findfirst(==(first_word), model.vocab)
    @test size(model.embeddings, 1) == 100
    @test isapprox(model.embeddings[1, idx], Float32(emb[first_word][1]); atol=1e-6)

    for w in ["human", "graph", "computer"]
        if haskey(emb, w)
            idx = findfirst(==(w), model.vocab)
            @test all(.!isnan.(model.embeddings[:, idx]))
        end
    end
end

@testset "pretrained gensim loader bin file" begin
    path = joinpath(@__DIR__, "data", "word2vec.bin")
    emb = Word2Vec.load_word2vec(path)

    @test length(keys(emb)) > 0  # basic check: embeddings are non-empty
    
    model = Word2Vec.from_pretrained(emb)

    # check dimensions
    first_word = first(model.vocab)
    idx = findfirst(==(first_word), model.vocab)
    @test size(model.embeddings, 1) == 100
    @test isapprox(model.embeddings[1, idx], Float32(emb[first_word][1]); atol=1e-6)

    for w in ["human", "graph", "computer"]
        if haskey(emb, w)
            idx = findfirst(==(w), model.vocab)
            @test all(.!isnan.(model.embeddings[:, idx]))
        end
    end
end
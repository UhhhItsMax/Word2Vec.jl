using Test
using LinearAlgebra: norm
using Word2Vec

@testset "Word2VecModel constructor" begin
    @testset "builds lookup structures and norms" begin
        vocab = ["one", "two"]
        embeddings = reshape(Float64[2, 2, 1, -2, 0, 0], 3, 2)

        model = Word2Vec.Word2VecModel(vocab, embeddings)

        @test model.word_to_index == Dict("one" => 1, "two" => 2)
        @test model.vector_norms ≈ [3.0, 2.0]
    end

    @testset "rejects mismatched shapes" begin
        vocab = ["one", "two"]
        bad_embeddings = ones(Float64, 3, 3)

        @test_throws ArgumentError Word2Vec.Word2VecModel(vocab, bad_embeddings)
    end

    @testset "rejects zero vectors" begin
        vocab = ["nonzero", " zero"]
        embeddings = Float64[0 1; 0 0]

        @test_throws ArgumentError Word2Vec.Word2VecModel(vocab, embeddings)
    end
end


@testset "embedding getters" begin
    vocab = ["one", "two"]
    embeddings = reshape(Float64[2, 2, 1, 1, 2, 3], 3, 2)
    model = Word2Vec.Word2VecModel(vocab, embeddings)

    @testset "by index returns view" begin
        emb = Word2Vec.get_embedding(model, "two")
        @test emb == Float64[1, 2, 3]
        @test parent(emb) === model.embeddings
    end

    @testset "correct vector norm" begin
        emb_norm = Word2Vec.get_embedding_norm(model, "one")
        @test emb_norm ≈ 3.0
    end

    @testset "unknown word throws" begin
        @test_throws KeyError Word2Vec.get_embedding(model, "missing")
    end
end

@testset "from_dict_data" begin
    embeddings_map = Dict(
        "alpha" => Float32[1, 0, 0.5],
        "beta" => Float32[-1, 2, 0],
    )

    model = Word2Vec.from_dict_data(embeddings_map)

    @test size(model.embeddings) == (3, 2)
    @test eltype(model.embeddings) === Float64
    @test Word2Vec.get_embedding(model, "alpha") == Float64[1, 0, 0.5]
    @test Word2Vec.get_embedding(model, "beta") == Float64[-1, 2, 0]
end

@testset "load_pretrained_model" begin
    data_dir = joinpath(@__DIR__, "data")
    bin_path = joinpath(data_dir, "word2vec.bin")
    txt_path = joinpath(data_dir, "word2vec.txt")

    @testset "binary embeddings" begin
        model = Word2Vec.load_pretrained_model(bin_path; fmt = :binary)

        @test size(model.embeddings) == (100, 12)
        @test model.vocab[1] == "system"

        system_vec = Word2Vec.get_embedding(model, "system")
        @test system_vec[1] ≈ -0.00053622725
        @test Word2Vec.get_embedding_norm(model, "system") ≈ norm(system_vec)
    end

    @testset "text embeddings auto-detected" begin
        model = Word2Vec.load_pretrained_model(txt_path)

        @test size(model.embeddings) == (100, 12)
        @test model.vocab[2] == "graph"

        graph_vec = Word2Vec.get_embedding(model, "graph")
        @test graph_vec[1] ≈ -0.0086196875
        @test graph_vec[3] ≈ 0.0051898835
    end
end

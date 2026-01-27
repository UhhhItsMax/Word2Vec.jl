using Test: @testset, @test, @test_throws 
using Word2Vec: cosine_similarity, similarity, Word2VecModel, analogy
using LinearAlgebra: dot

@testset "cosine_similarity" begin
    # Test 1: simple 2D vectors
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    @test isapprox(cosine_similarity(v1, v2), 0.0; atol=1e-8)

    # Test 2: same vector
    v = [1.0, 2.0, 3.0]
    @test isapprox(cosine_similarity(v, v), 1.0; atol=1e-8)

    # Test 3: opposite vector
    u = [-1.0, -2.0, -3.0]
    @test isapprox(cosine_similarity(v, u), -1.0; atol=1e-8)

    # Test 4: arbitrary vectors
    a = [1.0, 2.0]
    b = [2.0, 3.0]
    expected = dot(a, b) / (norm(a) * norm(b))
    @test isapprox(cosine_similarity(a, b), expected; atol=1e-8)

    # Test 5: zero vector triggers ArgumentError
    zero_vec = [0.0, 0.0]
    @test_throws ArgumentError cosine_similarity(v, zero_vec)
    @test_throws ArgumentError cosine_similarity(zero_vec, v)
    @test_throws ArgumentError cosine_similarity(zero_vec, zero_vec)
end

@testset "similarity" begin
    # normal embeddings ---
    vocab1 = ["king", "queen", "man", "woman"]
    embeddings1 = [
        1.0  0.9  0.5  0.4;
        0.5  0.6  0.3  0.2
    ]
    model1 = Word2VecModel(vocab1, embeddings1)

    # Normal similarity check
    @test similarity(model1, "king", "queen") ≈ 
        (dot(embeddings1[:,1], embeddings1[:,2]) / (norm(embeddings1[:,1])*norm(embeddings1[:,2])))

    # Words not in vocab
    @test_throws KeyError similarity(model1, "king", "prince")
    @test_throws KeyError similarity(model1, "prince", "queen")
end

@testset "analogy" begin

    # Dummy model
    vocab = ["king", "queen", "man", "woman", "prince"]
    embeddings = [
        1.0  0.9  0.5  0.4  0.95;
        0.5  0.6  0.3  0.2  0.45
    ]

    model = Word2VecModel(vocab, embeddings)

    # Normal analogy: king : queen ≈ man : ?
    top_word = analogy(model, "king", "queen", "man"; topk=1)
    @test top_word[1] == "woman"

    # Check topk
    top_words = analogy(model, "king", "queen", "man"; topk=2)
    @test length(top_words) == 2
    @test "king" ∉ top_words
    @test "queen" ∉ top_words
    @test "man" ∉ top_words

    # Words not in vocab
    @test_throws KeyError analogy(model, "king", "queen", "emperor")
    @test_throws KeyError analogy(model, "king", "prince", "emperor")
end
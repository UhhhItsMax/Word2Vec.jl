using Test
using Word2Vec

@testset "cbow.jl helpers" begin
    @testset "_flatten_corpus" begin
        # token vector (Vector{SubString{String}} from split)
        toks = split("the quick brown fox")
        flat = Word2Vec._flatten_corpus(toks)
        @test flat isa Vector{String}
        @test flat == ["the", "quick", "brown", "fox"]

        # sentence list
        sents = [split("hello world"), split("julia rocks")]
        flat2 = Word2Vec._flatten_corpus(sents)
        @test flat2 isa Vector{String}
        @test flat2 == ["hello", "world", "julia", "rocks"]

        # invalid type
        @test_throws ArgumentError Word2Vec._flatten_corpus(123)

        # bad sentence element
        bad = Any[split("ok sentence"), 123]
        @test_throws ArgumentError Word2Vec._flatten_corpus(bad)
    end

    @testset "_build_vocab_and_encode" begin
        tokens = ["a", "b", "a", "c", "b", "a"]

        vocab, w2i, idx = Word2Vec._build_vocab_and_encode(tokens; min_count=1)
        @test vocab isa Vector{String}
        @test w2i isa Dict{String,Int}
        @test idx isa Vector{Int}
        @test length(vocab) == length(w2i)
        @test length(idx) == length(tokens)

        # every kept word maps to valid indices
        @test all(1 .<= idx .<= length(vocab))

        # filtering with min_count
        vocab2, w2i2, idx2 = Word2Vec._build_vocab_and_encode(tokens; min_count=3) # only "a"
        @test vocab2 == ["a"]
        @test w2i2["a"] == 1
        @test idx2 == [1, 0, 1, 0, 0, 1]  # b,c filtered -> 0

        # empty after filtering should throw but will be handled in train_cbow
        vocab3, w2i3, idx3 = Word2Vec._build_vocab_and_encode(tokens; min_count=10^9)
        @test isempty(vocab3)
        @test isempty(w2i3)
        @test all(==(0), idx3)
    end

    @testset "_context_indices" begin
        idx_tokens = [1, 2, 3, 4, 5]

        # middle position, window=1
        ctx = Word2Vec._context_indices(idx_tokens, 3, 1)
        @test ctx == [2, 4]

        # edge position, window=2 (only right side exists)
        ctx2 = Word2Vec._context_indices(idx_tokens, 1, 2)
        @test ctx2 == [2, 3]

        # skips zeros
        idx_tokens2 = [1, 0, 3, 0, 5]
        ctx3 = Word2Vec._context_indices(idx_tokens2, 3, 2)
        @test ctx3 == [1, 5]

        # single element => empty context
        ctx4 = Word2Vec._context_indices([7], 1, 2)
        @test isempty(ctx4)
    end

        @testset "_softmax!" begin
        x = [0.0, 0.0, 0.0]
        out = zeros(Float64, 3)
        Word2Vec._softmax!(out, x)

        @test all(isfinite, out)
        @test isapprox(sum(out), 1.0; atol=1e-12)
        @test all(isapprox.(out, 1/3; atol=1e-12))

        # stability: large values
        x2 = [1000.0, 1001.0, 999.0]
        out2 = zeros(Float64, 3)
        Word2Vec._softmax!(out2, x2)

        @test all(isfinite, out2)
        @test isapprox(sum(out2), 1.0; atol=1e-12)
        @test argmax(out2) == 2
    end
end

@testset "CBOW training" begin
    @testset "happy path + shapes" begin
        tokens = split("the quick brown fox jumps over the lazy dog the fox is quick")
        m = train_cbow(tokens; dim=10, window=2, epochs=2, lr=0.05, seed=1)

        @test m isa Word2VecModel
        @test length(m.vocab) > 0
        @test size(m.embeddings) == (10, length(m.vocab))
        @test all(isfinite, m.embeddings)

        @test haskey(m.word_to_index, "fox")
        v = get_embedding(m, "fox")
        @test size(v) == (10,)
    end

    @testset "determinism (seed)" begin
        tokens = split("the quick brown fox jumps over the lazy dog the fox is quick")
        m1 = train_cbow(tokens; dim=8, window=2, epochs=2, lr=0.05, seed=123)
        m2 = train_cbow(tokens; dim=8, window=2, epochs=2, lr=0.05, seed=123)

        @test m1.vocab == m2.vocab
        @test isapprox(m1.embeddings, m2.embeddings; atol=1e-12, rtol=0)
    end

    @testset "sentence list input" begin
        sents = [split("hello world hello"), split("world of julia")]
        m = train_cbow(sents; dim=6, window=1, epochs=1, lr=0.05, seed=7)

        @test m isa Word2VecModel
        @test size(m.embeddings) == (6, length(m.vocab))
    end

    @testset "verbose branch (coverage)" begin
        tokens = split("a b a b a b a b")
        redirect_stdout(devnull) do
            train_cbow(tokens; dim=5, window=1, epochs=1, lr=0.05, seed=1, verbose=true)
        end
        @test true
    end

    @testset "empty context branch (single token)" begin
        m = train_cbow(["solo"]; dim=4, window=2, epochs=1, lr=0.05, seed=1)
        @test m isa Word2VecModel
        @test size(m.embeddings) == (4, length(m.vocab))
    end

    @testset "target==0 continue branch via min_count" begin
        toks = ["a", "a", "b"]
        m = train_cbow(toks; dim=4, window=1, epochs=1, lr=0.05, seed=1, min_count=2)
        @test "a" in m.vocab
        @test !("b" in m.vocab)
    end

    @testset "error branches" begin
        @test_throws ArgumentError train_cbow(String[])
        tokens = split("a b c")
        @test_throws ArgumentError train_cbow(tokens; min_count=10^9)
        @test_throws ArgumentError train_cbow(123)
        bad = Any[split("ok sentence"), 123]
        @test_throws ArgumentError train_cbow(bad)
    end
end
using Test: @testset, @test, @test_throws
using Word2Vec: save_word2vec, _save_word2vec_binary, _save_word2vec_text, Word2VecModel, load_binary_embeddings, load_word2vec

@testset "_save_word2vec_binary" begin

    # Construct a tiny model
    vocab = ["king", "queen"]
    embeddings = [
        0.1  0.2;
        0.3  0.4;
        0.5  0.6
    ]  # size = (dim=3, vocab_size=2)

    model = Word2VecModel(vocab, embeddings)

    mktempdir() do dir
        path = joinpath(dir, "test.bin")

        # Save
        _save_word2vec_binary(model, path)

        @test isfile(path)

        @testset "header correctness" begin
            open(path, "r") do io
                header = readline(io)
                @test header == "2 3"
            end
        end

        @testset "round-trip load" begin
            vocab2, emb2 = load_binary_embeddings(path)

            @test vocab2 == vocab
            @test size(emb2) == size(embeddings)

            # Values should match approximately (Float32 roundoff)
            @test emb2 ≈ embeddings atol=1f-6
        end

        @testset "binary float format" begin
            open(path, "r") do io
                readline(io)  # skip header

                # Read first word
                word_bytes = UInt8[]
                while true
                    c = read(io, UInt8)
                    c == 0x20 && break
                    push!(word_bytes, c)
                end
                word = String(word_bytes)
                @test word == "king"

                # Read raw vector bytes
                vec32 = Vector{Float32}(undef, 3)
                read!(io, vec32)

                @test eltype(vec32) == Float32
                @test vec32 ≈ Float32.(embeddings[:, 1])
            end
        end
    end
end


@testset "_save_word2vec_text" begin

    # Create a small model
    vocab = ["king", "queen", "man"]
    embeddings = [
        0.1 0.2 0.0;
        0.2 0.1 0.1;
        0.3 0.4 0.0;
        0.4 0.3 0.1;
        0.5 0.0 0.0
    ]  # 5 x 3 matrix
    model = Word2VecModel(vocab, embeddings)

    # Round-trip test using a temporary file
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "model.txt")
        _save_word2vec_text(model, path)

        # Load back using your loader
        model2 = load_word2vec(path)

        # Check vocab
        @test model2.vocab == vocab

        # Check embeddings
        @test size(model2.embeddings) == size(embeddings)
        @test model2.embeddings ≈ embeddings atol=1e-8  # approximate comparison for floating points
    end

    # Test header line is correct
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "model.txt")
        _save_word2vec_text(model, path)
        firstline = open(path) do io
            readline(io)
        end
        @test firstline == "$(length(vocab)) $(size(embeddings, 1))"
    end

    # Test that file is human-readable: contains the first word
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "model.txt")
        _save_word2vec_text(model, path)
        content = read(path, String)
        @test occursin("king", content)
    end

end


@testset "save_word2vec" begin

    txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
    txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")
    bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")

    @testset "round-trip text1 save/load" begin
        mktempdir() do d
            # Load original text model
            model = load_word2vec(txt_path_1)

            # Save to new temporary file in text format
            out_file = joinpath(d, "model.txt")
            save_word2vec(model, out_file; format=:text)

            # Reload
            model2 = load_word2vec(out_file)

            @test model2.vocab == model.vocab
            @test model2.embeddings ≈ model.embeddings  # approximate equality
        end
    end

    @testset "round-trip text2 save/load" begin
        mktempdir() do d
            # Load original text model
            model = load_word2vec(txt_path_2)

            # Save to new temporary file in text format
            out_file = joinpath(d, "model.txt")
            save_word2vec(model, out_file; format=:text)

            # Reload
            model2 = load_word2vec(out_file)

            @test model2.vocab == model.vocab
            @test model2.embeddings ≈ model.embeddings
        end
    end


    @testset "round-trip binary save/load" begin
        mktempdir() do d
            # Load original binary model
            model = load_word2vec(bin_path)

            # Save to new temporary file in binary format
            out_file = joinpath(d, "model.bin")
            save_word2vec(model, out_file; format=:binary)

            # Reload
            model2 = load_word2vec(out_file)

            @test model2.vocab == model.vocab
            @test model2.embeddings ≈ model.embeddings  # approximate equality for Float32 -> Float64 conversion
        end
    end

    @testset "invalid format raises error" begin
        mktempdir() do d
            out_file = joinpath(d, "bad_model.xyz")
            model = load_word2vec(txt_path_2)

            @test_throws ArgumentError save_word2vec(model, out_file; format=:foo)
        end
    end

end
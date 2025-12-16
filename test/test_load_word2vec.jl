using Test
using Word2Vec

@testset "load_word2vec" begin
    
    @testset "normal text models" begin
        txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
        txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")

        vocab, emb = Word2Vec.load_word2vec(txt_path_1)
        @test length(vocab) == 3
        @test vocab[1] == "king"
        @test emb[:, 1] == [0.1, 0.2, 0.3, 0.4, 0.5]
        @test vocab[2] == "queen"
        @test emb[:, 2] == [0.2, 0.1, 0.4, 0.3, 0.0]
        @test vocab[3] == "man"
        @test emb[:, 3] == [0.0, 0.1, 0.0, 0.1, 0.0]

        vocab, emb = Word2Vec.load_word2vec(txt_path_2)
        @test length(vocab) == 12
        @test vocab[1] == "system"
        @test emb[1, 1] == -0.00053622725
    end

    @testset "normal bin model" begin
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")

        vocab, emb = Word2Vec.load_binary_embeddings(bin_path)
        @test length(vocab) == 12
        @test vocab[1] == "system"
        @test emb[1, 1] == Float32(-0.00053622725)
    end

end

@testset "detect_embedding_format" begin

    @testset "normal models" begin
        txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")
        txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")

        @test Word2Vec.detect_embedding_format(txt_path_1) == :text
        @test Word2Vec.detect_embedding_format(bin_path) == :binary
        @test Word2Vec.detect_embedding_format(txt_path_2) == :text
    end

    @testset "header line" begin
        mktempdir() do d
            f = joinpath(d, "header.txt")
            write(f, "3 5\nking 0.1 0.2 0.3 0.4 0.5\n")
            @test Word2Vec.detect_embedding_format(f) == :text
        end
    end

    @testset "empty lines before content" begin
        mktempdir() do d
            f = joinpath(d, "empty_lines.txt")
            write(f, "\n\n\nqueen 1 2 3\n")
            @test Word2Vec.detect_embedding_format(f) == :text
        end
    end

    @testset "scientific notation" begin
        mktempdir() do d
            f = joinpath(d, "sci.txt")
            write(f, "atom 1e-3 2e+1 -3e-2\n")
            @test Word2Vec.detect_embedding_format(f) == :text
        end
    end

    @testset "word containing digits" begin
        mktempdir() do d
            f = joinpath(d, "digits.txt")
            write(f, "mp3player 0.1 0.2 0.3\n")
            @test Word2Vec.detect_embedding_format(f) == :text
        end
    end

    @testset "invalid text file should be binary" begin
        mktempdir() do d
            f = joinpath(d, "invalid.txt")
            # No line should match: word + floats
            write(f, """
            this is not a word2vec file
            hello world
            123 abc
            """)
            @test Word2Vec.detect_embedding_format(f) == :binary
        end
    end

    @testset "bin file should be binary" begin
        mktempdir() do d
            f = joinpath(d, "bin.bin")
            open(f, "w") do io
                write(io, UInt8[0xFF, 0xD8, 0x00, 0xFF])
            end
            @test Word2Vec.detect_embedding_format(f) == :binary
        end
    end

end

@testset "load_text_embeddings" begin

    @testset "mormal modes" begin
        txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
        txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")
    
        vocab, emb = Word2Vec.load_word2vec(txt_path_1)
        @test length(vocab) == 3
        @test vocab[1] == "king"
        @test emb[:, 1] == [0.1, 0.2, 0.3, 0.4, 0.5]
        @test vocab[2] == "queen"
        @test emb[:, 2] == [0.2, 0.1, 0.4, 0.3, 0.0]
        @test vocab[3] == "man"
        @test emb[:, 3] == [0.0, 0.1, 0.0, 0.1, 0.0]

        vocab, emb = Word2Vec.load_word2vec(txt_path_2)
        @test length(vocab) == 12
        @test vocab[1] == "system"
        @test emb[1,1] == -0.00053622725
        @test typeof(emb[1,1]) == Float64
    end

    @testset "simple valid file" begin
        mktempdir() do d
            f = joinpath(d, "simple.txt")
            open(f, "w") do io
                write(io,
                """
                king 0.1 0.2 0.3
                queen 1 2 3
                """)
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test length(vocab) == 2
            @test vocab[1] == "king"
            @test emb[:, 1] == [0.1, 0.2, 0.3]
            @test vocab[2] == "queen"
            @test emb[:, 2] == [1.0, 2.0, 3.0]
        end
    end

    @testset "header line is skipped" begin
        mktempdir() do d
            f = joinpath(d, "header.txt")
            open(f, "w") do io
                write(io,
                """
                3 5
                king 0.1 0.2 0.3 0.4 0.5
                """)
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test length(vocab) == 1
            @test vocab[1] == "king"
        end
    end

    @testset "empty and whitespace lines" begin
        mktempdir() do d
            f = joinpath(d, "empty.txt")
            open(f, "w") do io
                write(io,
                """

                queen   1 2 3

                """)
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test length(vocab) == 1
            @test vocab[1] == "queen"
            @test emb[:, 1] == [1.0, 2.0, 3.0]
        end
    end

    @testset "scientific notation" begin
        mktempdir() do d
            f = joinpath(d, "sci.txt")
            open(f, "w") do io
                write(io, "atom 1e-3 2e+1 -3e-2\n")
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test vocab[1] == "atom"
            @test emb[:, 1] ≈ [1e-3, 20.0, -0.03]
        end
    end

    @testset "word containing digits" begin
        mktempdir() do d
            f = joinpath(d, "digits.txt")
            open(f, "w") do io
                write(io, "mp3player 0.1 0.2 0.3\n")
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test length(vocab) == 1
            @test vocab[1] == "mp3player"
            @test emb[:, 1] == [0.1, 0.2, 0.3]
        end
    end

    @testset "malformed lines are skipped" begin
        mktempdir() do d
            f = joinpath(d, "bad.txt")
            open(f, "w") do io
                write(io,
                """
                king 0.1 0.2 0.3
                badline x y z
                123 1 2 3
                queen 4 5 6
                """)
            end

            vocab, emb = Word2Vec.load_text_embeddings(f)

            @test length(vocab) == 2
            @test vocab[1] == "king"
            @test vocab[2] == "queen"
        end
    end

    @testset "throws error when no vocab" begin
        mktempdir() do d
            f = joinpath(d, "dup.txt")
            open(f, "w") do io
                write(io,
                """
                123 1 1 1
                42 2 2 2
                """)
            end

             @test_throws ErrorException Word2Vec.load_text_embeddings(f)

        end
    end

end

@testset "load_binary_embeddings" begin

    @testset "normal model" begin
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")

        vocab, emb = Word2Vec.load_binary_embeddings(bin_path)

        @test length(vocab) == 12
        @test size(emb) == (100, 12)
        @test vocab[1] == "system"
        @test emb[1,1] == Float32(-0.00053622725)
        @test typeof(emb[1,1]) == Float64

    end


    @testset "simple valid file" begin
        mktempdir() do d
            f = joinpath(d, "weird.bin")

            open(f, "w") do io
                write(io, "2 2\n")

                write(io, codeunits("word_123"))
                write(io, UInt8(0x20))
                write(io, Float32[1.0, -1.0])

                write(io, codeunits("ümlaut"))
                write(io, UInt8(0x20))
                write(io, Float32[0.5, 0.5])
            end

            vocab, emb = Word2Vec.load_binary_embeddings(f)

            @test length(vocab) == 2
            @test vocab[1] == "word_123"
            @test vocab[2] == "ümlaut"
            @test emb[:, 1] == [Float32(1.0), Float32(-1.0)]
            @test emb[:, 2] == [Float32(0.5), Float32(0.5)]
        end
    end

    @testset "invalid header content" begin
        mktempdir() do d
            f = joinpath(d, "badheader.bin")
            open(f, "w") do io
                write(io, "wrong header\n")
            end

            @test_throws ErrorException Word2Vec.load_binary_embeddings(f)
        end
    end

    @testset "42 42 42" begin
        mktempdir() do d
            f = joinpath(d, "badheader.bin")
            open(f, "w") do io
                write(io, "wrong header header\n")
            end

            @test_throws ErrorException Word2Vec.load_binary_embeddings(f)
        end
    end
end


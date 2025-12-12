using Test
using Word2Vec

@testset "load_word2vec" begin
    
    @testset "normal models" begin
        txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")
        txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")

        emb = Word2Vec.load_word2vec(txt_path_1)
        @test length(emb) == 3
        @test emb["king"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        @test emb["queen"] == [0.2, 0.1, 0.4, 0.3, 0.0]
        @test emb["man"] == [0.0, 0.1, 0.0, 0.1, 0.0]

        emb = Word2Vec.load_word2vec(txt_path_2)
        @test length(emb) == 12
        @test emb["system"] !== nothing
        @test emb["system"][1] == -0.00053622725

        emb = Word2Vec.load_binary_embeddings(bin_path)
        @test length(emb) == 12
        @test emb["system"] !== nothing
        @test emb["system"][1] == Float32(-0.00053622725)
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
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")
        txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")

        emb = Word2Vec.load_text_embeddings(txt_path_1)
    
        @test length(emb) == 3
        @test emb["king"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        @test emb["queen"] == [0.2, 0.1, 0.4, 0.3, 0.0]
        @test emb["man"] == [0.0, 0.1, 0.0, 0.1, 0.0]

        emb = Word2Vec.load_text_embeddings(txt_path_2)

        @test length(emb) == 12
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

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 2
            @test emb["king"] == [0.1, 0.2, 0.3]
            @test emb["queen"] == [1.0, 2.0, 3.0]
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

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 1
            @test haskey(emb, "king")
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

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 1
            @test emb["queen"] == [1.0, 2.0, 3.0]
        end
    end

    @testset "scientific notation" begin
        mktempdir() do d
            f = joinpath(d, "sci.txt")
            open(f, "w") do io
                write(io, "atom 1e-3 2e+1 -3e-2\n")
            end

            emb = Word2Vec.load_text_embeddings(f)

            @test haskey(emb, "atom")
            @test emb["atom"] ≈ [1e-3, 20.0, -0.03]
        end
    end

    @testset "word containing digits" begin
        mktempdir() do d
            f = joinpath(d, "digits.txt")
            open(f, "w") do io
                write(io, "mp3player 0.1 0.2 0.3\n")
            end

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 1
            @test emb["mp3player"] == [0.1, 0.2, 0.3]
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
                numeric 1 2 3   # should be skipped (word is number)
                123 1 2 3
                queen 4 5 6
                """)
            end

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 2
            @test haskey(emb, "king")
            @test haskey(emb, "queen")
            @test !haskey(emb, "numeric")
            @test !haskey(emb, "123")
            @test !haskey(emb, "badline")
        end
    end

    @testset "duplicate words overwrite last" begin
        mktempdir() do d
            f = joinpath(d, "dup.txt")
            open(f, "w") do io
                write(io,
                """
                cat 1 1 1
                cat 2 2 2
                """)
            end

            emb = Word2Vec.load_text_embeddings(f)

            @test length(emb) == 1
            @test emb["cat"] == [2.0, 2.0, 2.0]
        end
    end

end

@testset "load_binary_embeddings" begin

    @testset "normal model" begin
        bin_path   = joinpath(@__DIR__, "data", "word2vec.bin")

        emb = Word2Vec.load_binary_embeddings(bin_path)

        @test length(emb) == 12
        @test emb["system"] !== nothing
        @test emb["system"][1] == Float32(-0.00053622725)

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

            emb = Word2Vec.load_binary_embeddings(f)

            @test haskey(emb, "word_123")
            @test haskey(emb, "ümlaut")
            @test emb["word_123"] == [Float32(1.0), Float32(-1.0)]
            @test emb["ümlaut"] == [Float32(0.5), Float32(0.5)]
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


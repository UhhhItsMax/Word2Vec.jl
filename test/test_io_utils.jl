using Test: @testset, @test, @test_throws
using Word2Vec: clean_line, read_wordlist, read_corpus_sentences, read_corpus_tokens

const _DATA = joinpath(@__DIR__, "data")

@testset "utils - read_wordlist" begin
    @testset "reads one word per line and strips whitespace" begin
        path = joinpath(_DATA, "wordlist_basic.txt")
        words = read_wordlist(path)

        @test words == ["apple", "banana", "cherry"]
        @test eltype(words) == String
    end

    @testset "ignores empty lines and comment lines (after stripping)" begin
        path = joinpath(_DATA, "wordlist_comments_blanks.txt")
        words = read_wordlist(path)

        @test words == ["word1", "word2"]
    end

    @testset "keeps internal # when not a leading comment" begin
        path = joinpath(_DATA, "wordlist_hash_inside.txt")
        words = read_wordlist(path)

        @test words == ["foo#bar", "baz#qux"]
    end

    @testset "throws if file does not exist" begin
        missing = joinpath(_DATA, "definitely_missing.txt")
        @test_throws SystemError read_wordlist(missing)
    end
end

@testset "utils - text_utils (clean_line)" begin
    @testset "returns nothing for empty/whitespace-only lines" begin
        @test clean_line("") === nothing
        @test clean_line("   \t\n ") === nothing
    end

    @testset "dash_rule drops lines that are only 3+ dashes (after stripping)" begin
        @test clean_line("---") === nothing
        @test clean_line("   ------   ") === nothing

        @test clean_line("--") == "--"
        @test clean_line("-") == "-"
    end

    @testset "lowercasing + punctuation stripping + whitespace collapsing" begin
        @test clean_line("  Hello,   WORLD!!  ") == "hello world"

        @test clean_line("Room 101.") == "room 101"

        @test clean_line("a,b;c") == "a b c"
    end

    @testset "options: lowercase=false / strip_punct=false / dash_rule=false" begin
        @test clean_line("Hello WORLD"; lowercase = false) == "Hello WORLD"
        @test clean_line("Hello, WORLD!"; strip_punct = false) == "hello, world!"

        @test clean_line("-----"; dash_rule = false) == "-----"
    end

    @testset "strip_punct can erase content -> returns nothing" begin
        @test clean_line("!!!") === nothing
        @test clean_line("   ...   ") === nothing
    end
end

@testset "utils - text_utils (read_corpus_sentences / read_corpus_tokens)" begin
    @testset "throws if corpus file does not exist" begin
        missing = joinpath(_DATA, "corpus_missing.txt")
        @test_throws ArgumentError read_corpus_sentences(missing)
        @test_throws ArgumentError read_corpus_tokens(missing)
    end

    @testset "reads sentences, skips dashed separators + blank lines, cleans punctuation" begin
        path = joinpath(_DATA, "corpus_mixed.txt")

        sents = read_corpus_sentences(path)
        @test sents isa Vector{Vector{String}}
        @test sents == [
            ["hello", "world"],
            ["this", "is", "line", "2"],
            ["room", "101"],
            ["foo", "bar", "baz"],
        ]

        toks = read_corpus_tokens(path)
        @test toks == ["hello", "world", "this", "is", "line", "2", "room", "101", "foo", "bar", "baz"]
    end

    @testset "respects forwarded kwargs (disable dash_rule)" begin
        path = joinpath(_DATA, "corpus_dashes_only.txt")

        sents1 = read_corpus_sentences(path)
        @test isempty(sents1)

        sents2 = read_corpus_sentences(path; dash_rule = false, strip_punct = false, lowercase = false)
        @test sents2 == [["---"], ["------"]]
    end

    @testset "respects forwarded kwargs (no punctuation stripping)" begin
        path = joinpath(_DATA, "corpus_punct.txt")

        sents = read_corpus_sentences(path; strip_punct = false)
        @test sents == [
            ["hello,", "world!"],
            ["a,b;c"],
        ]
    end
end

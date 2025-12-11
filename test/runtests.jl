using Word2Vec
using Test

@testset "Word2Vec.jl" begin

end

@testset "format detection" begin
    txt_path_1 = joinpath(@__DIR__, "data", "small_model.txt")
    #bin_path = joinpath(@__DIR__, "data", "word2vec.bin")
    #txt_path_2 = joinpath(@__DIR__, "data", "word2vec.txt")

    @test Word2Vec.detect_embedding_format(txt_path_1) == :text
    #@test Word2Vec.detect_embedding_format(bin_path) == :binary
    #@test Word2Vec.detect_embedding_format(txt_path_2) == :text
end
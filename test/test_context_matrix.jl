using Test: @testset, @test
using SparseArrays: nnz
using Word2Vec: get_co_occurence_counts, SparseContextMatrix, save_sparse_context_matrix, load_sparse_context_matrix

@testset "ContextMatrix" begin
    path = joinpath(@__DIR__, "data", "context_matrix.txt")

    @testset "co-occurrence counts" begin
        token_to_id = Dict("a" => 1, "b" => 2, "c" => 3, "d" => 4)
        coocs = get_co_occurence_counts(path, token_to_id, 1)

        @test coocs[(1, 2)] == 1
        @test coocs[(3, 2)] == 1
        @test coocs[(2, 3)] == 1
        @test coocs[(4, 3)] == 1
    end

    @testset "build matrix" begin
        cm = SparseContextMatrix(path; window_size = 1, min_count = 1)
        @test size(cm.mat) == (4, 4)
        @test nnz(cm.mat) == 4
        @test length(cm.vocab) == 4
        @test length(cm.token_to_id) == 4
    end

    @testset "serialization" begin
        path = joinpath(@__DIR__, "data", "context_matrix.txt")
        tmp = joinpath(mktempdir(), "scm.bin")

        cm = SparseContextMatrix(
            path;
            window_size = 1,
            min_count = 1,
        )

        save_sparse_context_matrix(tmp, cm)
        cm2 = load_sparse_context_matrix(tmp)

        @test size(cm2.mat) == size(cm.mat)
        @test nnz(cm2.mat) == nnz(cm.mat)
        @test cm2.mat == cm.mat
        @test cm2.vocab == cm.vocab
        @test cm2.token_to_id == cm.token_to_id
    end
end

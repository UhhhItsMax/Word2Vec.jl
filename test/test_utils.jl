using Test: @test, @testset 
using Word2Vec: center_rows!, center_rows, l2normalize_rows!, l2normalize_rows
using LinearAlgebra: norm

@testset "utils - math_utils (center_rows)" begin
    @testset "center_rows! centers columns in-place" begin
        X = [1.0 2.0;
             3.0 6.0;
             5.0 10.0]  # column means: [3, 6]

        Y = center_rows!(X)
        @test Y === X  # in-place return

        # column means should be ~0
        @test isapprox(sum(X[:, 1]), 0.0; atol=1e-12)
        @test isapprox(sum(X[:, 2]), 0.0; atol=1e-12)

        # exact expected centered result
        @test X ≈ [-2.0 -4.0;
                    0.0  0.0;
                    2.0  4.0]
    end

    @testset "center_rows returns a centered copy" begin
        X = [1.0 2.0;
             3.0 6.0]

        X_orig = copy(X)
        Y = center_rows(X)

        @test Y !== X               # new object
        @test X == X_orig           # original unchanged
        @test isapprox(sum(Y[:, 1]), 0.0; atol=1e-12)
        @test isapprox(sum(Y[:, 2]), 0.0; atol=1e-12)
    end

    @testset "center_rows supports integer input (returns Float)" begin
        X = [1 2;
             3 4]

        Y = center_rows(X)
        @test eltype(Y) <: AbstractFloat
        @test isapprox(sum(Y[:, 1]), 0.0; atol=1e-12)
        @test isapprox(sum(Y[:, 2]), 0.0; atol=1e-12)
    end
end


@testset "utils - normalize (l2normalize_rows)" begin
    @testset "l2normalize_rows! normalizes each row in-place" begin
        X = [3.0 4.0;
             5.0 12.0]  # norms: 5, 13

        Y = l2normalize_rows!(X)
        @test Y === X

        @test isapprox(norm(X[1, :]), 1.0; atol=1e-12)
        @test isapprox(norm(X[2, :]), 1.0; atol=1e-12)

        # direction should be preserved
        @test X[1, :] ≈ [0.6, 0.8]
        @test X[2, :] ≈ [5/13, 12/13]
    end

    @testset "l2normalize_rows! leaves zero rows unchanged" begin
        X = [0.0 0.0;
             3.0 4.0]

        l2normalize_rows!(X)

        @test X[1, :] == [0.0, 0.0]                 # unchanged
        @test isapprox(norm(X[2, :]), 1.0; atol=1e-12)
    end

    @testset "l2normalize_rows returns a normalized copy" begin
        X = [3.0 4.0;
             0.0 0.0]

        X_orig = copy(X)
        Y = l2normalize_rows(X)

        @test Y !== X
        @test X == X_orig

        @test isapprox(norm(Y[1, :]), 1.0; atol=1e-12)
        @test Y[2, :] == [0.0, 0.0]
    end

    @testset "l2normalize_rows supports integer input (returns Float)" begin
        X = [3 4;
             0 0]

        Y = l2normalize_rows(X)
        @test eltype(Y) <: AbstractFloat
        @test isapprox(norm(Y[1, :]), 1.0; atol=1e-12)
        @test Y[2, :] == [0.0, 0.0]
    end
end
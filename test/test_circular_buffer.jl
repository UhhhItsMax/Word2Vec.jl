using Test: @testset, @test, @test_throws
using Word2Vec: CircularBuffer, isempty, isfull

@testset "CircularBuffer" begin
    @testset "basic push/pop behavior" begin
        buf = CircularBuffer{Int}(3)

        @test length(buf) == 0
        @test isempty(buf)
        @test !isfull(buf)

        push!(buf, 1)
        push!(buf, 2)
        push!(buf, 3)

        @test length(buf) == 3
        @test isfull(buf)
        @test collect(buf) == [1, 2, 3]

        push!(buf, 4)

        @test length(buf) == 3
        @test collect(buf) == [2, 3, 4]

        @test collect(buf) == [2, 3, 4]
        @test length(buf) == 3
    end

    @testset "wraparound indexing and overwrite order" begin
        buf = CircularBuffer{Int}(2)

        push!(buf, 10)
        push!(buf, 20)
        push!(buf, 30)

        @test collect(buf) == [20, 30]
        @test buf[1] == 20
        @test buf[2] == 30

        push!(buf, 40)
        @test collect(buf) == [30, 40]
    end

    @testset "empty and bounds errors" begin
        buf = CircularBuffer{Int}(1)

        @test_throws BoundsError buf[1]

        push!(buf, 7)
        @test buf[1] == 7

        buf = CircularBuffer{Int}(1)
        @test isempty(buf)
        @test length(buf) == 0
    end
end

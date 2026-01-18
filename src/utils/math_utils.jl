export center_rows!, center_rows
using Statistics: mean

"""
    center_rows!(X) -> X

In-place mean-centering of each column (i.e., subtract column means from every row).
Assumes `X` is points-as-rows (n_points × dim).
"""
function center_rows!(X::AbstractMatrix{<:Real})
    μ = mean(X; dims=1)
    X .-= μ
    return X
end

center_rows(X::AbstractMatrix{<:Real}) = center_rows!(copy(float.(X)))
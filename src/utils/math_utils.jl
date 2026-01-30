"""
    center_rows!(X)

In-place mean-centering of each **column** of `X`.

# Arguments
- `X::AbstractMatrix{<:Real}`: Matrix of size `(n_points × dim)` where rows are points and columns are features.

# Returns
- The same matrix `X`, with each column mean-subtracted.

# Notes
- Centers data along the **rows** (i.e., subtracts the column means from every row).
- Modifies `X` in-place for memory efficiency.
"""
function center_rows!(X::AbstractMatrix{<:Real})
    μ = mean(X; dims = 1)
    X .-= μ
    return X
end


"""
    center_rows(X)

Return a **mean-centered copy** of the input matrix `X`.

# Arguments
- `X::AbstractMatrix{<:Real}`: Input matrix with rows as points and columns as features.

# Returns
- `X_centered::Matrix{Float64}`: A new matrix where each column has zero mean.

# Notes
- Does **not** modify the original `X`.
- Internally converts `X` to `Float64` and calls `center_rows!` on a copy.
"""
center_rows(X::AbstractMatrix{<:Real}) = center_rows!(copy(float.(X)))

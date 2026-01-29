"""
    l2normalize_rows(X)

Return a **row-wise L2-normalized copy** of the input matrix `X`.

# Arguments
- `X::AbstractMatrix{<:Real}`: Input matrix with rows as vectors to normalize.

# Returns
- `Matrix{Float64}`: A new matrix where each row has Euclidean norm 1 (rows of all zeros are unchanged).

# Notes
- Does **not** modify the original `X`.
- Internally converts `X` to `Float64` and calls `l2normalize_rows!` on a copy.
"""
function l2normalize_rows!(X::AbstractMatrix{<:Real})
    @inbounds for row in eachrow(X)
        s = sum(float.(row).^2)
        nrm = sqrt(s)
        nrm > 0 && (row .= float.(row) ./ nrm)
    end
    return X
end


"""
    l2normalize_rows(X)

Return a **row-wise L2-normalized copy** of the input matrix `X`.

# Arguments
- `X::AbstractMatrix{<:Real}`: Input matrix whose rows are to be normalized.

# Returns
- `Matrix{Float64}`: New matrix where each row has Euclidean norm 1 (rows of all zeros are unchanged).

# Notes
- The original matrix `X` is not modified.
- Conversion to `Float64` is applied internally.
- Uses `l2normalize_rows!` on a copy for the normalization.
"""
l2normalize_rows(X::AbstractMatrix{<:Real}) = l2normalize_rows!(copy(float.(X)))
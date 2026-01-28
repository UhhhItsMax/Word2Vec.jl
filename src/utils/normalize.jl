
"""
    l2normalize_rows!(X) -> X

In-place L2-normalization of each row of `X`.

After normalization, each row `X[i, :]` has Euclidean norm 1 (unless it was all zeros,
in which case the row is left unchanged).

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
    l2normalize_rows(X) -> Y

Return a copy of `X` with rows L2-normalized.
"""
l2normalize_rows(X::AbstractMatrix{<:Real}) = l2normalize_rows!(copy(float.(X)))
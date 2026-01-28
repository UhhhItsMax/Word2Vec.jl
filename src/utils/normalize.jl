
"""
    l2normalize_rows!(X) -> X

In-place L2-normalization of each row of `X`.

After normalization, each row `X[i, :]` has Euclidean norm 1 (unless it was all zeros,
in which case the row is left unchanged).

"""
function l2normalize_rows!(X::AbstractMatrix{<:Real})
    @inbounds for i in 1:size(X, 1)
        s = 0.0
        for j in 1:size(X, 2)
            v = float(X[i, j])
            s += v*v
        end
        nrm = sqrt(s)
        if nrm > 0
            for j in 1:size(X, 2)
                X[i, j] = float(X[i, j]) / nrm
            end
        end
    end
    return X
end

"""
    l2normalize_rows(X) -> Y

Return a copy of `X` with rows L2-normalized.
"""
l2normalize_rows(X::AbstractMatrix{<:Real}) = l2normalize_rows!(copy(float.(X)))
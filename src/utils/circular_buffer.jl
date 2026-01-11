module CircularBuffers

export CircularBuffer, capacity, isfull

"""
    CircularBuffer{T}

Fixed-capacity FIFO buffer that overwrites the oldest items when full.
"""
mutable struct CircularBuffer{T}
	data::Vector{T}
	head::Int
	len::Int

	function CircularBuffer{T}(capacity::Int) where T
		capacity > 0 || throw(ArgumentError("capacity must be positive"))
		data = Vector{T}(undef, capacity)
		return new{T}(data, 1, 0)
	end
end

"""
    capacity(buf::CircularBuffer)

Return the fixed capacity of `buf`.
"""
capacity(buf::CircularBuffer) = length(buf.data)

Base.length(buf::CircularBuffer) = buf.len
Base.isempty(buf::CircularBuffer) = buf.len == 0

"""
    isfull(buf::CircularBuffer)

Return `true` when `buf` is at capacity.
"""
isfull(buf::CircularBuffer) = buf.len == capacity(buf)

"""
    Base.push!(buf::CircularBuffer, item)

Push `item` into `buf`, overwriting the oldest entry when full.
"""
function Base.push!(buf::CircularBuffer, item)
	cap = capacity(buf)
	if buf.len < cap
		idx = (buf.head + buf.len - 1) % cap + 1
		@inbounds buf.data[idx] = item
		buf.len += 1
	else
		@inbounds buf.data[buf.head] = item
		buf.head = (buf.head % cap) + 1
	end
	return buf
end

"""
    Base.getindex(buf::CircularBuffer, i::Int)

Return the `i`-th item in FIFO order.
"""
function Base.getindex(buf::CircularBuffer, i::Int)
	1 <= i <= buf.len || throw(BoundsError(buf, i))
	cap = capacity(buf)
	idx = (buf.head + i - 2) % cap + 1
	return @inbounds buf.data[idx]
end

"""
    Base.iterate(buf::CircularBuffer, state::Tuple{Int, Int, Int}=(buf.head, 0, capacity(buf)))

Iterate items in FIFO order starting at the head.
"""
function Base.iterate(buf::CircularBuffer, state::Tuple{Int, Int, Int}=(buf.head, 0, capacity(buf)))
	idx, count, cap = state
	count >= buf.len && return nothing
	item = @inbounds buf.data[idx]
	next_idx = (idx % cap) + 1
	return (item, (next_idx, count + 1, cap))
end

end # module

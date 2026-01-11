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

capacity(buf::CircularBuffer) = length(buf.data)

Base.length(buf::CircularBuffer) = buf.len
Base.isempty(buf::CircularBuffer) = buf.len == 0

isfull(buf::CircularBuffer) = buf.len == capacity(buf)

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

function Base.getindex(buf::CircularBuffer, i::Int)
	1 <= i <= buf.len || throw(BoundsError(buf, i))
	cap = capacity(buf)
	idx = (buf.head + i - 2) % cap + 1
	return @inbounds buf.data[idx]
end

function Base.iterate(buf::CircularBuffer, state::Int=1)
	state > buf.len && return nothing
	return (buf[state], state + 1)
end

end # module

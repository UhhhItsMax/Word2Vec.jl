"""
    CircularBuffer{T}

A fixed-capacity **FIFO buffer** for elements of type `T` that overwrites the oldest entries when full.

# Fields
- `data::Vector{T}`: Internal storage of length `capacity`.
- `head::Int`: Index of the oldest element.
- `len::Int`: Current number of elements stored.

# Notes
- Supports indexing, iteration, and standard `Base` methods (`length`, `isempty`, `isfull`).
- Pushing an element when full overwrites the oldest value.
- Ideal for sliding-window tasks such as co-occurrence counting in NLP.
"""
mutable struct CircularBuffer{T}
    data::Vector{T}
    head::Int
    len::Int

    """
    	CircularBuffer{T}(capacity::Int)

    Create a new `CircularBuffer` of element type `T` with a fixed `capacity`.

    # Arguments
    - `capacity::Int` — Maximum number of elements the buffer can hold (must be ≥ 1).

    # Throws
    - `ArgumentError` if `capacity < 1`.

    # Notes
    - The buffer starts empty (`len = 0`) with `head = 1`.
    - Pushing elements beyond `capacity` overwrites the oldest entries in FIFO order.
    """
    function CircularBuffer{T}(capacity::Int) where {T}
        capacity > 0 || throw(ArgumentError("capacity must be positive"))
        data = Vector{T}(undef, capacity)
        return new{T}(data, 1, 0)
    end
end


"""
    capacity(buf::CircularBuffer)

Return the fixed capacity of `buf`.

# Notes
- The capacity is the maximum number of elements the buffer can hold.
- The current number of stored elements is `length(buf)`.
- Use `isfull(buf)` to check whether the buffer has reached capacity.
"""
capacity(buf::CircularBuffer) = length(buf.data)


"""
    Base.length(buf::CircularBuffer)

Return the current number of elements stored in the circular buffer `buf`.

# Arguments
- `buf::CircularBuffer`: The circular buffer.

# Returns
- Number of elements currently in the buffer (≤ `capacity(buf)`).

# Notes
- Use `capacity(buf)` to get the fixed maximum size.
- Use `isfull(buf)` to check if the buffer has reached its maximum capacity.
"""
Base.length(buf::CircularBuffer) = buf.len


"""
    Base.isempty(buf::CircularBuffer)

Check whether the circular buffer `buf` currently contains no elements.

# Arguments
- `buf::CircularBuffer`: The circular buffer.

# Returns
- `Bool`: `true` if the buffer is empty (`length(buf) == 0`), `false` otherwise.

# Notes
- Even if the buffer has a positive capacity, it is considered empty until elements are pushed.
"""
Base.isempty(buf::CircularBuffer) = buf.len == 0


"""
    isfull(buf::CircularBuffer)

Check whether the circular buffer `buf` has reached its fixed capacity.

# Arguments
- `buf::CircularBuffer`: The circular buffer.

# Returns
- `Bool`: `true` if the buffer is full (`length(buf) == capacity(buf)`), `false` otherwise.

# Notes
- When full, pushing a new element overwrites the oldest element in FIFO order.
- Use `length(buf)` to get the current number of stored elements.
"""
isfull(buf::CircularBuffer) = buf.len == capacity(buf)


"""
    Base.push!(buf::CircularBuffer, item)

Push `item` into the circular buffer `buf`.  

- If the buffer is not full, the item is appended at the end.  
- If the buffer is full, the oldest element is overwritten and the head pointer advances.

# Arguments
- `buf::CircularBuffer`: The circular buffer.
- `item`: The item to insert (of the buffer’s element type).

# Returns
- `buf`: The updated circular buffer (same instance).

# Notes
- The buffer maintains a fixed capacity; pushing beyond it overwrites old entries.
- Use `isfull(buf)` to check if the buffer is currently full.
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

Return the `i`-th item from the circular buffer `buf` in FIFO order.

# Arguments
- `buf::CircularBuffer`: The circular buffer.
- `i::Int`: Index of the item (1-based, relative to the oldest element).

# Returns
- `item`: The element stored at position `i` in FIFO order.

# Notes
- The oldest element in the buffer corresponds to `i = 1`.
- Supports standard Julia indexing semantics for 1-based arrays.

# Throws
- `BoundsError` if `i < 1` or `i > length(buf)`.
"""
function Base.getindex(buf::CircularBuffer, i::Int)
    1 <= i <= buf.len || throw(BoundsError(buf, i))
    cap = capacity(buf)
    idx = (buf.head + i - 2) % cap + 1
    return @inbounds buf.data[idx]
end


"""
    Base.iterate(buf::CircularBuffer, state::Tuple{Int, Int, Int}=(buf.head, 0, capacity(buf)))

Iterate over the elements of `buf` in FIFO order, starting from the oldest element.

# Arguments
- `buf::CircularBuffer`: The circular buffer to iterate over.
- `state::Tuple{Int, Int, Int}`: Internal iteration state `(current_index, count, capacity)`.
  Defaults to `(buf.head, 0, capacity(buf))` for starting a fresh iteration.

# Returns
- `item`: The next element in FIFO order.
- `state`: Updated state tuple for the next call to `iterate`.

# Notes
- Compatible with Julia's iteration interface, allowing `for item in buf` loops.
- Iterates exactly `length(buf)` elements, even if the buffer is full and wrapped around.
"""
function Base.iterate(buf::CircularBuffer, state::Tuple{Int, Int, Int} = (buf.head, 0, capacity(buf)))
    idx, count, cap = state
    count >= buf.len && return nothing
    item = @inbounds buf.data[idx]
    next_idx = (idx % cap) + 1
    return (item, (next_idx, count + 1, cap))
end

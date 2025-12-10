"""
    Word2VecLoader

A Julia module for loading Word2Vec embeddings from text or binary files,
with automatic format detection.

# Features
- Detects whether a file is in text or binary Word2Vec format.
- Loads embeddings into dictionaries mapping words (`String`) to vectors
  (`Vector{Float64}` for text, `Vector{Float32}` for binary).
- Handles optional header lines in text files.
- Provides a unified `load_word2vec` function.
- Safe parsing with error handling for malformed files.

# Dependencies
- Base Julia (no external packages required)
- Uses `DelimitedFiles` from the standard library for reading files

# Example
``julia
using Word2VecLoader

embeddings = load_word2vec("GoogleNews-vectors-negative300.bin")
vec = embeddings["king"]
"""

using DelimitedFiles

export load_word2vec

"""
    load_word2vec(path::String) :: Union{Dict{String, Vector{Float32}}, Dict{String, Vector{Float64}}}

Loads Word2Vec embeddings from a file, automatically detecting whether the file
is in text or binary format.

# Arguments
- `path::String`: Path to the Word2Vec embedding file.

# Returns
- `Dict{String, Vector{Float64}}` if the file is in text format.
- `Dict{String, Vector{Float32}}` if the file is in binary format.

# Notes
- The function uses `detect_embedding_format` to determine the file format.
- Text files are expected to have lines in the format `word float float ...`.
  Numeric header lines are skipped automatically.
- Binary files must follow the classic Word2Vec format with a header line
  `vocab_size dim` and subsequent word-vector entries.
- Errors may be thrown if the file does not match the expected format.
- The entire embedding dictionary is loaded into memory, which may require
  substantial RAM for large models.
"""
function load_word2vec(path::String)
    fmt = detect_format(path)
    if fmt == :text
        return load_text_embeddings(path)
    else
        return load_binary_embeddings(path)
    end
end

"""
    detect_embedding_format(path::String) :: Symbol

Detects the format of a Word2Vec embedding file.

# Arguments
- `path::String`: Path to the embedding file.

# Returns
- `:text` if the file appears to be in text format (`word float float ...`).
- `:binary` if the file appears to be in binary Word2Vec format.

# Notes
- The function reads the file line by line, skipping empty lines and optional
  header lines.
- It checks whether lines follow the pattern of a word followed by floating-point numbers.
- If a line cannot be interpreted as UTF-8, the function assumes the file is binary.
- If no text-like line is found, it defaults to `:binary`.
"""
function detect_embedding_format(path::String)::Symbol
    open(path, "r") do io
        while !eof(io)
            line = strip(readline(io; keep=true))

            isempty(line) && continue

            tokens = split(line)

            # Case 1: header-like → skip it
            if length(tokens) == 2 && all(t -> !isnothing(tryparse(Int, t)), tokens)
                continue
            end

            # Case 2: word + floats
            if length(tokens) >= 2
                word = tokens[1]
                vec = tokens[2:end]

                if tryparse(Float64, word) === nothing && all(t -> tryparse(Float64, t) !== nothing, vec)
                    return :text
                end
            end

            # If line contains non-UTF8 bytes, it's likely binary
            try
                String(line)
            catch
                return :binary
            end
        end
    end

    # Fallback: assume binary if nothing matched
    return :binary
end


"""
    load_text_embeddings(path::String) :: Dict{String, Vector{Float64}}

Loads Word2Vec embeddings from a text-format file.

# Arguments
- `path::String`: Path to the text-format Word2Vec embedding file.

# Returns
- `Dict{String, Vector{Float64}}`: A dictionary mapping words (`String`) to their embedding vectors (`Vector{Float64}`).

# Notes
- Each line in the file should have the format `word float float ...`.
- Lines where the first token is numeric (e.g., optional header) or where the vector
  tokens cannot be parsed as floating-point numbers are skipped automatically.
- This function reads the entire file into memory; very large embedding files may
  require substantial RAM.
"""
function load_text_embeddings(path::String)::Dict{String, Vector{Float64}}
    embeddings = Dict{String, Vector{Float64}}()

    open(path, "r") do io
        for line in eachline(io)
            tokens = split(line)
            if length(tokens) >= 2
                word, vec_tokens = tokens[1], tokens[2:end]

                # Ensure first token is a word, rest are floats
                if tryparse(Float64, word) === nothing && all(t -> tryparse(Float64, t) !== nothing, vec_tokens)
                    embeddings[word] = parse.(Float64, vec_tokens)
                end
            end
        end
    end

    return embeddings
end


"""
    load_binary_embeddings(path::String) :: Dict{String, Vector{Float32}}

Loads Word2Vec embeddings from a binary-format file.

# Arguments
- `path::String`: Path to the binary-format Word2Vec embedding file.

# Returns
- A dictionary mapping words (`String`) to their embedding vectors (`Vector{Float32}`).

# Notes
- The file must start with a header line containing `vocab_size dim`, where both
  values are integers. The function throws an error if the header is missing
  or malformed.
- The file is expected to follow the classic Word2Vec binary format: after the
  header, there are `vocab_size` entries, each consisting of a word (terminated
  by a space) and a vector of `dim` Float32 values.
- A single newline may follow each vector; the loader handles it if present.
"""
function load_binary_embeddings(path::String)::Dict{String, Vector{Float32}}
    embeddings = Dict{String, Vector{Float32}}()

    open(path, "r") do io
        # Read and validate header: "vocab_size dim\n"
        header = readline(io)
        tokens = split(header)
        if length(tokens) != 2
            error("Invalid binary Word2Vec file: header must have 2 tokens (vocab_size dim)")
        end

        vocab_size = tryparse(Int, tokens[1])
        dim = tryparse(Int, tokens[2])
        if vocab_size === nothing || dim === nothing
            error("Invalid binary Word2Vec header: tokens must be integers")
        end

        for _ in 1:vocab_size
            # Read word (terminated by space)
            word_bytes = UInt8[]
            while true
                c = read(io, UInt8)
                if c == 0x20  # space
                    break
                end
                push!(word_bytes, c)
            end
            word = String(word_bytes)

            # Read vector
            vec = read(io, Float32, dim)
            embeddings[word] = vec

            # Skip single newline after vector if present
            if !eof(io)
                peek_byte = read(io, UInt8; keep=true)
                peek_byte == 0x0A && read(io, UInt8)
            end
        end
    end

    return embeddings
end
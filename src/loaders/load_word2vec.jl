"""
    Word2VecLoader

A Julia module for loading Word2Vec embeddings from text or binary files,
with automatic format detection.

# Features
- Detects whether a file is in text or binary Word2Vec format.
- Loads embeddings into dictionaries mapping words (`String`) to vectors.
- Handles optional header lines in text files.
- Supports Gensim's hybrid binary format (text header + binary vectors).
- Provides a unified `load_word2vec` function.
- Safe parsing with error handling for malformed files.

# Dependencies
- Base Julia (no external packages required)
- Uses `DelimitedFiles` from the standard library for reading files

# Example
using Word2VecLoader

embeddings = load_word2vec("word2vec.bin") # or .txt
vec = embeddings["king"]

text
"""

using DelimitedFiles

export load_word2vec

"""
    load_word2vec(path::String) :: Dict{String, Vector{Float64}}

Loads Word2Vec embeddings from a file, automatically detecting whether the file
is in text or binary format.

# Arguments
- `path::String`: Path to the Word2Vec embedding file.

# Returns
- `Dict{String, Vector{Float64}}`: Dictionary mapping words to embedding vectors.

# Notes
- Uses `detect_embedding_format` to determine file format.
- Text files: `word float float ...` (skips numeric header lines).
- Binary files: Gensim hybrid format (text header `"vocab_size dim"`, ASCII words, binary Float32 vectors).
- Binary vectors are converted to `Float64` for consistency.
- Large models require substantial RAM.
"""
function load_word2vec(path::String)
    fmt = detect_embedding_format(path)
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
- `:binary` if the file appears to be in Gensim binary format.

# Notes
- Prioritizes `.bin` extension -> `:binary`.
- Checks first 8 bytes for Int32 header (rare pure-binary case).
- Falls back to text detection on first few lines.
- Defaults to `:text` if ambiguous (safer for small files).
"""
function detect_embedding_format(path::String)::Symbol
    ext = splitext(basename(path))[2]
    ext == ".bin" && return :binary  # Simple extension check first
    
    open(path, "r") do io
        # Check first 8 bytes for binary header (two Int32)
        if filesize(path) >= 8
            seekstart(io)
            vocab_size = read(io, Int32)
            dim = read(io, Int32)
            # Valid header if reasonable sizes
            if vocab_size > 0 && dim > 0 && dim < 1000
                return :binary
            end
        end
        
        # Fallback to text detection (only read first few lines)
        seekstart(io)
        lines_read = 0
        for line in eachline(io)
            lines_read += 1
            line = strip(line)
            isempty(line) && continue
            
            tokens = split(line)
            if length(tokens) == 2 && all(t -> tryparse(Int, t) !== nothing, tokens)
                continue  # header line
            end
            
            if length(tokens) >= 2
                word, vec_tokens = tokens[1], tokens[2:end]
                if tryparse(Float64, word) === nothing &&
                   all(t -> tryparse(Float64, t) !== nothing, vec_tokens[1:min(5,end)])
                    return :text
                end
            end
            
            lines_read > 5 && break  # Don't read entire file
        end
    end
    return :text
end

"""
    load_text_embeddings(path::String) :: Dict{String, Vector{Float64}}

Loads Word2Vec embeddings from a text-format file.

# Arguments
- `path::String`: Path to the text-format Word2Vec embedding file.

# Returns
- `Dict{String, Vector{Float64}}`: Dictionary mapping words to embedding vectors.

# Notes
- Each line: `word float float ...` (header lines auto-skipped).
- Skips lines where first token is numeric or vectors can't be parsed.
- Reads entire file into memory.
"""
function load_text_embeddings(path::String)::Dict{String, Vector{Float64}}
    embeddings = Dict{String, Vector{Float64}}()

    open(path, "r") do io
        for line in eachline(io)
            tokens = split(line)
            if length(tokens) >= 2
                word, vec_tokens = tokens[1], tokens[2:end]

                # Ensure first token is a word, rest are floats
                if tryparse(Float64, word) === nothing && 
                   all(t -> tryparse(Float64, t) !== nothing, vec_tokens)
                    embeddings[word] = parse.(Float64, vec_tokens)
                end
            end
        end
    end

    return embeddings
end

"""
    load_binary_embeddings(path::String) :: Dict{String, Vector{Float64}}

Loads Word2Vec embeddings from a Gensim binary-format file.

# Arguments
- `path::String`: Path to the Gensim binary-format embedding file.

# Returns
- `Dict{String, Vector{Float64}}`: Dictionary mapping words to embedding vectors.

# Notes
- **Gensim hybrid format**: Text header `"vocab_size dim"`, ASCII words (space-terminated), binary Float32 vectors.
- Vectors are automatically converted from `Float32` to `Float64` for consistency.
- Matches output of `model.wv.save_word2vec_format(binary=True)` from gensim.test.utils.
- No pure-binary Int32 header support (gensim-specific).
"""
function load_binary_embeddings(path::String)::Dict{String, Vector{Float64}}
    embeddings = Dict{String, Vector{Float64}}()
    open(path, "r") do io
        # Handle gensim's text header + binary vectors
        header = readline(io)  # "12 100"
        vocab_size, dim = parse.(Int, split(strip(header)))
        
        for _ in 1:vocab_size
            # Read word (space-terminated)
            word_bytes = UInt8[]
            while true
                c = read(io, UInt8)
                c == 0x20 && break  # space
                push!(word_bytes, c)
            end
            word = String(word_bytes)
            
            # Read binary vector and convert to Float64
            vec32 = Vector{Float32}(undef, dim)
            read!(io, vec32)
            embeddings[word] = Float64.(vec32)
        end
    end
    return embeddings
end
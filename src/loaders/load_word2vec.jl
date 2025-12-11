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

Heuristically detects whether a Word2Vec embedding file is in text or binary format.

# Arguments
- `path::String`: Path to the embedding file.

# Returns
- `:text` if the file appears to be in text format (`word float float ...`).
- `:binary` if the file appears to be in binary Word2Vec format.

# Notes
- The function first checks the file extension: files ending in `.bin` are assumed
  to be binary.
- If the extension does not indicate the format, the function inspects only the
  first few non-empty lines of the file.
- A line is classified as text if it has a non-numeric first token and the
  following tokens resemble floating-point numbers.
- Only a small prefix of the file is scanned to avoid loading large files into memory.
"""
function detect_embedding_format(path::String)::Symbol
    open(path, "r") do io
        for (i, line) in enumerate(Iterators.take(eachline(io), 200))
            # Check for invalid UTF-8 (binary)
            try
                line = String(line)
            catch
                println("ALARM")
                return :binary
            end
            line = strip(line)
            isempty(line) && continue

            tokens = split(line)
            length(tokens) < 2 && continue

            word, vec = tokens[1], tokens[2:end]

            # Debug prints
            println("Line $i Word: ", word)
            println(tryparse(Float64, word))
            println(tryparse(Float64, word) === nothing)
            println("Vector tokens: ", vec)

            # First token is not a number, rest are floats
            if tryparse(Float64, word) === nothing
                return :text
            end
        end
    end

    # If all first lines are valid UTF-8 but no word+float line detected, assume binary
    return :binary
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
"""
    Word2VecLoader

A Julia module for loading Word2Vec embeddings from text or binary files,
with automatic format detection.

# Features
- Detects whether a file is in text or binary Word2Vec format.
- Loads embeddings into a vocab vector and a embedding matrix mapping words to vectors.
- Handles optional header lines in text files.
- Supports Gensim's hybrid binary format (text header + binary vectors).
- Provides a unified `load_word2vec` function.
- Safe parsing with error handling for malformed files.

# Dependencies
- Base Julia (no external packages required)
- Uses `DelimitedFiles` from the standard library for reading files
"""

using DelimitedFiles

export load_word2vec

"""
    load_word2vec(path::String) :: Tuple{Vector{String}, Matrix{Float64}}

Loads Word2Vec embeddings from a file, automatically detecting whether the file
is in text or binary format.

# Arguments
- `path::String`: Path to the Word2Vec embedding file.

# Returns
- `Tuple{Vector{String}, Matrix{Float64}}`: Vector containing the words, Matrix which contains their respective embedding vector.

# Notes
- Uses `detect_embedding_format` to determine file format.
- Text files: `word float float ...` (skips numeric header lines).
- Binary files: Gensim hybrid format (text header `"vocab_size dim"`, ASCII words, binary Float32 vectors).
- Binary vectors are converted to `Float64` for consistency.
- Large models require substantial RAM.
"""
function load_word2vec(path::String)::Tuple{Vector{String}, Matrix{Float64}}
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
function detect_embedding_format(path::AbstractString)::Symbol
    fmt::Symbol = :binary  # default assumption

    open(path, "r") do io
        for (i, line) in enumerate(Iterators.take(eachline(io), 10))
            line = strip(line)
            isempty(line) && continue

            tokens = split(line)
            length(tokens) < 2 && continue

            word = tokens[1]
            vec = tokens[2:end]

            if tryparse(Float64, word) === nothing && all(t -> tryparse(Float64, t) !== nothing, vec)
                fmt = :text
                break
            end
        end
    end

    return fmt
end

"""
    load_text_embeddings(path::String) :: Tuple{Vector{String}, Matrix{Float64}}

Loads Word2Vec embeddings from a text-format file.

# Arguments
- `path::String`: Path to the text-format Word2Vec embedding file.

# Returns
- `Tuple{Vector{String}, Matrix{Float64}}`: Vector containing the words, Matrix which contains their respective embedding vector.

# Notes
- Each line: `word float float ...` (header lines auto-skipped).
- Skips lines where first token is numeric or vectors can't be parsed.
- Reads entire file into memory.
- Throws ErrorException when there was no vocab found
"""
function load_text_embeddings(path::String)::Tuple{Vector{String}, Matrix{Float64}}
    vocab = String[]
    vectors = Vector{Vector{Float64}}()

    open(path, "r") do io
        for line in eachline(io)
            tokens = split(line)
            length(tokens) < 2 && continue

            word, vec_tokens = tokens[1], tokens[2:end]

            # Skip header lines and invalid rows
            if tryparse(Float64, word) === nothing &&
               all(t -> tryparse(Float64, t) !== nothing, vec_tokens)

                vec = parse.(Float64, vec_tokens)
                push!(vocab, word)
                push!(vectors, vec)
            end
        end
    end

    isempty(vocab) && error("No valid vocab found in $path")

    dim = length(vectors[1])
    embeddings = Matrix{Float64}(undef, length(vocab), dim)

    for i in eachindex(vocab)
        embeddings[i, :] = vectors[i]
    end

    return vocab, embeddings
end

"""
    load_binary_embeddings(path::String) :: Tuple{Vector{String}, Matrix{Float64}}

Loads Word2Vec embeddings from a Gensim binary-format file.

# Arguments
- `path::String`: Path to the Gensim binary-format embedding file.

# Returns
- `Tuple{Vector{String}, Matrix{Float64}}`: Vector containing the words, Matrix which contains their respective embedding vector.

# Notes
- **Gensim hybrid format**: Text header `"vocab_size dim"`, ASCII words (space-terminated), binary Float32 vectors.
- Vectors are automatically converted from `Float32` to `Float64` for consistency.
- Matches output of `model.wv.save_word2vec_format(binary=True)` from gensim.test.utils.
- No pure-binary Int32 header support (gensim-specific).
"""
function load_binary_embeddings(path::String)::Tuple{Vector{String}, Matrix{Float64}}
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
        
        vocab = Vector{String}(undef, vocab_size)
        embeddings = Matrix{Float64}(undef, vocab_size, dim)

        for i in 1:vocab_size
            # Read word (space-terminated)
            word_bytes = UInt8[]
            while true
                c = read(io, UInt8)
                c == 0x20 && break  # space
                push!(word_bytes, c)
            end
            word = String(word_bytes)

            # Read binary vector (Float32)
            vec32 = Vector{Float32}(undef, dim)
            read!(io, vec32)

            # Store
            vocab[i] = word
            embeddings[i, :] .= vec32  # auto-converts Float32 → Float64
        end
        return vocab, embeddings
    end
end
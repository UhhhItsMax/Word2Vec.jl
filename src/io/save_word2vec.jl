"""
    save_word2vec(model::Word2VecModel, path::AbstractString; format::Symbol = :text)

Save a `Word2VecModel` to disk in Word2Vec-compatible text or binary format.

# Arguments
- `model::Word2VecModel`: The Word2Vec model to save.
- `path::AbstractString`: Destination file path.

# Keyword Arguments
- `format::Symbol`: Output format.
  - `:text`   — Human-readable text format (`word val1 val2 ...`).
  - `:binary` — Binary Word2Vec format using `Float32` vectors.
  Default: `:text`.

# Returns
- `AbstractString`: The output file path.

# Notes
- Text format is compatible with tools such as Gensim.
- Binary format follows the original Word2Vec C implementation layout.
- Binary files are typically smaller and faster to load, but not human-readable.

# Throws
- `ArgumentError`: If `format` is not one of `:text` or `:binary`.
"""
function save_word2vec(model::Word2VecModel, path::AbstractString; format::Symbol = :text)
    format === :text   && return _save_word2vec_text(model, path)
    format === :binary && return _save_word2vec_binary(model, path)

    throw(ArgumentError("Unknown format $format. Use :text or :binary."))
end


"""
    _save_word2vec_text(model::Word2VecModel, path::AbstractString)

Save a `Word2VecModel` to disk in the standard Word2Vec **text** format.

# Arguments
- `model::Word2VecModel`: The Word2Vec model to be saved.
- `path::AbstractString`: File path where the embeddings will be written.

# Returns
- `AbstractString`: The path to which the model was saved.

# Notes
- The output file starts with a header line of the form `vocab_size dim`.
- Each subsequent line contains a word followed by its embedding values:
  `word float float ...`.
- Embedding vectors are written in full precision (`Float64`) as stored in
  the model.
- Existing files at `path` will be overwritten.
- The entire model is written sequentially; no compression is applied.
"""
function _save_word2vec_text(model::Word2VecModel, path::AbstractString)
    vocab_size = length(model.vocab)
    dim = size(model.embeddings, 1)

    open(path, "w") do io
        # Header
        println(io, "$vocab_size $dim")

        # One word per line
        for (j, word) in enumerate(model.vocab)
            vec = model.embeddings[:, j]
            println(io, join([word; string.(vec)], " "))
        end
    end

    return path
end


"""
    _save_word2vec_binary(model::Word2VecModel, path::AbstractString)

Save a `Word2VecModel` to disk in the classic Word2Vec **binary** format.

# Arguments
- `model::Word2VecModel`: The Word2Vec model to be saved.
- `path::AbstractString`: Destination file path for the binary embedding file.

# Returns
- `AbstractString`: The path to which the model was written.

# Notes
- The file is written in the standard Word2Vec binary format:
  - A header line `vocab_size dim` followed by a newline.
  - For each vocabulary entry:
    - The word as raw bytes, terminated by a space (`0x20`).
    - The embedding vector stored as `Float32` values.
    - A trailing newline byte (`0x0A`).
- Embedding vectors are converted from `Float64` to `Float32` before writing,
  matching the original Word2Vec binary specification.
- The entire model is written sequentially and no compression is applied.
- Existing files at `path` will be overwritten.
"""
function _save_word2vec_binary(model::Word2VecModel, path::AbstractString)
    vocab_size = length(model.vocab)
    dim = size(model.embeddings, 1)

    open(path, "w") do io
        # Header
        write(io, "$vocab_size $dim\n")

        for (j, word) in enumerate(model.vocab)
            # Write word + space
            write(io, word)
            write(io, UInt8(0x20))  # space

            # Convert vector to Float32 (Word2Vec standard)
            vec32 = convert.(Float32, model.embeddings[:, j])
            write(io, vec32)
        end
    end

    return path
end

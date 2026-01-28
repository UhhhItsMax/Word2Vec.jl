
"""
	Word2VecModel

Unified in-memory representation for Word2Vec embeddings.

Fields:
- vocab::Vector{String}         	list of words
- embeddings::Matrix{Float64}   	size = (dim, vocab_size)
- vector_norms::Vector{Float64} 	norms of embedding vectors
- word_to_index::Dict{String,Int}  	maps words to column indices
"""
struct Word2VecModel{T<:Real}
    vocab::Vector{String}
    embeddings::Matrix{T}
    vector_norms::Vector{T}
    word_to_index::Dict{String,Int}
end

function Word2VecModel(vocab::Vector{String}, embeddings::Matrix{T}) where {T<:Real}
    size(embeddings, 2) == length(vocab) || throw(ArgumentError("embeddings must have one column per vocab entry"))

    word_to_index = Dict(word => idx for (idx, word) in enumerate(vocab))
    vector_norms = Vector{T}(undef, size(embeddings, 2))

    @inbounds for (j, col) in enumerate(eachcol(embeddings))
        n = norm(col)
        iszero(n) && throw(ArgumentError("embedding vector has zero norm for word $(vocab[j])"))
        vector_norms[j] = convert(T, n)
    end

    return Word2VecModel{T}(vocab, embeddings, vector_norms, word_to_index)
end


"""
	get_embedding(model::Word2VecModel, word::AbstractString)

Returns a view of the embedding vector for a given word.
"""
get_embedding(model::Word2VecModel, word::String) = @view model.embeddings[:, model.word_to_index[word]]


"""
	get_embedding_norm(model::Word2VecModel, word::AbstractString)

Returns the precomputed norm of an embedding vector for a given word.

Throws an error if the given word is not in the vocabulary of the model.
"""
get_embedding_norm(model::Word2VecModel, word::String) = model.vector_norms[model.word_to_index[word]]


"""
	from_dict_data(embeddings_map::Dict{String,Vector{T}})

Constructs a Word2VecModel from (word => vector) mappings.
"""
function from_dict_data(embeddings_map::Dict{String,Vector{T}}) where T<:AbstractFloat
	words = collect(keys(embeddings_map))
	dim = length(first(values(embeddings_map)))

	M = Array{T}(undef, dim, length(words))

	for (i, w) in enumerate(words)
		M[:, i] =  convert.(T, embeddings_map[w])
	end

	return Word2VecModel(words, M)
end

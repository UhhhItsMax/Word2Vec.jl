export Word2VecModel, from_pretrained_model, get_embedding, get_embedding_norm

using LinearAlgebra: norm, dot


"""
	Word2VecModel

Unified in-memory representation for Word2Vec embeddings.

Fields:
- vocab::Vector{String}         	list of words
- embeddings::Matrix{Float32}   	size = (dim, vocab_size)
- vector_norms::Vector{Float32} 	norms of embedding vectors
- word_to_index::Dict{String,Int}  	maps words to column indices
"""
struct Word2VecModel
	vocab::Vector{String}
	embeddings::Matrix{Float32}
	vector_norms::Vector{Float32}
	word_to_index::Dict{String,Int}

	function Word2VecModel(vocab::Vector{String}, embeddings::Matrix{Float32})
		size(embeddings, 2) == length(vocab) || throw(ArgumentError("embeddings must have one column per vocab entry"))

		word_to_index = Dict(word => idx for (idx, word) in enumerate(vocab))
		vector_norms = Vector{Float32}(undef, size(embeddings, 2))

		@inbounds for (j, col) in enumerate(eachcol(embeddings))
			n = norm(col)
			n == 0 && throw(ArgumentError("embedding vector has zero norm for word $(vocab[j])"))
			vector_norms[j] = Float32(n)
		end

		return new(vocab, embeddings, vector_norms, word_to_index)
	end
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

	M = Array{Float32}(undef, dim, length(words))

	for (i, w) in enumerate(words)
		M[:, i] = Float32.(embeddings_map[w])
	end

	return Word2VecModel(words, M)
end


"""
	load_pretrained_model(path::String; file_type::Symbol = :auto) -> Word2VecModel

Loads embeddings from disk and returns a dense Word2Vec model.

# Arguments
- `path`: Filesystem path to the embeddings.
- `file_type`: `:auto` to detect, or explicitly `:binary` / `:text`.
"""
function load_pretrained_model(path::String; fmt::Symbol = :auto)
	#TODO: refactor loaders to directly populate a preallocated embedding matrix instead of reading data into hash-map 
	
	if fmt !== :binary && fmt !== :text && fmt !== :auto
		throw(ArgumentError("file_type must be :auto, :binary or :text"))
	end

	if fmt === :auto
		 fmt = detect_embedding_format(path)
	end

	embeddings_map = (fmt === :binary) ? load_binary_embeddings(path) : load_text_embeddings(path)

	return from_dict_data(embeddings_map)
end



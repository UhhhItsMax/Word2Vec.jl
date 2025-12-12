export Word2VecModel, from_pretrained

"""
	Word2VecModel

Unified in-memory representation for Word2Vec embeddings.

Fields:
- vocab::Vector{String}         list of words
- embeddings::Matrix{Float32}   size = (dim, vocab_size)
"""
struct Word2VecModel
	vocab::Vector{String}
	embeddings::Matrix{Float32}
end


"""
	from_pretrained(embeddings::Dict{String,Vector{T}})

Constructs a Word2VecModel from pretrained vectors.

Accepts Dict{String,Vector{Float32}} or Dict{String,Vector{Float64}}.
"""
function from_pretrained(emb::Dict{String,Vector{T}}) where T<:AbstractFloat
	words = collect(keys(emb))
	dim = length(first(values(emb)))

	M = Array{Float32}(undef, dim, length(words))

	for (i, w) in enumerate(words)
		M[:, i] = Float32.(emb[w])
	end

	return Word2VecModel(words, M)
end
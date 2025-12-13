module Word2Vec

export Word2VecModel,
       load_word2vec,
       get_embedding,
       get_approximate_word,
       get_related_words,
       get_analogy_word,
       train_skipgram,
       train_cbow,
       conec_embedding,
       from_pretrained

include("models/Word2VecModel.jl")

include("loaders/load_word2vec.jl")

include("training/skipgram.jl")
include("training/cbow.jl")

include("conec/conec.jl")

include("utils/math_utils.jl")

end

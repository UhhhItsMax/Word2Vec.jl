module Word2Vec

export Word2VecModel,
       load_word2vec,
       get_embedding,
       train_skipgram,
       train_cbow,
       conec_embedding

include("models/Word2VecModel.jl")

include("loaders/text_loader.jl")
include("loaders/binary_loader.jl")

include("training/skipgram.jl")
include("training/cbow.jl")

include("conec/conec.jl")

include("utils/math_utils.jl")

end

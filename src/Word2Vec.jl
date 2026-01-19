module Word2Vec

export Word2VecModel,
       load_word2vec,
       save_word2vec,
       get_embedding,
       train_cbow,
       from_pretrained,
       load_pretrained_model,
       analogy,
       similarity,
       benchmark_cbow_for_dim,
       benchmark_cbow_for_epochs,
       benchmark_cbow_for_window


include("models/Word2VecModel.jl")

include("loaders/load_word2vec.jl")
include("savers/save_word2vec.jl")

include("training/cbow.jl")

include("conec/conec.jl")

include("evaluation/evaluation.jl")
include("benchmarking/benchmarking.jl")

end

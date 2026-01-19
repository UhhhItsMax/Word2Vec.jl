module Word2Vec

export Word2VecModel,
       load_word2vec,
       save_word2vec,
       get_embedding,
       train_cbow,
       conec_embedding,
       from_pretrained,
       load_pretrained_model,
       embedding_points,
       tsne_embeddings,
       ConEcModel,
       conec_embeddings_for_file,
       SparseContextMatrix,
       load_sparse_context_matrix,
       save_sparse_context_matrix,
       analogy,
       similarity,
       benchmark_cbow_for_dim,
       benchmark_cbow_for_epochs,
       benchmark_cbow_for_window,
       benchmark_model_quality


include("models/Word2VecModel.jl")

include("loaders/load_word2vec.jl")
include("savers/save_word2vec.jl")

include("utils/circular_buffer.jl")

include("training/cbow.jl")

include("conec/context_matrix.jl")
include("conec/conec.jl")

include("utils/math_utils.jl")
include("utils/normalize.jl")
include("utils/io_utils.jl")
include("visualization/tsne.jl")
include("visualization/plotting.jl")

include("evaluation/evaluation.jl")
include("benchmarking/benchmarking.jl")

end

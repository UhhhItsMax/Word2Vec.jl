module Word2Vec

using Plots: plot, display, scatter, annotate!
using BenchmarkTools: @benchmark, Trial, minimum
using SparseArrays: SparseMatrixCSC, sparse, spzeros, nzrange, rowvals, nonzeros, nnz
using Serialization: serialize, deserialize
using LinearAlgebra: norm, dot, mul!
using Random: shuffle!, seed!, MersenneTwister
using Statistics: mean
using TSne: tsne
using Distances: evaluate, SqEuclidean

include("models/Word2VecModel.jl")
include("loaders/load_word2vec.jl")
include("savers/save_word2vec.jl")
include("training/cbow.jl")
include("conec/context_matrix.jl")
include("conec/conec.jl")
include("utils/circular_buffer.jl")
include("utils/math_utils.jl")
include("utils/normalize.jl")
include("utils/io_utils.jl")
include("visualization/tsne.jl")
include("visualization/plotting.jl")
include("evaluation/evaluation.jl")
include("benchmarking/benchmarking.jl")

export Word2VecModel,
    get_embedding,
    get_embedding_norm,
    from_dict_data,
    load_word2vec,
    save_word2vec,
    train_cbow,
    read_corpus_sentences,
    read_corpus_tokens,
    read_wordlist,
    conec_embedding,
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
    SimilarityTest,
    AnalogyTest,
    benchmark_model_quality,
    benchmark_conec_for_window,
    benchmark_conec_for_local_corpus_size,
    benchmark_conec_for_dim

end

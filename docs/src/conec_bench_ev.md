# ConEc Embedding Benchmarking

This page describes **quantitative evaluation** tools for ConEc embeddings, including runtime benchmarking across window size, local corpus size, and CBOW embedding dimensions.

---

### Benchmarking by Window Size

Compute ConEc embeddings across multiple context window sizes.

#### Example

```julia
# Assuming `cm` is a ConEcModel and local_corpus.txt exists
window_results = benchmark_conec_for_window(cm, "local_corpus.txt", [1,2,5])
```

#### Notes

- Returns a `Dict{Int, BenchmarkTools.Trial}` mapping window size to runtime.
- Plots a line graph of computation time (ms) vs window size automatically.

### Benchmarking by Local Corpus Size

Compute ConEc embeddings across multiple local corpora of varying sizes.

#### Example

```julia
local_paths = ["local_small.txt", "local_medium.txt", "local_large.txt"]
corpus_results = benchmark_conec_for_local_corpus_size(cm, local_paths)
```

#### Notes

- Returns a `Dict{String, BenchmarkTools.Trial}` mapping corpus file to runtime.
- Helps evaluate how computation scales with the number of words/sentences in local documents.

### Benchmarking by Embedding Dimension

Compute ConEc embeddings for ConEc models built with CBOW embeddings of different dimensions.

#### Example

```julia
models = [cm50, cm100, cm200]  # ConEcModels trained with 50, 100, 200 dimensions
dims = [50, 100, 200]
dim_results = benchmark_conec_for_dim(models, "local_corpus.txt", dims)
```

#### Notes

- Returns a `Dict{Int, BenchmarkTools.Trial}` mapping embedding dimension to runtime.
- plots computation time vs embedding dimension.
- assesses runtime impact of increasing CBOW embedding size.

### General Notes

- All benchmarking functions return `BenchmarkTools.Trial` objects for precise runtime measurement.
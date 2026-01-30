# Word2Vec Model Evaluation & Benchmarking

This page describes **qualitative and quantitative evaluation** tools for a Word2VecModel, including similarity tests, analogy queries, and CBOW training benchmarks.

---

## Quality Evaluation

The module provides different utilites to evaluate and benchmark the quality of a given Word2vec model.

### Cosine Similarity

You can compute the cosine similarity between two words in a trained Word2Vec model.

#### Example

```@example
using Word2Vec # hide
vocab = ["king", "queen", "man", "woman"] # hide
emb = randn(5, 4) # hide
model = Word2VecModel(vocab, emb) # hide
sim_score = similarity(model, "king", "queen")
println("Cosine similarity king ↔ queen: ", sim_score)
```

#### Notes

- Returns a Float64 in $[-1.0, 1.0]$.
- Throws a KeyError if a word is not in the vocabulary.
- Throws ArgumentError if a vector has zero norm.

### Analogy

The `analogy` function allows you to solve word analogies using a trained Word2Vec model. 

#### Example

```@example
using Word2Vec # hide
vocab = ["king", "queen", "man", "woman"] # hide
emb = randn(5, 4) # hide
model = Word2VecModel(vocab, emb) # hide
# Predict "queen" given king : man :: ? : woman
preds = analogy(model, "king", "man", "woman"; topk=1)
println(preds)
```

#### Notes

- Cosine similarity is used to rank candidate words.
- Input words $(a, b, c)$ are excluded from the results.
- Embedding vectors must be non-zero; otherwise similarity is undefined.


### Benchmarking

You can use the following function and structs to perform quality benchmarks on a given Word2Vec model.

#### Example

```julia
# Define similarity tests
sim_tests = [
    SimilarityTest("king", "queen"; higher_than = ["man", "dog"]),
    SimilarityTest("cat", "dog"; higher_than = ["car", "tree"])
]

# Define analogy tests
ana_tests = [
    AnalogyTest("king", "queen", "man", ["woman"]),
    AnalogyTest("paris", "france", "berlin", ["germany", "Deutschland"])
]

# Benchmark model
results = benchmark_model_quality(
    model;
    similarity_tests = sim_tests,
    analogy_tests = ana_tests,
    topk = 5
)

println("Similarity accuracy: ", results.similarity.accuracy)
println("Analogy accuracy: ", results.analogy.accuracy)
```

#### Notes

- SimilarityTest checks that $w_{1}$ is more similar to $w_{2}$ than to any word in higher_than.
- AnalogyTest checks vector arithmetic predictions $a : b \approx c : ?$ and allows for multiple valid answers.
- benchmark_model_quality returns a NamedTuple containing:
    - similarity — Number of passed tests, total tests, accuracy, and individual results.
    - analogy — Same as above but for analogy tests.
- Any errors during evaluation (e.g., missing words) are counted as failed tests.
- Use topk to adjust how many candidate words are considered for analogies.

## Quantity benchmarking

THe module provides utilities to **benchmark CBOW training performance** across different hyperparameters.  
This helps understand how embedding size, context window, or number of epochs affects runtime.

### Examples

```julia
sentences = [
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"]
]

# Benchmark across different number of epochs
epoch_results = benchmark_cbow_for_epochs(
    sentences,
    [1, 2, 5, 10];
    dim = 50,
    window = 2,
    lr = 0.05,
    min_count = 1,
    seed = 42
)

# Benchmark across embedding dimensions
dim_results = benchmark_cbow_for_dim(
    sentences,
    [10, 25, 50, 100];
    window = 2,
    epochs = 5
)

# Benchmark across context window sizes
window_results = benchmark_cbow_for_window(
    sentences,
    [1, 2, 5];
    dim = 50,
    epochs = 5
)
```

### Notes

- All benchmarking functions return a `Dict{Int, BenchmarkTools.Trial}` mapping the parameter value to its timing result.
- The functions automatically plot training time vs. the tested parameter for quick visualization.
- Use these benchmarks to optimize CBOW training settings for your corpus size and hardware.

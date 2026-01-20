```@meta
CurrentModule = Word2Vec
```
# CBOW training

The core of Word2Vec.jl is the Continuous Bag-of-Words (CBOW) algorithm. CBOW trains a Word2VecModel by predicting a target word from the average of its surrounding context words. This is the foundation for embedding queries and extensions like ConEc.

CBOW uses a full softmax with cross-entropy loss and is optimized for small corpora. Training produces the input embedding matrix W_in, stored in model.embeddings, which can be used for similarity, analogy, or ConEc computations.

## Training example

For an example corpus:

```julia
# Example corpus (sentences or flat token stream)
sentences = [["the", "quick", "brown", "fox"],
             ["fox", "jumps", "over", "the", "lazy", "dog"]]

# Train CBOW model
model = train_cbow(sentences; dim=50, window=5, epochs=10, lr=0.05, seed=42, verbose=true)
```

For an real corpus inside a file:

```julia
# Preserve sentence boundaries
sentences = read_corpus_sentences("corpus.txt")
model1 = train_cbow(sentences)

# Or flatten all tokens into a single stream
tokens = read_corpus_tokens("corpus.txt")
model2 = train_cbow(tokens)
```

## Key arguments

- dim — Embedding dimensionality (columns of model.embeddings). Default = 50
- window — Symmetric context window size around the target word. Default = 2
- epochs — Number of passes over the corpus. Default = 5
- lr — Learning rate for stochastic gradient updates. Default = 0.05
- min_count — Ignore words with frequency below this threshold. Default = 1
- seed — RNG seed for reproducibility. Default = 42
- verbose — If true, prints average loss per epoch. Default = false

## Notes

- Context windows do not cross sentence boundaries when using read_corpus_sentences.
- Words filtered out by min_count are ignored.
- train_cbow throws an error if the corpus is empty or if all tokens are filtered out.
- The resulting Word2VecModel contains:
    - vocab — List of words
    - embeddings — Input word vectors (W_in)
    - vector_norms — Precomputed norms for similarity computations
    - word_to_index — Maps words to embedding columns

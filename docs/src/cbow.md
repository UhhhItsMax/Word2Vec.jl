```@meta
CurrentModule = Word2Vec
```
# CBOW training

The core of Word2Vec.jl is the Continuous Bag-of-Words (CBOW) algorithm. CBOW trains a Word2VecModel by predicting a target word from the average of its surrounding context words. This is the foundation for embedding queries and extensions like ConEc.

CBOW uses a full softmax with cross-entropy loss and is optimized for small corpora. Training produces the input embedding matrix W_in, stored in model.embeddings, which can be used for similarity, analogy, or ConEc computations.


```@docs
train_cbow
```

## Training example

For an example corpus:

```@example
using Word2Vec # hide
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
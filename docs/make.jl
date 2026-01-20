using Word2Vec
using Documenter

DocMeta.setdocmeta!(Word2Vec, :DocTestSetup, :(using Word2Vec); recursive=true)

makedocs(
    modules = [Word2Vec],
    checkdocs = :exports,
    authors = "Maximilian Hans <hans.maximilian@icloud.com>, Paul Mathias Nelde <paul.nelde@fu-berlin.de>, Mika Paul Merten <merten@campus.tu-berlin.de>",
    sitename = "Word2Vec.jl",
    format = Documenter.HTML(
        canonical = "https://uhhhitsmax.github.io/Word2Vec.jl/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Word2VecModel" => "model.md",
        "Model I/O" => "io.md",
        "CBOW Training" => "cbow.md",
        "Evaluation and Benchmarking for Word2VecModel" => "w2v_bench_ev.md",
        "Visualization" => "visualization.md",
        "ConEc Embeddings" => "conec.md",
        "API Reference" => "api.md"
    ],
)

deploydocs(
    repo = "github.com/UhhhItsMax/Word2Vec.jl.git",
    devbranch = "main",

    devurl = "dev",
    versions = [
        "stable" => "v^",
    ],

    forcepush = true,
)
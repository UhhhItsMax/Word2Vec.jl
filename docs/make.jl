using Word2Vec
using Documenter

DocMeta.setdocmeta!(Word2Vec, :DocTestSetup, :(using Word2Vec); recursive=true)

makedocs(
    modules = [Word2Vec],
    checkdocs = :exports,
    authors = "Maximilian Hans <hans.maximilian@icloud.com>, Paul Mathias Nelde <paulnelde@gmail.com>, Mika Paul Merten <merten@campus.tu-berlin.de>",
    sitename = "Word2Vec.jl",
    format = Documenter.HTML(
        canonical = "https://uhhhitsmax.github.io/Word2Vec.jl/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "CBOW Training" => "cbow.md",
        "ConEc Embeddings" => "conec.md",
        "Visualization" => "visualization.md",
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
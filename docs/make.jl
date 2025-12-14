using Word2Vec
using Documenter

DocMeta.setdocmeta!(Word2Vec, :DocTestSetup, :(using Word2Vec); recursive=true)

makedocs(
    modules = [Word2Vec],
    authors = "Maximilian Hans <hans.maximilian@icloud.com>",
    sitename = "Word2Vec.jl",
    format = Documenter.HTML(
        canonical = "https://uhhhitsmax.github.io/Word2Vec.jl/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
)

deploydocs(
    repo = "github.com/UhhhItsMax/Word2Vec.jl.git",
    devbranch = "main",

    devurl = "docs/dev",
    versions = [
        "docs/stable" => "v^",
    ],

    forcepush = true,
)
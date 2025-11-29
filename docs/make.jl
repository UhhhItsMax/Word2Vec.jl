using Word2Vec
using Documenter

DocMeta.setdocmeta!(Word2Vec, :DocTestSetup, :(using Word2Vec); recursive=true)

makedocs(;
    modules=[Word2Vec],
    authors="Maximilian Hans hans.maximilian@icloud.com",
    sitename="Word2Vec.jl",
    format=Documenter.HTML(;
        canonical="https://UhhhItsMax.github.io/Word2Vec.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/UhhhItsMax/Word2Vec.jl",
    devbranch="main",
)

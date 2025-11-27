using Word2Vec
using Documenter

DocMeta.setdocmeta!(Word2Vec, :DocTestSetup, :(using Word2Vec); recursive=true)

makedocs(;
    modules=[Word2Vec],
    authors="Maximilian Willem Hans <m.hans@tu-berlin.de>, Jan-Erik Hein <jan-erik.hein@campus.tu-berlin.de>, Mika Paul Merten <merten@campus.tu-berlin.de>, Paul Mathias Nelde <paul.nelde@fu-berlin.de>",
    sitename="Word2Vec.jl",
    format=Documenter.HTML(;
        canonical="https://UhhhItsMax.github.io/Word2Vec.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
)

deploydocs(;
    repo="github.com/UhhhItsMax/Word2Vec.jl",
    devbranch="main",
)

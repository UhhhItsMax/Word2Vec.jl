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

let landing = normpath(joinpath(@__DIR__, "..", "site", "index.html")),
    target  = normpath(joinpath(@__DIR__, "build", "index.html"))

    if isfile(landing)
        mkpath(dirname(target))
        cp(landing, target; force = true)
    else
        @warn "Landing page not found at $(landing). Create site/index.html to enable the Pages landing page."
    end
end

deploydocs(
    repo = "github.com/UhhhItsMax/Word2Vec.jl.git",
    devbranch = "main",
    devurl = "docs/dev",
    versions = [
        "docs/stable" => "v^",
    ],
    forcepush = true,
)
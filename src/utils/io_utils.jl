
"""
    read_wordlist(path::AbstractString) -> Vector{String}

Read a word list file (one word per line). Empty lines and lines starting with `#` are ignored.
"""
function read_wordlist(path::AbstractString)
    out = String[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(out, s)
    end
    return out
end

"""
    clean_line(line::AbstractString;
               lowercase::Bool=true,
               strip_punct::Bool=true,
               dash_rule::Bool=true) -> Union{String,Nothing}

Clean a single text line for tokenization.

- Trims whitespace.
- Optionally lowercases.
- If the stripped line consists only of dashes (`-`):
  - If `dash_rule=true` and the line has 3+ dashes, drop it (treat as separator) => `nothing`.
  - Otherwise keep the dash line verbatim (even if `strip_punct=true`).
- If `strip_punct=true`, replaces non-alphanumeric characters with spaces.
- Collapses internal whitespace to a single space.

Returns `nothing` if the line becomes empty (or is dropped by the dash rule),
otherwise returns the cleaned line as a `String`.
"""
function clean_line(line::AbstractString;
    lowercase::Bool=true,
    strip_punct::Bool=true,
    dash_rule::Bool=true
)
    s = strip(String(line))
    isempty(s) && return nothing

    if occursin(r"^-+$", s)
        if dash_rule && occursin(r"^-{3,}$", s)
            return nothing
        else
            return s
        end
    end

    if lowercase
        s = Base.Unicode.lowercase(s)
    end

    if strip_punct
        s = replace(s, r"[^A-Za-z0-9\s]+" => " ")
    end

    s = replace(s, r"\s+" => " ")
    s = strip(s)

    return isempty(s) ? nothing : s
end

"""
    read_corpus_sentences(path::AbstractString; kwargs...) -> Vector{Vector{String}}

Read a corpus file where each line is treated as a sentence.

Each non-empty cleaned line becomes `split(cleaned_line)` and is appended as a sentence.
This preserves sentence boundaries so context windows do not cross sentences.
Keyword arguments are forwarded to `clean_line`.
"""
function read_corpus_sentences(path::AbstractString; kwargs...)
    isfile(path) || throw(ArgumentError("corpus file does not exist: $path"))

    sentences = Vector{Vector{String}}()
    for ln in eachline(path)
        cleaned = clean_line(ln; kwargs...)
        cleaned === nothing && continue
        toks = split(cleaned)
        isempty(toks) && continue
        push!(sentences, toks)
    end
    return sentences
end

"""
    read_corpus_tokens(path::AbstractString; kwargs...) -> Vector{String}

Read a corpus file and return a single flat token stream (`Vector{String}`).

This reads line-by-line, cleans each line with `clean_line`, splits into tokens,
and concatenates all tokens into one vector (sentence boundaries are ignored).
"""
function read_corpus_tokens(path::AbstractString; kwargs...)
    sents = read_corpus_sentences(path; kwargs...)
    out = String[]
    for s in sents
        append!(out, s)
    end
    return out
end
"""
    read_wordlist(path::AbstractString)

Read a word list from a text file, one word per line.

# Arguments
- `path::AbstractString`: Path to the file containing the word list.

# Returns
- `Vector{String}`: Words in the file, in order, with empty lines and lines starting with `#` ignored.

# Notes
- Leading and trailing whitespace on each line is stripped.
- Lines that are empty or start with `#` are skipped.
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
    clean_line(line::AbstractString; lowercase=true, strip_punct=true, dash_rule=true) 

Clean a single line of text for tokenization.

# Arguments
- `line::AbstractString`: Input text line.
- `lowercase::Bool=true`: Convert text to lowercase if true.
- `strip_punct::Bool=true`: Replace non-alphanumeric characters with spaces if true.
- `dash_rule::Bool=true`: Drop lines consisting of 3 or more dashes (`---`) if true.

# Returns
- `String` if the line has meaningful content after cleaning.
- `nothing` if the line is empty or dropped by the dash rule.

# Notes
- Leading/trailing whitespace is trimmed.
- Internal whitespace is collapsed to a single space.
- Lines of dashes are treated according to `dash_rule`.
"""
function clean_line(
        line::AbstractString;
        lowercase::Bool = true,
        strip_punct::Bool = true,
        dash_rule::Bool = true
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
    read_corpus_sentences(path::AbstractString; kwargs...)

Read a text corpus where each line is treated as a separate sentence.

# Arguments
- `path::AbstractString`: Path to the corpus file.
- `kwargs...`: Optional keyword arguments forwarded to `clean_line` (e.g., `lowercase`, `strip_punct`, `dash_rule`).

# Returns
- `Vector{Vector{String}}`: A vector of sentences, each represented as a vector of tokens.

# Notes
- Empty or dropped lines (according to `clean_line`) are skipped.
- Sentence boundaries are preserved: context windows do not cross lines.
- Each line is tokenized by splitting on whitespace after cleaning.
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
    read_corpus_tokens(path::AbstractString; kwargs...)

Read a corpus file and return a flat token vector.

# Arguments
- `path::AbstractString`: Path to the corpus file.
- `kwargs...`: Optional keyword arguments forwarded to `clean_line` (e.g., `lowercase`, `strip_punct`, `dash_rule`).

# Returns
- `Vector{String}`: A single flat list of all tokens in the corpus.

# Notes
- Lines are cleaned using `clean_line` and split into tokens.
- Sentence boundaries are ignored; all tokens are concatenated into one vector.
- Empty lines or lines dropped by `clean_line` are skipped.
"""
function read_corpus_tokens(path::AbstractString; kwargs...)
    sents = read_corpus_sentences(path; kwargs...)
    out = String[]
    for s in sents
        append!(out, s)
    end
    return out
end

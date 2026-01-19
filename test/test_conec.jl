using Test
using Word2Vec
using SparseArrays
using LinearAlgebra: norm


# Test data: small synthetic corpus for reproducible tests
const GLOBAL_CORPUS_TEXT = """
the quick brown fox jumps over the lazy dog
the quick brown fox jumps over a lazy dog
fox jumps quickly over the lazy brown dog
quick brown fox over jumps the lazy dog
"""

const LOCAL_CORPUS_TEXT = """
lazy dog chases quick fox over brown hill
brown fox jumps over lazy dog again
"""

# Create temp files for testing
function create_test_files()
    global_corpus_path = tempname() * ".txt"
    local_corpus_path = tempname() * ".txt"
    
    write(global_corpus_path, GLOBAL_CORPUS_TEXT)
    write(local_corpus_path, LOCAL_CORPUS_TEXT)
    
    return global_corpus_path, local_corpus_path
end

@testset "ConEc: Basic functionality" begin
    global_path, local_path = create_test_files()
    
    try
        # 1. Train a tiny word2vec model on global corpus
        tokens = split(GLOBAL_CORPUS_TEXT)
        w2v = train_cbow(tokens; dim=10, window=2, epochs=3, lr=0.1, 
                        min_count=1, verbose=false)
        
        @test w2v isa Word2VecModel
        @test size(w2v.embeddings) == (10, length(w2v.vocab))
        @test length(w2v.vocab) >= 10  # at least these words: the,quick,brown,etc.
        
        # 2. Build ConEc model with global context
        conec_model = build_conec_global(w2v, global_path; 
                                       window_size=2, min_count=1, a=0.6)
        @test conec_model isa ConEcModel
        @test conec_model.w2v === w2v
        @test conec_model.a ≈ 0.6
        
        # 3. Check global context matrix has expected properties
        global_cm = conec_model.global_cm
        @test global_cm isa SparseContextMatrix{Float64}
        @test nnz(global_cm.mat) > 0
        @test length(global_cm.vocab) >= 10
        
    finally
        rm(global_path)
        rm(local_path)
    end
end

@testset "ConEc: Embedding computation" begin
    global_path, local_path = create_test_files()
    
    try
        tokens = split(GLOBAL_CORPUS_TEXT)
        w2v = train_cbow(tokens; dim=5, window=1, epochs=5, lr=0.2, 
                        min_count=1, seed=42, verbose=false)
        
        conec_model = build_conec_global(w2v, global_path; 
                                       window_size=1, min_count=1, a=0.5)
        
        # 4. Compute ConEc embeddings for local document
        local_embs = conec_embeddings_for_file(conec_model, local_path; 
                                             window_size=1, min_count=1)
        
        @test local_embs isa Dict{String,Vector{Float64}}
        @test length(local_embs) >= 8  # expect: lazy,dog,chases,quick,fox,over,brown,hill,again
        
        # Check known words have reasonable embeddings (non-zero, reasonable norm)
        for word in ["quick", "fox", "lazy", "dog", "brown"]
            @test haskey(local_embs, word)
            emb = local_embs[word]
            @test length(emb) == 5
            @test norm(emb) > 0.01  # non-trivial embedding
            @test all(isfinite, emb)
        end
        
        # 5. OOV word test: "hill" and "chases" should only use local context
        for oov_word in ["hill", "chases"]
            @test haskey(local_embs, oov_word)
            emb = local_embs[oov_word]
            @test norm(emb) > 0.01
        end
        
    finally
        rm(global_path)
        rm(local_path)
    end

    @testset "ConEc: a=1.0 fallback to local context" begin
        global_path, local_path = create_test_files()

        try
            # Train w2v ONLY on global corpus
            tokens = split(GLOBAL_CORPUS_TEXT)
            w2v = train_cbow(
                tokens;
                dim = 6,
                window = 1,
                epochs = 5,
                lr = 0.2,
                min_count = 1,
                seed = 123,
                verbose = false,
            )

            # Force pure-global weighting
            conec_model = build_conec_global(
                w2v,
                global_path;
                window_size = 1,
                min_count = 1,
                a = 1.0,
            )

            embs = conec_embeddings_for_file(
                conec_model,
                local_path;
                window_size = 1,
                min_count = 1,
            )

            # "hill" and "chases" do not occur in the global corpus
            for word in ["hill", "chases"]
                @test haskey(embs, word)

                v = embs[word]
                @test length(v) == 6
                @test norm(v) > 0.0
                @test all(isfinite, v)
            end

        finally
            rm(global_path)
            rm(local_path)
        end
    end

end

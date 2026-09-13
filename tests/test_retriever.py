from langchain_core.documents import Document

from pr_agent.rag.retriever import BM25Index, HybridRetriever, render_chunks, tokenize_code
from pr_agent.rag.splitter import chunk_file, language_for


def test_tokenize_splits_camel_and_snake_case_and_keeps_the_whole_token():
    tokens = tokenize_code("parseHTTPResponse")
    assert "parsehttpresponse" in tokens
    assert {"parse", "http", "response"} <= set(tokens)

    tokens = tokenize_code("max_retry_count")
    assert "max_retry_count" in tokens
    assert {"max", "retry", "count"} <= set(tokens)


def _docs():
    return [
        Document(page_content="def validate_token(token): return token.is_valid",
                 metadata={"path": "auth.py", "chunk": 0, "id": "auth.py:0",
                           "start_line": 1, "end_line": 2}),
        Document(page_content="def render_template(name): return Template(name)",
                 metadata={"path": "views.py", "chunk": 0, "id": "views.py:0",
                           "start_line": 1, "end_line": 2}),
    ]


def test_bm25_ranks_the_matching_chunk_first():
    index = BM25Index(_docs())
    hits = index.search("validate token", k=2)
    assert hits[0][0].metadata["path"] == "auth.py"
    assert hits[0][1] > 0


def test_bm25_handles_an_empty_corpus_and_empty_query():
    assert BM25Index([]).search("anything", k=5) == []
    assert BM25Index(_docs()).search("!!!", k=5) == []


def test_bm25_matches_an_identifier_written_in_another_case():
    # A reviewer writing "validateToken" should still reach validate_token.
    hits = BM25Index(_docs()).search("validateToken", k=2)
    assert hits and hits[0][0].metadata["path"] == "auth.py"


def test_hybrid_retriever_without_vectors_falls_back_to_bm25():
    retriever = HybridRetriever(_docs(), vectorstore=None)
    hits = retriever.retrieve("render a template", k=1)
    assert len(hits) == 1
    assert hits[0].path == "views.py"
    assert hits[0].source == "bm25"


def test_retrieve_many_deduplicates_across_queries():
    retriever = HybridRetriever(_docs())
    hits = retriever.retrieve_many(["validate token", "token validation"], k=5)
    paths = [h.path for h in hits]
    assert len(paths) == len(set(paths))


def test_retrieve_ignores_a_blank_query():
    assert HybridRetriever(_docs()).retrieve("   ", k=3) == []


def test_hybrid_retriever_survives_a_broken_vector_store():
    class Exploding:
        def similarity_search(self, *_a, **_k):
            raise RuntimeError("backend down")

    hits = HybridRetriever(_docs(), vectorstore=Exploding()).retrieve("validate token", k=1)
    assert hits and hits[0].path == "auth.py"


def test_chunk_metadata_line_numbers_point_at_the_real_lines():
    content = "\n".join(f"line {i}" for i in range(1, 61))
    docs = chunk_file("notes.txt", content, chunk_size=100, chunk_overlap=0)
    assert len(docs) > 1
    for doc in docs:
        start = doc.metadata["start_line"]
        actual = content.splitlines()[start - 1]
        assert doc.page_content.splitlines()[0] == actual


def test_chunk_file_ignores_empty_content():
    assert chunk_file("empty.py", "   \n\n") == []


def test_language_detection_by_extension():
    assert language_for("a/b/c.py") is not None
    assert language_for("a/b/c.ts") is not None
    assert language_for("a/b/c.unknownext") is None


def test_render_chunks_respects_the_character_budget():
    retriever = HybridRetriever(_docs())
    chunks = retriever.retrieve("validate token", k=2)
    assert "auth.py" in render_chunks(chunks)
    assert render_chunks([]) == "(no relevant code found in the index)"
    assert "budget" in render_chunks(chunks, max_chars=5)

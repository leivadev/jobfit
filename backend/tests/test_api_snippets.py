from backend.api.snippets import build_snippet


def test_build_snippet_strips_leading_position():
    job_text = "Backend Developer We are looking for a strong engineer."

    snippet = build_snippet(job_text, "Backend Developer")

    assert snippet == "We are looking for a strong engineer."


def test_build_snippet_falls_back_to_full_text_when_position_not_a_prefix():
    job_text = "Some job text that does not start with the position."

    snippet = build_snippet(job_text, "Backend Developer")

    assert snippet == job_text


def test_build_snippet_returns_short_text_unchanged():
    job_text = "Backend Developer Short description."

    snippet = build_snippet(job_text, "Backend Developer", max_chars=280)

    assert snippet == "Short description."


def test_build_snippet_truncates_long_text_at_word_boundary():
    description = " ".join(["word"] * 100)
    job_text = f"Backend Developer {description}"

    snippet = build_snippet(job_text, "Backend Developer", max_chars=20)

    assert snippet == "word word word word…"
    assert len(snippet) <= 21


def test_build_snippet_does_not_truncate_text_at_the_limit():
    description = "x" * 20
    job_text = f"Backend Developer {description}"

    snippet = build_snippet(job_text, "Backend Developer", max_chars=20)

    assert snippet == description

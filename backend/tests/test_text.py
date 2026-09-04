import pytest
from text import build_job_text, chunk_text, clean_text

# Real row from lang-uk/recruitment-dataset-job-descriptions-english
# (Position="10 + Blockchain Nodes / Masternodes to set up"), preserving its
# literal \r\n line endings and *Requirements* asterisk-markdown.
REAL_DESCRIPTION = (
    "*Requirements*\r\n\r\n"
    "We're looking for a long term collaboration with someone that has an "
    "experience in crypto, masternodes, nodes, validators etc. We need to set up:"
    "\r\n\r\nKyber Network\r\nNebulas\r\nSecretNetwork\r\nTron\r\nAion\r\nDeFiChain\r\n"
    "EOS\r\nTomoChain\r\nElrond\r\nIRISnet\r\nIoTeX\r\nTerra\r\nChainX\r\nThorchain\r\n\r\n"
    "Succesful candidates will have an opportunity to get more jobs and long "
    "term collaboration."
)

# Real row, Position="1C bas erp" — 271 tokens under all-MiniLM-L6-v2, so it
# splits into a full 256-token chunk plus a 15-token remainder.
REAL_LONG_DESCRIPTION = (
    "We are looking for an experienced 1C Analyst to join our team. You will be "
    "responsible for analyzing and optimizing the functionality of BAS ERP "
    "subsystems.\r\n\r\nRequirements:\r\n"
    "● Knowledge of the principles of operation of the main subsystems of BAS ERP.\r\n"
    "● Strong knowledge of the subject area, including accounting, tax and "
    "management accounting, production, warehousing and logistics.\r\n"
    "● Ability to formalize client requirements and identify gaps between "
    "typical system functionality and these requirements.\r\n"
    "● Experience in building and modeling customer processes in BAS ERP.\r\n"
    "● Understanding of 1C system metadata, such as directories, registers, documents.\r\n"
    "● Experience with workflow systems such as Jira, Trello, Worksection.\r\n\r\n"
    "Responsibilities:\r\n"
    "● Studying the functionality of BAS ERP subsystems and identifying their "
    "compliance with customer requirements.\r\n"
    "● Analyze and identify gaps between the current system functionality and "
    "customer needs.\r\n"
    "● Formalization of customer requirements and preparation of analysis reports.\r\n"
    "● Development and modeling of customer processes in BAS ERP.\r\n"
    "● Collaboration with developers and testers to implement the changes.\r\n"
    "● Ensuring high quality and meeting project deadlines.\r\n\r\n"
    "We are looking for an independent and responsible person with high analytical "
    "thinking. Experience in working with BAS ERP subsystems and knowledge of "
    "accounting systems will be a great advantage. You should have good "
    "communication skills and the ability to work effectively in a team."
)


def test_clean_text_normalizes_crlf_and_strips_asterisk_markdown():
    result = clean_text(REAL_DESCRIPTION)

    assert "\r" not in result
    assert "*" not in result
    assert result.startswith("Requirements We're looking for a long term collaboration")
    assert "  " not in result


def test_clean_text_strips_line_start_bullet_markers():
    text = "Responsibilities:\n- First duty\n* Second duty\n• Third duty"

    result = clean_text(text)

    assert result == "Responsibilities: First duty Second duty Third duty"


def test_clean_text_collapses_repeated_whitespace():
    text = "Too   many\t\tspaces\n\n\nand   blank lines"

    result = clean_text(text)

    assert result == "Too many spaces and blank lines"


def test_build_job_text_joins_position_and_description():
    result = build_job_text("Python Developer", "Line1\r\nLine2")

    assert result == "Python Developer\nLine1\r\nLine2"


@pytest.fixture(scope="module")
def minilm_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.integration
def test_chunk_text_returns_single_chunk_under_max_tokens(minilm_tokenizer):
    text = clean_text(REAL_DESCRIPTION)
    assert len(minilm_tokenizer.encode(text, add_special_tokens=False)) < 256

    chunks = chunk_text(text, minilm_tokenizer, max_tokens=256)

    assert len(chunks) == 1


@pytest.mark.integration
def test_chunk_text_splits_into_non_overlapping_windows_over_max_tokens(minilm_tokenizer):
    text = clean_text(REAL_LONG_DESCRIPTION)
    token_count = len(minilm_tokenizer.encode(text, add_special_tokens=False))
    assert token_count > 256

    chunks = chunk_text(text, minilm_tokenizer, max_tokens=256)
    chunk_lengths = [
        len(minilm_tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks
    ]

    assert len(chunks) > 1
    assert all(length <= 256 for length in chunk_lengths)
    assert sum(chunk_lengths) == token_count

from text import build_job_text, clean_text

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


def test_clean_text_strips_bullet_before_inline_emphasis_on_same_line():
    text = "* Bold heading with *inline emphasis* on the bullet line"

    result = clean_text(text)

    assert result == "Bold heading with inline emphasis on the bullet line"


def test_clean_text_collapses_repeated_whitespace():
    text = "Too   many\t\tspaces\n\n\nand   blank lines"

    result = clean_text(text)

    assert result == "Too many spaces and blank lines"


def test_build_job_text_joins_position_and_description():
    result = build_job_text("Python Developer", "Line1\r\nLine2")

    assert result == "Python Developer\nLine1\r\nLine2"

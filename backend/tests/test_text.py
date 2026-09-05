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


def test_clean_text_drops_bilingual_mirror_section():
    # Real dataset shape: an English section followed by a full Ukrainian
    # mirror translation of the same content, under a Ukrainian header.
    text = (
        "Requirements\r\n\r\n"
        "We need a Python Developer with 3+ years experience.\r\n\r\n"
        "Вимоги\r\n\r\n"
        "Потрібен Python Developer з досвідом 3+ роки."
    )

    result = clean_text(text)

    assert result == "Requirements We need a Python Developer with 3+ years experience."


def test_clean_text_drops_cyrillic_header_glued_to_english_block():
    # Real dataset shape: a Ukrainian header line joined to an English
    # paragraph by a single line break, not a blank-line paragraph gap.
    text = "Про компанію CHI Software\nWe build products for clients worldwide."

    result = clean_text(text)

    assert result == "We build products for clients worldwide."


def test_clean_text_strips_cyrillic_parenthetical_from_title():
    # Real dataset shape: 'Business Analyst (Бізнес-аналітик)' — the
    # Ukrainian translation of the title is parenthetical, not the whole line.
    text = "Business Analyst (Бізнес-аналітик)"

    result = clean_text(text)

    assert result == "Business Analyst"


def test_clean_text_keeps_english_half_of_slash_separated_title():
    # Real dataset shape: 'Account manager / Менеджер по роботі з клієнтами'.
    text = "Account manager / Менеджер по роботі з клієнтами"

    result = clean_text(text)

    assert result == "Account manager"


def test_clean_text_keeps_english_line_with_few_cyrillic_loanwords():
    text = "We use Jira and Confluence for planning."

    result = clean_text(text)

    assert result == text

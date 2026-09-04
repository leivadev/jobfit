import pandas as pd
import pytest
from dataset import deduplicate_jobs, filter_jobs

LONG_DESC = "x" * 250
SHORT_DESC = "too short"


def make_row(company, position, keyword="Python", desc=LONG_DESC, exp_years="1-3"):
    return {
        "id": f"{company}-{position}",
        "Position": position,
        "Long Description": desc,
        "Company Name": company,
        "Exp Years": exp_years,
        "Primary Keyword": keyword,
    }


@pytest.fixture
def jobs_df():
    return pd.DataFrame(
        [
            make_row("Acme Corp", "Python Developer", keyword="Python"),
            make_row("Acme Corp ", "Python Developer", keyword="Python"),
            make_row("Globex", "QA Engineer", keyword="QA"),
            make_row("Globex", "DevOps Engineer", keyword="DevOps"),
            make_row("Initech", "iOS Developer", keyword="iOS"),
            make_row("Initech", "Android Developer", keyword="Android"),
            make_row("Umbrella", "Data Analyst", keyword="Data Analyst"),
            make_row("Umbrella", "Data Engineer", keyword="Data Engineer"),
            make_row("Wayne Ent.", "Ruby Developer", keyword="Ruby"),
            make_row("Wayne Ent.", "Golang Developer", keyword="Golang"),
            make_row("Stark Ind.", "SQL Developer", keyword="SQL"),
            make_row("Stark Ind.", "Scala Developer", keyword="Scala"),  # not in keyword list
            make_row("Wonka Co.", "Short Desc Job", keyword="Java", desc=SHORT_DESC),
            make_row("Wonka Co.", "Empty Desc Job", keyword="Java", desc=""),
        ]
    )


KEYWORDS = [
    "QA", "QA Automation", "DevOps", "iOS", "Android",
    "Data Analyst", "Data Engineer", "Data Science",
    "JavaScript", ".NET", "Java", "Node.js", "PHP", "Python", "C++", "Ruby", "Golang", "SQL",
]


def test_filter_jobs_keeps_only_listed_keywords(jobs_df):
    result = filter_jobs(jobs_df, KEYWORDS)

    assert "Scala" not in result["Primary Keyword"].values


def test_filter_jobs_drops_short_and_empty_descriptions(jobs_df):
    result = filter_jobs(jobs_df, KEYWORDS, min_desc_chars=200)

    assert "Short Desc Job" not in result["Position"].values
    assert "Empty Desc Job" not in result["Position"].values


def test_filter_jobs_keeps_matching_rows(jobs_df):
    result = filter_jobs(jobs_df, KEYWORDS, min_desc_chars=200)

    assert "QA Engineer" in result["Position"].values
    assert len(result[result["Position"] == "Python Developer"]) == 2


def test_deduplicate_jobs_removes_case_and_whitespace_variants(jobs_df):
    filtered = filter_jobs(jobs_df, KEYWORDS, min_desc_chars=200)
    result = deduplicate_jobs(filtered)

    python_rows = result[result["Position"] == "Python Developer"]
    assert len(python_rows) == 1


def test_deduplicate_jobs_preserves_first_seen_casing(jobs_df):
    filtered = filter_jobs(jobs_df, KEYWORDS, min_desc_chars=200)
    result = deduplicate_jobs(filtered)

    python_row = result[result["Position"] == "Python Developer"].iloc[0]
    assert python_row["Company Name"] == "Acme Corp"


def test_deduplicate_jobs_keeps_distinct_company_position_pairs(jobs_df):
    filtered = filter_jobs(jobs_df, KEYWORDS, min_desc_chars=200)
    result = deduplicate_jobs(filtered)

    assert len(result) == len(filtered) - 1


def test_deduplicate_jobs_normalizes_internal_whitespace():
    df = pd.DataFrame(
        [
            make_row("Acme  Corp", "Python Developer"),
            make_row("Acme Corp", "Python  Developer"),
        ]
    )

    result = deduplicate_jobs(df)

    assert len(result) == 1
    assert result.iloc[0]["Company Name"] == "Acme  Corp"

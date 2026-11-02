from dataclasses import dataclass

EXP_YEARS_SCALE = ["no_exp", "1y", "2y", "3y", "5y"]
_EXP_YEARS_RANK = {bucket: rank for rank, bucket in enumerate(EXP_YEARS_SCALE)}

# Every `Primary Keyword` value in the source dataset (45), not just the IT
# subset — see docs/adr/0008-index-all-primary-keywords.md.
KEYWORD_VOCABULARY = frozenset(
    {
        ".NET",
        "Android",
        "Artist",
        "Block-chain",
        "Business Analyst",
        "C++",
        "Data Analyst",
        "Data Engineer",
        "Data Science",
        "Design",
        "DevOps",
        "Flutter",
        "Golang",
        "HR",
        "Java",
        "JavaScript",
        "Lead",
        "Lead Generation",
        "Marketing",
        "Node.js",
        "Other",
        "PHP",
        "Product Manager",
        "Product Owner",
        "Project Manager",
        "Python",
        "QA",
        "QA Automation",
        "React",
        "Recruiter",
        "Ruby",
        "Rust",
        "SAP",
        "SEO",
        "SQL",
        "Sales",
        "Salesforce",
        "Scala",
        "Scrum Master",
        "Security",
        "Support",
        "Sysadmin",
        "Technical Writing",
        "Unity",
        "iOS",
    }
)


@dataclass(frozen=True)
class CandidateSignals:
    """The candidate's own declared experience/skills, from `/recommend`'s optional form fields."""

    exp_years: str | None = None
    keywords: list[str] | None = None


class InvalidCandidateSignalError(ValueError):
    """Raised when `exp_years` or `keywords` falls outside its controlled vocabulary."""


def validate_candidate_signals(signals: CandidateSignals) -> None:
    """Raise `InvalidCandidateSignalError` if either declared signal is out of vocabulary."""
    if signals.exp_years is not None and signals.exp_years not in _EXP_YEARS_RANK:
        raise InvalidCandidateSignalError(f"Invalid exp_years: {signals.exp_years!r}")

    if signals.keywords is not None:
        invalid = [
            keyword for keyword in signals.keywords if keyword not in KEYWORD_VOCABULARY
        ]
        if invalid:
            raise InvalidCandidateSignalError(f"Invalid keywords: {invalid!r}")


def exp_distance(candidate_exp_years: str | None, job_exp_years: str) -> int | None:
    """Symmetric ordinal distance between two Exp Years buckets on `EXP_YEARS_SCALE`.

    `None` when the candidate didn't declare `exp_years` (absent signal, not a
    distance of 0), or when either bucket isn't on the scale (e.g. Job
    Metadata predating this vocabulary) — an unrecognized bucket means
    "unknown", not "maximally distant".
    """
    if candidate_exp_years is None:
        return None

    candidate_rank = _EXP_YEARS_RANK.get(candidate_exp_years)
    job_rank = _EXP_YEARS_RANK.get(job_exp_years)
    if candidate_rank is None or job_rank is None:
        return None

    return abs(candidate_rank - job_rank)


def keyword_match(candidate_keywords: list[str] | None, job_keyword: str) -> bool:
    """Whether any of the candidate's declared keywords overlaps the job's keyword.

    `False` when the candidate declared no keywords (absent signal, not a
    negative match).
    """
    if not candidate_keywords:
        return False
    return job_keyword in candidate_keywords

import pytest

from backend.domain.matching import (
    EXP_YEARS_SCALE,
    KEYWORDS,
    CandidateSignals,
    InvalidCandidateSignalError,
    exp_distance,
    keyword_match,
    validate_candidate_signals,
)


def test_exp_years_scale_is_the_five_ordered_buckets():
    assert EXP_YEARS_SCALE == ["no_exp", "1y", "2y", "3y", "5y"]


def test_keywords_is_the_eighteen_value_controlled_vocabulary():
    assert KEYWORDS == {
        ".NET",
        "Android",
        "C++",
        "Data Analyst",
        "Data Engineer",
        "Data Science",
        "DevOps",
        "Golang",
        "Java",
        "JavaScript",
        "Node.js",
        "PHP",
        "Python",
        "QA",
        "QA Automation",
        "Ruby",
        "SQL",
        "iOS",
    }


def test_exp_distance_is_none_when_candidate_did_not_declare_exp_years():
    assert exp_distance(None, "3y") is None


def test_exp_distance_is_zero_for_exact_match():
    assert exp_distance("2y", "2y") == 0


def test_exp_distance_is_symmetric_ordinal_distance():
    assert exp_distance("no_exp", "5y") == 4
    assert exp_distance("5y", "no_exp") == 4
    assert exp_distance("1y", "3y") == 2


def test_keyword_match_is_false_when_candidate_declared_no_keywords():
    assert keyword_match(None, "Python") is False
    assert keyword_match([], "Python") is False


def test_keyword_match_is_true_when_job_keyword_is_among_candidate_keywords():
    assert keyword_match(["Python", "SQL"], "Python") is True


def test_keyword_match_is_false_when_job_keyword_is_not_among_candidate_keywords():
    assert keyword_match(["Python", "SQL"], "Java") is False


def test_exp_distance_is_none_when_job_bucket_is_not_on_the_scale():
    assert exp_distance("2y", "not-a-bucket") is None


def test_validate_candidate_signals_accepts_no_signals():
    validate_candidate_signals(CandidateSignals())


def test_validate_candidate_signals_accepts_valid_exp_years_and_keywords():
    validate_candidate_signals(CandidateSignals(exp_years="3y", keywords=["Python", "SQL"]))


def test_validate_candidate_signals_rejects_exp_years_outside_the_scale():
    with pytest.raises(InvalidCandidateSignalError):
        validate_candidate_signals(CandidateSignals(exp_years="10y"))


def test_validate_candidate_signals_rejects_keyword_outside_the_vocabulary():
    with pytest.raises(InvalidCandidateSignalError):
        validate_candidate_signals(CandidateSignals(keywords=["Python", "Haskell"]))

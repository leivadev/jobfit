from pathlib import Path

import numpy as np
import pytest
from text import clean_text

from backend.domain.embeddings import BiEncoder, _chunk_text

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "fixtures"

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

# Real row, Position='Business Analyst (Бізнес-аналітик)' — 2281 tokens under
# all-MiniLM-L6-v2 (near the dataset's real max of ~5200 tokens / ~12,500 chars,
# see ADR-0007), so it splits into several 256-token chunks.
REAL_LONG_DESCRIPTION = "**Who we need**\r\n\r\nMagicFuse (The daughter company of TechMagic focused on Salesforce development) is looking for a Junior Strong BA with 2+ years of experience to join our Salesforce Delivery.\r\n\r\n**Requirements**\r\n\r\n**Must have**\r\n- At least 2 years of experience in Business Analysis\r\n- Good understanding of SDLC and Agile methodologies\r\n- Experience in direct communication with the clients\r\n- Experience in writing user stories and use cases\r\n- Experience in preparing mockups\r\n- Experience working with Jira and Confluence\r\n- Solid self-organization skills\r\n- Technical skills: understanding of API, basic knowledge of SQL queries\r\n\r\n**Will be a plus**\r\n- QA background\r\n- Experience with Salesforce-based projects\r\n\r\n**Responsibilities**\r\n- Requirements elicitation\r\n- Preparing user stories, acceptance criteria, and technical requirements\r\n- Regular clarification of requirements with the clients and dev team\r\n- Requirements management\r\n- Support in the UAT process\r\n- Knowledge base documentation and maintenance\r\n\r\n**About Project**\r\n\r\n**Customer**\r\n\r\nOur customer is one of the world’s most successful hotel companies in the world. The hotel has 124 properties in 47 countries. For the second consecutive year, the hotel was named the Best Luxury Hotel Chain in the World by Business Traveller magazine.\r\n\r\n**Product**\r\n\r\nThis is a growing product ecosystem. Approximately 2-3 years in development; however, the stack is up to date, and no legacy\r\n\r\n**Stage**\r\n\r\nThis is a growing startup.\r\n\r\n**Project team**\r\n\r\nLviv team: Delivery Manager, Marketing Cloud Specialist, Project Manager, 5 Salesforce devs, 1 Business Analyst, Salesforce Administrator, 3 QA Engineers. Team on the client’s side: only management and several support experts\r\n\r\n**Project Technologies**\r\n\r\nApex, LWC, Lightning Components, Sales Cloud, Service Cloud, Experience Cloud\r\n\r\n**Work Schedule**\r\n\r\nFlexible hours, 40 hours per week; work from the office or remotely\r\n\r\n**Interview Stages**\r\n- 1-st stage — interview with our Recruiter\r\n- 2-nd stage — Technical Interview\r\n\r\n**Our Benefits**\r\n- Opportunity to work with the most trending technologies, no legacy, no bureaucracy\r\n- Work from anywhere (fully remotely or in our office)\r\n- Paid vacations and sick leaves, additional days off, relocation bonus\r\n- Wellness: Medical insurance/sports compensation/ health check-up+flu vaccination at your choice\r\n- Education: regular tech talks, educational courses, paid certifications, English classes\r\n- Fun: own football team, budget for team lunches, branded gifts\r\n- One of the best IT employers in Lviv based on DOU rating\r\n\r\n**Хто нам потрібен**\r\n\r\nMagicFuse (дочірня компанія TechMagic, яка зосереджена на розробці Salesforce) шукає Junior Strong BA з досвідом понад 2 роки, щоб приєднатися до нашої Salesforce Delivery.\r\n\r\n**Обов’язкові вимоги:**\r\n- Принаймні 2 роки досвіду в бізнес-аналізі\r\n- Добре розуміння методології SDLC і Agile\r\n- Досвід прямого спілкування з клієнтами\r\n- Досвід написання історій користувачів і випадків використання\r\n- Досвід підготовки макетів\r\n- Досвід роботи з Jira та Confluence\r\n- Гарні навички самоорганізації\r\n- Технічні навики: розуміння API, базові знання запитів SQL\r\n\r\n**Буде плюсом**\r\n- Досвід роботи QA\r\n- Досвід роботи з проектами на базі Salesforce\r\n\r\n**Обов’язки**\r\n- Виявлення вимог\r\n- Підготовка історій користувачів, критеріїв прийому та технічних вимог\r\n- Регулярне уточнення вимог з клієнтами та командою розробників\r\n- Управління вимогами\r\n- Підтримка в процесі UAT\r\n- Документація та супровід бази знань\r\n\r\n**Про проект**\r\n\r\n**Замовник**\r\n\r\nНаш клієнт є однією з найуспішніших готельних компаній у світі. Готель має 124 помешкання в 47 країнах. Другий рік поспіль готель був названий журналом Business Traveller найкращою мережею готелів класу люкс у світі.\r\n\r\n**Продукт**\r\n\r\nЦе зростаюча продуктова екосистема. Приблизно 2-3 роки в розробці; однак стек оновлений і не має спадщини\r\n\r\n**Етап**\r\n\r\nЦе зростаючий стартап.\r\n\r\n**Команда проекту**\r\n\r\nЛьвівська команда: Delivery Manager, Marketing Cloud Specialist, Project Manager, 5 Salesforce devs, 1 Business Analyst, Salesforce Administrator, 3 QA Engineers. Команда з боку клієнта: тільки керівництво і кілька експертів підтримки\r\n\r\n**Проектні технології**\r\n\r\nApex, LWC, Lightning Components, Sales Cloud, Service Cloud, Experience Cloud\r\n\r\n**Робочий розклад**\r\n\r\nГнучкий графік, 40 годин на тиждень; робота з офісу або віддалено\r\n\r\n**Етапи співбесіди**\r\n\r\n1-й етап — співбесіда з нашим рекрутером\r\n2-й етап — Технічна співбесіда\r\n\r\n**Наші переваги**\r\n- Можливість працювати з найбільш трендовими технологіями, без бюрократії\r\n- Працюйте з будь-якого місця (повністю віддалено або в нашому офісі)\r\n- Оплачувані відпустки та лікарняні, додаткові вихідні, компенсація релокації\r\n- Wellness: медична страховка/спортивна компенсація/огляд+щеплення від грипу на ваш вибір\r\n- Освіта: регулярні технічні лекції, навчальні курси, платні сертифікати, уроки англійської\r\n- Розваги: власна футбольна команда, бюджет на командні обіди, фірмові подарунки\r\n- Один з найкращих IT-роботодавців Львова за рейтингом DOU"


@pytest.fixture(scope="module")
def bi_encoder():
    return BiEncoder()


def test_bi_encoder_pins_exact_model_revision(bi_encoder):
    assert (
        bi_encoder.model[0].auto_model.config._commit_hash == BiEncoder.MODEL_REVISION
    )


def test_encode_returns_one_384_vector_per_text(bi_encoder):
    texts = [
        "Python Developer\nBuild backend services.",
        "QA Engineer\nWrite test plans.",
    ]

    embeddings = bi_encoder.encode(texts)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_encode_is_not_normalized(bi_encoder):
    embeddings = bi_encoder.encode(["Python Developer\nBuild backend services."])

    norm = np.linalg.norm(embeddings[0])
    assert norm != pytest.approx(1.0, abs=1e-3)


def test_encode_mean_pools_chunks_for_long_text(bi_encoder):
    long_text = "Python Developer\n" + " ".join(
        f"requirement number {i}" for i in range(1000)
    )
    chunks = _chunk_text(
        long_text, bi_encoder.model.tokenizer, BiEncoder.MAX_CHUNK_TOKENS
    )
    assert len(chunks) > 1

    # Each chunk is itself under max_tokens, so encode() sees it as a single
    # chunk and returns its raw pooled vector unmodified.
    chunk_vectors = np.concatenate(
        [bi_encoder.encode([chunk]) for chunk in chunks], axis=0
    )
    expected = chunk_vectors.mean(axis=0)
    actual = bi_encoder.encode([long_text])[0]

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_encode_matches_raw_encoding_for_short_text(bi_encoder):
    text = "Python Developer\nShort description."
    device = str(next(bi_encoder.model.parameters()).device)

    actual = bi_encoder.encode([text])[0]
    expected = bi_encoder._encode_raw([text], device=device)[0]

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_encode_output_unchanged_after_refactor(bi_encoder):
    """Same chunking, pooling, and numbers as before the move from app/embeddings.py."""
    texts = [
        "Python Developer\nBuild backend services.",
        "QA Engineer\nWrite test plans.",
        "Senior Engineer\n" + " ".join(f"requirement number {i}" for i in range(600)),
    ]
    expected = np.load(FIXTURES_DIR / "bi_encoder_baseline.npy")

    actual = bi_encoder.encode(texts)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_chunk_text_returns_single_chunk_under_max_tokens(bi_encoder):
    text = clean_text(REAL_DESCRIPTION)
    tokenizer = bi_encoder.model.tokenizer
    assert len(tokenizer.encode(text, add_special_tokens=False)) < 256

    chunks = _chunk_text(text, tokenizer, max_tokens=256)

    assert len(chunks) == 1


def test_chunk_text_splits_into_non_overlapping_windows_near_dataset_max(bi_encoder):
    text = clean_text(REAL_LONG_DESCRIPTION)
    tokenizer = bi_encoder.model.tokenizer
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    assert token_count > 256

    chunks = _chunk_text(text, tokenizer, max_tokens=256)
    chunk_lengths = [
        len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks
    ]

    assert len(chunks) > 2
    assert all(length <= 256 for length in chunk_lengths)
    # decode() isn't a perfect inverse of encode(), so re-encoded chunk
    # lengths can drift slightly from the original slice sizes; what must
    # hold is that no chunk silently exceeds max_tokens when re-tokenized.
    assert sum(chunk_lengths) == pytest.approx(token_count, abs=len(chunks) * 2)

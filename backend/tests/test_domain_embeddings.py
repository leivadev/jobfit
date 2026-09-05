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

# Real row, Position='Software Engineer (Back-end)' — 2518 tokens under
# all-MiniLM-L6-v2 (near the dataset's real max of ~5200 tokens / ~12,500 chars,
# see ADR-0007), so it splits into several 256-token chunks.
REAL_LONG_DESCRIPTION = "Do you have a passion about writing scalable code and delivering amazing product experiences? Are you ambitious and full of energy, and ready to start a new chapter in your career? We are looking for a talented Back-end Software Engineer to join our team on a full-time basis and assist us in disrupting one of the biggest industries out there!\r\n\r\nGet to know us\r\n\r\nYourHero (Douleutaras/YourPro) is the leading tech-enabled company for home services that solves the hassle of hiring a reliable professional. We launched in 2015 and since then we have helped more than 700.000 customers find the right expert for their job across 4 European countries. By leveraging technology and our unique business model we are fixing and improving it bit by bit and day by day. This is how we are serving our mission every day, creating an ecosystem of thousands of growing entrepreneurs that solve consumer problems.\r\n\r\nWe want to be the definitive services company - the app you go to every time you need a job done. Today, we possess the best KPIs in the industry, have generated more than €100m for our 5000+ professionals, possess an NPS of 70+, collected 80.000+ customer reviews, already expanded internationally and are supported by top-tier European investors.\r\n\r\nAnd we’re just getting started.\r\n\r\nThe scale of the opportunity ahead of us is immense. The global services market is valued at €6 Trillion, yet less than 5% of that is online. Contrast that with the digital disruption of countless other industries - from banking and travel to retail and communications - and it’s clear that our journey in the services sector has only just begun.\r\n\r\nIn 2020 our international journey began. From 2021 onwards, we are launching new cities in new countries, every year. Join us in this incredible journey and be a major part in disrupting the €6 Trillion services market and becoming the global leader under our international brand name YourHero.\r\n\r\n\r\nOur future goals\r\n\r\nOur vision is to build a product that the world has never seen before, by innovating in the home services market and offering our users a truly disruptive experience. We feel that finding a home expert should not be a laborious task but should be as easy as eating a cake (yummy!). Thus we envision to build a product that fundamentally changes the way home tasks are performed. Using this product innovation we plan to rapidly expand to several countries in the next 2 years, making Douleutaras a global player in the home services industry!\r\n\r\n\r\nHow we innovate in Douleutaras\r\n\r\nWe cannot revolutionize and disrupt the home services market without first starting a revolution ourselves – internally! We are Douleutaras, we encourage autonomy, ethos, innovation and we like to move fast!\r\n\r\nWe seek our team members to have an explosive career path and a constant boost of knowledge, and we have set up specific evolution steps to make sure we achieve it – together.\r\n\r\nOur teams are small, empowered and autonomous. We allow teams to define the methodology desired and used. From Scrum to Kanban, as well as other practices such as test-driven methodologies and pair programming, you will be the one defining it and you will be the one guiding us towards it!\r\n\r\nJoining Douleutaras, you will find great colleagues to assist you technically and psychologically to achieve your goals! Our environment enhances knowledge, friendliness, and excellence - even if it means spending a bit more time to achieve perfection!\r\n\r\nWe build the product together! We believe our product needs input from every source, so we encourage open expression, ideation, and participation in order to achieve a great user experience. After all, we eat our own food, we are all users of Douleutaras having a first class experience of the quality of services that we offer!\r\n\r\nLastly, we always give back. We encourage open source contribution and provide the time to achieve it. What comes around goes around, or else, give and it will be given to you :)\r\n\r\n\r\nTools we use to build...\r\n\r\nCutting edge technology is what we use and we do not fear experimentation if this leads to product innovation or just a great tech time in the office :) Our amazing internal monitoring systems are here to protect us by providing an isolated environment on which we can go nuts! And nuts we do go!\r\n\r\nThis is an outline of our current tech:\r\n\r\n1) Our web app is in React and Angular\r\n2) Our Back-end is in Python running on the Django framework\r\n3) We use Nativescript for our mobile app\r\n4) All our applications are Dockerized\r\n5) Our servers work on Digital Ocean and AWS\r\n6) We like serverless technologies as we are really greedy when it comes to computing power\r\n7) We use databases of all sorts to suit our needs: PostgreSQL, ElasticSearch, Memcached.\r\n8) Data Science is viciously used as it makes us stand out from the crowd! Python here my friends!\r\n\r\nIf you are curious to find out more about our tech stack, send us an email. We love wasting time and talking tech!\r\n\r\n\r\nAbout this role\r\n\r\nAs a Software Engineer (Back-end) you will be building core capabilities and services for Douleutaras and you must be passionate about software engineering and awesome product experiences. You should care deeply about writing solid code, you closely follow industry trends and the open source community, you are curious and an avid learner. You have a strong opinion about technologies and are willing to test new ideas out. You like to move fast and get code into production because you know that your work has a positive impact on the lives of end users.\r\n\r\nYou will also be in the driving seat for installing awesome infrastructure to assist the business geographical expansion plans which require more tech power and better delivery pipelines! Bring in some horsepower and cloud capacity please!!!\r\n\r\nOur product’s backend is in pure Python and we strictly follow Python’s best practices and coding principles. We currently make full use of the Django framework in conjunction with Django’s REST framework and numerous Django applications.\r\n\r\nWe rely on PostgreSQL for our database needs, so you will need concrete knowledge of its optimizations and host configuration. Search is done through a cluster of ElasticSearch nodes and we constantly try to optimize search times. All our asynchronous tasks are performed through Celery.\r\n\r\nOur deployment pipeline lives on Gitlab CI and functions with a hybrid of Ansible playbooks deploying to cloud instances and Docker containers. It is rock solid with zero downtime but there is always room for improvement.\r\n\r\nOur Front End is a combination of Angular, React and TypeScript with a tight focus on writing optimized code for the best possible user experience. Implementation follows strict UX/UI principles communicated by our talented Design team. We use Gulp to automate tasks such as bundling and minifying, SASS/SCSS for our CSS and CoffeeScript where we have deemed necessary.\r\n\r\nOur Front End code is deployed alongside the back-end code. It goes through a Gitlab CI pipeline later bundled in with the Python application’s code for delivery.\r\n\r\nOur mobile application is built on NativeScript using Angular to share core codebase with Front End in the most efficient way and allows us to build native mobile apps for iOS and Android drastically speeding up development and release cycle. Native iOS and Android development tools are also used to build platform- and hardware-specific components. We also strictly follow human interface design guidelines to have the most intuitive and easy-to-use mobile application for our customers.\r\n\r\n\r\nHave we not persuaded you yet? Some more points:\r\n1) Your opinion will be highly appreciated and valued. We only hire creative and open-minded people who are not afraid to listen to new ideas.\r\n2) You will never walk alone! We all belong in two teams; an agile team to develop features based on our roadmap and a tech team which makes sure we buddy in the journey of code excellence\r\n3) You will deliver high quality scalable and well-tested code by following our comprehensive coding standards and follow our coding principles.\r\n4) You will define our infrastructure and the tools we use! As a technology company, we constantly seek to upgrade our tools. Bring in your ideas!\r\n\r\nReady to create a disruptive service and blast through with amazing tech? Then this might be the right job for you!\r\n\r\n\r\nAbout you\r\n\r\n1) You communicate with candor and directness and you welcome feedback without getting defensive\r\n2) You go out of your way for your team players and do not throw the ball over the fence.\r\n3) You project passion, positive energy and enthusiasm in whatever you do and you always champion a “can do better” mentality\r\n4) You think in a “scalable” way and are proud of the code you produce. 5) Admittedly, nothing is perfect and so refactoring is your friend. You always like to leave the code better than you found it, in fact, it’s how you code every day\r\n5) You act like an owner and are selfless while being accountable and goal-oriented\r\n\r\nResponsibilities\r\n\r\n1) Participate in the agile feature/product design process working with cross-functional teams including: Product Management, Design and Operations\r\n2) Collaborate with other engineers to share best practices and knowledge of emerging technologies\r\n3) Implement features, products, and enhancements that improve the user experience\r\n4) Maintain and monitor our CI/CD pipeline\r\n5) Design, build and maintain core infrastructure pieces that allow Douleutaras to scale and support thousands of concurrent users\r\n6) Plan the growth of our infrastructure\r\n7) Improve the deployment process to make it as boring as possible\r\n8) Monitor the application’s stability and analyze performance metrics\r\n9) Deliver fully tested and documented code\r\n\r\nRequirements\r\n\r\nIdeally you will have\r\n1) 1+ year of proven experience in developing web applications with Python\r\n2) 2+ years of professional experience with enterprise architecture and developing highly scalable websites/services\r\n3) Strong Understanding of OOP principles/design patterns\r\n4) Proven REST services experience\r\n5) Source control systems experience (Git)\r\n6) Experience in relational databases\r\n7) Proficient with Linux administration\r\n8) Hands-on experience developing, releasing, and maintaining large-scale software applications\r\n9) Good written, verbal, and collaboration skills\r\n10) Excellent command of English, both written and verbal\r\n11) Experience working with a remote and multicultural team\r\n\r\nAnd you will also\r\n\r\n1) Be a self-starter with a strong work ethic and a passion for problem-solving\r\n2) Be flexible and able to adapt to changing priorities and technologies\r\n3) Think analytically and methodically with attention to detail\r\n4) Refactor problematic, incomplete parts and constantly improve our codebase\r\n5) Read, understand & debug code\r\n6) Follow coding standards and apply good practices\r\n7) Deliver testable, efficient, reusable, high quality and easy-to-read code\r\n\r\nBonus Points (if you have the below just pass by our office :) )\r\n\r\n1) Golang, JS experience\r\n2) Experience with Pyramid/Flask/Tornado or another Python framework\r\n3) Experience in working with Agile Methodologies\r\n4) Experience in a peak performance organization, preferably a tech startup\r\n\r\nProjects you could work on:\r\n\r\n1) Building the back-end infrastructure and API for our Mobile Applications\r\n2) Improving the communication flow between the consumer and the home experts\r\n3) Building organizational and scheduling features for our home experts\r\n4) Improving our CI/CD pipelines\r\n5) Moving our infrastructure from Digitalocean to AWS\r\n6) Architecting our infrastructure to have the capacity for a global expansion\r\n7) Installing monitoring and alert services\r\n\r\nBenefits\r\n\r\nOur core belief and reason for existence is the care and attention we show our users, our community, and our employees.\r\n\r\n1) A transparent and motivating environment where you can thrive\r\n2) Impact the lives of millions of customers and of the professionals, we partner with\r\n3) Being a part of an international team of domain experts in all functions\r\n4) Competitive compensation and potential for stock options\r\n5) Group Health Insurance program\r\n6) Workstation of your choice\r\n7) An annual budget for Douleutaras/YourHero/YourPro services\r\n8) A bespoke development plan that will allow you to utilise your potential and grow quickly\r\n9) The pleasure of being part of a high-performance team that works hard but has plenty of fun in the process"


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

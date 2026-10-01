"""Assisted apply: the answer layer, with the provider stubbed. No browser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobhunt import apply

QUESTIONS = [
    {"label": "First Name", "required": True,
     "fields": [{"name": "first_name", "type": "input_text", "values": []}]},
    {"label": "Resume/CV", "required": True,
     "fields": [{"name": "resume", "type": "input_file", "values": []},
                {"name": "resume_text", "type": "textarea", "values": []}]},
    {"label": "What is your Notice Period?", "required": True,
     "fields": [{"name": "question_1", "type": "input_text", "values": []}]},
    {"label": "Do you have 3+ years of experience?", "required": True,
     "fields": [{"name": "question_2", "type": "multi_value_single_select",
                 "values": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]}]},
]


class Stub:
    name = "stub"

    def __init__(self, reply):
        self.reply, self.sent = reply, None

    def complete(self, model, system, user, max_tokens, json_mode=False):
        self.sent = user
        return json.dumps(self.reply)


def test_flatten_drops_the_paste_in_twin_of_the_resume_upload():
    names = [q["name"] for q in apply.flatten_questions(QUESTIONS)]
    assert names == ["first_name", "resume", "question_1", "question_2"]


def test_standard_fields_come_from_the_file_and_never_reach_the_model():
    stub = Stub([{"name": "question_1", "answer": "30 days", "source": "details"},
                 {"name": "question_2", "answer": "No", "source": "profile"}])
    answers = apply.answer_questions(apply.flatten_questions(QUESTIONS),
                                     {"first_name": "Asha"}, {}, "SDE", stub, "m")

    assert answers["first_name"] == {"answer": "Asha", "source": "details"}
    assert '"first_name"' not in stub.sent.split("QUESTIONS:")[1]
    assert "resume" not in answers
    assert answers["question_2"]["answer"] == "No"


def test_an_answer_outside_the_offered_options_is_left_for_the_human():
    stub = Stub([{"name": "question_1", "answer": "", "source": "needs_input"},
                 {"name": "question_2", "answer": "Almost", "source": "profile"}])
    answers = apply.answer_questions(apply.flatten_questions(QUESTIONS),
                                     {}, {}, "SDE", stub, "m")

    assert answers["first_name"]["source"] == "needs_input"
    assert answers["question_1"] == {"answer": "", "source": "needs_input"}
    assert answers["question_2"] == {"answer": "", "source": "needs_input"}


def test_summary_counts_only_real_submissions_and_shows_blanks():
    qa = [{"label": "Notice <Period>", "required": True, "answer": ""},
          {"label": "Current CTC", "required": True, "answer": "5 LPA"}]
    entries = [
        {"job_id": "greenhouse:x:1", "title": "SDE", "company": "X",
         "url": "https://example.com", "submitted": True, "note": "", "qa": qa},
        {"job_id": "greenhouse:x:2", "title": "SDE 2", "company": "X",
         "url": "https://example.com", "submitted": False, "note": "filled, you did not submit", "qa": qa},
    ]
    subject, doc = apply.summary(entries)

    assert subject.startswith("Applied to 1 of 2 jobs")
    assert "Notice &lt;Period&gt; *" in doc and "left blank" in doc and "5 LPA" in doc
    assert "filled, you did not submit" in doc


def test_recording_an_application_marks_applied_only_when_submitted(tmp_path):
    from jobhunt.store import Store
    store = Store(tmp_path / "seen.json")
    base = {"title": "SDE", "company": "X", "url": "u", "note": "", "qa": []}
    store.record_application({**base, "job_id": "greenhouse:x:1", "submitted": True})
    store.record_application({**base, "job_id": "greenhouse:x:2", "submitted": False})

    assert store.data["greenhouse:x:1"]["applied"] is True
    assert store.data["greenhouse:x:2"]["applied"] is False
    assert store.stats()["applied"] == 1

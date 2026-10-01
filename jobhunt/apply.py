"""Assisted apply: pre-fill an application form, then hand over to the human.

The agent still never submits. It opens the real form in a visible browser,
fills what it can from applicant.json and the profile, flags what it could not
answer, and stops. You review, solve the CAPTCHA, and press submit yourself.

Greenhouse only for now: it is the one ATS that publishes its form questions
(`?questions=true`), and its input ids match the API field names, so filling
is exact instead of guessed from page text.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from . import llm
from .fetch import TIMEOUT, UA
from .providers import Provider

ANSWER_MAX_TOKENS = 4000

# Answered straight from applicant.json — no model call, no chance of drift.
DIRECT_FIELDS = ("first_name", "last_name", "email", "phone")
FILE_FIELDS = ("resume", "cover_letter")

APPLICANT_PROMPT = """Extract the applicant's contact details from this resume.

Return ONLY a JSON object, no prose. Use "" for anything the resume does not
state — never guess:
{
  "first_name": str,
  "last_name": str,
  "email": str,
  "phone": str,            // digits with country code if shown
  "linkedin": str,         // full URL
  "github": str,           // full URL
  "website": str,
  "current_employer": str,
  "current_location": str  // city
}"""

ANSWER_SYSTEM = """You fill in a job application form for one applicant.

Hard rule: never invent a fact. Every answer must come from APPLICANT DETAILS
or the CANDIDATE PROFILE. If neither supports an answer, return an empty
`answer` with `source` "needs_input" — a blank the human fills in is fine, a
wrong answer sent to a recruiter is not.

For a question with `options`, `answer` must be exactly one of the option
labels. Answer yes/no questions truthfully even when the truthful answer hurts
the application (e.g. "3+ years of experience?" for a 1.5-year candidate is No).

Keep free-text answers short and factual, the way a person types into a form.

Return ONLY a JSON array, one object per question, no prose:
[{"name": str, "answer": str, "source": "details" | "profile" | "needs_input"}]
Echo `name` back exactly as given."""


def build_applicant(resume: bytes, provider: Provider, model: str) -> dict:
    """Resume PDF -> the contact half of applicant.json."""
    data = llm.parse_json(provider.complete_document(
        model, APPLICANT_PROMPT, resume, ANSWER_MAX_TOKENS))
    if not isinstance(data, dict):
        raise ValueError("applicant extraction did not return a JSON object")
    return data


def fetch_form(job_id: str) -> dict:
    """`greenhouse:<slug>:<id>` -> {url, title, company, description, questions}."""
    ats, slug, gh_id = job_id.split(":", 2)
    if ats != "greenhouse":
        raise ValueError(f"assisted apply supports greenhouse only, not {ats!r}")
    r = requests.get(
        f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{gh_id}?questions=true",
        headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        raise ValueError(f"greenhouse HTTP {r.status_code} for {job_id} (posting closed?)")
    body = r.json()
    return {
        "url": body.get("absolute_url") or "",
        "title": (body.get("title") or "").strip(),
        "company": body.get("company_name") or slug,
        "description": body.get("content") or "",
        "questions": flatten_questions(body.get("questions") or []),
    }


def flatten_questions(questions: list[dict]) -> list[dict]:
    """One row per fillable field: {name, label, required, type, options}."""
    out = []
    for q in questions:
        for f in q.get("fields") or []:
            name = f.get("name") or ""
            # The *_text twins of the file fields are paste-in alternatives to
            # an upload; the upload is what gets used.
            if name in ("resume_text", "cover_letter_text"):
                continue
            out.append({
                "name": name,
                "label": (q.get("label") or "").strip(),
                "required": bool(q.get("required")),
                "type": f.get("type") or "",
                "options": [str(v.get("label")) for v in (f.get("values") or [])],
            })
    return out


def answer_questions(questions: list[dict], applicant: dict, profile: dict,
                     job_title: str, provider: Provider, model: str) -> dict[str, dict]:
    """Field name -> {answer, source}. Standard fields skip the model."""
    answers: dict[str, dict] = {}
    for_model = []
    for q in questions:
        name = q["name"]
        if name in FILE_FIELDS:
            continue
        if name in DIRECT_FIELDS:
            value = str(applicant.get(name) or "")
            answers[name] = {"answer": value,
                             "source": "details" if value else "needs_input"}
        else:
            for_model.append(q)

    if for_model:
        raw = provider.complete(
            model, ANSWER_SYSTEM,
            f"APPLICANT DETAILS:\n{json.dumps(applicant, ensure_ascii=False)}\n\n"
            f"CANDIDATE PROFILE:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
            f"APPLYING FOR: {job_title}\n\n"
            f"QUESTIONS:\n{json.dumps(for_model, ensure_ascii=False)}",
            ANSWER_MAX_TOKENS, json_mode=True)
        got = {str(r.get("name")): r for r in llm._as_list(llm.parse_json(raw))}
        for q in for_model:
            r = got.get(q["name"]) or {}
            answer = str(r.get("answer") or "").strip()
            # An answer that is not one of the offered options cannot be
            # selected, so it is no answer at all.
            if q["options"] and answer not in q["options"]:
                answer = ""
            source = str(r.get("source") or "needs_input") if answer else "needs_input"
            answers[q["name"]] = {"answer": answer, "source": source}
    return answers


def fill_form(form: dict, answers: dict[str, dict], applicant: dict,
              headless: bool = False, screenshot: str | Path | None = None,
              hold: bool = True) -> dict[str, Any]:
    """Open the form and fill it. Never clicks submit.

    Returns {filled, skipped, failed, seconds}. With `hold`, the browser stays
    open until the human closes it — that is where review and submit happen.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "pip install playwright && python -m playwright install chromium") from None

    by_name = {q["name"]: q for q in form["questions"]}
    result: dict[str, Any] = {"filled": [], "skipped": [], "failed": []}
    started = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.goto(form["url"], wait_until="networkidle")

        def combobox(selector: str, label: str) -> None:
            box = page.locator(selector)
            box.click()
            box.fill(label)
            # Prefix match: the country list labels its options "India +91".
            page.get_by_role("option", name=re.compile(rf"^{re.escape(label)}\b")
                             ).first.click(timeout=4000)

        # Not in the questions API, but the form requires it.
        country = str(applicant.get("country") or "")
        if country and page.locator("#country").count():
            try:
                combobox("#country", country)
                result["filled"].append("country")
            except Exception as e:
                result["failed"].append(f"country ({type(e).__name__})")

        resume = Path(str(applicant.get("resume_path") or ""))
        if resume.is_file() and page.locator("#resume").count():
            page.locator("#resume").set_input_files(str(resume))
            result["filled"].append("resume")
        else:
            result["skipped"].append("resume")

        for name, a in answers.items():
            q = by_name[name]
            if not a["answer"]:
                result["skipped"].append(q["label"])
                continue
            selector = f'[id="{name}"]'
            try:
                if q["options"]:
                    combobox(selector, a["answer"])
                else:
                    page.locator(selector).fill(a["answer"])
                result["filled"].append(q["label"])
            except Exception as e:  # field renamed, option not rendered
                result["failed"].append(f"{q['label']} ({type(e).__name__})")

        result["seconds"] = round(time.time() - started, 1)
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True)
        if hold and not headless:
            print("  form is filled. Review it, complete the blanks, solve the "
                  "CAPTCHA and submit.\n  Close the browser window when done.")
            page.wait_for_event("close", timeout=0)
        browser.close()
    return result

"""Assisted apply: pre-fill an application form, then hand over to the human.

The agent still never submits. It opens the real form in a visible browser,
fills what it can from applicant.json and the profile, flags what it could not
answer, and stops. You review, solve the CAPTCHA, and press submit yourself.

Greenhouse only for now: it is the one ATS that publishes its form questions
(`?questions=true`), and its input ids match the API field names, so filling
is exact instead of guessed from page text.
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from . import llm
from .fetch import TIMEOUT, UA, strip_html
from .providers import Provider

ANSWER_MAX_TOKENS = 4000

# Answered straight from applicant.json — no model call, no chance of drift.
DIRECT_FIELDS = ("first_name", "last_name", "email", "phone")
FILE_FIELDS = ("resume",)
COVER_LETTER = "cover_letter_text"
JD_CHARS = 6000

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

Questions come in two kinds. Treat them differently.

FACTS about the applicant — pay, notice period, location, employer, links,
years of experience, visa or sponsorship status, legal agreements, past
employment at this company, certifications held. Answer only from APPLICANT
DETAILS or the CANDIDATE PROFILE. Never invent or assume one: if neither states
it, return an empty `answer` with `source` "needs_input". A blank the human
fills in is fine, a wrong fact sent to a recruiter is not.

WRITING — a cover letter, "why this role", "describe a project", "tell us
about yourself", anything asking for prose. Write it, tailored to the JOB
DESCRIPTION, using only real experience from the profile, with `source`
"generated". Always in the first person ("I built", never "Siddharth built"
or "he built") — the applicant is the one sending it. Plain and concrete: no "I am writing to express my interest", no
"I am excited to", no flattery about the company. A cover letter is 120-160
words and opens with a concrete reason the applicant fits this role; other
written answers are 2-4 sentences.

For a question with `options`, `answer` must be exactly one of the option
labels. Answer yes/no questions truthfully even when the truthful answer hurts
the application (e.g. "3+ years of experience?" for a 1.5-year candidate is No).

Leave an optional question blank when the honest answer is "nothing to add".
Keep factual answers short, the way a person types into a form.

Return ONLY a JSON array, one object per question, no prose:
[{"name": str, "answer": str,
  "source": "details" | "profile" | "generated" | "needs_input"}]
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
        # `absolute_url` can be the company's own careers page with the form
        # buried in an iframe. The embed URL is always the bare Greenhouse form.
        "form_url": f"https://job-boards.greenhouse.io/embed/job_app?for={slug}&token={gh_id}",
        "title": (body.get("title") or "").strip(),
        "company": body.get("company_name") or slug,
        "description": body.get("content") or "",
        "questions": flatten_questions(body.get("questions") or []),
        # A city picker some boards add. Like country, it is on the form but
        # not in `questions`.
        "asks_location": bool(body.get("location_questions")),
    }


def flatten_questions(questions: list[dict]) -> list[dict]:
    """One row per fillable field: {name, label, required, type, options}."""
    out = []
    for q in questions:
        for f in q.get("fields") or []:
            name = f.get("name") or ""
            # Each file field has a paste-in twin. The resume goes up as a
            # file; the cover letter is written per job, so it is pasted.
            if name in ("resume_text", "cover_letter"):
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
                     job_title: str, provider: Provider, model: str,
                     job_description: str = "") -> dict[str, dict]:
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
            f"JOB DESCRIPTION:\n{strip_html(job_description)[:JD_CHARS]}\n\n"
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


_CONFIRMED = re.compile(r"thank you for applying|application (has been |was )?"
                        r"(submitted|received)", re.I)


SUBMIT_WAIT_S = 25


def missing_required(form: dict, answers: dict[str, dict], applicant: dict) -> list[str]:
    """Labels of required fields that would go out empty."""
    missing = []
    for q in form["questions"]:
        if not q["required"]:
            continue
        if q["name"] == "resume":
            if not Path(str(applicant.get("resume_path") or "")).is_file():
                missing.append(q["label"])
        elif not (answers.get(q["name"]) or {}).get("answer"):
            missing.append(q["label"])
    return missing


def _why_not(page) -> str:
    """Say what the form is showing after a submit that did not confirm."""
    body = ""
    try:
        body = page.locator("body").inner_text(timeout=2000)
    except Exception:
        pass
    if re.search(r"security code|verification code", body, re.I):
        return ("the form asked for a security code emailed to you — open the "
                "link, enter the code and submit")
    try:
        if page.locator('iframe[title*="challenge"]').first.is_visible(timeout=500):
            return "a CAPTCHA challenge appeared — open the link and finish it"
    except Exception:
        pass
    errors = []
    try:
        for text in page.locator('[role="alert"], [id$="-error"], [class*="error"]'
                                 ).all_inner_texts():
            text = " ".join(text.split())
            if text and text not in errors:
                errors.append(text)
    except Exception:
        pass
    if errors:
        return "the form rejected it: " + "; ".join(errors)[:300]
    return "no confirmation page after clicking submit — open the link and check"


def fill_form(form: dict, answers: dict[str, dict], applicant: dict,
              headless: bool = False, screenshot: str | Path | None = None,
              hold: bool = True, submit: bool = False) -> dict[str, Any]:
    """Open the form and fill it. Clicks submit only when `submit` is set.

    `submit` presses the button once, and only when every required field has
    an answer and every fill succeeded. It does nothing about a CAPTCHA
    challenge or an emailed security code: without a confirmation page the
    application is reported as not submitted and left for the human.

    Returns {filled, skipped, failed, seconds, submitted, note}. With `hold`,
    the browser stays open until the human closes it — that is where review
    and submit happen, and `submitted` reports whether they pressed it.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "pip install playwright && python -m playwright install chromium") from None

    by_name = {q["name"]: q for q in form["questions"]}
    result: dict[str, Any] = {"filled": [], "skipped": [], "failed": [],
                              "submitted": False, "note": ""}
    started = time.time()
    visited: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.on("framenavigated",
                lambda f: visited.append(f.url) if f == page.main_frame else None)
        page.set_default_timeout(5000)
        page.goto(form.get("form_url") or form["url"], wait_until="networkidle",
                  timeout=30000)

        def confirmed() -> bool:
            if any("confirmation" in u for u in visited):
                return True
            try:
                return bool(_CONFIRMED.search(page.locator("body").inner_text(timeout=2000)))
            except Exception:
                return False

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

        city = str(applicant.get("current_location") or "")
        if page.locator("#candidate-location").count():
            try:
                # Typed, not filled: the suggestions load per keystroke.
                box = page.locator("#candidate-location")
                box.click()
                box.press_sequentially(city, delay=60)
                page.locator('[id^="react-select-candidate-location-option"]'
                             ).first.click(timeout=6000)
                result["filled"].append("location (city)")
            except Exception as e:
                result["failed"].append(f"Location (City) ({type(e).__name__})")

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
                if name == COVER_LETTER:
                    # The textarea only exists after "Enter manually".
                    page.locator('button[data-testid="cover_letter-text"]').click()
                    page.locator(selector).fill(a["answer"])
                elif q["options"]:
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
        if submit:
            blanks = missing_required(form, answers, applicant)
            if blanks or result["failed"]:
                result["note"] = "not submitted: " + (
                    f"required fields empty ({'; '.join(blanks)})" if blanks
                    else f"could not fill {'; '.join(result['failed'])}")
            else:
                page.locator('button[type="submit"]').first.click()
                deadline = time.time() + SUBMIT_WAIT_S
                while time.time() < deadline and not confirmed():
                    page.wait_for_timeout(500)
                result["submitted"] = confirmed()
                if not result["submitted"]:
                    result["note"] = "not submitted: " + _why_not(page)
                    # The form is filled and the code is in the human's inbox:
                    # keep the window so finishing is one code and one click,
                    # not a second run.
                    if not headless and hold:
                        print(f"  {result['note']}\n  Finish it in the open window, "
                              "or close the window to skip this job.")
                        while not page.is_closed():
                            if confirmed():
                                result["submitted"], result["note"] = True, ""
                                break
                            try:
                                page.wait_for_timeout(700)
                            except Exception:  # window closed mid-wait
                                break
                if screenshot:
                    page.screenshot(path=str(screenshot), full_page=True)
        elif hold and not headless:
            print("  form is filled. Review it, complete the blanks, solve the "
                  "CAPTCHA and submit.\n  Close the browser window when done.")
            while not page.is_closed():
                if confirmed():
                    result["submitted"] = True
                try:
                    page.wait_for_timeout(700)
                except Exception:  # window closed mid-wait
                    break
            result["note"] = "" if result["submitted"] else "filled, you did not submit"
        else:
            result["note"] = "filled as a preview, not submitted"
        if not page.is_closed():
            browser.close()
    return result


# ---------------------------------------------------------------- summary ---

def summary(entries: list[dict]) -> tuple[str, str]:
    """Email of what was applied to, with every field asked and answered.

    Each entry: {job_id, title, company, url, submitted, note, qa:[{label,
    answer, required}]}.
    """
    today = datetime.now().strftime("%d %b %Y")
    sent = sum(1 for e in entries if e["submitted"])
    subject = (f"Applied to {sent} of {len(entries)} job"
               f"{'s' if len(entries) != 1 else ''} — {today}")
    cards = []
    for e in entries:
        rows = "".join(
            f'<tr><td style="padding:6px 10px 6px 0;color:#8b93a3;font-size:13px;'
            f'vertical-align:top;width:52%;">{html.escape(q["label"])}'
            f'{" *" if q["required"] else ""}</td>'
            f'<td style="padding:6px 0;color:{"#e6e8ec" if q["answer"] else "#d29922"};'
            f'font-size:13px;vertical-align:top;">'
            f'{html.escape(q["answer"]) or "left blank"}</td></tr>' for q in e["qa"])
        status = ("Submitted" if e["submitted"] else html.escape(e["note"] or "Not submitted"))
        color = "#3fb950" if e["submitted"] else "#d29922"
        cards.append(
            f'<div style="background:#171a21;border:1px solid #262b36;border-radius:12px;'
            f'padding:18px;margin-bottom:14px;">'
            f'<a href="{html.escape(e["url"])}" style="color:#7c9cff;font-size:16px;'
            f'font-weight:700;text-decoration:none;">{html.escape(e["title"])}</a>'
            f'<div style="color:#8b93a3;font-size:13px;margin-top:4px;">'
            f'{html.escape(e["company"])} · {html.escape(e["job_id"])}</div>'
            f'<div style="color:{color};font-size:13px;font-weight:700;margin-top:8px;">'
            f'{status}</div>'
            f'<table style="border-collapse:collapse;width:100%;margin-top:10px;">{rows}</table>'
            f'</div>')
    doc = (f'<!doctype html><html><body style="margin:0;padding:20px;background:#0f1115;'
           f"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;\">"
           f'<div style="max-width:640px;margin:0 auto;">'
           f'<div style="color:#e6e8ec;font-size:22px;font-weight:800;">Application summary</div>'
           f'<div style="color:#8b93a3;font-size:13px;margin:6px 0 20px 0;">{today} · '
           f'{sent} submitted · {len(entries) - sent} not submitted · * required field</div>'
           f'{"".join(cards)}</div></body></html>')
    return subject, doc

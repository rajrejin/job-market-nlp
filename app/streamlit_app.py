"""Streamlit UI: Market Trends and CV Analyser. Talks to the API over HTTP only."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("JOBNLP_API_URL", "http://localhost:8000")
KEY = os.environ.get("JOBNLP_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}

st.set_page_config(page_title="German job market NLP", layout="wide")


def get(path: str, **params: object) -> dict:  # type: ignore[type-arg]
    r = httpx.get(f"{API}{path}", params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()  # type: ignore[no-any-return]


def post(path: str, **kw: object) -> dict:  # type: ignore[type-arg]
    r = httpx.post(f"{API}{path}", headers=HEADERS, timeout=600, **kw)  # type: ignore[arg-type]
    if r.status_code >= 400:
        st.error(r.json().get("error", {}).get("message", "request failed"))
        st.stop()
    return r.json()  # type: ignore[no-any-return]


market_tab, cv_tab = st.tabs(["Market Trends", "CV Analyser"])

with market_tab:
    try:
        profile = get("/v1/market/profile")
    except Exception:
        st.warning("Market results are not available. Run `jobnlp market` and start the API.")
        st.stop()
    st.caption(
        f"{profile['postings']} German postings, published {profile['published_min']} to "
        f"{profile['published_max']}. Only postings active at harvest time are visible, so the "
        "window favours long-running postings; trends need repeated snapshots. Postings are a "
        "proxy for employer demand, not for hiring."
    )
    group_by = st.selectbox(
        "Group by",
        [
            "overall",
            "region",
            "state",
            "category",
            "data_role",
            "experience",
            "language",
            "remote",
            "week",
        ],
    )
    data = get("/v1/market/skills", group_by=group_by, top=25)
    df = pd.DataFrame(data["rows"])
    if df.empty:
        st.info("No group passes the minimum size.")
    else:
        group = st.selectbox("Group", sorted(df.group.astype(str).unique()))
        sub = df[df.group.astype(str) == group]
        st.write(f"Postings in group: {int(sub.postings_in_group.iloc[0])}")
        st.dataframe(
            sub[["label", "share", "ci_low", "ci_high", "postings_with_skill"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(sub.set_index("label").share)
    if data["suppressed_groups"]:
        st.caption(
            f"Suppressed (fewer than {data['min_group_size']} postings): "
            + ", ".join(g["group"] for g in data["suppressed_groups"])
        )
    st.caption(data["note"])
    st.subheader("Role profiles (skill clusters)")
    rp = profile["role_profiles"]
    st.caption(
        f"NMF with {rp['k']} profiles; stability across seeds (mean adjusted Rand index): "
        f"{rp['stability_mean_ari']}"
    )
    for p in rp["profiles"]:
        st.write(
            f"Profile {p['profile']} ({p['postings']} postings): " + ", ".join(p["top_skills"])
        )
    st.subheader("Data and ML roles")
    st.json(profile["data_roles"])

with cv_tab:
    st.info(
        "Privacy: your CV is processed by the local API on this machine; nothing is sent "
        "to an external service, stored, or logged. Suggestions come from a local model."
    )
    job_text = st.text_area("Job description", height=220)
    mode = st.radio("CV input", ["Paste text", "Upload file (txt, pdf, docx)"], horizontal=True)
    cv_text = st.text_area("CV text", height=220) if mode == "Paste text" else ""
    upload = (
        st.file_uploader("CV file", type=["txt", "pdf", "docx"]) if mode != "Paste text" else None
    )
    want_llm = st.checkbox("Generate suggestions with the local LLM (slower)", value=True)
    if st.button("Analyse") and job_text:
        with st.spinner("Analysing..."):
            if upload is not None:
                res = post(
                    "/v1/gap/upload",
                    data={"job_text": job_text, "suggestions": want_llm},
                    files={"cv_file": (upload.name, upload.getvalue())},
                )
            else:
                res = post(
                    "/v1/gap",
                    json={"job_text": job_text, "cv_text": cv_text, "suggestions": want_llm},
                )
        st.metric("Match score", "n/a" if res["score"] is None else f"{res['score']:.2f}")
        st.caption(res["disclaimer"])
        st.caption(
            "Formula: weighted coverage. Must-have weight 1.0, nice-to-have 0.4; "
            "covered counts 1.0, related 0.5, missing 0."
        )
        st.write(f"Languages: job {res['job_lang']}, CV {res['cv_lang']}")
        skills = pd.DataFrame(res["skills"])
        if not skills.empty:
            st.dataframe(
                skills[["label", "necessity", "status", "related_to", "importance"]].sort_values(
                    ["status", "importance"], ascending=[True, False]
                ),
                use_container_width=True,
                hide_index=True,
            )
        for s in res.get("suggestions", []):
            status = "passed guards" if s["guard_passed"] else "guard fallback"
            with st.expander(f"{s['skill_label']} ({status}, {s['attempts']} attempt(s))"):
                if s["rewrite"]:
                    st.write("Original bullet:", s["original_bullet"])
                    st.write("Suggested truthful rewrite:", s["rewrite"])
                if s["suggestion"]:
                    st.write("Suggestion to do (not a claim to write):", s["suggestion"])
                if s["violations"]:
                    st.warning("Removed by guards: " + "; ".join(s["violations"]))
        for n in res.get("notes", []):
            st.caption(n)

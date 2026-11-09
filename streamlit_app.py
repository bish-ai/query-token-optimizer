"""QueryWise Streamlit app: use-case classification and LLM cost estimates."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "models.json"

st.set_page_config(
    page_title="QueryWise — LLM Recommender",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

TRAINING_EXAMPLES: dict[str, list[str]] = {
    "simple": [
        "extract names dates amounts and key fields from text",
        "classify messages into categories or labels",
        "rewrite a sentence and fix grammar or spelling",
        "convert a list into a structured format",
        "translate this phrase or short passage",
        "identify sentiment positive negative or neutral",
        "remove duplicates and normalize these records",
        "answer a simple factual question from the provided text",
    ],
    "general": [
        "answer a general question and explain a topic clearly",
        "summarize this article document or long report",
        "help me understand this subject and give an overview",
        "compare these options and provide useful information",
        "give me recommendations for everyday planning",
        "explain the main points in simple language",
        "respond to a question using the context provided",
        "provide a helpful overview of this topic",
    ],
    "writing": [
        "write a professional email to a customer",
        "draft a blog post story essay or marketing copy",
        "edit this paragraph for clarity and style",
        "make this message more persuasive and engaging",
        "change the tone to be friendly concise or formal",
        "brainstorm creative ideas and headlines",
        "write a product description or social media post",
        "proofread and improve this draft",
    ],
    "coding": [
        "write a javascript python or typescript function",
        "debug this code and fix the error or exception",
        "explain this program and how the algorithm works",
        "refactor this function and improve its performance",
        "create a sql query or api endpoint",
        "review this code and identify a software bug",
        "write unit tests for this module",
        "help me build a web application component",
    ],
    "reasoning": [
        "solve this difficult multi step logic problem",
        "analyze evidence and evaluate competing hypotheses",
        "derive a rigorous proof or mathematical solution",
        "plan a complex project with constraints and dependencies",
        "reason through a difficult strategic decision",
        "perform a detailed quantitative analysis",
        "find edge cases in this complicated system design",
        "compare scenarios and explain tradeoffs step by step",
    ],
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give",
    "help", "i", "in", "into", "is", "it", "me", "of", "on", "or", "please",
    "this", "that", "the", "these", "to", "using", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9+#_.-]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


@st.cache_data(show_spinner=False)
def train_classifier() -> tuple[dict[str, dict[str, Any]], set[str], int]:
    """Train a tiny multinomial Naive Bayes classifier from example phrases."""
    classes: dict[str, dict[str, Any]] = {}
    vocabulary: set[str] = set()
    total_examples = 0

    for label, examples in TRAINING_EXAMPLES.items():
        counts: Counter[str] = Counter()
        for example in examples:
            tokens = tokenize(example)
            counts.update(tokens)
            vocabulary.update(tokens)
        classes[label] = {
            "counts": counts,
            "total_words": sum(counts.values()),
            "example_count": len(examples),
        }
        total_examples += len(examples)

    return classes, vocabulary, total_examples


def classify_use_case(text: str) -> tuple[str, float, list[tuple[str, float]]]:
    classes, vocabulary, total_examples = train_classifier()
    tokens = tokenize(text)
    scored: list[tuple[str, float]] = []

    for label, stats in classes.items():
        log_score = math.log(stats["example_count"] / total_examples)
        denominator = stats["total_words"] + len(vocabulary)
        for token in tokens:
            # Laplace smoothing prevents unseen words from zeroing a class score.
            log_score += math.log((stats["counts"].get(token, 0) + 1) / denominator)
        scored.append((label, log_score))

    max_score = max(score for _, score in scored)
    weights = [(label, math.exp(score - max_score)) for label, score in scored]
    normalizer = sum(weight for _, weight in weights) or 1.0
    probabilities = sorted(
        [(label, weight / normalizer) for label, weight in weights],
        key=lambda item: item[1],
        reverse=True,
    )
    best_label, confidence = probabilities[0]
    return best_label, confidence, probabilities[1:3]


def load_catalog() -> dict[str, Any]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data.get("useCases"), list) or not isinstance(data.get("models"), list):
            raise ValueError("Catalog must include useCases and models lists.")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not load the model catalog: {exc}")
        st.stop()


def estimate_tokens(text: str) -> int:
    # Approximation only; actual tokenization depends on model and language.
    return math.ceil(len(text.strip()) / 4) if text.strip() else 0


def estimate_cost(model: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * float(model["inputPerMillion"])
        + output_tokens * float(model["outputPerMillion"])
    ) / 1_000_000


def money(value: float) -> str:
    return f"${value:.7f}" if value < 0.0001 else f"${value:.5f}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(ellipse at 75% 4%, rgba(86,100,204,.18), transparent 38rem), #0b1020;
            color: #edf1ff;
          }
          [data-testid="stHeader"] { background: transparent; }
          .block-container { max-width: 900px; padding-top: 2rem; }
          .qw-brand { color:#a7b5ff; font-weight:800; letter-spacing:.02em; }
          .qw-hero { padding:2rem 0 1.3rem; }
          .qw-eyebrow { color:#aebafa; font-size:.72rem; font-weight:750; letter-spacing:.14em; }
          .qw-hero h1 { font-size:clamp(2.5rem,7vw,4.4rem); line-height:.98; letter-spacing:-.06em; margin:.8rem 0; }
          .qw-hero h1 span { color:#a7b5ff; }
          .qw-hero p, .qw-muted { color:#9aa8c3; line-height:1.65; }
          .qw-card { padding:1.2rem 1.4rem; margin:.5rem 0 1rem; border:1px solid rgba(165,184,232,.16); border-radius:16px; background:rgba(17,25,46,.78); }
          .qw-recommend { border-color:rgba(115,229,193,.35); background:rgba(16,31,39,.88); }
          .qw-model { display:flex; justify-content:space-between; gap:1rem; padding:.9rem 0; border-bottom:1px solid rgba(165,184,232,.12); }
          .qw-model:last-child { border-bottom:0; }
          .qw-model small { color:#9aa8c3; }
          .qw-price { text-align:right; white-space:nowrap; font-weight:700; }
          .qw-footer { color:#8491aa; font-size:.8rem; border-top:1px solid rgba(165,184,232,.15); padding-top:1rem; margin-top:2rem; }
          div.stButton > button[kind="primary"] { border:0; color:#10172b; font-weight:750; background:linear-gradient(100deg,#b8c2ff,#8ea1ff); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    catalog = load_catalog()
    use_cases = catalog["useCases"]
    use_case_by_id = {item["id"]: item for item in use_cases}

    st.markdown('<div class="qw-brand">◈ QUERYWISE</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="qw-hero">
          <div class="qw-eyebrow">LLM MODEL MATCHING, MADE SIMPLE</div>
          <h1>Find the right model.<br><span>Skip the guesswork.</span></h1>
          <p>Describe your task to detect its use case, compare estimated token costs, and find a suitable low-cost model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("What are you working on?")
        prompt = st.text_area(
            "Describe your task or paste your prompt",
            height=155,
            max_chars=50_000,
            placeholder="For example: Summarize this report in five bullet points...",
            help="The prompt is processed in this Streamlit session and is not sent to an AI model.",
        )
        input_tokens = estimate_tokens(prompt)
        st.caption(f"Estimated input: {input_tokens:,} tokens · Rough estimate (about 4 characters per token).")

        names = {item["id"]: item["label"] for item in use_cases}
        use_case_options = ["auto", *names.keys()]
        selected = st.selectbox(
            "Use case",
            options=use_case_options,
            format_func=lambda value: "✨ Auto-detect with ML" if value == "auto" else names[value],
            help="The built-in Naive Bayes classifier runs locally in Python. You can also select a category manually.",
        )
        output_tokens = st.select_slider(
            "Expected answer length (tokens)",
            options=[150, 500, 1200, 2500],
            value=500,
            format_func=lambda value: {150: "Short · 150", 500: "Medium · 500", 1200: "Long · 1,200", 2500: "Very long · 2,500"}[value],
        )
        submitted = st.button("Find my best-fit model →", type="primary", use_container_width=True)

    if submitted:
        if not prompt.strip():
            st.warning("Enter a task description or prompt first.")
        else:
            if selected == "auto":
                chosen_use_case, confidence, alternatives = classify_use_case(prompt)
            else:
                chosen_use_case, confidence, alternatives = selected, None, []

            matching = [
                model for model in catalog["models"]
                if chosen_use_case in model.get("useCases", [])
            ]
            for model in matching:
                model["estimatedCost"] = estimate_cost(model, input_tokens, output_tokens)
            matching.sort(key=lambda model: model["estimatedCost"])
            recommendation = matching[0] if matching else None

            st.markdown("---")
            st.markdown('<div class="qw-eyebrow">YOUR RECOMMENDATION</div>', unsafe_allow_html=True)
            st.subheader(names.get(chosen_use_case, chosen_use_case))

            if confidence is not None:
                st.caption(
                    f"Auto-detected by the in-app Naive Bayes classifier · "
                    f"{confidence:.0%} relative confidence (not a validated accuracy score)."
                )
                if alternatives:
                    alt_text = " · ".join(f"{names.get(label, label)} {probability:.0%}" for label, probability in alternatives)
                    st.caption(f"Other possible categories: {alt_text}")

            if recommendation is None:
                st.info("No model is configured for this use case. Add one to data/models.json.")
            else:
                st.markdown('<div class="qw-card qw-recommend">', unsafe_allow_html=True)
                st.markdown(f"### :green[{recommendation['name']}]")
                st.write(f"{recommendation['provider']} · lowest estimated cost among configured models for {names.get(chosen_use_case, chosen_use_case).lower()}.")
                st.metric("Estimated cost for this prompt", money(recommendation["estimatedCost"]))
                st.caption(
                    f"Approx. {input_tokens:,} input + {output_tokens:,} output tokens. "
                    f"Rates: ${recommendation['inputPerMillion']:g}/1M input · "
                    f"${recommendation['outputPerMillion']:g}/1M output."
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("Cost comparison")
                for model in matching:
                    col_name, col_cost = st.columns([3, 1])
                    with col_name:
                        mark = "  ·  RECOMMENDED" if model["id"] == recommendation["id"] else ""
                        st.markdown(f"**{model['name']}**{mark}")
                        st.caption(
                            f"{model['provider']} · ${model['inputPerMillion']:g}/1M input · "
                            f"${model['outputPerMillion']:g}/1M output"
                        )
                    with col_cost:
                        st.markdown(f"<div class='qw-price'>{money(model['estimatedCost'])}<br><small>this prompt</small></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="qw-footer">
          Estimates use a rough character-based token approximation and the sample rates in data/models.json.
          Verify current vendor pricing before budgeting. QueryWise does not call an LLM or send your prompt to a provider.<br><br>
          © 2026 Bishal Mondal
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

# QueryWise — Streamlit deployment

QueryWise is a Streamlit app that classifies a task into a use case with a small, in-app Naive Bayes text classifier, then recommends the lowest-estimated-cost configured model for that category. It estimates cost from input/output token counts and the prices in `data/models.json`.

## Run locally

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app and select this repository and branch.
3. Set **Main file path** to `streamlit_app.py`.
4. Deploy. Streamlit installs dependencies from `requirements.txt`.

The app reads the model catalog from `data/models.json`, so keep that file in the repo and update its rates and use-case mappings as needed.

## Notes

- The built-in classifier is a lightweight demo trained from example phrases in `streamlit_app.py`; its displayed confidence is relative, not a validated accuracy metric. Improve it with a labeled dataset and evaluation before relying on it in production.
- Token count is a character-based approximation, not the tokenizer for a specific provider/model.
- Catalog prices are sample values and are not fetched live. Confirm current provider prices before budgeting.
- The prompt is used in the current Streamlit session for local classification; no external LLM API is called.
- The original Node.js version remains available via `npm start`.

© 2026 Bishal Mondal

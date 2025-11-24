"""
Streamlit app: AI Data Readiness Assessment (Azure OpenAI)

This file is the updated version that uses the provided AzureOpenAI client
initialization pattern (load_dotenv + AzureOpenAI client) instead of the
previous `openai` global configuration.

Usage:
  1. Install dependencies:
     pip install streamlit pandas python-dotenv openpyxl xlrd
     # If you are using the Azure OpenAI helper client referenced here, make
     # sure the package that provides `AzureOpenAI` is installed in your env.

  2. Set environment variables (or create a .env file):
     AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com
     AZURE_OPENAI_API_KEY=YOUR_API_KEY
     AZURE_OPENAI_DEPLOYMENT_NAME=YOUR_DEPLOYMENT_NAME
     AZURE_OPENAI_API_VERSION=2024-12-01

  3. Run:
     streamlit run streamlit_azure_ai_data_readiness_canvas.py

Notes:
 - This file keeps the same user experience but wires the LLM calls through
   the provided `AzureOpenAI` client object. The code attempts to be
   backwards-compatible with the previous ChatCompletion-style call by
   creating a `chat.completions.create(...)` call on the client.

"""

import os
import io
import json
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

# Try to import the AzureOpenAI client wrapper the user requested.
try:
    # Replace the import below with whichever package provides AzureOpenAI in
    # your environment. The user-provided snippet referenced `AzureOpenAI`.
    from openai import AzureOpenAI  # keep this as-is if your env provides it
except Exception:
    AzureOpenAI = None

# ---------------------- Helper: Azure OpenAI setup ----------------------

def init_azure_openai():
    load_dotenv()

    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01")

    if not (AZURE_API_KEY and AZURE_API_BASE and AZURE_DEPLOYMENT):
        raise EnvironmentError(
            "Missing Azure OpenAI config. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT_NAME in the environment."
        )

    if AzureOpenAI is None:
        raise RuntimeError(
            "AzureOpenAI client is not available in this environment. Make sure the package that provides `AzureOpenAI` is installed and the import path is correct."
        )

    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
    )

    return client, AZURE_DEPLOYMENT

# ---------------------- Data inspection utilities ----------------------

def read_uploaded_file(uploaded) -> pd.DataFrame:
    """Read CSV/Excel upload into a DataFrame."""
    filename = uploaded.name.lower()
    bytes_io = io.BytesIO(uploaded.getvalue())
    if filename.endswith(".csv"):
        return pd.read_csv(bytes_io)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(bytes_io)
    else:
        # try csv as fallback
        try:
            return pd.read_csv(bytes_io)
        except Exception:
            raise ValueError("Unsupported file type: " + uploaded.name)


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        ser = df[c]
        dtype = str(ser.dtype)
        nonnull = ser.notnull().sum()
        total = len(ser)
        missing_pct = 1 - (nonnull / total) if total > 0 else 0
        unique_count = ser.nunique(dropna=True)
        sample_values = ser.dropna().astype(str).unique()[:5].tolist()
        cols.append({
            'column': c,
            'dtype': dtype,
            'missing_pct': round(missing_pct, 3),
            'unique_count': int(unique_count),
            'sample_values': sample_values
        })
    return pd.DataFrame(cols)


def detect_potential_target_columns(df: pd.DataFrame) -> List[str]:
    # heuristic: columns with names like target, label, outcome, class, churn
    candidates = []
    keywords = ['target', 'label', 'class', 'outcome', 'y', 'churn', 'is_', 'has_']
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords):
            candidates.append(c)
    # also add columns with low unique_count (e.g., binary)
    for c in df.columns:
        if c not in candidates:
            try:
                if df[c].nunique(dropna=True) <= 20:
                    candidates.append(c)
            except Exception:
                continue
    # dedupe preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

# ---------------------- Prompt builder & LLM call ----------------------

def build_system_prompt() -> str:
    return (
        "You are an expert data scientist focused on assessing whether a dataset is 'AI-ready' for ML/AI projects."
        " Provide a clear assessment: readiness level (Not ready, Partially ready, Ready), key issues, and concrete remediation steps and sample preprocessing code or ideas."
    )


def build_user_prompt(summary: Dict[str, Any], answers: Dict[str, Any], sample_rows: List[Dict[str, Any]]) -> str:
    # summary: header summary DataFrame -> json-able
    summary_json = json.dumps(summary, indent=2)
    answers_json = json.dumps(answers, indent=2)
    sample_json = json.dumps(sample_rows, indent=2)

    prompt = (
        f"Dataset summary:\n{summary_json}\n\n"
        f"User questionnaire answers:\n{answers_json}\n\n"
        f"Sample rows (up to 5):\n{sample_json}\n\n"
        "Please provide:\n"
        "1) Readiness assessment (Not ready / Partially ready / Ready) with a 1-2 sentence justification.\n"
        "2) Top 5 issues prioritized (missing data, label imbalance, privacy, time-series gaps, wrong types, duplicates, leakage, etc.). For each issue include severity (High/Medium/Low).\n"
        "3) Concrete remediation steps and short example code snippets (pandas/sklearn) for cleaning/feature engineering/labeling to address the issues.\n"
        "4) Suggested next steps: what additional data or labels to collect, validation experiments to run, simple baseline model suggestions.\n"
        "5) If privacy/PII concerns exist, highlight them and suggest safer alternatives (pseudonymization, removing columns, synthetic data).\n"
    )
    return prompt


def call_azure_chat(client, deployment_name: str, prompt: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=deployment_name,          # updated param (model instead of engine)
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return f"LLM error: {e}"

    # New Azure SDK response structure
    choice = response.choices[0].message

    # Safely extract message content or function call
    if hasattr(choice, "content") and choice.content:
        return choice.content

    # If it returns a function call (tool call)
    if hasattr(choice, "function_call") and choice.function_call:
        return str(choice.function_call)

    # Fallback
    return str(choice)


# ---------------------- Streamlit App UI ----------------------

def main():
    st.set_page_config(page_title="AI Data Readiness — Azure OpenAI", layout="wide")
    st.title("AI Data Readiness Assessment")
    st.caption("Upload a dataset (CSV / XLSX). The app will inspect the data, ask a few questions, then use Azure OpenAI to provide an assessment and remediation steps.")

    with st.sidebar:
        st.header("Azure OpenAI settings")
        st.write("The app uses these environment variables (set them before running).")
        st.code("AZURE_OPENAI_ENDPOINT\nAZURE_OPENAI_API_KEY\nAZURE_OPENAI_DEPLOYMENT_NAME\n")
        show_settings = st.checkbox("Show current env (local) values", value=False)
        if show_settings:
            st.write({
                'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
                'AZURE_OPENAI_DEPLOYMENT_NAME': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            })

    uploaded = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])

    if uploaded is None:
        st.info("Upload a file to begin. You can also use the example dataset below.")
        if st.button("Use example: Iris dataset (small)"):
            from sklearn import datasets
            iris = datasets.load_iris(as_frame=True).frame
            uploaded_df = iris
            st.session_state['df'] = uploaded_df
    else:
        try:
            df = read_uploaded_file(uploaded)
            st.session_state['df'] = df
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    if 'df' in st.session_state:
        df: pd.DataFrame = st.session_state['df']
        st.subheader("Preview & summary")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(df.head(10))
        with c2:
            st.write(f"Rows: {len(df)}  Columns: {len(df.columns)}")
            cs = column_summary(df)
            st.dataframe(cs)

        # Dynamic questionnaire
        st.subheader("Quick AI-readiness questionnaire")
        suggested_targets = detect_potential_target_columns(df)
        target_column = st.selectbox("Select target/label column (if any)", options=["(none)"] + suggested_targets, index=1 if suggested_targets else 0)

        is_time_series = st.radio("Is this dataset time-series/has an explicit timestamp column?", options=["No", "Yes — has timestamp column"], index=0)
        timestamp_col = None
        if is_time_series.startswith("Yes"):
            timestamp_col = st.selectbox("Select timestamp column", options=["(none)"] + df.columns.tolist())

        has_identifiers = st.radio("Does the dataset include personal identifiers (PII) like name, email, phone, user_id)?", options=["No", "Yes"], index=0)
        pii_cols = []
        if has_identifiers == "Yes":
            pii_cols = st.multiselect("Select PII columns", options=df.columns.tolist())

        label_quality = None
        if target_column != "(none)":
            label_quality = st.selectbox("Label quality (if target chosen)", options=["High (well-labeled)", "Medium (some noise)", "Low (many errors/missing)", "No labels / needs labeling"]) 

        missing_policy = st.selectbox("What's an acceptable missing-data policy?", options=["Drop rows with missing values", "Fill with simple imputation (mean/median)", "More advanced imputation required", "Depends on column (custom)"])

        business_goal = st.text_area("Briefly describe the business problem / use case you'd like AI to solve (1-2 sentences)")

        # collect questionnaire answers
        answers = {
            'target_column': target_column,
            'is_time_series': is_time_series,
            'timestamp_col': timestamp_col,
            'has_pii': has_identifiers == 'Yes',
            'pii_columns': pii_cols,
            'label_quality': label_quality,
            'missing_policy': missing_policy,
            'business_goal': business_goal.strip()
        }

        # Prepare sample rows (up to 5) as list of dicts
        sample_rows = df.head(5).fillna('').to_dict(orient='records')

        st.markdown("---")
        col_run, col_opts = st.columns([1, 2])
        with col_opts:
            st.write("Optional LLM settings")
            temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1)
            max_tok = st.slider("Max tokens", min_value=128, max_value=2048, value=800, step=64)
        with col_run:
            run_btn = st.button("Run AI-readiness assessment")

        if run_btn:
            with st.spinner("Initializing Azure OpenAI and preparing prompt..."):
                try:
                    client, deployment = init_azure_openai()
                except Exception as e:
                    st.error(f"Azure OpenAI config error: {e}")
                    st.stop()

                summary_table = column_summary(df).to_dict(orient='records')
                prompt = build_user_prompt(summary_table, answers, sample_rows)

                st.subheader("Prompt preview (sent to LLM)")
                st.code(prompt[:2000] + ("\n..." if len(prompt) > 2000 else ""))

                with st.spinner("Calling Azure OpenAI..."):
                    try:
                        ai_response = call_azure_chat(client, deployment, prompt, temperature=temp, max_tokens=max_tok)
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
                        st.stop()

                st.subheader("AI assessment & suggestions")
                st.write(ai_response)

                # allow download
                report_text = f"=== AI Data Readiness Report ===\n\nBusiness goal:\n{business_goal}\n\nData summary:\n{json.dumps(summary_table, indent=2)}\n\nQuestionnaire answers:\n{json.dumps(answers, indent=2)}\n\nLLM assessment:\n{ai_response}\n"
                st.download_button("Download report (txt)", data=report_text, file_name="ai_data_readiness_report.txt")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for demonstration. Use responsibly. Remove PII before sending data to external APIs unless allowed.")


if __name__ == '__main__':
    main()

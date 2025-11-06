import os
import json
import streamlit as st
from typing import Callable, Dict, Any, List
from dotenv import load_dotenv

from openai import AzureOpenAI

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

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_BASE,
    api_version=AZURE_API_VERSION,
)

def db_tool(query: str) -> str:
    """Simulated DB tool. In this deployment DB access is disabled."""
    return (
        "DB_ACCESS_DISABLED: I do not have database access in this deployment. "
        "If you need DB queries, please configure a DB connector and re-run."
    )

def idea_evaluator_tool(idea_text: str) -> str:
    """Call the LLM to evaluate an idea and return structured JSON (string)."""
    prompt = f"""
You are a concise product/market analyst assistant.

Input idea:
`{idea_text}`

Task:
1) List 4-6 concise follow-up questions that would help decide market impact and feasibility.
2) Give a short (1-paragraph) market-impact assessment (low/medium/high — explain why).
3) Provide 4 concrete, prioritized suggestions (what to validate first, experiments to run, MVP scope).

Output strictly as JSON with keys:
{{
  "follow_up_questions": ["q1", "q2", ...],
  "assessment": "one-paragraph string",
  "impact_level": "low|medium|high",
  "suggestions": ["s1", "s2", "s3", "s4"]
}}

Do not output anything outside the JSON. Keep answers concise.
"""
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        resp_text = resp.choices[0].message.content.strip()
    except Exception as e:
        resp_text = f"LLM error: {e}"

    try:
        parsed = json.loads(resp_text)
        return json.dumps(parsed, indent=2)
    except Exception:
        start = resp_text.find("{")
        end = resp_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = resp_text[start:end+1]
            try:
                parsed = json.loads(candidate)
                return json.dumps(parsed, indent=2)
            except Exception:
                pass
    return json.dumps({
        "raw_output": resp_text,
        "note": "LLM output could not be parsed as JSON. Consider tightening the prompt or using a structured output parser.",
    }, indent=2)

TOOLS: Dict[str, Callable[[str], str]] = {
    "DB": db_tool,
    "IdeaEvaluator": idea_evaluator_tool,
}

FUNCTIONS_METADATA = [
    {
        "name": "DB",
        "description": "Execute a database query or request. In this environment DB access may be disabled.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL or natural language query for the DB."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "IdeaEvaluator",
        "description": "Evaluate a startup/product idea. Returns a concise JSON with follow-ups, assessment, impact_level and suggestions.",
        "parameters": {
            "type": "object",
            "properties": {
                "idea_text": {"type": "string", "description": "Short description of the idea to evaluate."}
            },
            "required": ["idea_text"],
        },
    },
]

class SimpleAgent:
    def __init__(self, client: AzureOpenAI, deployment: str):
        self.client = client
        self.deployment = deployment
        self.system_prompt = "You are a helpful, concise assistant. Use tools when appropriate."
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def handle(self, user_message: str) -> str:
        """Add user message, handle function call if needed, and return assistant response."""
        self.messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=self.messages,
                functions=FUNCTIONS_METADATA,
                function_call="auto",
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as e:
            return f"LLM error: {e}"

        choice = response.choices[0].message

        func_call = getattr(choice, "function_call", None)
        if not func_call and isinstance(choice, dict):
            func_call = choice.get("function_call")

        if func_call:
            func_name = getattr(func_call, "name", None) or func_call.get("name")
            func_args_raw = getattr(func_call, "arguments", None) or func_call.get("arguments", "{}")

            try:
                func_args = json.loads(func_args_raw or "{}") if isinstance(func_args_raw, str) else func_args_raw
            except Exception:
                func_args = {"_raw": func_args_raw}

            if func_name in TOOLS:
                arg_value = None
                if isinstance(func_args, dict) and len(func_args) > 0:
                    arg_value = next(iter(func_args.values()))
                    if not isinstance(arg_value, str):
                        arg_value = json.dumps(arg_value)
                else:
                    arg_value = str(func_args)

                try:
                    tool_result = TOOLS[func_name](arg_value)
                except Exception as e:
                    tool_result = f"Tool execution error: {e}"

                # Add tool messages to history
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": func_name, "arguments": func_args_raw}
                })
                self.messages.append({"role": "function", "name": func_name, "content": tool_result})

                # Ask model for the final assistant response
                try:
                    final_resp = self.client.chat.completions.create(
                        model=self.deployment,
                        messages=self.messages,
                        temperature=0.2,
                        max_tokens=800,
                    )
                except Exception as e:
                    return f"LLM error when finalizing: {e}"

                final_msg = final_resp.choices[0].message.content.strip()
                self.messages.append({"role": "assistant", "content": final_msg})
                return final_msg
            else:
                return f"Model requested unknown function '{func_name}'."

        # ---- Normal assistant message (no function call) ----
        assistant_reply = getattr(choice, "content", None) or choice.get("content", "")
        assistant_reply = assistant_reply.strip() if assistant_reply else "(no reply)"
        self.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

st.set_page_config(page_title="Idea Chat bot (function-calling)", page_icon="💡")
st.title("Idea Chat bot — function-calling demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = SimpleAgent(client=client, deployment=AZURE_DEPLOYMENT)
agent = st.session_state.agent

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up? (ask about idea or DB)"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        reply = agent.handle(prompt)

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

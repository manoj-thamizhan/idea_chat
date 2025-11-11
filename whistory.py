# chat2.py
import os
import json
import copy
import datetime
import streamlit as st
from typing import List, Dict, Any, Callable
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

SYSTEM_PROMPT = (
    "You are a Product Market Analyst assistant. Follow the protocol below exactly:\n"
    "1) When a user asks to build, design, launch, or evaluate a product or feature, start by asking a single concise follow-up question.\n"
    "2) Ask follow-up questions one at a time and wait for the user's reply before asking the next.\n"
    "3) After each user reply, either: a) ask the next single follow-up question, or b) provide a single focused suggestion (one recommendation) based on the information so far.\n"
    "4) When you have sufficient information, send exactly one separate message that describes the MARKET IMPACT (size, direction, risks, and a short recommendation). That market-impact message must be its own assistant message.\n"
    "5) Keep messages concise, actionable, and numbered when listing short points. If the user asks you to change behavior, confirm and continue following this one-at-a-time protocol.\n"
    "6) If the user did not ask for product work (for example: asking for definitions), answer normally but stay in role as a Product Market Analyst."
    "7) If the idea or thing does not make any sense just reply straight that it is sort of dumb or not works out"
    "8) You have access to DB tool in order to answer any questions for the DB side"
)

TOOLS: Dict[str, Callable[[str], str]] = {
    "DB": db_tool,
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
]


class SimpleAgent:
    def __init__(self, client: AzureOpenAI, deployment: str):
        self.client = client
        self.deployment = deployment
        self.system_prompt = SYSTEM_PROMPT
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
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

                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": func_name, "arguments": func_args_raw}
                })
                self.messages.append({"role": "function", "name": func_name, "content": tool_result})

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

        assistant_reply = getattr(choice, "content", None) or choice.get("content", "")
        assistant_reply = assistant_reply.strip() if assistant_reply else "(no reply)"
        self.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply


# --- Persistence helpers ---
DATA_DIR = os.getenv("CRITIQ_DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
SAVED_FILE = os.path.join(DATA_DIR, "saved_chats.json")
CURRENT_FILE = os.path.join(DATA_DIR, "current_messages.json")


def load_json_file(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json_file(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save {path}: {e}")


st.set_page_config(page_title="Critiq agent", page_icon="💡", layout="wide")

default_saved = {
}

loaded_saved = load_json_file(SAVED_FILE, default_saved)
loaded_current = load_json_file(CURRENT_FILE, [])

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = loaded_saved

if "messages" not in st.session_state:
    # load previously saved current messages (if any) otherwise start empty
    st.session_state.messages = loaded_current if isinstance(loaded_current, list) else []

if "agent" not in st.session_state:
    # create a working agent by default
    st.session_state.agent = SimpleAgent(client=client, deployment=AZURE_DEPLOYMENT)
    # if we loaded current messages, inject them into the agent so its internal state matches the UI
    for m in st.session_state.messages:
        st.session_state.agent.messages.append(m)
agent = st.session_state.agent

# ensure disk reflects what we loaded (no-op if same)
save_json_file(SAVED_FILE, st.session_state.saved_chats)
save_json_file(CURRENT_FILE, st.session_state.messages)

with st.sidebar:
    st.title("💡 Critiq Agent")
    if st.button("🆕 New Chat"):
        msgs = st.session_state.messages
        if msgs and any(m.get("role") == "user" for m in msgs):
            first_user = next((m for m in msgs if m.get("role") == "user"), None)
            if first_user:
                prefix = first_user["content"].strip().split("\n")[0][:40]
                title = f"{prefix} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            else:
                title = f"Chat {len(st.session_state.saved_chats)+1} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if title in st.session_state.saved_chats:
                title = f"{title} (copy)"
            st.session_state.saved_chats[title] = copy.deepcopy(msgs)
            save_json_file(SAVED_FILE, st.session_state.saved_chats)

        st.session_state.messages = []
        st.session_state.agent.reset()
        save_json_file(CURRENT_FILE, st.session_state.messages)

    st.markdown("---")
    st.markdown("### 💬 Chat History")
    if st.session_state.saved_chats:
        for i, (title, msgs) in enumerate(list(st.session_state.saved_chats.items())):
            if st.button(title, key=f"load_chat_{i}"):
                st.session_state.messages = copy.deepcopy(msgs)
                st.session_state.agent.reset()
                for m in st.session_state.messages:
                    st.session_state.agent.messages.append(m)
                save_json_file(CURRENT_FILE, st.session_state.messages)
    else:
        st.write("No history yet")

    st.markdown("---")
    if st.button("💾 Save current chat"):
        msgs = st.session_state.messages
        if msgs and any(m.get("role") == "user" for m in msgs):
            first_user = next((m for m in msgs if m.get("role") == "user"), None)
            if first_user:
                prefix = first_user["content"].strip().split("\n")[0][:40]
                title = f"{prefix} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            else:
                title = f"Chat {len(st.session_state.saved_chats)+1} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if title in st.session_state.saved_chats:
                title = f"{title} (copy)"
            st.session_state.saved_chats[title] = copy.deepcopy(msgs)
            save_json_file(SAVED_FILE, st.session_state.saved_chats)
            st.success(f"Saved chat as: {title}")
    st.markdown("---")

st.title("Critiq agent")

# render messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role", "user")):
        st.markdown(message.get("content", ""))

# input handling: when user submits, append to messages and autosave to disk
if prompt := st.chat_input("Describe your product idea or ask for an analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # persist current messages immediately
    save_json_file(CURRENT_FILE, st.session_state.messages)

    with st.spinner("Thinking like a product-market analyst..."):
        reply = agent.handle(prompt)

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # persist after assistant reply as well
    save_json_file(CURRENT_FILE, st.session_state.messages)

import os
import json
import streamlit as st
from typing import List, Dict, Any,Callable
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

st.set_page_config(page_title="Critiq agent", page_icon="💡")
st.title("Critiq agent")

if "messages" not in st.session_state:
    st.session_state.messages = []


if "agent" not in st.session_state:
    st.session_state.agent = SimpleAgent(client=client, deployment=AZURE_DEPLOYMENT)

agent = st.session_state.agent

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your product idea or ask for an analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking like a product-market analyst..."):
        reply = agent.handle(prompt)

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})



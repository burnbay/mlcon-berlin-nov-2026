"""The oMLX twin of test_mcp_client_ollama.py: the same MCP client (it starts mcp_server.py over stdio, lists its
tools and lets the model chain tool calls in an agentic loop), but the LLM is Qwen3.5-4B served by oMLX's
OpenAI-compatible /v1/chat/completions instead of Ollama.

Needs oMLX running on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and OMLX_API_KEY in
the environment or the repo-root .env. Needs `pip install mcp httpx requests python-dotenv`.

    python test_mcp_client_omlx.py
"""
import asyncio
import json
import os
import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}

OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"   # oMLX names Hugging Face cache models "org--repo"


def call_omlx(messages, tools=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(OMLX_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling oMLX: {e}")
        return None


def convert_mcp_tool_to_openai_format(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }
    }


async def run_mcp_demo_with_omlx():
    import sys

    python_path = sys.executable
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    server_params = StdioServerParameters(
        command=python_path,
        args=[server_script],
        env=None
    )

    print("Starting MCP Server...")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(tools_list.tools)} tools:")
            for tool in tools_list.tools:
                print(f"   - {tool.name}: {tool.description}")
            print()

            openai_tools = [
                convert_mcp_tool_to_openai_format(tool)
                for tool in tools_list.tools
            ]

            questions = [
                "What is the weather in the capital of the United States?",
                "What is the population of France and what's the weather like in Paris?",
                "Tell me about Japan - what's its capital and what's the weather there?"
            ]

            for i, question in enumerate(questions, 1):
                print(f"\n{'=' * 60}")
                print(f"Question {i}: {question}")
                print('=' * 60)

                messages = [{"role": "user", "content": question}]

                # Agentic loop: allow the model to chain tool calls
                max_iterations = 10
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1

                    response_data = call_omlx(messages, tools=openai_tools)

                    if not response_data:
                        print("Failed to get response from oMLX")
                        break

                    assistant_message = response_data["choices"][0]["message"]

                    tool_calls = assistant_message.get("tool_calls") or []

                    if tool_calls:
                        messages.append(assistant_message)

                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            # The OpenAI format sends the arguments as a JSON string
                            function_args = json.loads(tool_call["function"]["arguments"] or "{}")

                            print(f"\nCalling tool: {function_name}")
                            print(f"   Input: {json.dumps(function_args, indent=2)}")

                            try:
                                result = await session.call_tool(function_name, function_args)

                                result_text = ""
                                for content in result.content:
                                    if hasattr(content, "text"):
                                        result_text += content.text

                                print(f"   Result: {result_text[:200]}...")

                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": result_text
                                })

                            except Exception as e:
                                print(f"   Error calling tool: {e}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": f"Error: {str(e)}"
                                })
                    else:
                        content = assistant_message.get("content", "")
                        if content:
                            print(f"\nAnswer:\n{content}")
                        break

                if iteration >= max_iterations:
                    print(f"\nReached maximum iterations ({max_iterations})")

            print(f"\n\n{'=' * 60}")
            print("Demo completed successfully!")
            print('=' * 60)


async def check_omlx_availability():
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models", headers=HEADERS, timeout=5)
        response.raise_for_status()

        models_data = response.json()
        available_models = [m["id"] for m in models_data.get("data", [])]

        if MODEL not in available_models:
            print(f"Warning: Model '{MODEL}' not found in oMLX")
            print(f"Available models: {', '.join(available_models)}")
            print("\nDownload mlx-community/Qwen3.5-4B-MLX-4bit into the Hugging Face cache, then reload oMLX's models")
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to oMLX at http://127.0.0.1:8000")
        print(f"Make sure oMLX is running on port 8000")
        print(f"Error details: {e}")
        return False


if __name__ == "__main__":
    print("MCP Client with oMLX - Open Source LLM Demo")
    print(f"Model: {MODEL}")
    print()

    if not asyncio.run(check_omlx_availability()):
        print("\nCannot proceed without oMLX running")
        exit(1)

    try:
        asyncio.run(run_mcp_demo_with_omlx())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

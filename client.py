from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["MathServer.py"],
                "transport":  "stdio"
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    import os
    # os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY_1")

    tools = await client.get_tools()
    model = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=os.getenv('DEEPSEEK_API_KEY_1'), base_url="https://api.deepseek.com", max_retries=2)
    agent = create_react_agent(
        model, tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (8 + 2) * 12 ?"}]}
    )
    print("Maths Response: ",math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather in California?"}]}
    )
    print("Weather Response: ",weather_response['messages'][-1].content)


asyncio.run(main())
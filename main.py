import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

load_dotenv()


SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C:\\Python313\\Scripts\\uv.exe",
        "args": ["run", "fastmcp", "run", "D:/mcp_math_server/main.py"]
    },
    "expense": {
        "transport": "streamable_http",
        "url": "https://soft-black-jellyfish.fastmcp.app/mcp"
    }
}
async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    #print(tools)

    named_tools = {}
    for tool in tools:  
        named_tools[tool.name] = tool

    #print(named_tools)
    print('Available tools:', named_tools.keys())

    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)

    prompt = "Add an expense - Rs 800 for groceries on 26th Nov"
    #prompt = "What is the capital of India?"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply: ",response.content)
        return 
    
    tool_message = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        print(f"\n-> Executing remote tool: {selected_tool}")
        print(f" with args: ", selected_tool_args)

        # print(f"Selected tool: {selected_tool}")
        # print(f"Selected tool args: {selected_tool_args}")

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        # print(f"Tool result: {tool_result}")

        tool_message.append(ToolMessage(tool_call_id=selected_tool_id, content = json.dumps(result)))
    
    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_message])
    print(f"Final response: {final_response.content}")
    
if __name__ == "__main__":
    asyncio.run(main())

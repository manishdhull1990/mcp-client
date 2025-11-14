import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

load_dotenv()


SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C:\\Python313\\Scripts\\uv.exe",
        "args": ["run", "fastmcp", "run", "D:/mcp_math_server/main.py"]
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

    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)

    # prompt = "What is the product of 12 and 15 using the math tool?"
    prompt = "What is the capital of India?"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply: ",response.content)
        return 
    
    selected_tool = response.tool_calls[0]["name"]
    selected_tool_args = response.tool_calls[0]["args"]
    selected_tool_id = response.tool_calls[0]["id"]

    # print(f"Selected tool: {selected_tool}")
    # print(f"Selected tool args: {selected_tool_args}")

    tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
    # print(f"Tool result: {tool_result}")

    tool_message = ToolMessage(content=tool_result, tool_call_id=selected_tool_id)
    final_response = await llm_with_tools.ainvoke([prompt, response, tool_message])

    print(f"Final response: {final_response.content}")
    
if __name__ == "__main__":
    asyncio.run(main())

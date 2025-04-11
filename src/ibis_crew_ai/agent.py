"""Main agent code."""

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ibis_crew_ai.crew.crew import DevCrew

LOCATION = 'us-central1'
LLM = 'gemini-2.0-flash-001'


@tool
def coding_tool(code_instructions: str) -> str:
    """Use this tool to write a python program given a set of requirements and or instructions."""
    inputs = {'code_instructions': code_instructions}
    return DevCrew().crew().kickoff(inputs=inputs)


tools = [coding_tool]

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=4096, streaming=True,
).bind_tools(tools)


# 3. Define workflow components
def should_continue(state: MessagesState) -> str:
    """Determines whether to use the crew or end the conversation."""
    last_message = state['messages'][-1]
    return 'dev_crew' if last_message.tool_calls else END


def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response."""
    system_message = (
        "You are an expert Lead Software Engineer Manager.\n"
        "Your role is to speak to a user and understand what kind of code they need to "
        "build.\n"
        "Part of your task is therefore to gather requirements and clarifying ambiguity "
        "by asking followup questions. Don't ask all the questions together as the user "
        "has a low attention span, rather ask a question at the time.\n"
        "Once the problem to solve is clear, you will call your tool for writing the "
        "solution.\n"
        "Remember, you are an expert in understanding requirements but you cannot code, "
        "use your coding tool to generate a solution. Keep the test cases if any, they "
        "are useful for the user."
    )

    messages_with_system = [{'type': 'system', 'content': system_message}] + state['messages']
    # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
    response = llm.invoke(messages_with_system, config)
    return {'messages': response}


# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node('agent', call_model)
workflow.add_node('dev_crew', ToolNode(tools))
workflow.set_entry_point('agent')

# 5. Define graph edges
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('dev_crew', 'agent')

# 6. Compile the workflow
agent = workflow.compile()

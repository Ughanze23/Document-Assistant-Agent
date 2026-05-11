from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from src.schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from src.prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]


    actions_taken: Annotated[List[str],add_messages]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used



def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)
    

    #Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt()
    formated_prompt = prompt.invoke({
        "user_input": state.get("user_input", ""),
        "conversation_history": history,
    })
    intent_response = structured_llm.invoke(formated_prompt)
    
    #Add conditional logic to set next_step based on intent
    next_step = intent_response.intent_type 

    if intent_response.intent_type == "qa":
        next_step = "qa"
    elif intent_response.intent_type == "summarization":
        next_step = "summarization"
    elif intent_response.intent_type == "calculation":
        next_step = "calculation"
    else:
        next_step = "qa" # Default to QA if intent is unknown, could also choose to end or ask for clarification

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent_response,
        "next_step": next_step
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the summarization_agent function. Refer to README.md Task 2.3
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    result, tools_used = invoke_react_agent(SummarizationResponse, [], llm, tools)

    prompt_template = get_chat_prompt_template("summarization")
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",

    }


# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")
    result, tools_used = invoke_react_agent(CalculationResponse, [], llm, tools)

    prompt_template = get_chat_prompt_template("calculation")
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()
   
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }



def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(
        UpdateMemoryResponse
    )

    response = structured_llm.invoke(prompt_with_history)
    return {
        "conversation_summary":  response.summary,# TODO: Extract summary from response
        "active_documents":  response.active_documents,# TODO: Update with the current active documents
        "next_step":  "end"# TODO: Update the next step to end
    }

    def should_continue(state: AgentState) -> str:
        """Router function"""
        return state.get("next_step", "end")

    def create_workflow(llm, tools):
        """
        Creates the LangGraph agents.
        Compiles the workflow with an InMemorySaver checkpointer to persist state.
        """
        workflow = StateGraph(AgentState)

        # TODO: Add all the nodes to the workflow by calling workflow.add_node(...)
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("qa", qa_agent)
        workflow.add_node("summarization", summarization_agent)
        workflow.add_node("calculation", calculation_agent)
        workflow.add_node("update_memory", update_memory)

        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            should_continue,
            {
                # TODO: Map the intent strings to the correct node names
                "qa": "qa",
                "summarization": "summarization",
                "calculation": "calculation",   
                "end": END
            }
        )

        memory = InMemorySaver()
    
        workflow.add_edge("qa", "update_memory")
        workflow.add_edge("summarization", "update_memory")
        workflow.add_edge("calculation", "update_memory")   

        workflow.add_edge("update_memory", END)

        return workflow.compile(memory)
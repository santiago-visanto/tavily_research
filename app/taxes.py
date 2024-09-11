from typing import TypedDict, List, Annotated, Literal, Dict, Union, Optional 
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field
from tavily import AsyncTavilyClient
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
import os
from libs.pdf_generator import generate_pdf_from_md
from dotenv import load_dotenv

load_dotenv()

# Define the research state
class ResearchState(TypedDict):
    report: str
    documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    messages: Annotated[list[AnyMessage], add_messages]

# Define the structure for the model's response, which includes citations.
class Citation(BaseModel):
    source_id: str = Field(
        ...,
        description="The url of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources. Include any relevant sources in the answer as markdown hyperlinks. For example: 'This is a sample text ([url website](url))'"
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
    
# Add Tavily's arguments to enhance the web search tool's capabilities
class TavilyQuery(BaseModel):
    query: str = Field(description="sub query")
    topic: str = Field(description="type of search, should be 'general' or 'news'")
    days: int = Field(description="number of days back to run 'news' search")
    raw_content: bool = Field(description="include raw content from found sources, use it ONLY if you need more information besides the summary content provided")
    include_domains: List[str] = Field(
        default=[
            "https://estatuto.co/129",
            "https://www.secretariasenado.gov.co/senado/basedoc/estatuto_tributario.html",
            "https://accounter.co/normatividad/se-reglamenta-el-art-107-2-del-et-referente-a-la-deducciones-por-contribuciones-a-educacion-de-los-empleados-decreto-1013-de-2020.html",
            "https://xperta.legis.co/",
            "https://cortesuprema.gov.co/",
            "https://dian.gov.co",
            "https://estatuto.co/129"
        ],
        description="list of domains to include in the research"
    )

# Define the args_schema for the tavily_search tool using a multi-query approach, enabling more precise queries for Tavily.
class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")

@tool("tavily_search", args_schema=TavilySearchInput, return_direct=True)
async def tavily_search(sub_queries: List[TavilyQuery]):
    """
    Realiza búsquedas web utilizando el servicio Tavily.

    Esta función toma una lista de consultas y realiza búsquedas web para cada una,
    utilizando los parámetros especificados en cada consulta. Los resultados de todas
    las búsquedas se combinan y se devuelven.

    Args:
        sub_queries (List[TavilyQuery]): Una lista de objetos TavilyQuery, cada uno
        especificando los parámetros para una búsqueda individual.

    Returns:
        List[Dict]: Una lista de resultados de búsqueda combinados de todas las consultas.
    """
    search_results = []
    for query in sub_queries:
        response = await tavily_client.search(
            query=query.query,
            topic=query.topic,
            days=query.days,
            include_raw_content=query.raw_content,
            max_results=5,
            include_domains=query.include_domains
        )
        search_results.extend(response['results'])
    
    return search_results

tools = [tavily_search]
tools_by_name = {tool.name: tool for tool in tools}
tavily_client = AsyncTavilyClient()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools(tools)

# Define an async custom tool node to store Tavily's search results for improved processing and filtering.
async def tool_node(state: ResearchState):
    docs = state.get('documents', {})
    docs_str = ""
    msgs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        new_docs = await tool.ainvoke(tool_call["args"])
        if isinstance(new_docs, list):
            for doc in new_docs:
                if isinstance(doc, dict) and 'url' in doc:
                    # Make sure that this document was not retrieved before
                    if doc['url'] not in docs:
                        docs[doc['url']] = doc
                        docs_str += json.dumps(doc)
                else:
                    # If doc is not a dict or doesn't have 'url', we treat it as a string
                    docs_str += str(doc)
        else:
            # If new_docs is not a list, we treat it as a single response
            docs_str += str(new_docs)
        
        msgs.append(ToolMessage(content=f"Found the following new documents: {docs_str}", tool_call_id=tool_call["id"]))
    
    return {"messages": msgs, "documents": docs}
    
# Invoke the model with research tools to gather information about the company.     
def call_model(state: ResearchState):
    prompt = f"""You are an expert assistant in Colombian tax legislation. Your task is to search for and collect relevant information on the tax deductibility of employee education expenses paid by companies in Colombia.

Please search and collect:

1. Current tax laws and regulations related to this topic.
2. Specific articles of the Colombian Tax Statute, especially article 107-2 and any other relevant articles.
3. Recent DIAN concepts or resolutions on this matter.
4. Relevant jurisprudence of the Colombian high courts.
5. Analysis of tax experts on this topic.

Focus on the most recent and relevant information. Be sure to include the dates of the sources found.

Respond in Spanish language\n
    """
    messages = state['messages'] + [SystemMessage(content=prompt)]
    # print("state['messages']:",state['messages'])
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
    

# Define the function that decides whether to continue research using tools or proceed to writing the report
def should_continue(state: ResearchState) -> Literal["tools", "write_report"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user with citations)
    return "write_report"

# Define the function to write the report based on the retrieved documents.
def write_report(state: ResearchState):
    # Create the prompt
    prompt = f"""oday's date is {datetime.now().strftime('%d/%m/%Y')}\n.
You are an expert on Colombian tax legislation. Your task is to analyze the information collected and provide a detailed report in Spanish on the tax deductibility of employee education expenses paid by companies in Colombia.

Context: Two people are debating whether the payment of employee education generates a tax deduction in Colombia for the employer. 
One claims that it does not apply according to article 107-2 of the Colombian Tax Statute, while the other says it does.

Based on the documents and information gathered, please:

1. determine which of the two people is right and why.
2. Explain in detail the conditions under which these expenses may be deductible, if applicable.
3. Cite the specific laws, regulations or concepts that support your analysis.
4. Mention any recent changes in legislation that may affect this situation.
5. Provide recommendations for companies considering offering this benefit.
6. Analyzes concepts of obsolescence

Use only the most relevant and recent information from the documents provided to develop your response. Make sure your analysis is comprehensive and up to date with current legislation.

    Here are all the documents you gathered so far:\n{state['documents']}\n
    Use only the relevant and most recent documents to provide a comprehensive answer to the tax query.
    Respond in Spanish language""" 
    messages = [state['messages'][-1]] + [SystemMessage(content=prompt)]
    response = model.with_structured_output(QuotedAnswer).invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [AIMessage(content=f"Generated Report:\n{response.answer}")], "report": response.answer}

def generete_pdf(state: ResearchState):
    directory = "reports"
    file_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
    msg = generate_pdf_from_md(state['report'], filename=f'{directory}/{file_name}.pdf')
    return {"messages": [AIMessage(content=msg)]}

# Define a graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("research", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("write_report", write_report)
workflow.add_node("generate_pdf", generete_pdf)
# Set the entrypoint as route_query
workflow.set_entry_point("research")

# Determine which node is called next
workflow.add_conditional_edges(
    "research",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# Add a normal edge from `tools` to `route_query`.
# This means that after `tools` is called, `route_query` node is called next.
workflow.add_edge("tools", "research")
workflow.add_edge("write_report", "generate_pdf")  # Option in the future, to add another step and filter the documents retrieved using rerhank before writing the report
workflow.add_edge("generate_pdf", END)  # Option in the future, to add another step and filter the documents retrieved using rerhank before writing the report

app = workflow.compile()
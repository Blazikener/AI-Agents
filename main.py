from llama_index.llms.groq import Groq
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os
import pandas as pd
from prompt import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pdf import canada_engine

load_dotenv()

llm = Groq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df,
    llm=llm,
    verbose=True,
    instruction_str=instruction_str,
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    context=context
)  

while (prompt := input("Enter a prompt (e to exit): ")) != "e":
    result = agent.query(prompt)
    print(result)
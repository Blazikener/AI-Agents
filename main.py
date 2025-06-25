from llama_index.llms.groq import Groq
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os
import pandas as pd
from prompt import new_prompt, instruction_str

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

response = population_query_engine.query("What is the population of Canada?")
print(response)

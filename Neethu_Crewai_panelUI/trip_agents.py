##** credits: https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner **
##************************************************************************
## using the model "llama3-8b-8192" from chatGroq API. You can change the model name.
## Also, tried with the models from Hugging face.
## used the tools ScrapeWebsiteTool and SerperDevTool from crewai_tools
##************************************************************************

from crewai import Agent
from langchain_groq import ChatGroq
import os
from calculator_tools import CalculatorTools
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_community.llms import HuggingFaceEndpoint
from langchain.agents import load_tools


from dotenv import load_dotenv
load_dotenv()

tools = load_tools(["human"])

serper_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])
search_venues_tool = ScrapeWebsiteTool(website_url="https://olympics.com/en/paris-2024/venues")
search_eateries_tool =ScrapeWebsiteTool(website_url="https://www.theinfatuation.com/paris/guides/where-eat-paris-new")
search_pubs_tool= ScrapeWebsiteTool(website_url="https://www.visitparisregion.com/en/inspiration/top-experiences/irish-pubs-paris")

# using the model from Groq cloud API-- > https://console.groq.com
# need to put the API key in the .env file
llm = ChatGroq(temperature=0.1, groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")

## Using the model from Hugging Face. You can try with different models changing the repo_id.
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, temperature=0.2, huggingfacehub_api_token=os.environ["HF_TOKEN"]
# )

class TripAgents():

  def travels_representative(self):
    return Agent(
        role='Travels Representative',
        goal='Ask questions one by one and take inputs from the customer/user regarding their travel plan',
        backstory='An experienced representative in understanding and collectiong information from the customer',
        verbose=True,
        llm=llm,
        tools = tools,
        allow_delegation=False,)

  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        # tools=[
        #     serper_tool,

        # ],
        verbose=True,
        llm=llm,
        allow_delegation=False)

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        # tools=[
        #     serper_tool,
        #     search_venues_tool,
        #     search_eateries_tool,
        #     search_pubs_tool,
        # ],
        verbose=True,
        llm=llm,
        allow_delegation=False)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        # tools=[
        #     serper_tool,
        #     search_venues_tool,
        #     search_eateries_tool,
        #     search_pubs_tool,
        #     CalculatorTools.calculate,
        # ],
        verbose=True,
        llm=llm,
        allow_delegation=False)

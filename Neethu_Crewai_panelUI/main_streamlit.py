from crewai import Crew
from trip_agents import TripAgents
from trip_tasks import TripTasks
import datetime
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional
import streamlit as st
from streamlit_chat import message
from langchain.agents import load_tools
from langchain_core.agents import AgentAction, AgentFinish
import os
from langchain_groq import ChatGroq
from crewai.process import Process


llm = ChatGroq(temperature=0.1, groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")
import threading

#sidebar elements
st.sidebar.title("Sidebar")
provider_name = st.sidebar.radio("Choose the provider:", ("OpenAI", "Groq"))

if provider_name == "OpenAI":
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
else:
    model_name = st.sidebar.radio("Choose a model:", ("LLAMA3-8B", "LLAMA3-70B"))





user_input = None
initiate_chat_task_created = False


# def initiate_chat(message):

#     global initiate_chat_task_created
#     # Indicate that the task has been created
#     initiate_chat_task_created = True
#     trip_crew = TripCrew()
#     result = trip_crew.run()
#     return result


# def callback(contents: str, user: str):
#     # location = text_input_src.value
#     # cities = text_input_dest.value
#     # date_range = date_ranges.value
#     # interests = text_area_input.value
#     # trip_crew = TripCrew(location, cities, date_range, interests)
#     # result = trip_crew.run()
#     # instance.send(result, user="assistant", respond=False)
#     global initiate_chat_task_created
#     global user_input

#     if not initiate_chat_task_created:
#         thread = threading.Thread(target=initiate_chat, args=(contents,))
#         thread.start()

#     else:
#         user_input = contents

avators = {"travels_representative_agent":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "city_selector_agent":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "local_expert_agent":"https://cdn-icons-png.freepik.com/512/9408/9408201.png",
            "travel_concierge_agent":"https://cdn-icons-png.flaticon.com/512/320/320336.png"}

class MyCustomHandler(BaseCallbackHandler):

    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    # def on_chain_start(
    #     self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    # ) -> None:
    #     """Print out that we are entering a chain."""
    #     st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
    #     st.chat_message("assistant").write(inputs['input'])
   
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(outputs['output'])
    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        for line in action.log.split("\n"):
                st.session_state.messages.append({"role": self.agent_name , "content": line})
                st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(line)
                print(line)
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        st.session_state.messages.append({"role": self.agent_name, "content": input_str})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(input_str)
        



class TripCrew:

    def __init__(self):
        pass
        
    # def __init__(self, origin, cities, date_range, interests):
    #     self.cities = cities
    #     self.origin = origin
    #     self.interests = interests
    #     self.date_range = date_range

    def run(self):

        humantool = load_tools(["human"], callbacks=[MyCustomHandler("human_tool")])

        agents = TripAgents(humantool)
        tasks = TripTasks()

        travels_representative_agent = agents.travels_representative()
        travels_representative_agent.callbacks = [MyCustomHandler("travels_representative_agent")]
        city_selector_agent = agents.city_selection_agent()
        city_selector_agent.callbacks = [MyCustomHandler("city_selector_agent")]
        local_expert_agent = agents.local_expert()
        local_expert_agent.callbacks = [MyCustomHandler("local_expert_agent")]
        travel_concierge_agent = agents.travel_concierge()
        travel_concierge_agent.callbacks = [MyCustomHandler("travel_concierge_agent")]

        # humantool = agents.humantool()
        # humantool.callbacks = [MyCustomHandler("human_tool")]
        

        collectDetails = tasks.collect_details(
            travels_representative_agent,
        )
        identify_task = tasks.identify_task(
            city_selector_agent,
            # self.origin,
            # self.cities,
            # self.interests,
            # self.date_range
        )
        identify_task.context = [collectDetails]
        gather_task = tasks.gather_task(
            local_expert_agent,
            # self.origin,
            # self.interests,
            # self.date_range
        )
        gather_task.context = [collectDetails, identify_task]
        plan_task = tasks.plan_task(
            travel_concierge_agent,
            # self.origin,
            # self.interests,
            # self.date_range
        )
        plan_task.context = [collectDetails, identify_task, gather_task]
        crew = Crew(
            agents=[
                travels_representative_agent, city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[collectDetails, identify_task, gather_task, plan_task],
            full_output=False,
            verbose=True,
            embedder= {
            "provider": "huggingface",
            "config":{
                "model": 'all-MiniLM-L6-v2'
            }}
        )

        result = crew.kickoff()
        # chat_interface.send("## Final Result\n"+str(result), user="assistant", respond=False)

        return result

st.title("💬 Paris Trip Planner") 

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": """Hi welcome to Paris Trip planner.
    Please provide trip details including your trip origin, planned date and interests"""}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
# Define a variable to enable/disable chat_input()
if 'is_chat_input_disabled' not in st.session_state:
    st.session_state.is_chat_input_disabled = False
if prompt := st.chat_input("your message", disabled=st.session_state.is_chat_input_disabled) or st.session_state.is_chat_input_disabled:

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    trip_crew = TripCrew()
    result = trip_crew.run()

    result = f"## Here is the Final Result \n\n {result}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
    st.session_state.is_chat_input_disabled = True
    

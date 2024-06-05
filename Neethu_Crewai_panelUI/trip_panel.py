from crewai import Crew
from trip_agents import TripAgents
from trip_tasks import TripTasks
import datetime
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional

import panel as pn 
pn.extension(design="bootstrap")

import threading

# Sidebar elements

today = datetime.datetime.now().date()
next_year = today.year + 1
jan_16_next_year = datetime.date(next_year, 1, 10)

title = pn.pane.Markdown("Enter Your Trip Details")
text_input_src = pn.widgets.TextInput(name='Where are you currently located', placeholder='Japan')
text_input_dest = pn.widgets.TextInput(name='City and Country you are interested to visit', placeholder='Paris')
# date_range = pn.widgets.DateRangePicker(
#     name='Date Range Picker', 
#     start=today, 
#     end=jan_16_next_year,
#     value = (today, today + datetime.timedelta(days=6)),
# )
date_ranges = pn.widgets.TextInput(name='Probable dates of visit', placeholder='July 2024')
text_area_input = pn.widgets.TextAreaInput(name='High level interests and hobbies or extra details about your trip?', 
                                           placeholder='2 adults who love swimming, dancing, hiking, and eating')
button = pn.widgets.Button(name='Submit', button_type='primary')


user_input = None
initiate_chat_task_created = False

def initiate_chat(message):

    global initiate_chat_task_created
    # Indicate that the task has been created
    initiate_chat_task_created = True

    location = text_input_src.value
    cities = text_input_dest.value
    date_range = date_ranges.value
    interests = text_area_input.value
    trip_crew = TripCrew(location, cities, date_range, interests)
    result = trip_crew.run()


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # location = text_input_src.value
    # cities = text_input_dest.value
    # date_range = date_ranges.value
    # interests = text_area_input.value
    # trip_crew = TripCrew(location, cities, date_range, interests)
    # result = trip_crew.run()
    # instance.send(result, user="assistant", respond=False)
    global initiate_chat_task_created
    global user_input

    if not initiate_chat_task_created:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()

    else:
        user_input = contents

avators = {"city_selector_agent":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "local_expert_agent":"https://cdn-icons-png.freepik.com/512/9408/9408201.png",
            "travel_concierge_agent":"https://cdn-icons-png.flaticon.com/512/320/320336.png"}

class MyCustomHandler(BaseCallbackHandler):

    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""

        chat_interface.send(inputs['input'], user="assistant", respond=False)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
    
        chat_interface.send(outputs['output'], user=self.agent_name, avatar=avators[self.agent_name], respond=False)



class TripCrew:

    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        city_selector_agent.callbacks = [MyCustomHandler("city_selector")]
        local_expert_agent = agents.local_expert()
        local_expert_agent.callbacks = [MyCustomHandler("local_expert")]
        travel_concierge_agent = agents.travel_concierge()
        travel_concierge_agent.callbacks = [MyCustomHandler("travel_concierge")]

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            full_output=False,
            verbose=True,
        )

        result = crew.kickoff()
        chat_interface.send("## Final Result\n"+str(result), user="assistant", respond=False)

        return result



# def on_Submit(event):
#     if not event:
#         return
#     trip_crew = TripCrew(location, cities, date_range, interests)
#     result = trip_crew.run()

# pn.bind(on_Submit, button, watch=True)

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(
    {"user": "System", "value": "Fill the details in sidebar and click send"},
    respond=False,
)
template = pn.template.BootstrapTemplate(
    sidebar=[title, text_input_src, text_input_dest, date_ranges, text_area_input], main=[chat_interface]
)
template.servable()
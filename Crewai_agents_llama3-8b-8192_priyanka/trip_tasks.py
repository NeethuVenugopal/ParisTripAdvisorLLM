##*********************************************************************
##** credits: https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner **
## added the expected_output in all the tasks.
##********************************************************************

from crewai import Task
from textwrap import dedent
from datetime import date

class TripTasks():

  def identify_task(self, agent, origin, cities, interests, range):
    return Task(description=dedent(f"""
        Analyze and select the best city for the trip based 
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple cities, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses. 
        
        Your final answer must be a detailed
        report on the chosen city, and everything you found out
        about it, including the actual flight costs, weather 
        forecast and attractions.
        {self.__tip_section()}

        Traveling from: {origin}
        City Options: {cities}
        Trip Date: {range}
        Traveler Interests: {interests}
        """),
        expected_output = dedent("""
        A detailed report that includes:
        1. The chosen city for the trip.
        2. Weather forecast for the trip dates.
        3. Upcoming cultural or seasonal events.
        4. Estimated travel costs, including flight and accommodation.
        5. Key attractions and activities related to traveler interests.
        """),
        agent=agent)

  def gather_task(self, agent, origin, interests, range):
    return Task(description=dedent(f"""
        As a local expert on this city you must compile an 
        in-depth guide for someone traveling there and wanting 
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what 
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forecasts, and
        high level costs.
        
        The final answer must be a comprehensive city guide, 
        rich in cultural insights and practical tips, 
        tailored to enhance the travel experience.
        {self.__tip_section()}

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
        """),
        expected_output = dedent("""
        A comprehensive city guide that includes:
        1. Key attractions: Detailed descriptions of must-visit places.
        2. Local customs: Insights into the local culture and etiquette.
        3. Special events: Information on upcoming events during the trip dates.
        4. Daily activity recommendations: Suggested itineraries for each day.
        5. Hidden gems: Less known but highly recommended spots.
        6. Cultural hotspots: Important cultural and historical sites.
        7. Weather forecasts: Predicted weather for the trip dates.
        8. High-level costs: Estimated costs for major activities and experiences.
        9. Practical tips: Advice on transportation, safety, and other practical matters.
        """),
        agent=agent)

  def plan_task(self, agent, origin, interests, range):
    return Task(description=dedent(f"""
        Expand this guide into a travel 
        itinerary with detailed per-week plans, including 
        weather forecasts, places to eat, packing suggestions, safety advice
        and a budget breakdown.
        
        You MUST suggest actual places to visit, actual hotels 
        to stay and actual restaurants to go to.
        
        This itinerary should cover all aspects of the trip, 
        from arrival to departure, integrating the city guide
        information with practical travel logistics.
        
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give it a reason why you picked
        # up each place, what make them special! {self.__tip_section()}

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
        """),
        expected_output = dedent("""
        A travel itinerary formatted as markdown that includes:
        1. Daily Schedule: Detailed plans for each day, including morning, afternoon, and evening activities.
        2. Hidden gems: Uncover unique festivals, hidden gems/treasures and hidden restaurants frequented by locals, quirky shops, and off-the-beaten-path experiences (e.g., attending a local sports event, visiting a community art space, exploring lesser-known historical sites.
        3. Weather Forecasts: Predicted weather conditions for each day.
        4. Safety tips: Which places to avoid and links to nearby police stations and embasssies.
        5. Dining Recommendations: Specific restaurants for each meal, with reasons for selection.
        6. Accommodation: Recommended hotels or places to stay, with reasons for selection.
        7. Packing Suggestions: Recommended clothing and items to pack based on weather and activities.
        8. Budget Breakdown: Estimated costs for accommodation, dining, activities, and other expenses.
        9. Eco-friendly travel options: Provide the cheaper and eco-friendly travel options.
        10. Special Reasons: Explanation of why each place is chosen and what makes it special.
        """),
        agent=agent)

  def __tip_section(self):
    return "If you do your BEST WORK, I'll tip you $100!"

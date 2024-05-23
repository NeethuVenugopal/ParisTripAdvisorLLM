import json
import os

import requests
from crewai import Agent, Task
from langchain.tools import tool
from unstructured.partition.html import partition_html
from crewai_tools import ScrapeWebsiteTool


class BrowserTools():

  @tool("Scrape website content")
  def scrape_and_summarize_website(website):
    """Useful to scrape and summarize a website content"""
    # To enable scrapping any website it finds during it's execution
    tool = ScrapeWebsiteTool()

    # Initialize the tool with the website URL, so the agent can only scrap the content of the specified website
    tool = ScrapeWebsiteTool(website_url=website)

    # Extract the text from the site
    text = tool.run()
    
    # url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
    # payload = json.dumps({"url": website})
    # headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
    # response = requests.request("POST", url, headers=headers, data=payload)
    elements = partition_html(text=text)
    content = "\n\n".join([str(el) for el in elements])
    content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
    summaries = []
    for chunk in content:
      agent = Agent(
          role='Principal Researcher',
          goal=
          'Do amazing researches and summaries based on the content you are working with',
          backstory=
          "You're a Principal Researcher at a big company and you need to do a research about a given topic.",
          allow_delegation=False)
      task = Task(
          agent=agent,
          description=
          f'Analyze and summarize the content bellow, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}',
          expected_output = 'a summary of the content',
      ) 
      summary = task.execute()
      summaries.append(summary)
    return "\n\n".join(summaries)

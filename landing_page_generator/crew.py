from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.browser_tools import BrowserTools
from tools.file_tools import FileTools
from tools.search_tools import SearchTools
from tools.template_tools import TemplateTools
from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import FileManagementToolkit

import os

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL_NAME"],
    base_url=os.environ["OPENAI_API_BASE"]
)

@CrewBase
class LandingPageCrew:
  agents_config = "config/agents.yaml"  
  tasks_config = "config/tasks.yaml"

  def __init__(self):
    self._toolkit = FileManagementToolkit(
      root_dir='workdir',
      selected_tools=["read_file", "list_directory"]
    )    

  @agent
  def idea_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config["idea_analyst"],
      tools=[
        SearchTools.search_internet,
        BrowserTools.scrape_and_summarize_website,
      ],
      llm = llm      
    )   

  @agent
  def communications_strategist(self) -> Agent:
    return Agent(
      config=self.agents_config["communications_strategist"],
      tools=[
        SearchTools.search_internet,
        BrowserTools.scrape_and_summarize_website,
      ],
      llm = llm      
    )
  
  @agent
  def react_developer(self) -> Agent:
    return Agent(
      config=self.agents_config["react_developer"],
      tools=[
          SearchTools.search_internet,
          BrowserTools.scrape_and_summarize_website,
          TemplateTools.learn_landing_page_options,
          TemplateTools.copy_landing_page_template_to_project_folder,
          FileTools.write_file
      ] + self._toolkit.get_tools(),
      llm = llm      
    )
  
  @agent
  def content_editor(self) -> Agent:
    return Agent(
      config=self.agents_config["content_editor"],
      tools=[
          SearchTools.search_internet,
          BrowserTools.scrape_and_summarize_website,
      ],
      llm = llm      
    )
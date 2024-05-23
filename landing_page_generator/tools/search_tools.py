from langchain_community.tools import DuckDuckGoSearchRun,tool

search_tool = DuckDuckGoSearchRun()

class SearchTools():

  @tool('Search the internet')
  def search_internet(query: str):
    """Useful to search the internet
    about a a given topic and return relevant results"""
    return search_tool.run(query)
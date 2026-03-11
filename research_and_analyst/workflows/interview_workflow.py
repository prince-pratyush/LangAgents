from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import get_buffer_string

from research_and_analyst.schemas.models import InterviewState, SearchQuery
from research_and_analyst.prompt_library.prompt_locator import (
    ANALYST_ASK_QUESTIONS,
    GENERATE_SEARCH_QUERY,
    GENERATE_ANSWERS,
    WRITE_SECTION,
)
from research_and_analyst.logger import GLOBAL_LOGGER
from research_and_analyst.exception.custom_exception import ResearchAnalystException


class InterviewGraphBuilder:
    """
    Constructs and compiles the LangGraph Interview workflow.

    The graph runs one full analyst-expert interview and produces a
    written report section from the gathered context.

    Flow
    ----
    ask_question → search_web → generate_answer → save_interview → write_section
    """

    def __init__(self, llm, tavily_search):
        self.llm = llm
        self.tavily_search = tavily_search
        self.memory = MemorySaver()
        self.logger = GLOBAL_LOGGER.bind(module="InterviewGraphBuilder")

    # ------------------------------------------------------------------
    # Node 1 — Analyst generates a question
    # ------------------------------------------------------------------
    def _generate_question(self, state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]

        try:
            self.logger.info("Generating analyst question", analyst=analyst.name)
            system_prompt = ANALYST_ASK_QUESTIONS.render(goals=analyst.persona)
            question = self.llm.invoke(
                [SystemMessage(content=system_prompt)] + messages
            )
            self.logger.info(
                "Question generated", preview=question.content[:120]
            )
            return {"messages": [question]}

        except Exception as e:
            self.logger.error("Error generating analyst question", error=str(e))
            raise ResearchAnalystException(
                "Failed to generate analyst question", cause=e
            ) from e

    # ------------------------------------------------------------------
    # Node 2 — Web search
    # ------------------------------------------------------------------
    def _search_web(self, state: InterviewState):
        try:
            self.logger.info("Generating search query from conversation")
            structured_llm = self.llm.with_structured_output(SearchQuery)
            search_prompt = GENERATE_SEARCH_QUERY.render()
            search_query = structured_llm.invoke(
                [SystemMessage(content=search_prompt)] + state["messages"]
            )

            self.logger.info(
                "Performing web search", query=search_query.search_query
            )
            search_docs = self.tavily_search.invoke(search_query.search_query)

            if not search_docs:
                self.logger.warning("No search results returned")
                return {"context": ["[No search results found.]"]}

            formatted = "\n\n---\n\n".join(
                f'<Document href="{doc.get("url", "#")}"/>\n'
                f'{doc.get("content", "")}\n</Document>'
                for doc in search_docs
            )
            self.logger.info("Web search completed", results=len(search_docs))
            return {"context": [formatted]}

        except Exception as e:
            self.logger.error("Error during web search", error=str(e))
            raise ResearchAnalystException(
                "Failed during web search", cause=e
            ) from e

    # ------------------------------------------------------------------
    # Node 3 — Expert generates an answer
    # ------------------------------------------------------------------
    def _generate_answer(self, state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]
        context = state.get("context", ["[No context available.]"])

        try:
            self.logger.info("Generating expert answer", analyst=analyst.name)
            system_prompt = GENERATE_ANSWERS.render(
                goals=analyst.persona, context=context
            )
            answer = self.llm.invoke(
                [SystemMessage(content=system_prompt)] + messages
            )
            answer.name = "expert"
            self.logger.info(
                "Expert answer generated", preview=answer.content[:120]
            )
            return {"messages": [answer]}

        except Exception as e:
            self.logger.error("Error generating expert answer", error=str(e))
            raise ResearchAnalystException(
                "Failed to generate expert answer", cause=e
            ) from e

    # ------------------------------------------------------------------
    # Node 4 — Save interview transcript
    # ------------------------------------------------------------------
    def _save_interview(self, state: InterviewState):
        try:
            messages = state["messages"]
            interview = get_buffer_string(messages)
            self.logger.info(
                "Interview transcript saved", message_count=len(messages)
            )
            return {"interview": interview}

        except Exception as e:
            self.logger.error("Error saving interview transcript", error=str(e))
            raise ResearchAnalystException(
                "Failed to save interview transcript", cause=e
            ) from e

    # ------------------------------------------------------------------
    # Node 5 — Write a report section from the interview
    # ------------------------------------------------------------------
    def _write_section(self, state: InterviewState):
        context = state.get("context", ["[No context available.]"])
        analyst = state["analyst"]

        try:
            self.logger.info(
                "Writing report section", analyst=analyst.name
            )
            system_prompt = WRITE_SECTION.render(focus=analyst.description)
            section = self.llm.invoke(
                [SystemMessage(content=system_prompt)]
                + [
                    HumanMessage(
                        content=f"Use this source to write your section: {context}"
                    )
                ]
            )
            self.logger.info(
                "Section written", length=len(section.content)
            )
            return {"sections": [section.content]}

        except Exception as e:
            self.logger.error("Error writing report section", error=str(e))
            raise ResearchAnalystException(
                "Failed to generate report section", cause=e
            ) from e

    # ------------------------------------------------------------------
    # Build and compile the graph
    #
    # Flow
    # ----
    # START
    #   │
    #   ▼
    # ask_question      ← analyst generates first question using their persona
    #   │
    #   ▼
    # search_web        ← LLM converts the question into a search query → Tavily
    #   │
    #   ▼
    # generate_answer   ← expert answers using the retrieved web context
    #   │
    #   ▼
    # save_interview    ← full conversation buffered into a transcript string
    #   │
    #   ▼
    # write_section     ← LLM writes a structured markdown section from context
    #   │
    #   ▼
    # END
    # ------------------------------------------------------------------
    def build(self):
        try:
            self.logger.info("Building interview graph")
            builder = StateGraph(InterviewState)

            builder.add_node("ask_question",    self._generate_question)
            builder.add_node("search_web",      self._search_web)
            builder.add_node("generate_answer", self._generate_answer)
            builder.add_node("save_interview",  self._save_interview)
            builder.add_node("write_section",   self._write_section)

            builder.add_edge(START,             "ask_question")
            builder.add_edge("ask_question",    "search_web")
            builder.add_edge("search_web",      "generate_answer")
            builder.add_edge("generate_answer", "save_interview")
            builder.add_edge("save_interview",  "write_section")
            builder.add_edge("write_section",   END)

            graph = builder.compile(checkpointer=self.memory)
            self.logger.info("Interview graph compiled successfully")
            return graph

        except Exception as e:
            self.logger.error("Error building interview graph", error=str(e))
            raise ResearchAnalystException(
                "Failed to build interview graph", cause=e
            ) from e


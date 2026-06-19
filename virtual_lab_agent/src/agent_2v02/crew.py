import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from agent_2v02.models import VirtualLabReport
from agent_2v02.tools.custom_tool import ManufacturingCostCalculatorTool

# LLM is read from the MODEL env var (set in .env), defaulting to OpenAI's
# gpt-4o. Accepts any LiteLLM model string, e.g. "gpt-4o", "openai/gpt-4o",
# or "anthropic/claude-opus-4-8".
MODEL = os.getenv("MODEL", "gpt-4o")
LLM_INSTANCE = LLM(model=MODEL)


@CrewBase
class Agent2V02():
    """Virtual Lab Agent (Agent 2).

    Consumes the Material Theory Agent's proposal, simulates the synthesis
    reaction + yield, estimates manufacturing cost, and emits a structured
    VirtualLabReport for the Procurement Agent (Agent 3).
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def process_simulation_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['process_simulation_scientist'],  # type: ignore[index]
            llm=LLM_INSTANCE,
            verbose=True,
        )

    @agent
    def manufacturing_cost_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['manufacturing_cost_analyst'],  # type: ignore[index]
            llm=LLM_INSTANCE,
            tools=[ManufacturingCostCalculatorTool()],
            verbose=True,
        )

    @task
    def reaction_simulation_task(self) -> Task:
        return Task(
            config=self.tasks_config['reaction_simulation_task'],  # type: ignore[index]
        )

    @task
    def manufacturing_cost_task(self) -> Task:
        return Task(
            config=self.tasks_config['manufacturing_cost_task'],  # type: ignore[index]
            context=[self.reaction_simulation_task()],
            output_pydantic=VirtualLabReport,
            output_file='output/virtual_lab_report.json',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Virtual Lab crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

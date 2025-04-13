"""Set up crew ai."""

from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class DevCrew:
    """Developer crew."""

    agents_config: dict[str, Any]
    tasks_config: dict[str, Any]

    llm = 'vertex_ai/gemini-2.0-flash-001'

    @agent
    def senior_engineer_agent(self) -> Agent:
        """Senior engineer agent."""
        return Agent(config=self.agents_config.get('senior_engineer_agent'),
                     allow_delegation=False,
                     verbose=True,
                     llm=self.llm)

    @agent
    def chief_qa_engineer_agent(self) -> Agent:
        """Chief QA engineer agent."""
        return Agent(config=self.agents_config.get('chief_qa_engineer_agent'),
                     allow_delegation=True,
                     verbose=True,
                     llm=self.llm)

    @task
    def code_task(self) -> Task:
        """Code task."""
        return Task(config=self.tasks_config.get('code_task'),
                    agent=self.senior_engineer_agent())

    @task
    def evaluate_task(self) -> Task:
        """Evaluate task."""
        return Task(config=self.tasks_config.get('evaluate_task'),
                    agent=self.chief_qa_engineer_agent())

    @crew
    def crew(self) -> Crew:
        """Create the Dev Crew."""
        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)

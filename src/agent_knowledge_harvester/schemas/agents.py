from pydantic import BaseModel, Field

from agent_knowledge_harvester.config import AnalysisStage


class AgentRoleSpec(BaseModel):
    role_id: str
    name: str
    mission: str
    model_stage: AnalysisStage
    system_prompt: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    quality_gates: list[str] = Field(default_factory=list)
    handoff_notes: list[str] = Field(default_factory=list)


class AgentHandoffSpec(BaseModel):
    from_role: str
    to_role: str
    artifact: str
    contract: str


class MultiAgentBlueprint(BaseModel):
    name: str = "Frontier Agent Knowledge Harvesting Team"
    purpose: str
    roles: list[AgentRoleSpec] = Field(default_factory=list)
    handoffs: list[AgentHandoffSpec] = Field(default_factory=list)


class TeamStageTrace(BaseModel):
    role_id: str
    status: str = "pending"
    model_stage: AnalysisStage | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TeamRunTrace(BaseModel):
    run_name: str
    blueprint_name: str
    output_dir: str
    stages: list[TeamStageTrace] = Field(default_factory=list)

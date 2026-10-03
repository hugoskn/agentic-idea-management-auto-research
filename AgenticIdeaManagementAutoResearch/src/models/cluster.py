from pydantic import BaseModel


class IdeaAssignment(BaseModel):
    idea_id: str
    reason: str


class Cluster(BaseModel):
    id: str
    name: str
    description: str
    fundamentally_different: bool
    members: list[IdeaAssignment]


class ClusteringResult(BaseModel):
    solution_space_summary: str
    clusters: list[Cluster]

    def cluster_of(self, idea_id: str) -> Cluster | None:
        return next((c for c in self.clusters if any(m.idea_id == idea_id for m in c.members)), None)

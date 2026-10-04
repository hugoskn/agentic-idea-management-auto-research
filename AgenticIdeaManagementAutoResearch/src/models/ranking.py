from pydantic import BaseModel, Field


class ClusterRank(BaseModel):
    cluster_id: str
    rank: int = Field(ge=1, description="Dense rank, 1 = most promising; ties allowed.")


class ClusterRanking(BaseModel):
    cluster_ranks: list[ClusterRank]
    rationale: str = Field(description="1-3 sentences citing specific clusters.")


class IdeaRank(BaseModel):
    idea_id: str
    rank: int = Field(ge=1, description="Dense rank within the cluster, 1 = most promising; ties allowed.")


class IdeaRanking(BaseModel):
    idea_ranks: list[IdeaRank]
    rationale: str = Field(description="1-3 sentences.")


class RankingResult(BaseModel):
    clusters: ClusterRanking
    ideas: dict[str, IdeaRanking] = Field(default_factory=dict, description="Within-cluster rankings of untested ideas, keyed by cluster id.")

    def cluster_rank(self, cluster_id: str) -> int | None:
        return next((r.rank for r in self.clusters.cluster_ranks if r.cluster_id == cluster_id), None)

    def idea_rank(self, idea_id: str) -> str | None:
        for ranking in self.ideas.values():
            for r in ranking.idea_ranks:
                if r.idea_id == idea_id:
                    return f"{r.rank}/{len(ranking.idea_ranks)}"
        return None

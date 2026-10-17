from pydantic import BaseModel


class ClusterCandidate(BaseModel):
    algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float


class ClusteringResult(BaseModel):
    candidates: list[ClusterCandidate]
    best: ClusterCandidate
    null_silhouette: float
    has_structure: bool
    n_rows_used: int
    dropped_columns: list[str]
    row_index: list[int]
    labels: list[int]
    n_clusters: int
    refinements: list[str] = []
    cat_weight: float = 1.0

class SegmentFeature(BaseModel):
    feature: str
    kind: str
    effect: float
    detail: str


class SegmentProfile(BaseModel):
    segment_id: int
    size: int
    share: float
    features: list[SegmentFeature]
    summary: str

class SegmentTitle(BaseModel):
    """Sortie demandée au LLM."""
    title: str
    description: str


class NamedSegment(BaseModel):
    profile: SegmentProfile
    title: str
    description: str
    text_source: str  # "llm" ou "default"


class ClusteringReport(BaseModel):
    result: ClusteringResult
    verdict: str
    segments: list[NamedSegment]
    corrections: list[str]
    plot_path: str | None = None
    run_id: str | None = None

class SegmentTitle(BaseModel):
    """Sortie demandée au LLM : un titre seulement."""
    title: str
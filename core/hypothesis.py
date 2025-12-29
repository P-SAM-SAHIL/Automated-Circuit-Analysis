from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class CircuitHypothesis:
    """Stores circuit-level hypothesis (not just single head)"""
    target_layer: int
    target_head: int
    circuit_path: List[Tuple[int, int]]
    mechanism: str
    predicted_behavior: str
    confidence: float = 0.0
    validation_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.validation_scores is None:
            self.validation_scores = {}

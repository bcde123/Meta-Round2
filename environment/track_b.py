from typing import List, Dict, Set
from .rule_engine import RuleEngine, EpisodeState, ViolationReport

class ComplianceChecker:
    def __init__(self, standards_path: str):
        self.engine = RuleEngine(standards_path)
        self.state = EpisodeState()
        
    def reset(self, files: Dict[str, str], rules_active: List[int]) -> ViolationReport:
        self.state = EpisodeState()
        # Deterministic initial obligations are important for RL: the same
        # candidate refactor should receive the same compliance reward when
        # scored repeatedly. Episode variety already comes from the generator's
        # code corruptions and active-rule budget.
        initial_triggers = list(rules_active[:min(len(rules_active), 15)])
        for rid in initial_triggers:
            self.state.triggered_rules.add(rid)
            self.state.outstanding_obligations.add(rid)
            
        return ViolationReport(
            newly_triggered=set(initial_triggers),
            newly_resolved=set(),
            still_outstanding=set(initial_triggers),
            conflict_flags=[]
        )
        
    def step(self, action: dict, diff: str) -> ViolationReport:
        self.state.steps_taken += 1
        return self.engine.process_action(action, diff, self.state)
        
    def get_score(self) -> float:
        return self.engine.compute_compliance_score(self.state)
        
    def get_outstanding(self) -> List[int]:
        return list(self.state.outstanding_obligations)

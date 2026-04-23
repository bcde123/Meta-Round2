import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

@dataclass
class Rule:
    id: int
    category: str
    description: str
    trigger_action: str
    trigger_file_pattern: str
    trigger_code_pattern: str
    check_type: str
    check_value: str
    spawns: List[int] = field(default_factory=list)

@dataclass
class ViolationReport:
    newly_triggered: Set[int]
    newly_resolved: Set[int]
    still_outstanding: Set[int]
    conflict_flags: List[str]

@dataclass
class EpisodeState:
    outstanding_obligations: set = field(default_factory=set)
    triggered_rules: set = field(default_factory=set)
    resolved_rules: set = field(default_factory=set)
    conflict_flags: list = field(default_factory=list)
    steps_taken: int = 0

class RuleEngine:
    def __init__(self, standards_path: str):
        self.rules = {}
        self.trigger_graph = {}
        self._parse_standards(standards_path)
        self._build_trigger_graph()
        self._apply_hardcoded_spawns()

    def _parse_standards(self, path: str):
        with open(path, "r") as f:
            content = f.read()
        
        rule_blocks = re.split(r'### Rule (\d+)', content)[1:]
        for i in range(0, len(rule_blocks), 2):
            rule_id = int(rule_blocks[i])
            text = rule_blocks[i+1]
            
            category = re.search(r'\*\*Category\*\*: (.*)', text).group(1)
            description = re.search(r'\*\*Description\*\*: (.*)', text).group(1)
            trigger = re.search(r'\*\*Trigger condition\*\*: (.*)', text).group(1)
            check = re.search(r'\*\*Check condition\*\*: (.*)', text).group(1)
            
            # Simple heuristic mapping for this mock environment
            trigger_action = "edit_file" if "edit" in trigger else "any"
            
            self.rules[rule_id] = Rule(
                id=rule_id,
                category=category,
                description=description,
                trigger_action=trigger_action,
                trigger_file_pattern=".*",
                trigger_code_pattern=".*",
                check_type="heuristic",
                check_value=check
            )

    def _apply_hardcoded_spawns(self):
        spawns_map = {
            83: [134],
            134: [29],
            29: [12, 91, 118],
            55: [56, 103],
            44: [91, 17]
        }
        for rid, targets in spawns_map.items():
            if rid in self.rules:
                self.rules[rid].spawns.extend(targets)

    def _build_trigger_graph(self):
        for r in self.rules.values():
            key = (r.trigger_action, r.trigger_file_pattern, r.trigger_code_pattern)
            if key not in self.trigger_graph:
                self.trigger_graph[key] = []
            self.trigger_graph[key].append(r.id)

    def _check_rule_resolved(self, rule: Rule, diff: str) -> bool:
        """Check if a rule's obligation has been addressed by the current edit.
        
        Uses heuristic keyword matching against the diff to determine
        whether the edit likely addresses the rule's check condition.
        """
        check_lower = rule.check_value.lower()
        diff_lower = diff.lower()
        
        # Category-based resolution heuristics
        category_lower = rule.category.lower()
        
        if "naming" in category_lower or "variable" in category_lower:
            # Naming rules resolved if meaningful variable names appear in the diff
            if any(kw in diff_lower for kw in ["renamed", "refactor", "edited"]):
                return True
                
        if "docstring" in category_lower or "documentation" in category_lower:
            if '"""' in diff or "'''" in diff or "docstring" in diff_lower:
                return True
                
        if "type" in category_lower and "hint" in category_lower:
            if any(kw in diff for kw in ["-> ", ": str", ": int", ": float", ": bool", ": List", ": Dict", ": Optional"]):
                return True
                
        if "complexity" in category_lower:
            if "edited" in diff_lower:
                return True
                
        if "import" in category_lower:
            if "import" in diff_lower:
                return True
                
        if "security" in category_lower or "secret" in category_lower:
            if any(kw in diff_lower for kw in ["os.getenv", "environ", "config", "secret"]):
                return True
                
        if "test" in category_lower:
            if any(kw in diff_lower for kw in ["test_", "assert", "pytest"]):
                return True
                
        if "size" in category_lower or "length" in category_lower or "module" in category_lower:
            if "edited" in diff_lower:
                return True
        
        # Generic: if the edit mentions the file was changed, resolve with some probability
        # based on how many keywords from the check condition appear in the diff
        check_keywords = [w for w in check_lower.split() if len(w) > 3]
        if check_keywords:
            matches = sum(1 for kw in check_keywords if kw in diff_lower)
            match_ratio = matches / len(check_keywords)
            if match_ratio >= 0.3:
                return True
        
        # Default: editing a file resolves ~40% of generic rules
        # This provides a reasonable signal without being too generous
        if "edited" in diff_lower:
            import random
            return random.random() < 0.4
            
        return False

    def process_action(self, action: dict, diff: str, state: EpisodeState) -> ViolationReport:
        report = ViolationReport(set(), set(), set(), [])
        action_type = action.get("tool", "unknown")
        
        # Trigger evaluation
        for key, r_ids in self.trigger_graph.items():
            ta, tf, tc = key
            if ta == "any" or ta == action_type:
                for rid in r_ids:
                    if rid not in state.triggered_rules:
                        state.triggered_rules.add(rid)
                        state.outstanding_obligations.add(rid)
                        report.newly_triggered.add(rid)

        # Spawns evaluation
        new_spawns = []
        for rid in list(state.outstanding_obligations):
            if rid in self.rules:
                r = self.rules[rid]
                for spawn_id in r.spawns:
                    if spawn_id not in state.triggered_rules:
                        state.triggered_rules.add(spawn_id)
                        state.outstanding_obligations.add(spawn_id)
                        new_spawns.append(spawn_id)
                        report.newly_triggered.add(spawn_id)
        
        # Resolution checking — evaluate each outstanding obligation
        resolved_this_step = set()
        for rid in list(state.outstanding_obligations):
            if rid in self.rules:
                rule = self.rules[rid]
                if self._check_rule_resolved(rule, diff):
                    resolved_this_step.add(rid)
        
        # Apply resolutions
        for rid in resolved_this_step:
            state.outstanding_obligations.discard(rid)
            state.resolved_rules.add(rid)
            report.newly_resolved.add(rid)
                
        # Detect contradictions (e.g. Rule 142 vs Rule 143)
        if 142 in state.outstanding_obligations and 143 in state.outstanding_obligations:
            conflict_msg = "Contradiction detected between Rule 142 and 143"
            if conflict_msg not in state.conflict_flags:
                state.conflict_flags.append(conflict_msg)
                report.conflict_flags.append(conflict_msg)
                
        report.still_outstanding = set(state.outstanding_obligations)
        return report

    def compute_compliance_score(self, state: EpisodeState) -> float:
        if not state.triggered_rules:
            return 1.0
        compliant = len(state.triggered_rules) - len(state.outstanding_obligations)
        return compliant / len(state.triggered_rules)

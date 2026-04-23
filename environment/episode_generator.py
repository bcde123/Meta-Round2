import os
import ast
import random
import uuid
import astor
from typing import Dict, List, Tuple

class CurriculumManager:
    def __init__(self):
        self.history = []
        self.level = 1
        
    def add_result(self, reward: float):
        self.history.append(reward)
        if len(self.history) > 150:
            self.history.pop(0)
            
        if self._should_escalate():
            if self.level < 4:
                self.level += 1
                self.history = []
                
    def _should_escalate(self) -> bool:
        if len(self.history) < 150:
            return False
            
        windows = [
            self.history[0:50],
            self.history[50:100],
            self.history[100:150]
        ]
        
        for w in windows:
            if sum(w) / len(w) <= 0.7:
                return False
        return True

class EpisodeGenerator:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.curriculum = CurriculumManager()
        
    def _load_base_files(self) -> Dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(self.base_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    path = os.path.join(root, filename)
                    with open(path, "r") as f:
                        rel_path = os.path.relpath(path, self.base_dir)
                        files[rel_path] = f.read()
        return files

    def _inject_circular_imports(self, files: Dict[str, str]):
        if "api.py" in files and "utils.py" in files:
            files["api.py"] = "from .utils import *\n" + files["api.py"]
            files["utils.py"] = "from .api import *\n" + files["utils.py"]

    def _rename_to_cryptic(self, files: Dict[str, str]):
        class RenameTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if random.random() < 0.4:
                    node.name = random.choice(["x1", "tmp2", "_zzz", "do_stuff"])
                self.generic_visit(node)
                return node
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store) and random.random() < 0.3:
                    node.id = random.choice(["v1", "temp", "val"])
                return node

        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                tree = RenameTransformer().visit(tree)
                ast.fix_missing_locations(tree)
                files[filename] = astor.to_source(tree)
            except SyntaxError:
                pass

    def _remove_type_hints(self, files: Dict[str, str]):
        class TypeHintRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.returns = None
                self.generic_visit(node)
                return node
            def visit_arg(self, node):
                node.annotation = None
                return node
            def visit_AnnAssign(self, node):
                val = node.value if node.value is not None else ast.Constant(value=None)
                return ast.Assign(targets=[node.target], value=val)

        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                tree = TypeHintRemover().visit(tree)
                ast.fix_missing_locations(tree)
                files[filename] = astor.to_source(tree)
            except SyntaxError:
                pass

    def _bloat_with_dead_code(self, files: Dict[str, str]):
        dead_code = "\n".join([
            "    # This is dead code",
            "    _tmp_x = 42",
            "    for _i in range(10):",
            "        _tmp_x += _i"
        ])
        for filename in list(files.keys()):
            if random.random() < 0.5:
                lines = files[filename].split("\n")
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if line.strip().startswith("def ") and random.random() < 0.3:
                        new_lines.extend(dead_code.split("\n"))
                while len(new_lines) < 310:
                    idx = random.randint(0, max(1, len(new_lines)-10))
                    block = new_lines[idx:idx+10]
                    new_lines.extend(block)
                files[filename] = "\n".join(new_lines)

    def _hardcode_secrets(self, files: Dict[str, str]):
        for filename, content in files.items():
            modified = content.replace('os.getenv("API_KEY", "default_secret_key")', '"password123"')
            modified = modified.replace('os.getenv("DB_URL", "sqlite:///./tasks.db")', '"root:supersecret@localhost:5432/db"')
            files[filename] = modified

    def _break_import_paths(self, files: Dict[str, str]):
        for filename, content in files.items():
            if "api.py" in filename:
                files[filename] = content.replace("from .utils import", "from ..utils import")
            if "main.py" in filename:
                files[filename] = content.replace("from .api import", "from api_broken import")

    def _remove_docstrings(self, files: Dict[str, str]):
        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        if ast.get_docstring(node):
                            node.body.pop(0)
                files[filename] = astor.to_source(tree)
            except SyntaxError:
                pass

    def _add_god_function(self, files: Dict[str, str]):
        if "utils.py" in files:
            god_function = "def god_function(data, date_str, current_date):\n"
            god_function += "    # MERGED FUNCTION\n"
            god_function += "    try:\n"
            god_function += "        dt = datetime.fromisoformat(date_str)\n"
            god_function += "    except ValueError as e:\n"
            god_function += "        logger.error(f'Error parsing date {date_str}: {e}')\n"
            god_function += "        raise ValueError('Invalid date format. Expected ISO 8601.')\n"
            god_function += "    delta = dt - current_date\n"
            god_function += "    if delta.days < 0:\n"
            god_function += "        prio = 'OVERDUE'\n"
            god_function += "    else:\n"
            god_function += "        prio = 'NORMAL'\n"
            god_function += "    if not data.get('title') or len(data.get('title', '')) > 100:\n"
            god_function += "        valid = False\n"
            god_function += "    else:\n"
            god_function += "        valid = True\n"
            god_function += "    return dt, prio, valid\n"
            
            files["utils.py"] += "\n\n" + god_function

    def generate(self) -> dict:
        files = self._load_base_files()
        
        corruptions = [
            self._inject_circular_imports,
            self._rename_to_cryptic,
            self._remove_type_hints,
            self._bloat_with_dead_code,
            self._hardcode_secrets,
            self._break_import_paths,
            self._remove_docstrings,
            self._add_god_function
        ]
        
        num_corruptions = random.randint(4, 8)
        chosen_corruptions = random.sample(corruptions, num_corruptions)
        
        for corr in chosen_corruptions:
            corr(files)
            
        num_rules = min(150, 20 + (self.curriculum.level - 1) * 40)
        rules_active = list(range(1, num_rules + 1))
        
        return {
            "files": files,
            "rules_active": rules_active,
            "curriculum_level": self.curriculum.level,
            "episode_id": str(uuid.uuid4())
        }

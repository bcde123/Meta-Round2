import os
import pwd
import getpass

# Hard patch pwd.getpwuid to never raise KeyError for the current user
def dummy_getpwuid(uid):
    return ('huggingface', 'x', uid, 1000, 'HuggingFace user', '/home/huggingface', '/bin/sh')

pwd.getpwuid = dummy_getpwuid

# Hard patch getpass.getuser to immediately return the dummy user
def dummy_getuser():
    return "huggingface"

getpass.getuser = dummy_getuser

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["LOGNAME"] = "huggingface"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor"

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import torch

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker
from environment.track_c import GreenCodeEvaluator
from environment.co2_calculator import generate_dashboard_data
from environment.rubrics import build_green_rubric, CodeAction, CodeObservation

app = FastAPI(
    title="Green-Code Optimizer",
    description="OpenEnv RL environment that trains a code agent to refactor Python for energy efficiency (CPU + memory) — with a CO₂-savings dashboard.",
    version="1.0.0",
)

# CORS for HuggingFace Spaces / external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResetRequest(BaseModel):
    curriculum_level: Optional[int] = None

class ActionRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]

# ── Anti-cheat constants (shared with training/train_grpo.py) ─────────────────
PROTECTED_FILES = frozenset({"conftest.py", "test_main.py", "test_api.py",
                             "test_utils.py", "pytest.ini", "setup.cfg"})
EFFICIENCY_PENALTY_PER_STEP = 0.01
HACK_PENALTY = -1.0


class EpisodeContext:
    def __init__(self, base_dir: str):
        self.generator = EpisodeGenerator(base_dir)
        self.track_a = CodeQualityEvaluator()
        self.track_b = ComplianceChecker(os.path.join(os.path.dirname(base_dir), "ENGINEERING_STANDARDS.md"))
        self.files = {}
        self.orig_files = {}  # immutable baseline for comparison
        self.active_rules = []
        self.standards_path = os.path.join(os.path.dirname(base_dir), "ENGINEERING_STANDARDS.md")
        self.steps_remaining = 70
        self.steps_taken = 0
        self.hack_detected = False
        self.hack_reason = ""
        
active_episodes: Dict[str, EpisodeContext] = {}
BASE_DIR = os.path.join(os.path.dirname(__file__), "environment/base_codebase")

@app.post("/reset")
async def reset_env(req: Optional[ResetRequest] = None):
    ctx = EpisodeContext(BASE_DIR)
    if req and req.curriculum_level is not None:
        ctx.generator.curriculum.level = req.curriculum_level
        
    episode_data = ctx.generator.generate()
    episode_id = episode_data["episode_id"]
    ctx.files = episode_data["files"]
    ctx.orig_files = {k: v for k, v in ctx.files.items()}  # snapshot
    ctx.active_rules = episode_data["rules_active"]
    
    ctx.track_a.evaluate(ctx.files)
    report = ctx.track_b.reset(ctx.files, episode_data["rules_active"])
    
    active_episodes[episode_id] = ctx
    
    observation = {
        "files": ctx.files,
        "active_rules_count": len(episode_data["rules_active"]),
        "steps_remaining": ctx.steps_remaining,
        "violation_report": {
            "newly_triggered": list(report.newly_triggered),
            "newly_resolved": list(report.newly_resolved),
            "still_outstanding": list(report.still_outstanding),
            "conflict_flags": report.conflict_flags
        },
        "curriculum_level": episode_data["curriculum_level"]
    }
    
    return {
        "episode_id": episode_id,
        "observation": observation,
        "info": {}
    }

@app.post("/step")
async def step_env(req: ActionRequest):
    episode_id = req.episode_id
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
        
    ctx = active_episodes[episode_id]
    action = req.action
    tool = action.get("tool")
    args = action.get("args", {})
    
    diff = ""
    if tool == "edit_file":
        filename = args.get("filename")
        content = args.get("content")
        if filename and content:
            # ── Protected-file guard ──────────────────────────────────────────
            basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
            if basename in PROTECTED_FILES:
                ctx.hack_detected = True
                ctx.hack_reason = f"Edited protected file: {filename}"
                # Terminate episode immediately
                reward = HACK_PENALTY
                info = {"hack_detected": True, "hack_reason": ctx.hack_reason,
                        "final_reward": reward}
                del active_episodes[episode_id]
                return {
                    "observation": {"files": ctx.files, "steps_remaining": 0,
                                    "violation_report": {"newly_triggered": [],
                                                         "newly_resolved": [],
                                                         "still_outstanding": [],
                                                         "conflict_flags": []}},
                    "reward": reward, "done": True, "info": info,
                }

            ctx.files[filename] = content
            diff = f"Edited {filename}"
            ctx.track_a.evaluate(ctx.files)
            report = ctx.track_b.step(action, diff)
            
    elif tool == "run_tests":
        ctx.track_a.evaluate(ctx.files)
        report = ctx.track_b.step(action, "Ran tests")
        
    elif tool == "check_compliance":
        report = ctx.track_b.step(action, "Checked compliance")
        
    elif tool == "read_file":
        report = ctx.track_b.step(action, "Read file")
        
    else:
        report = ctx.track_b.step(action, "Unknown/Finish")

    ctx.steps_remaining -= 1
    ctx.steps_taken += 1
    done = ctx.steps_remaining <= 0 or tool == "finish"
    
    observation = {
        "files": ctx.files,
        "steps_remaining": ctx.steps_remaining,
        "violation_report": {
            "newly_triggered": list(report.newly_triggered),
            "newly_resolved": list(report.newly_resolved),
            "still_outstanding": list(report.still_outstanding),
            "conflict_flags": report.conflict_flags
        }
    }
    
    reward = None
    info = {}
    if done:
        # ── Code-quality telemetry (not used directly in the reward) ─────────
        code_score = ctx.track_a.evaluate(ctx.files)

        # ── Efficiency penalty ───────────────────────────────────────────────
        p_efficiency = EFFICIENCY_PENALTY_PER_STEP * ctx.steps_taken

        # ── Canonical Green-Code Rubric ──────────────────────────────────────
        # Single source of truth shared by training, baselines, and /step:
        # R = (syntax_gate ∧ hack_gate) × (0.70·green + 0.30·compliance)
        rubric = build_green_rubric()
        rubric_score = rubric(
            action=CodeAction(updated_files=ctx.files),
            observation=CodeObservation(
                orig_files=ctx.orig_files,
                active_rules=ctx.active_rules,
                standards_path=ctx.standards_path,
            ),
        )
        rubric_children = dict(rubric.named_rubrics())
        s_test = rubric_children["syntax_gate"].last_score or 0.0
        green_score = rubric_children["green"].last_score or 0.0
        compliance_score = rubric_children["compliance"].last_score or 0.0
        reward = rubric_score - p_efficiency

        info = {
            "final_reward": reward,
            "s_test": s_test,
            "code_score": {
                "test_pass_rate": code_score.test_pass_rate,
                "lint_improvement": code_score.lint_improvement,
                "complexity_reduction": code_score.complexity_reduction,
                "module_size_compliance": code_score.module_size_compliance,
                "total": code_score.total
            },
            "compliance_score": compliance_score,
            "green_score": green_score,
            "rubric": {
                name: child.last_score
                for name, child in rubric_children.items()
            },
            "efficiency_penalty": p_efficiency,
            "steps_taken": ctx.steps_taken,
        }
        
        del active_episodes[episode_id]
        
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/")
def root():
    """Project info page."""
    gpu_available = torch.cuda.is_available()
    return {
        "project": "Green-Code Optimizer",
        "tagline": "RL agent that refactors Python for energy efficiency, not readability.",
        "description": (
            "Most refactoring agents optimize for readability. This one minimizes "
            "CPU cycles and peak memory while preserving program logic. A graphlet "
            "analyzer models control-flow patterns (nested loops, expensive calls, "
            "deep branching) so the agent learns which structures are 'expensive' "
            "and swaps them for 'cheap' alternatives. CPU-time savings are converted "
            "into real-world CO₂ savings (kg/year, tree-equivalents, car-km)."
        ),
        "hackathon": "OpenEnv India Hackathon 2026 — Meta PyTorch",
        "theme": "Green AI — sustainable code via reinforcement learning",
        "reward_formula": "R = S_test × (0.70·green_score + 0.30·compliance_score) − P_efficiency",
        "green_score_components": {
            "graphlet_score": "Avoidance of expensive control-flow patterns",
            "cpu_improvement": "Relative CPU-time reduction vs. original",
            "memory_improvement": "Relative peak-memory reduction vs. original",
        },
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "training_method": "GRPO (Group Relative Policy Optimization) via Unsloth",
        "adapter": "https://huggingface.co/shreeyanshi03/constrained-refactor-adapter-1.5b",
        "gpu_available": gpu_available,
        "inference_available": gpu_available,
        "endpoints": {
            "GET /demo": "🌱 Live demo page — start here for a 30-second pitch",
            "GET /": "This page — project info",
            "GET /health": "Health check",
            "GET /health/green": "Green-code subsystem status",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "POST /reset": "Start a new episode (corrupted codebase + active rules)",
            "POST /step": "Submit an edit — receive new state + reward (Gym API)",
            "GET /state/{episode_id}": "Episode metadata (Gym API: reset/step/state)",
            "GET /rubric": "Composable rubric tree (named children, formula)",
            "POST /infer": "Run the trained agent (requires GPU)",
            "GET /dashboard/co2/{episode_id}": "CO₂-savings dashboard (HTML for browsers, JSON for clients)",
        },
    }


@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": "green-code-optimizer",
        "gpu_available": gpu_available,
        "inference_ready": gpu_available,
        "endpoints": ["/reset", "/step", "/state", "/rubric", "/infer", "/health",
                      "/health/green", "/dashboard/co2/{episode_id}"],
    }


@app.get("/state")
@app.get("/state/{episode_id}")
def get_state(episode_id: Optional[str] = None):
    """Gym-style state endpoint (RFC 001).

    Returns episode metadata for the active episode. With no `episode_id`,
    returns a summary of all active episodes — useful for monitoring.
    """
    if episode_id is None:
        return {
            "active_episodes": list(active_episodes.keys()),
            "n_active": len(active_episodes),
        }
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    ctx = active_episodes[episode_id]
    return {
        "episode_id": episode_id,
        "step_count": ctx.steps_taken,
        "steps_remaining": ctx.steps_remaining,
        "done": ctx.steps_remaining <= 0 or ctx.hack_detected,
        "hack_detected": ctx.hack_detected,
        "hack_reason": ctx.hack_reason,
        "n_files": len(ctx.files),
        "curriculum_level": getattr(ctx.generator.curriculum, "level", None),
    }


@app.get("/rubric")
def get_rubric_tree():
    """Expose the composable rubric tree as JSON.

    Returns the named children of the Green-Code Rubric so judges (and
    training infrastructure) can introspect what the agent is being scored on.
    """
    try:
        from environment.rubrics import build_green_rubric, OPENENV_AVAILABLE
        rubric = build_green_rubric()
        children = [
            {"path": name, "type": child.__class__.__name__}
            for name, child in rubric.named_rubrics()
        ]
        return {
            "openenv_core": OPENENV_AVAILABLE,
            "root": rubric.__class__.__name__,
            "formula": (
                "R = (syntax_gate ∧ hack_gate) × "
                "(0.70·green + 0.30·compliance)"
            ),
            "green_formula": (
                "green = 0.40·graphlet + 0.35·cpu + 0.25·memory"
            ),
            "children": children,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health/green")
def health_green():
    """Health check for Track C green-code subsystem."""
    return {"track_c": "enabled", "graphlet_analyzer": "active"}


def _render_co2_dashboard_html(payload: Dict[str, Any]) -> str:
    """Render an HTML/CSS dashboard for the CO2 savings payload.

    Used for live demos and judging — visiting /dashboard/co2/{id} from a
    browser returns this rich page; cURL/JSON clients still get JSON.
    """
    g = payload["green_score"]
    e = payload["execution_metrics"]
    co2 = payload["co2_savings"]
    eid = payload.get("episode_id", "—")

    pct_cpu = max(0, min(100, int(g["cpu_improvement"] * 100)))
    pct_mem = max(0, min(100, int(g["memory_improvement"] * 100)))
    pct_graphlet = max(0, min(100, int(g["graphlet_score"] * 100)))
    pct_total = max(0, min(100, int(g["total"] * 100)))

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>🌱 CO₂ Savings Dashboard — {eid[:8]}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root {{ --green:#22c55e; --leaf:#16a34a; --bg:#0b1220; --card:#111827; --txt:#e5e7eb; --muted:#94a3b8; --accent:#34d399; }}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;padding:32px;min-height:100vh;background-image:radial-gradient(circle at top right,rgba(34,197,94,.08),transparent 50%)}}
.container{{max-width:1100px;margin:0 auto}}
h1{{font-size:28px;font-weight:700;display:flex;align-items:center;gap:12px}}
h1 .leaf{{font-size:32px}}
.subtitle{{color:var(--muted);margin-top:6px;font-size:14px}}
.episode{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;color:var(--accent);margin-top:4px}}
.headline{{background:linear-gradient(135deg,#064e3b 0%,#022c22 100%);border:1px solid #10b981;border-radius:16px;padding:36px;margin:24px 0;box-shadow:0 0 60px rgba(34,197,94,.15)}}
.headline .big{{font-size:64px;font-weight:800;color:var(--accent);line-height:1;letter-spacing:-2px}}
.headline .sub{{color:#a7f3d0;font-size:18px;margin-top:8px}}
.headline .equiv{{display:flex;gap:32px;margin-top:24px;flex-wrap:wrap}}
.headline .equiv > div{{display:flex;align-items:center;gap:8px;color:#d1fae5;font-size:15px}}
.headline .equiv .num{{font-size:24px;font-weight:700;color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:24px 0}}
.card{{background:var(--card);border:1px solid #1f2937;border-radius:12px;padding:20px}}
.card h3{{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
.bar-row{{margin:12px 0}}
.bar-row .label{{display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px}}
.bar-row .label span:last-child{{color:var(--accent);font-weight:600}}
.bar{{background:#1f2937;border-radius:99px;height:8px;overflow:hidden}}
.bar > div{{background:linear-gradient(90deg,#10b981,#34d399);height:100%;border-radius:99px;transition:width .8s ease}}
.metric{{font-size:32px;font-weight:700;color:var(--accent)}}
.metric .unit{{font-size:14px;color:var(--muted);margin-left:6px;font-weight:400}}
.delta{{display:inline-block;margin-top:8px;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600}}
.delta.good{{background:#064e3b;color:#a7f3d0}}
.delta.bad {{background:#7f1d1d;color:#fecaca}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}
table td{{padding:8px 0;border-bottom:1px solid #1f2937;font-size:14px}}
table td:last-child{{text-align:right;color:var(--accent);font-weight:600}}
.footer{{margin-top:32px;color:var(--muted);font-size:12px;text-align:center}}
.footer a{{color:var(--accent);text-decoration:none}}
</style></head><body>
<div class="container">
  <h1><span class="leaf">🌱</span>Green-Code Optimizer — CO₂ Savings Dashboard</h1>
  <div class="subtitle">Real-time impact analysis of an RL-refactored Python codebase</div>
  <div class="episode">episode_id: {eid}</div>

  <div class="headline">
    <div class="big">{co2['kg_per_year']:.2f}<span style="font-size:24px;color:#a7f3d0;margin-left:8px">kg CO₂ / year</span></div>
    <div class="sub">saved by refactoring this codebase (assuming {payload['assumptions']['runs_per_day']:,} executions/day)</div>
    <div class="equiv">
      <div>🌳 <span class="num">{co2['equivalent_trees']:.1f}</span> tree-years absorbed</div>
      <div>🚗 <span class="num">{co2['equivalent_car_km']:.0f}</span> km of car travel avoided</div>
      <div>⚡ <span class="num">{co2['grams_per_day']:.1f}</span> g/day reduction</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Green Score Components</h3>
      <div class="bar-row"><div class="label"><span>Graphlet score</span><span>{pct_graphlet}%</span></div><div class="bar"><div style="width:{pct_graphlet}%"></div></div></div>
      <div class="bar-row"><div class="label"><span>CPU improvement</span><span>{pct_cpu}%</span></div><div class="bar"><div style="width:{pct_cpu}%"></div></div></div>
      <div class="bar-row"><div class="label"><span>Memory improvement</span><span>{pct_mem}%</span></div><div class="bar"><div style="width:{pct_mem}%"></div></div></div>
      <div class="bar-row"><div class="label"><span><b>Total green score</b></span><span><b>{pct_total}%</b></span></div><div class="bar"><div style="width:{pct_total}%"></div></div></div>
    </div>

    <div class="card">
      <h3>Runtime Metrics</h3>
      <table>
        <tr><td>Original CPU time</td><td>{e['original']['cpu_time_ms']} ms</td></tr>
        <tr><td>Refactored CPU time</td><td>{e['refactored']['cpu_time_ms']} ms</td></tr>
        <tr><td><b>CPU saved per run</b></td><td><b>{e['cpu_saved_ms']} ms</b></td></tr>
        <tr><td>Original peak memory</td><td>{e['original']['peak_memory_mb']} MB</td></tr>
        <tr><td>Refactored peak memory</td><td>{e['refactored']['peak_memory_mb']} MB</td></tr>
        <tr><td><b>Memory saved per run</b></td><td><b>{e['memory_saved_mb']} MB</b></td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Carbon Assumptions</h3>
      <table>
        <tr><td>Grid carbon intensity</td><td>{payload['assumptions']['carbon_intensity_g_per_kwh']} gCO₂/kWh</td></tr>
        <tr><td>CPU TDP per core</td><td>{payload['assumptions']['cpu_tdp_watts']} W</td></tr>
        <tr><td>Daily executions</td><td>{payload['assumptions']['runs_per_day']:,}</td></tr>
        <tr><td>Files in codebase</td><td>{payload['file_count']['refactored']}</td></tr>
      </table>
    </div>
  </div>

  <div class="footer">
    Powered by <a href="https://github.com/meta-pytorch/openenv">OpenEnv</a> · Built for the <b>OpenEnv India Hackathon 2026</b> · Want raw data? Fetch this URL with <code>Accept: application/json</code>
  </div>
</div></body></html>"""


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    """Interactive demo page — judges & non-technical viewers can see the
    pitch live without needing to call the API. Generates one fresh episode,
    runs evaluation, renders before/after with the CO₂ dashboard inline."""
    generator = EpisodeGenerator(BASE_DIR)
    ep = generator.generate()
    orig_files = generator._load_base_files()
    corrupted = ep["files"]
    evaluator = GreenCodeEvaluator()

    # Pick the smallest corrupted file for clean side-by-side rendering
    fname = min(corrupted, key=lambda f: len(corrupted[f]))
    before = corrupted[fname]
    after = orig_files.get(fname, "")

    green = evaluator.evaluate(corrupted, orig_files)
    payload = generate_dashboard_data(green, corrupted, orig_files)
    payload["episode_id"] = ep["episode_id"]

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    co2 = payload["co2_savings"]
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/><title>🌱 Green-Code Optimizer — Live Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0b1220;color:#e5e7eb;font-family:'Inter',-apple-system,sans-serif;padding:32px;background-image:radial-gradient(circle at top right,rgba(34,197,94,.08),transparent 50%)}}
.container{{max-width:1280px;margin:0 auto}}
h1{{font-size:28px;font-weight:700;margin-bottom:8px}}
h1 .leaf{{margin-right:12px}}
.tagline{{color:#94a3b8;margin-bottom:32px;font-size:15px}}
.cta{{background:linear-gradient(135deg,#064e3b,#022c22);border:1px solid #10b981;border-radius:16px;padding:32px;margin-bottom:32px}}
.cta .big{{font-size:48px;font-weight:800;color:#34d399;letter-spacing:-1px;line-height:1}}
.cta .sub{{color:#a7f3d0;font-size:16px;margin-top:8px}}
.cta .equiv{{display:flex;gap:32px;margin-top:20px;flex-wrap:wrap;color:#d1fae5}}
.cta .equiv b{{color:#fff}}
.diff{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px}}
.code-card{{background:#111827;border:1px solid #1f2937;border-radius:12px;overflow:hidden}}
.code-card h3{{padding:16px 20px;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;border-bottom:1px solid #1f2937;display:flex;justify-content:space-between;align-items:center}}
.code-card.bad h3{{color:#fca5a5}}
.code-card.good h3{{color:#a7f3d0}}
.tag{{font-size:10px;padding:2px 8px;border-radius:99px;font-weight:700}}
.tag.bad{{background:#7f1d1d;color:#fecaca}}
.tag.good{{background:#064e3b;color:#a7f3d0}}
pre{{padding:16px 20px;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5;color:#e5e7eb;max-height:400px}}
.cta-link{{display:inline-block;margin-top:32px;background:#10b981;color:#022c22;padding:12px 24px;border-radius:8px;font-weight:600;text-decoration:none}}
.cta-link:hover{{background:#34d399}}
@media (max-width:900px){{.diff{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="container">
  <h1><span class="leaf">🌱</span>Green-Code Optimizer — Live Demo</h1>
  <div class="tagline">An RL agent refactors energy-inefficient Python. Below is a freshly generated episode showing what the agent saw and what an ideal refactor would save.</div>

  <div class="cta">
    <div class="big">{co2['kg_per_year']:.2f} kg CO₂ / year</div>
    <div class="sub">would be saved if this codebase were optimised (at {payload['assumptions']['runs_per_day']:,} runs/day)</div>
    <div class="equiv">
      <span>🌳 <b>{co2['equivalent_trees']:.1f}</b> tree-years</span>
      <span>🚗 <b>{co2['equivalent_car_km']:.0f}</b> km car travel</span>
      <span>⚡ <b>{co2['grams_per_day']:.1f}</b> g/day</span>
    </div>
  </div>

  <h2 style="font-size:20px;margin-bottom:16px">📄 What the agent sees: <code style="color:#34d399">{fname}</code></h2>
  <div class="diff">
    <div class="code-card bad">
      <h3>Before refactor (corrupted) <span class="tag bad">energy-inefficient</span></h3>
      <pre>{_esc(before[:2500])}</pre>
    </div>
    <div class="code-card good">
      <h3>After refactor (target) <span class="tag good">green</span></h3>
      <pre>{_esc(after[:2500])}</pre>
    </div>
  </div>

  <a class="cta-link" href="/dashboard/co2/{ep['episode_id']}">🔍 See full CO₂ dashboard →</a>
  &nbsp;<a class="cta-link" style="background:transparent;border:1px solid #10b981;color:#34d399" href="/docs">📖 API docs</a>
</div></body></html>""")


@app.get("/dashboard/co2/{episode_id}")
async def dashboard_co2(episode_id: str, request: Request):
    """CO₂-savings dashboard for an active episode.

    Content-negotiates: browsers (Accept: text/html) get a rich HTML page;
    JSON clients (curl, scripts) get the raw payload.
    """
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    ctx = active_episodes[episode_id]
    orig_files = ctx.generator.generate()["files"]
    updated_files = ctx.files

    evaluator = GreenCodeEvaluator()
    green_score = evaluator.evaluate(orig_files, updated_files)
    payload = generate_dashboard_data(green_score, orig_files, updated_files)
    payload["episode_id"] = episode_id

    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return HTMLResponse(_render_co2_dashboard_html(payload))
    return payload


class InferRequest(BaseModel):
    observation: Dict[str, Any]


@app.post("/infer")
async def infer(req: InferRequest):
    """Run the trained agent on an observation and return the next action.
    
    Requires GPU. On CPU-only Spaces, returns an error with instructions
    to run inference on the Lightning Studio instead.
    """
    if not torch.cuda.is_available():
        return JSONResponse(
            status_code=503,
            content={
                "error": "GPU required for inference",
                "detail": "This Space runs on CPU. Inference requires GPU even on the 1.5B model.",
                "alternatives": {
                    "adapter": "https://huggingface.co/shreeyanshi03/constrained-refactor-adapter-1.5b",
                    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    "instructions": "Load the adapter with peft and run inference on a GPU machine.",
                },
                "environment_endpoints_work": True,
                "try_these": ["POST /reset", "POST /step", "GET /health"],
            },
        )
    try:
        from inference import run_inference

        action = run_inference(req.observation)
        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

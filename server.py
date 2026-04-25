from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uuid
import torch

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker

app = FastAPI(
    title="Constrained Refactor Gauntlet",
    description="OpenEnv RL environment for multi-file code refactoring with 150 cascading rules",
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

class EpisodeContext:
    def __init__(self, base_dir: str):
        self.generator = EpisodeGenerator(base_dir)
        self.track_a = CodeQualityEvaluator()
        self.track_b = ComplianceChecker(os.path.join(os.path.dirname(base_dir), "ENGINEERING_STANDARDS.md"))
        self.files = {}
        self.steps_remaining = 70
        
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
        code_score = ctx.track_a.evaluate(ctx.files)
        compliance_score = ctx.track_b.get_score()
        reward = code_score.total * compliance_score
        
        info = {
            "final_reward": reward,
            "code_score": {
                "test_pass_rate": code_score.test_pass_rate,
                "lint_improvement": code_score.lint_improvement,
                "complexity_reduction": code_score.complexity_reduction,
                "module_size_compliance": code_score.module_size_compliance,
                "total": code_score.total
            },
            "compliance_score": compliance_score
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
        "project": "Constrained Refactor Gauntlet",
        "description": "OpenEnv RL environment: refactor a Python codebase while obeying 150 cascading engineering rules",
        "hackathon": "Meta PyTorch OpenEnv Hackathon",
        "reward_formula": "CodeScore × ComplianceScore",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "training_method": "GRPO (Group Relative Policy Optimization) via Unsloth",
        "adapter": "https://huggingface.co/shreeyanshi03/constrained-refactor-adapter",
        "gpu_available": gpu_available,
        "inference_available": gpu_available,
        "endpoints": {
            "GET /": "This page — project info",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "POST /reset": "Start a new episode",
            "POST /step": "Take an action in the environment",
            "POST /infer": "Run trained agent on an observation (requires GPU)",
        },
    }


@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": "constrained-refactor-gauntlet",
        "gpu_available": gpu_available,
        "inference_ready": gpu_available,
        "endpoints": ["/reset", "/step", "/infer", "/health"],
    }


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
                "detail": "This Space runs on CPU. The 7B model requires GPU for inference.",
                "alternatives": {
                    "adapter": "https://huggingface.co/shreeyanshi03/constrained-refactor-adapter",
                    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
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

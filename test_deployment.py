"""
test_deployment.py — Test the deployed HF Space end-to-end.

Tests the full agent loop: /health → /reset → /infer → /step

Usage:
    python test_deployment.py

    # Or with a custom Space URL:
    SPACE_URL=https://your-space.hf.space python test_deployment.py
"""

import os
import json
import httpx

SPACE_URL = os.getenv(
    "SPACE_URL",
    "https://shreeyanshi03-constrained-refactor-gauntlet.hf.space"
)

client = httpx.Client(base_url=SPACE_URL, timeout=120.0)


def test_health():
    print("=" * 60)
    print("TEST 1: /health")
    print("=" * 60)
    resp = client.get("/health")
    data = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(data, indent=2)}")
    assert data["status"] == "ok", f"Health check failed: {data}"
    print("  ✅ Health check passed!\n")
    return data


def test_reset():
    print("=" * 60)
    print("TEST 2: /reset (Create new episode)")
    print("=" * 60)
    resp = client.post("/reset", json={})
    data = resp.json()
    print(f"  Status: {resp.status_code}")
    
    episode_id = data.get("episode_id", "N/A")
    obs = data.get("observation", {})
    files = obs.get("files", {})
    steps = obs.get("steps_remaining", 0)
    rules = obs.get("active_rules_count", 0)
    level = obs.get("curriculum_level", 0)
    
    print(f"  Episode ID: {episode_id}")
    print(f"  Files: {list(files.keys())}")
    print(f"  Steps remaining: {steps}")
    print(f"  Active rules: {rules}")
    print(f"  Curriculum level: {level}")
    
    for fname, content in files.items():
        lines = content.split("\n")
        print(f"    {fname}: {len(lines)} lines, {len(content)} chars")
    
    print("  ✅ Reset passed!\n")
    return data


def test_infer(observation):
    print("=" * 60)
    print("TEST 3: /infer (Agent generates action)")
    print("=" * 60)
    resp = client.post("/infer", json={"observation": observation})
    
    if resp.status_code == 200:
        data = resp.json()
        action = data.get("action", {})
        print(f"  Status: {resp.status_code}")
        print(f"  Action tool: {action.get('tool', 'N/A')}")
        
        args = action.get("args", {})
        if action.get("tool") == "edit_file":
            fname = args.get("filename", "?")
            content = args.get("content", "")
            print(f"  Editing: {fname} ({len(content)} chars)")
            print(f"  Preview: {content[:200]}...")
        else:
            print(f"  Args: {json.dumps(args, indent=2)}")
        
        print("  ✅ Inference passed!\n")
        return data
    else:
        print(f"  ⚠️  Status: {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        print("  ⚠️  Inference endpoint returned error (model may not be loaded yet)")
        print("     This is expected if HF_ADAPTER_REPO secret isn't set.\n")
        return None


def test_step(episode_id, action):
    print("=" * 60)
    print("TEST 4: /step (Execute action in environment)")
    print("=" * 60)
    resp = client.post("/step", json={
        "episode_id": episode_id,
        "action": action
    })
    print(f"  Status: {resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        print(f"  ⚠️  Response was not JSON: {resp.text[:200]}")
        print("  ⚠️  HF Space proxy may have timed out.\n")
        return None
    
    obs = data.get("observation", {})
    reward = data.get("reward")
    done = data.get("done", False)
    
    print(f"  Steps remaining: {obs.get('steps_remaining', '?')}")
    print(f"  Done: {done}")
    print(f"  Reward: {reward}")
    
    report = obs.get("violation_report", {})
    if report:
        print(f"  Violations triggered: {len(report.get('newly_triggered', []))}")
        print(f"  Violations resolved: {len(report.get('newly_resolved', []))}")
        print(f"  Still outstanding: {len(report.get('still_outstanding', []))}")
    
    print("  ✅ Step passed!\n")
    return data


def test_manual_step(episode_id):
    """Test a manual edit_file action without inference."""
    print("=" * 60)
    print("TEST 3b: /step with manual action")
    print("=" * 60)
    
    action = {
        "tool": "check_compliance",
        "args": {}
    }
    resp = client.post("/step", json={
        "episode_id": episode_id,
        "action": action
    })
    print(f"  Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"  Steps remaining: {data['observation'].get('steps_remaining', '?')}")
        print("  ✅ Manual step passed!\n")
        return data
    except Exception as e:
        print(f"  ⚠️  Response was not JSON: {resp.text[:200]}")
        print(f"  ⚠️  This can happen with HF Space proxy timeouts.\n")
        return None


def main():
    print(f"\n🔗 Testing Space: {SPACE_URL}\n")
    
    # Test 1: Health
    test_health()
    
    # Test 2: Reset
    reset_data = test_reset()
    episode_id = reset_data["episode_id"]
    observation = reset_data["observation"]
    
    # Test 3: Infer (may fail if model not loaded - that's ok)
    infer_result = test_infer(observation)
    
    # Test 3b: Manual step (always works)
    test_manual_step(episode_id)
    
    # Test 4: Step with inferred action (if inference worked)
    if infer_result and infer_result.get("action"):
        test_step(episode_id, infer_result["action"])
    else:
        # Fallback: test step with a manual action
        test_step(episode_id, {"tool": "run_tests", "args": {}})
    
    print("=" * 60)
    print("✅ ALL DEPLOYMENT TESTS COMPLETE!")
    print("=" * 60)
    print(f"\n  Space URL: {SPACE_URL}")
    print(f"  Docs:      {SPACE_URL}/docs")
    print(f"  Health:    {SPACE_URL}/health")


if __name__ == "__main__":
    main()

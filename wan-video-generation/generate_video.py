#!/usr/bin/env python3
"""
Text-to-Video script using Wan2.2-T2V-A14B (Wan-AI).

What it does:
 - logs into Hugging Face if HF_TOKEN is present
 - clones Wan2.2 repo (if missing) and downloads the model weights
 - runs the Wan2.2 generate.py (basic single-GPU fallback command)
 - finds generated video files, copies them into ./artifacts
 - saves artifacts via transformerlab's lab.save_artifact()

Notes:
 - This script tries to be conservative (defaults to 480P).
 - For large / 720P runs you should follow Wan2.2's multi-GPU or offload instructions
   (see the model README for recommended flags). :contentReference[oaicite:1]{index=1}
"""

import os
import sys
import subprocess
import shutil
import glob
import time
from datetime import datetime
from pathlib import Path

# TransformerLab import
from lab import lab

# Hugging Face helper
from huggingface_hub import login as hf_login

DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
WAN_REPO_GIT = "https://github.com/Wan-Video/Wan2.2.git"

def setup_environment():
    cfg = {
        "model_name": os.getenv("MODEL_NAME", DEFAULT_MODEL),
        "wan_repo_dir": os.getenv("WAN_REPO_DIR", "./Wan2.2"),
        "model_dir": os.getenv("MODEL_DIR", "./Wan2.2-T2V-A14B"),
        "prompt": os.getenv("PROMPT",
                            "A short 5-second cinematic explainer video presenting the benefits of Transformer Lab: "
                            "clean reproducible ML infra, simple artifact tracking, automatic experiment provenance, "
                            "collaboration-ready tasks, and easy model orchestration. Friendly narrator voice, "
                            "on-screen minimal motion graphics, and clear bullets."),
        "size": os.getenv("SIZE", "640*480"),   # 480p default for lower memory
        "offload": os.getenv("OFFLOAD", "True"),
        "use_prompt_extend": os.getenv("USE_PROMPT_EXTEND", "False"),
        "output_dir": os.getenv("OUTPUT_DIR", "./artifacts"),
        "hf_token": os.getenv("HF_TOKEN", None),
        "seed": int(os.getenv("SEED", str(int(time.time() % 100000)))),
    }
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg

def run_cmd(cmd, cwd=None, env=None, check=True):
    lab.log(f"Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, env=env or os.environ, capture_output=True, text=True)
    if result.returncode != 0:
        lab.log("Command stdout:\n" + (result.stdout or "<none>"))
        lab.log("Command stderr:\n" + (result.stderr or "<none>"))
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result

def ensure_wan_repo(cfg):
    if not Path(cfg["wan_repo_dir"]).exists():
        lab.log(f"Cloning Wan2.2 repo to {cfg['wan_repo_dir']} ...")
        run_cmd(["git", "clone", WAN_REPO_GIT, cfg["wan_repo_dir"]])
    else:
        lab.log(f"Wan2.2 repo already exists at {cfg['wan_repo_dir']}")

def download_model_weights(cfg):
    # The Wan README suggests using huggingface-cli download or modelscope; we use huggingface-cli for simplicity.
    if Path(cfg["model_dir"]).exists() and any(Path(cfg["model_dir"]).iterdir()):
        lab.log(f"Model directory {cfg['model_dir']} already populated.")
        return

    lab.log("Downloading model weights via huggingface-cli. This may take a long time.")
    # ensure huggingface-cli is installed
    run_cmd([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)

    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.download",  # fallback to huggingface_hub CLI module
        cfg["model_name"],
        "--local-dir", cfg["model_dir"],
    ]
    # If a token is present, pass it via env to huggingface-cli
    env = os.environ.copy()
    if cfg["hf_token"]:
        env["HF_TOKEN"] = cfg["hf_token"]

    # Some installations prefer huggingface-cli executable; try best-effort:
    try:
        run_cmd(cmd, env=env)
    except Exception as e:
        lab.log("huggingface_hub CLI method failed, trying huggingface-cli (if available)...")
        run_cmd(["huggingface-cli", "download", cfg["model_name"], "--local-dir", cfg["model_dir"]], env=env)

def install_requirements(cfg):
    req_file = Path(cfg["wan_repo_dir"]) / "requirements.txt"
    if req_file.exists():
        lab.log("Installing Wan2.2 requirements (may be large)...")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        lab.log("No requirements.txt found in Wan2.2 repo; skipping pip install -r requirements.txt")

def run_generation(cfg):
    # Build generate.py command (single-GPU conservative fallback)
    gen_py = Path(cfg["wan_repo_dir"]) / "generate.py"
    if not gen_py.exists():
        raise FileNotFoundError(f"generate.py not found in {cfg['wan_repo_dir']}. Check Wan repo layout.")

    # Use a simple inference command. For production or high-res, use the repo's multi-GPU instructions.
    cmd = [
        sys.executable, str(gen_py),
        "--task", "t2v-A14B",
        "--size", cfg["size"],
        "--ckpt_dir", cfg["model_dir"],
        "--offload_model", "True" if cfg["offload"].lower() in ["true", "1", "yes"] else "False",
        "--convert_model_dtype",
        "--prompt", cfg["prompt"],
        "--seed", str(cfg["seed"])
    ]

    # Optionally enable prompt extension flags (disabled by default)
    if cfg["use_prompt_extend"].lower() in ["true", "1", "yes"]:
        cmd += ["--use_prompt_extend"]

    # Run generation (this may stream a lot of stdout/stderr)
    run_cmd(cmd, cwd=cfg["wan_repo_dir"])

def collect_and_save_artifacts(cfg):
    lab.log("Searching for generated video files (mp4 / webm / gif)...")
    # Search common places for generated videos in the repo / working tree
    candidates = []
    for pattern in ("**/*.mp4", "**/*.webm", "**/*.gif", "**/*.mov", "**/*.mkv"):
        candidates.extend(glob.glob(str(Path(cfg["wan_repo_dir"]) / pattern), recursive=True))

    # Fallback: also search cwd
    candidates.extend(glob.glob("**/*.mp4", recursive=True))
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)

    if not candidates:
        lab.error("No generated video files were found. Check the Wan2.2 generate logs for output path.")
        return []

    saved = []
    for src in candidates[:5]:  # save top 5 most-recent candidates
        fname = f"{Path(src).stem}_{int(time.time())}{Path(src).suffix}"
        dst = Path(cfg["output_dir"]) / fname
        shutil.copy(src, dst)
        lab.log(f"Copied generated file to {dst}")
        artifact_path = lab.save_artifact(str(dst), fname)
        saved.append(artifact_path)

    return saved

def main():
    cfg = setup_environment()
    lab.init()
    lab.set_config(cfg)
    lab.log("🚀 Starting Wan2.2 video generation (Transformer Lab explainer)")
    lab.log(f"Model: {cfg['model_name']}  Size: {cfg['size']}  Prompt: {cfg['prompt'][:120]}...")
    start = datetime.now()

    try:
        # HF login (optional)
        if cfg["hf_token"]:
            lab.log("Logging into Hugging Face with provided token...")
            hf_login(token=cfg["hf_token"])

        cfg["wan_repo_dir"] = os.path.abspath(cfg["wan_repo_dir"])
        cfg["model_dir"] = os.path.abspath(cfg["model_dir"])
        cfg["output_dir"] = os.path.abspath(cfg["output_dir"])

        ensure_wan_repo(cfg)
        install_requirements(cfg)
        if not Path(cfg["model_dir"]).exists() or not any(Path(cfg["model_dir"]).iterdir()):
            download_model_weights(cfg)
        else:
            lab.log("Model weights already present; skipping download.")

        lab.update_progress(20)
        lab.log("Running generation (this step may take a while)...")
        run_generation(cfg)
        lab.update_progress(80)

        artifacts = collect_and_save_artifacts(cfg)
        lab.update_progress(100)
        duration = datetime.now() - start
        lab.finish(f"Video generation finished in {duration}. Saved {len(artifacts)} artifact(s).")
        return {"status": "success", "artifacts": artifacts}

    except Exception as e:
        lab.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    main()

"""
upload_adapter.py — Push the trained LoRA adapter to HuggingFace Hub.

Usage (run on the training machine where grpo_output/final_adapter exists):

    # Set your HF token
    export HF_TOKEN="hf_..."

    # Upload to your HF repo
    python upload_adapter.py --repo-id "shreeyanshi123/constrained-refactor-adapter" 

    # Or with custom adapter path
    python upload_adapter.py --repo-id "shreeyanshi123/constrained-refactor-adapter" \
                             --adapter-path ./grpo_output/final_adapter
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID, e.g. 'username/constrained-refactor-adapter'",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.path.join(os.path.dirname(__file__), "grpo_output", "final_adapter"),
        help="Path to the adapter directory (default: grpo_output/final_adapter)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the repo private",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN environment variable not set!")
        print("   Run: export HF_TOKEN='hf_...'")
        return

    adapter_path = os.path.abspath(args.adapter_path)
    if not os.path.isdir(adapter_path):
        print(f"❌ Adapter directory not found: {adapter_path}")
        print("   Train first with: python training/train_grpo.py")
        return

    # List files to upload
    files = os.listdir(adapter_path)
    print(f"📦 Adapter directory: {adapter_path}")
    print(f"   Files: {files}")
    total_size = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in files
        if os.path.isfile(os.path.join(adapter_path, f))
    )
    print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")

    # Create repo if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            token=token,
            exist_ok=True,
        )
        print(f"✅ Repo ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"⚠️  Repo creation note: {e}")

    # Upload the adapter
    print(f"\n🚀 Uploading adapter to {args.repo_id}...")
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload GRPO-trained LoRA adapter for Constrained Refactor Gauntlet",
    )

    print(f"\n✅ Upload complete!")
    print(f"   🔗 https://huggingface.co/{args.repo_id}")
    print(f"\n   To use in Docker, set:")
    print(f"     HF_ADAPTER_REPO={args.repo_id}")


if __name__ == "__main__":
    main()

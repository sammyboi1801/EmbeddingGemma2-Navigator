"""Create (if needed) and upload the Hugging Face Space that serves the model.

usage:  hf auth login                      # once, with a write token
        python space/deploy.py <hf-user>/<space-name>

Uploads only what the Docker image needs, plus space/README.md as the Space's README.
"""
import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["Dockerfile", "requirements-server.txt", "server.py",
         "eg2_autoregressive_decoding.py", "checkpoints/chat.pt"]


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python space/deploy.py <hf-user>/<space-name>")
    repo_id = sys.argv[1]
    api = HfApi()
    print("logged in as", api.whoami()["name"])

    url = api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT, allow_patterns=FILES,
                      commit_message="Deploy model server")
    api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=os.path.join(ROOT, "space", "README.md"),
                    path_in_repo="README.md", commit_message="Space configuration")

    owner, name = repo_id.split("/")
    host = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    print(f"Space:   {url}")
    print(f"Backend: https://{host}.hf.space   (check /health once the build finishes)")


if __name__ == "__main__":
    main()

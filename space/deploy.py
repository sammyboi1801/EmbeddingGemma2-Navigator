"""Upload the model server to a Hugging Face Space (Docker SDK).

usage:  hf auth login                      # once, with a write token
        python space/deploy.py <hf-user>/<space-name>

Uploads only what the Docker image needs, plus space/README.md as the Space's README. If the Space
already exists (e.g. an older Docker Space reused for this project), files from its previous app
are removed in the same commit. Creating a new Docker Space may require a Hugging Face PRO plan.
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

    if api.repo_exists(repo_id, repo_type="space"):
        url = f"https://huggingface.co/spaces/{repo_id}"
        keep = set(FILES) | {"README.md", ".gitattributes"}
        stale = [f for f in api.list_repo_files(repo_id, repo_type="space") if f not in keep]
        print(f"updating existing Space; removing {len(stale)} file(s) from its previous app:", stale)
    else:
        url = api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False)
        stale = []
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT, allow_patterns=FILES,
                      delete_patterns=stale or None, commit_message="Deploy model server")
    api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=os.path.join(ROOT, "space", "README.md"),
                    path_in_repo="README.md", commit_message="Space configuration")

    owner, name = repo_id.split("/")
    host = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    print(f"Space:   {url}")
    print(f"Backend: https://{host}.hf.space   (check /health once the build finishes)")


if __name__ == "__main__":
    main()

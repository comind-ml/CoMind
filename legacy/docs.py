from pathlib import Path
import json

doc_dir = Path(__file__).parent / "docs"

competitions = doc_dir.glob("*")

num_discussions = 0
num_kernels = 0

for competition in competitions:
    discussion_dir = competition / "docs" / "discussions"
    kernel_dir = competition / "docs" / "kernels"

    with open(discussion_dir / "info.json", "r") as f:
        discussion_info = json.load(f)
        num_discussions += len(discussion_info)
    
    with open(kernel_dir / "info.json", "r") as f:
        kernel_info = json.load(f)
        num_kernels += len(kernel_info)

print(f"Total number of discussions: {num_discussions}")
print(f"Total number of kernels: {num_kernels}")
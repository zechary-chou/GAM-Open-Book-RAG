import argparse
from pathlib import Path
import json
import shutil

parser = argparse.ArgumentParser(description="Find and process bad results.")
parser.add_argument('-r', '--root', required=True, help='Input results dir name in ./results/hotpotqa')
parser.add_argument('-o', '--output', required=True, help='Output dir name within ./results/underperform')
args = parser.parse_args()

root = Path(f"./results/hotpotqa/{args.root}")
dest = Path(f"./results/underperform/{args.output}")
avg = 0
count = 0
for qa_file in root.rglob("qa_result.json"):
    try:
        with open(qa_file, encoding="utf-8") as f:
            data = json.load(f)
            avg += data["f1"]
            count += 1
            # if data["f1"] <= 0.1:
            #     print(qa_file.parent)
            #     count += 1
            #     shutil.copytree(qa_file.parent, dest / qa_file.parent.name, dirs_exist_ok=True)
    except UnicodeDecodeError as e:
        print(f"Encoding error in {qa_file}: {e}")
        continue
    except PermissionError as e:
        print(f"Permission error in {qa_file}: {e}")
        continue
if count > 0:
    avg /= count
print("count: ", count)
print("avg: ", avg)
# print("totalCount: ", totalCount)


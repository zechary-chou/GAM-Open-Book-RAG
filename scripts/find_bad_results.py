from pathlib import Path
import json
import shutil
root = Path("./results/hotpotqa/eval_400_qwen2.5-0.5b-instruct")
dest = Path("./results/underperform/hotpotqa_400_qwen2.5-0.5b-instruct")
avg = 0
count = 0
for qa_file in root.rglob("qa_result.json"):
    try:
        with open(qa_file, encoding="utf-8") as f:
            data = json.load(f)
            avg += data["f1"]
            count += 1
            if data["f1"] <= 0.1:
                print(qa_file.parent)
                # count += 1
                # shutil.copytree(qa_file.parent,dest / qa_file.parent.name, dirs_exist_ok=True)


    except UnicodeDecodeError as e:
        print(f"Encoding error in {qa_file}: {e}")
        continue
    except PermissionError as e:
        print(f"Permission error in {qa_file}: {e}")
        continue
avg /= 128
print("count: ",count)
print("avg: ", avg)
# print("totalCount: ", totalCount)


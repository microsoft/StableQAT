from datasets import load_dataset
import json

out_file = "fineweb_edu.jsonl"

# use name="sample-100BT" to use the 100BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
json.dump(list(fw), open(out_file, 'w', encoding='utf-8'), ensure_ascii=False)
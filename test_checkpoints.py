import json

batches = [
    "batch_20260202_234755",
    "batch_20260202_234751",
    "batch_20260202_234738",
    "batch_20260202_234746",
    "batch_20260202_234742",
    "batch_20260202_234733",
]

for b in batches:
    path = f"contract-translator/output/{b}/checkpoint.json"
    try:
        data = json.load(open(path))
        print(
            f'✓ {b} - Batch: {data["batch_id"]}, Processed: {len(data["processed_indices"])}, Total: {data["total_contracts"]}'
        )
    except Exception as e:
        print(f"✗ {b} - ERROR: {e}")

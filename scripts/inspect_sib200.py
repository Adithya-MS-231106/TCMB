from datasets import load_from_disk

path = "data/sib200"

print("Loading SIB-200...")
dataset = load_from_disk(path)

print("\nAvailable splits:")
for split, data in dataset.items():
    print(f"\n{split}: {len(data)} examples")
    print("Columns:", data.column_names)

    if len(data) > 0:
        print("First example:")
        print(data[0])
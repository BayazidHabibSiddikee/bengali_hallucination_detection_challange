import json

with open("kernel-metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

ds = set(meta.get("dataset_sources", []))
ds.add("ajmainmahtab/bangla-natural-language-inference-dataset")
ds.add("disisbig/bengali-wikipedia-articles")
ds.add("mahdihasanqurishi/banglahallueval-qa")

meta["dataset_sources"] = list(ds)

with open("kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)


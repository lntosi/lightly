from pathlib import Path
import json

# create metadata schema.json
schema = [
    {
        "name": "Video Name",
        "path": "video_name",
        "defaultValue": "undefined",
        "valueDataType": "CATEGORICAL_STRING",
    }
]

schema_path = Path("metadata/schema.json")
schema_path.parent.mkdir(parents=True, exist_ok=True)

with open(schema_path, "w") as f:
    json.dump(schema, f, indent=4)

videos = Path("data/").glob("*.avi")

for fname in videos:
    metadata = {
        "file_name": str(fname),
        "type": "video",
        "metadata": {"video_name": str(fname.stem)},
    }

    lightly_metadata_fname = "metadata" / fname.with_suffix(
        ".json"
    )
    lightly_metadata_fname.parent.mkdir(parents=True, exist_ok=True)

    with open(lightly_metadata_fname, "w") as f:
        json.dump(metadata, f, indent=4)

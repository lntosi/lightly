from ultralytics import YOLO
from pathlib import Path
import json

model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

predictions_rooth_path = Path("predictions")
task_name = "object_detection_pedestrian"
predictions_path = Path(predictions_rooth_path / task_name)

important_classes = {
    "person": 0,
    "bicycle": 1,
    "car": 2
}

# "motorcycle": 3,
# "bus": 5,
# "traffic light": 9,
# "stop sign": 11,
# "parking meter": 12,

classes = list(important_classes.values())

# create tasks.json
tasks_json_path = predictions_rooth_path / "tasks.json"
tasks_json_path.parent.mkdir(parents=True, exist_ok=True)

with open(tasks_json_path, "w") as f:
    json.dump([task_name], f)


# create schema.json
schema = {"task_type": "object-detection", "categories": []}
for key, val in important_classes.items():
    cat = {"id": val, "name": key}
    schema["categories"].append(cat)

schema_path = predictions_path / "schema.json"
schema_path.parent.mkdir(parents=True, exist_ok=True)

with open(schema_path, "w") as f:
    json.dump(schema, f, indent=4)


videos = Path("data/").glob("*.avi")

for video in videos:
    results = model.predict(video, conf=0.1)
    predictions = [result.boxes.boxes for result in results]

    # convert filename to lightly format
    # 'data/passageway1-c0.avi' --> 'data/passageway1-c0-0001-avi.json'
    number_of_frames = len(predictions)
    padding = len(str(number_of_frames))  # '1234' --> 4 digits
    fname = video

    for idx, prediction in enumerate(predictions):
        fname_prediction = (
            f"{fname.parents[0] / fname.stem}-{idx:0{padding}d}-{fname.suffix[1:]}.json"
        )

        # NOTE: prediction file_name must be a .png file as the Lightly Worker
        # treats extracted frames from videos as PNGs
        lightly_prediction = {
            "file_name": str(Path(fname_prediction).with_suffix(".png")),
            "predictions": [],
        }

        for pred in prediction:
            x0, y0, x1, y1, conf, class_id = pred

            # skip predictions thare are not part of the important_classes
            if class_id in important_classes.values():
                # note that we need to conver form x0, y0, x1, y1 to x, y, w, h format
                pred = {
                    "category_id": int(class_id),
                    "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    "score": float(conf),
                }
                lightly_prediction["predictions"].append(pred)

                # create the prediction file for the image
                path_to_prediction = predictions_path / Path(
                    fname_prediction
                ).with_suffix(".json")

                path_to_prediction.parents[0].mkdir(parents=True, exist_ok=True)
                with open(path_to_prediction, "w") as f:
                    json.dump(lightly_prediction, f, indent=4)

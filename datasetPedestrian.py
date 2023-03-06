from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType
from lightly.openapi_generated.swagger_client import DatasourcePurpose
import json


# Create the Lightly client to connect to the API.
client = ApiWorkflowClient(
    token="LIGHTLY_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset(dataset_name="datasetPedestrian",
                      dataset_type=DatasetType.VIDEOS)
dataset_id = client.dataset_id

# Configure the Input datasource.
client.set_gcs_config(
    resource_path="gs://dtlake-poc/pedestrians_input/data/",
    project_id="PROJECT_ID",
    credentials=json.dumps(
        json.load(open("FILE_JSON"))),
    purpose=DatasourcePurpose.INPUT,
)
# Configure the Lightly datasource.
client.set_gcs_config(
    resource_path="gs://dtlake-poc/pedestrians_lightly/",
    project_id="PROJECT_ID",
    credentials=json.dumps(
        json.load(open("FILE_JSON"))),
    purpose=DatasourcePurpose.LIGHTLY,
)


scheduled_run_id = client.schedule_compute_worker_run(
    worker_config={
        # "enable_training": True,
    },
    selection_config={
        "n_samples": 100,
        "strategies": [
            {
                # strategy to find diverse objects
                "input": {
                    "type": "EMBEDDINGS",
                    "task": "object_detection_pedestrian",
                },
                "strategy": {
                    "type": "DIVERSITY",
                },
            },
            {
                # strategy to balance the class ratios
                "input": {
                    "type": "PREDICTIONS",
                    "name": "CLASS_DISTRIBUTION",
                    "task": "object_detection_pedestrian",
                },
                "strategy": {
                    "type": "BALANCE",
                    "target": {
                        'person': 0.33,
                        'bicycle': 0.34,
                        'car': 0.33,
                    }
                },
            },
            {
                # strategy to use prediction score (Active Learning)
                "input": {
                    "type": "SCORES",
                    "task": "object_detection_pedestrian",
                    "score": "object_frequency"
                },
                "strategy": {
                    "type": "WEIGHTS"
                },
            },
            {
                # strategy to use prediction score (Active Learning)
                "input": {
                    "type": "SCORES",
                    "task": "object_detection_pedestrian",
                    "score": "objectness_least_confidence"
                },
                "strategy": {
                    "type": "WEIGHTS"
                },
            },
            {
                # strategy to balance across videos
                "input": {
                    "type": "METADATA",
                    "key": "video_name"
                },
                "strategy": {
                    "type": "BALANCE",
                    "target": {
                        "passageway1-c0": 0.34,
                        "terrace1-c0": 0.33,
                        "terrace1-c2": 0.33,
                    }

                },
            }
        ],
    },
    lightly_config={
        # "trainer": {
        #    "max_epochs": 5,
        # },
        # "loader": {"batch_size": 128},
    },
)
print(scheduled_run_id)

# You can use this code to track and print the state of the Lightly Worker.
# The loop will end once the run has finished, was canceled, or failed.
for run_info in client.compute_worker_run_info_generator(
    scheduled_run_id=scheduled_run_id
):
    print(
        f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'"
    )

if run_info.ended_successfully():
    print("SUCCESS")
else:
    print("FAILURE")

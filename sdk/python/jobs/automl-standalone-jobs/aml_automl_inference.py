import json
from pathlib import Path
import base64
import numpy as np
from sklearn.metrics import top_k_accuracy_score as topk_acc
import mlflow
import mlflow.pyfunc
import pandas as pd
import tqdm
from mlflow.tracking.client import MlflowClient

import aml_automl_ic_pipeline as aml_train


def main():

    ml_client = aml_train.configure()

    # Obtain the tracking URL from MLClient
    MLFLOW_TRACKING_URI = ml_client.workspaces.get(
        name=ml_client.workspace_name
    ).mlflow_tracking_uri

    # Set the MLFLOW TRACKING URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"\nCurrent tracking uri: {mlflow.get_tracking_uri()}")


    # Initialize MLFlow client
    mlflow_client = MlflowClient()

    print(MLFLOW_TRACKING_URI)

    job_name = 'sco-sg-segmented-batch1-v1_aml_0326'

    # Get the parent run
    mlflow_parent_run = mlflow_client.get_run(job_name)

    print("Parent Run: ")
    print(mlflow_parent_run)
    # Print parent run tags. 'automl_best_child_run_id' tag should be there.
    print(mlflow_parent_run.data.tags.keys())

    # Get the best model's child run

    best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
    print(f"Found best child run id: {best_child_run_id}")

    best_run = mlflow_client.get_run(best_child_run_id)

    print("Best child run: ")
    print(best_run)

    print(pd.DataFrame(best_run.data.metrics, index=[0]).T)

    local_dir = Path('./artifact_downloads')
    local_dir.mkdir(parents=True, exist_ok=True)
    mlflow_model_dir = local_dir / "outputs/mlflow-model"
    if not mlflow_model_dir.exists():
        local_path = mlflow_client.download_artifacts(best_run.info.run_id, "outputs", local_dir)
        print(f"Artifacts downloaded in: {local_path}")

    dataset_parent_dir = Path("/work/irisml/Irisml-Internal/tmp_dir")
    dataset_name = "sco-sg-segmented-batch1-v1"

    test_set_folder = dataset_parent_dir / dataset_name / 'test'
    labels, test_image_paths = get_labels(test_set_folder / 'images.json')
    test_image_paths = [test_set_folder / path for path in test_image_paths]
    print(test_image_paths[:2])

    model = load_model(mlflow_model_dir)
    predictions = inference(test_image_paths, model)
    print(predictions, labels)
    metrics = {}
    for k in [1,3,5]:
        metrics.update({f'top{k}_acc': topk_acc(labels, predictions, k=k, labels=np.arange(predictions.shape[1]))})
    print(metrics)


def get_test_image_paths(data_dir: Path):
    res = list(data_dir.glob('*.jpg'))
    print(f'{len(res)} images for inference')
    return res


def load_model(mlflow_model_dir):
    return  mlflow.pyfunc.load_model(mlflow_model_dir)


def get_labels(coco_path):
    with open(coco_path) as f:
        coco = json.load(f)
        img_id_to_cate_id = {a['image_id']: a['category_id'] for a in coco['annotations']}
        labels = [img_id_to_cate_id[im['id']] - 1 for im in coco['images']]
        image_paths = [im['file_name'] for im in coco['images']]

    print(f'{len(image_paths)} images for inference')

    return labels, image_paths

def inference(image_paths, model, batch_size=64):
    def read_image(image_path):
        with open(image_path, "rb") as f:
            return f.read()

    predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i: i+batch_size]
        test_df = pd.DataFrame(
            data=[
                base64.encodebytes(read_image(image_path)).decode("utf-8")
                for image_path in batch
            ],
            columns=["image"],
        )
        batch_result = model.predict(test_df).to_json(orient="records")
        probs = np.array([img['probs'] for img in json.loads(batch_result)])
        predictions.append(probs)


    return np.concatenate(predictions, axis=0)



if __name__ == '__main__':
    main()
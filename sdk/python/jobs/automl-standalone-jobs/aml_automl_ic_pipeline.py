# Import required libraries
import argparse
import os, json
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.automl import SearchSpace, ClassificationPrimaryMetrics
from azure.ai.ml.sweep import (
    Choice,
    Choice,
    Uniform,
    BanditPolicy,
)

from azure.ai.ml import automl

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml import Input

from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError


compute_name = "tkjinnc6s"
experiment_name = "aml-automl-ic-experiments"
postfix = '0326'

def configure():
    credential = DefaultAzureCredential()
    ml_client = None
    try:
        ml_client = MLClient.from_config(credential)
    except Exception as ex:
        print(ex)
        # Enter details of your AML workspace
        print('use my own config')
        subscription_id = "a10177d2-ae16-41b0-9a88-7d9e4ed460a8"
        resource_group = "vision"
        workspace = "CustVisITing"
        ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    config_compute(ml_client)

    return ml_client


def config_compute(ml_client):
    try:
        _ = ml_client.compute.get(compute_name)
        print("Found existing compute target.")
    except ResourceNotFoundError:
        print("Creating a new compute target...")
        compute_config = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_NC6",
            idle_time_before_scale_down=120,
            min_instances=0,
            max_instances=4,
        )
        ml_client.begin_create_or_update(compute_config).result()


def create_ml_table_file(filename):
    """Create ML Table definition"""

    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def save_ml_table_file(output_path, mltable_file_contents):
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


def coco_to_jsonl_ic(coco_path, img_subfolder, uri_folder_data_asset, jsonl_path=''):
    if not jsonl_path:
        jsonl_path = Path(str(coco_path).replace('.json', '.jsonl'))
    if not jsonl_path.parent.exists():
        jsonl_path.parent.mkdirs(parents=False, exist_ok=True)

    jsonl_line_sample = {
      "image_url": uri_folder_data_asset.path,
      "label": "",
    }
    with open(jsonl_path, 'w') as fout:
        with open(coco_path) as fin:
            coco = json.load(fin)
            cate_id_to_name = {c['id']: c['name'] for c in coco['categories']}
            img_id_to_cate_name = {a['image_id']: cate_id_to_name[a['category_id']] for a in coco['annotations']}
            for im in coco['images']:
                line = dict(jsonl_line_sample)
                line['image_url']  += (img_subfolder + '/' + im['file_name'])
                line['label'] = img_id_to_cate_name[im['id']]
                fout.write(json.dumps(line) + "\n")
    return jsonl_path


def prepare_data(ml_client, dataset_name, desp, dataset_dir, train_folder,
    train_coco_fnm, val_folder, val_coco_fnm):

    # upload data
    my_data = Data(
      path=dataset_dir,
      type=AssetTypes.URI_FOLDER,
      description=desp,
      name=dataset_name,
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)

    # convert coco to JSONL
    train_mltable_path = dataset_dir / train_folder
    val_mltable_path = dataset_dir / val_folder
    train_jsonl_path = coco_to_jsonl_ic(train_mltable_path / train_coco_fnm, train_folder, uri_folder_data_asset, jsonl_path='')
    val_jsonl_path = coco_to_jsonl_ic(val_mltable_path / val_coco_fnm, val_folder, uri_folder_data_asset, jsonl_path='')

    # Create and save train mltable
    train_mltable_file_contents = create_ml_table_file(
        os.path.basename(train_jsonl_path)
    )
    save_ml_table_file(train_mltable_path, train_mltable_file_contents)

    # Create and save validation mltable
    validation_mltable_file_contents = create_ml_table_file(
        os.path.basename(val_jsonl_path)
    )
    save_ml_table_file(val_mltable_path, validation_mltable_file_contents)

    # Training MLTable defined locally, with local data to be uploaded
    my_training_data_input = Input(type=AssetTypes.MLTABLE, path=train_mltable_path)

    # Validation MLTable defined locally, with local data to be uploaded
    my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=val_mltable_path)

    # WITH REMOTE PATH: If available already in the cloud/workspace-blob-store
    # my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/train")
    # my_validation_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/valid")
    return my_training_data_input, my_validation_data_input


def submit_experiment(ml_client, experiment_name, train_input, val_input, job_name, tags={}):
    # Create the AutoML job with the related factory-function.

    image_classification_job = automl.image_classification(
        compute=compute_name,
        name=job_name,
        experiment_name=experiment_name,
        training_data=train_input,
        validation_data=val_input,
        target_column_name="label",
        primary_metric="accuracy",
        tags=tags,
    )

    image_classification_job.set_limits(
        max_trials=20,
        max_concurrent_trials=4,
    )


    # Submit the AutoML job
    returned_job = ml_client.jobs.create_or_update(
        image_classification_job
    )  # submit the job to the backend

    print(f"Created job: {returned_job}")


def main():

    parser = argparse.ArgumentParser(description="AML Automl IC")
    # parser.add_argument('job_filepath', type=pathlib.Path)
    parser.add_argument('--dataset_dir', '-d', type=Path, default='')
    parser.add_argument('--dataset_name', '-n', type=str, default='')
    parser.add_argument('--dataset_description', '-dd', type=str)

    parser.add_argument('--train_folder', '-tr', type=str, default='train')
    parser.add_argument('--val_folder', '-vr', type=str, default='val')

    parser.add_argument('--train_coco_fnm', '-tc', type=str, default='images.json')
    parser.add_argument('--val_coco_fnm', '-vc', type=str, default='images.json')


    # parser.add_argument('--verbose', '-v', action='store_true')
    # parser.add_argument('--very_verbose', '-vv', action='store_true')
    # parser.add_argument('--no_cache', action='store_true')

    args = parser.parse_args()

    ml_client = configure()

    train_input, val_input = prepare_data(ml_client, args.dataset_name, args.dataset_description, args.dataset_dir, args.train_folder,
      args.train_coco_fnm, args.val_folder, args.val_coco_fnm)

    submit_experiment(ml_client, experiment_name, train_input, val_input, job_name=f'{args.dataset_name}_aml_{postfix}')

if __name__ == "__main__":
    main()
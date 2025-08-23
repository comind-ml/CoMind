from pathlib import Path
from comind.config import Config
from .api import authenticate_kaggle_api
import json
import html
import kagglehub
import os
import shutil

def download_model(cfg: Config, model_id: str) -> Path:
    raw_model_id = model_id
    if len(model_id.split('/')) != 4:
        model_id = '/'.join(model_id.split('/')[:4])
        print(f"Warning: Model ID is not in the correct format, truncating to 4 parts: {model_id}")
    
    download_dir = cfg.agent_external_data_dir / model_id
    model_id = '/'.join([part[0].lower() + part[1:] for part in model_id.split('/')])

    if download_dir.exists() and len(list(download_dir.iterdir())) > 0:
        print(f"Download directory {download_dir} already exists, skipping download")
        return download_dir
    
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        target_dir = kagglehub.model_download(model_id)
        shutil.move(target_dir, download_dir)

        api = authenticate_kaggle_api()
        model_name = "/".join(model_id.split('/')[:2])
        metadata = api.model_get(model_name).to_dict()
        with open(cfg.agent_external_data_dir / raw_model_id / 'model-metadata.json', 'w') as f:
            json.dump(metadata, f)

    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        return None
    
    return download_dir

def get_model_metadata(cfg: Config, model_id: str) -> dict:
    download_dir = cfg.agent_external_data_dir / model_id

    assert download_dir.exists(), f"Download directory {download_dir} does not exist. Please run download_model first."

    api = authenticate_kaggle_api()
    metadata_file = api.get_model_metadata_file(download_dir)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_dataset_metadata(cfg: Config, dataset_id: str) -> dict:
    download_dir = cfg.agent_external_data_dir / dataset_id

    assert download_dir.exists(), f"Download directory {download_dir} does not exist. Please run download_dataset first."

    api = authenticate_kaggle_api()
    metadata_file = api.get_dataset_metadata_file(download_dir)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def download_dataset(cfg: Config, dataset_id: str) -> Path:
    download_dir = cfg.agent_external_data_dir / dataset_id

    if download_dir.exists() and len(list(download_dir.iterdir())) > 0:
        print(f"Download directory {download_dir} already exists, skipping download")
        return download_dir
    
    download_dir.mkdir(parents=True, exist_ok=True)

    api = authenticate_kaggle_api()
    api.dataset_download_files(dataset=dataset_id, path=download_dir, quiet=False, unzip=True)
    api.dataset_metadata(dataset=dataset_id, path=download_dir)

    with open(download_dir / 'dataset-metadata.json', 'r') as f:
        content = f.read()
        metadata = json.loads(html.unescape(content))
    
    with open(download_dir / 'dataset-metadata.json', 'w') as f:
        f.write(metadata)

    return download_dir
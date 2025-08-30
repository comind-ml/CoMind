import json
import requests
import pandas as pd
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import time

from .api import authenticate_kaggle_api
from comind.config import Config
from comind.utils import MetricValue, WorstMetricValue

def _init_session() -> tuple[requests.Session, dict]:
    session = requests.Session()
    url = "https://www.kaggle.com/competitions/"
    session.get(url)
    return session, { 'x-xsrf-token': session.cookies.get_dict()['XSRF-TOKEN'] }

def str_to_datetime(date_str: str) -> datetime:
    """Convert a string to a datetime object"""
    return pd.to_datetime(date_str).to_pydatetime()

def get_competition_info(competition_id: str) -> dict:
    session, headers = _init_session()
    url = "https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition"

    payload = {
        "competitionName": competition_id
    }

    response = session.post(url, data=json.dumps(payload), headers=headers)
    return response.json()

def list_kernels(competition_id: str) -> list[dict]:
    session, headers = _init_session()

    url = "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels"
    competition_info = get_competition_info(competition_id)

    payload = {
        "kernelFilterCriteria": {
            "search": "",
            "listRequest": {
                "competitionId": competition_info['id'],
                "sortBy": "HOTNESS",
                "pageSize": 50,
                "group": "EVERYONE",
                "page": 1,
                "modelIds": [],
                "modelInstanceIds": [],
                "excludeKernelIds": [],
                "tagIds": "",
                "excludeResultsFilesOutputs": False,
                "wantOutputFiles": False,
                "excludeNonAccessedDatasources": True
            }
        },
        "detailFilterCriteria": {
            "deletedAccessBehavior": "RETURN_NOTHING",
            "unauthorizedAccessBehavior": "RETURN_NOTHING",
            "excludeResultsFilesOutputs": False,
            "wantOutputFiles": False,
            "kernelIds": [],
            "outputFileTypes": [],
            "includeInvalidDataSources": False
        },
        "readMask": "pinnedKernels"
    }

    page = 1
    kernels = []

    while True:
        payload['kernelFilterCriteria']['listRequest']['page'] = page
        response = session.post(url, data=json.dumps(payload), headers=headers).json()

        if 'kernels' not in response or len(response['kernels']) == 0:
            break 
        
        id = response['kernels'][0]['id']
        if id in [kernel['id'] for kernel in kernels]:
            break

        deadline = str_to_datetime(competition_info['deadline'])
        for kernel in response['kernels']:
            created_at = str_to_datetime(kernel['scriptVersionDateCreated'])
            
            if created_at > deadline:
                continue
            kernels.append(kernel)
            
        page += 1
    
    return kernels

def download_kernel_output(cfg: Config, kernel_id: str) -> Path:
    api = authenticate_kaggle_api()
    download_dir = cfg.agent_external_data_dir / kernel_id
    if download_dir.exists() and len(list(download_dir.iterdir())) > 0:
        print(f"Download directory {download_dir} already exists, skipping download")
        return download_dir
    
    download_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel=kernel_id, path=download_dir, quiet=False)
    return download_dir

def download_kernels(cfg: Config, is_lower_better: bool) -> list[Path]:
    kernels = list_kernels(cfg.competition_id)

    hottest_kernels = sorted(deepcopy(kernels), key=lambda x: x["totalVotes"] if "totalVotes" in x else 0, reverse=True)

    best_kernels = sorted(deepcopy(kernels), key=lambda x: MetricValue(x["bestPublicScore"], maximize=not is_lower_better) if "bestPublicScore" in x else WorstMetricValue(), reverse=True)

    k = cfg.agent_max_referred_kernels
    hottest_kernels = hottest_kernels[:min(k, len(hottest_kernels))]
    best_kernels = best_kernels[:min(k, len(best_kernels))]

    if len(best_kernels) > 0:
        print(best_kernels[0]['bestPublicScore'])

    # Remove duplicates by scriptUrl
    seen_urls = set()
    unique_kernels = []
    for kernel in hottest_kernels + best_kernels:
        if kernel['scriptUrl'] not in seen_urls:
            seen_urls.add(kernel['scriptUrl'])
            unique_kernels.append(kernel)
    kernels = unique_kernels

    print(f"Downloading {len(kernels)} kernels")

    api = authenticate_kaggle_api()
    download_dir = cfg.agent_external_data_dir

    download_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for kernel in kernels:
        kernel_ref = kernel['scriptUrl'].replace('/code/', '')
        target_dir = download_dir / kernel_ref

        if target_dir.exists() and len(list(target_dir.iterdir())) > 0:
            print(f"Kernel {kernel_ref} already exists, skipping download")
            with open(target_dir / 'kernel-metadata.json', 'r') as f:   
                metadata = json.load(f)
            results.append(target_dir / metadata['code_file'])
            continue

        while True:
            wait_time = 1
            try: 
                api.kernels_pull(kernel_ref, target_dir, metadata=True, quiet=False)

                with open(target_dir / 'kernel-metadata.json', 'r') as f:
                    metadata = json.load(f)

                if "bestPublicScore" in kernel:
                    metadata['score'] = kernel['bestPublicScore'] 
                
                assert "dataset_sources" in metadata, "Dataset sources not found in metadata"
                
                with open(target_dir / 'kernel-metadata.json', 'w') as f:
                    json.dump(metadata, f)
                
                api.kernels_output(kernel=kernel_ref, path=target_dir, quiet=False)
                results.append(target_dir / metadata['code_file'])
                break

            except Exception as e:
                wait_time *= 2
                print(f"Error downloading kernel {kernel_ref}: {e}, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        
        time.sleep(1)

    return results

def get_kernel_metadata(kernel_path: Path) -> dict:
    if kernel_path.is_dir():
        with open(kernel_path / 'kernel-metadata.json', 'r') as f:
            metadata = json.load(f)
    else:
        with open(kernel_path.parent / 'kernel-metadata.json', 'r') as f:
            metadata = json.load(f)

    return metadata
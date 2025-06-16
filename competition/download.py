import json
import os
import shutil
import webbrowser
import requests
import argparse

from pathlib import Path
from tqdm import tqdm

from comind.utils.generic import str_to_datetime, read_jupyter_notebook

def authenticate_kaggle_api() -> "KaggleApi":
    """Authenticates the Kaggle API and returns an authenticated API object, or raises an error if authentication fails."""
    try:
        # only import when necessary; otherwise kaggle asks for API key on import
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competitions_list()  # a cheap op that requires authentication
        return api
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        raise PermissionError(
            "Kaggle authentication failed! Please ensure you have valid Kaggle API credentials "
            "configured. Refer to the Kaggle API documentation for guidance on setting up "
            "your API token."
        ) from e

def download_dataset(
    competition_id: str,
    download_dir: Path,
    quiet: bool = False,
    force: bool = False,
):
    """Downloads the competition data as a zip file using the Kaggle API."""

    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    print(f"Downloading the dataset for `{competition_id}` to `{download_dir}`...")

    api = authenticate_kaggle_api()

    # only import when necessary; otherwise kaggle asks for API key on import

    while True:
        try:
            api.competition_download_files(
                competition=competition_id,
                path=download_dir,
                quiet=quiet,
                force=force,
            )
        except Exception as e:
            if "Forbidden" in str(e):
                print("You must accept the competition rules before downloading the dataset.")

                response = input("Would you like to open the competition page in your browser now? (y/n): ")
                if response.lower() != "y":
                    raise RuntimeError("You must accept the competition rules before downloading the dataset.")

                webbrowser.open(f"https://www.kaggle.com/c/{competition_id}/rules")
                input("Press Enter to continue after you have accepted the rules...")
                continue
            
            raise e
        break
        
    zip_files = list(download_dir.glob("*.zip"))

    assert (
        len(zip_files) == 1
    ), f"Expected to download a single zip file, but found {len(zip_files)} zip files."

    zip_file = zip_files[0]

    shutil.unpack_archive(zip_file, download_dir / "input")

def _init_session() -> tuple[requests.Session, dict]:
    session = requests.Session()
    url = "https://www.kaggle.com/competitions/"
    session.get(url)
    return session, { 'x-xsrf-token': session.cookies.get_dict()['XSRF-TOKEN'] }

def download_description(
    competition_id: str,
    download_dir: Path,
    force: bool = False,
):
    """ Downloads the competition description."""
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    print(f"Downloading the description for `{competition_id}` to `{download_dir}`...")

    competition_info = _get_competition_info(competition_id)

    if not (download_dir / "description.md").exists() or force:
        session, headers = _init_session()
        url = "https://www.kaggle.com/api/i/competitions.PageService/ListPages"

        payload = {
            "competitionId": competition_info['id']
        }

        response = session.post(url, data=json.dumps(payload), headers=headers).json()

        content = ""
        for page in response['pages']:
            if page['name'] == 'foundational-rules':
                continue
            content += f"# {page['postTitle']}\n"
            content += page['content'] + "\n\n"

        with open(download_dir / "description.md", 'w', encoding='utf-8') as f:
            f.write(content)
    
        print(f"Description has been saved to `{download_dir / 'description.md'}`")
    else:
        print(f"Description already exists in `{download_dir / 'description.md'}`")

def _get_competition_info(competition_id: str) -> dict:
    session, headers = _init_session()
    url = "https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition"

    payload = {
        "competitionName": competition_id
    }

    response = session.post(url, data=json.dumps(payload), headers=headers)
    return response.json()

def _list_kernels(competition_id: str, exclude_after_deadline: bool = True) -> list[dict]:
    session, headers = _init_session()

    url = "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels"
    competition_info = _get_competition_info(competition_id)

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
            
            if exclude_after_deadline and created_at > deadline:
                continue
            kernels.append(kernel)
            
        page += 1
    
    return kernels

def _list_topics(competition_id: str, exclude_after_deadline: bool = True) -> list[dict]:
    session, headers = _init_session()
    url = "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId"

    competition_info = _get_competition_info(competition_id)

    payload = {
        "category": "TOPIC_LIST_CATEGORY_ALL",
        "group": "TOPIC_LIST_GROUP_ALL",
        "customGroupingIds": [],
        "author": "TOPIC_LIST_AUTHOR_UNSPECIFIED",
        "myActivity": "TOPIC_LIST_MY_ACTIVITY_UNSPECIFIED",
        "recency": "TOPIC_LIST_RECENCY_UNSPECIFIED",
        "filterCategoryIds": [],
        "searchQuery": "",
        "sortBy": "TOPIC_LIST_SORT_BY_UNSPECIFIED",
        "page": 1,
        "forumId": competition_info['forumId']
    }

    page = 1
    topics = []

    while True:
        payload['page'] = page
        response = session.post(url, data=json.dumps(payload), headers=headers).json()

        if 'topics' not in response or len(response['topics']) == 0:
            break

        deadline = str_to_datetime(competition_info['deadline'])
        for topic in response['topics']:
            posted_at = str_to_datetime(topic['postDate'])
            if exclude_after_deadline and posted_at > deadline:
                continue
            topics.append(topic)
    
        page += 1
    
    return topics

def read_discussion(discussion: dict) -> str:
    def get_content(obj) -> str:
        if "rawMarkdown" in obj:
            return obj["rawMarkdown"]
        elif "content" in obj:
            return obj["content"]
        else:
            return ""

    def walk(obj) -> list[str]:
        comments_field = "comments" if "comments" in obj else "replies"
        if comments_field not in obj:
            return []
        result = []
        for comment in obj[comments_field]:
            if "isDeleted" in comment and comment["isDeleted"]:
                continue
            if 'author' not in comment or 'displayName' not in comment['author']:
                continue
            if 'tier' not in comment['author']:
                comment['author']['tier'] = 'N/A'
            result.append(f"    + ({comment['author']['displayName']} <TIER: {comment['author']['tier']}>) " + get_content(comment))
            result += ["    " + line for line in walk(comment)]
        return result

    discussion = discussion['forumTopic']
    
    if 'authorPerformanceTier' not in discussion:
        discussion['authorPerformanceTier'] = 'N/A'
    result = [f"# {discussion['name']}"]

    result += [f"({discussion['authorUserDisplayName']} <TIER: {discussion['authorPerformanceTier']}>) " + get_content(discussion["firstMessage"])]
    result += walk(discussion)
    return "\n".join(result)

def download_kernels(
    competition_id: str,
    download_dir: Path,
    exclude_after_deadline: bool = True,
    force: bool = False,
) -> Path:
    """Downloads all kernels for a competition using Kaggle API with pagination."""
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    api = authenticate_kaggle_api()

    kernel_dir = download_dir / "kernels"
    kernel_dir.mkdir(exist_ok=True)

    kernels = _list_kernels(competition_id, exclude_after_deadline)

    info = []

    # Download each kernel in current page
    for kernel in tqdm(kernels, desc="Downloading kernels"):
        try:
            kernel_ref = kernel['scriptUrl'].replace('/code/', '')
            file_name = kernel_ref.split('/')[-1]
            kernel_path = kernel_dir / f"{kernel['scriptVersionId']}.txt"

            metadata = {
                "title": kernel['title'],
                "votes": kernel["totalVotes"] if "totalVotes" in kernel else 0,
                "id": str(kernel["scriptVersionId"]),
                "created_at": kernel['scriptVersionDateCreated'],
                "bestPublicScore": kernel["bestPublicScore"] if "bestPublicScore" in kernel else None,
            }

            info.append(metadata)
            
            if not kernel_path.exists() or force:
                api.kernels_pull(
                    kernel_ref,
                    kernel_dir,
                    metadata=False,
                    quiet=False
                )

                raw_path = kernel_dir / f"{file_name}.ipynb"
                content = read_jupyter_notebook(raw_path)
                os.remove(raw_path)
                with open(kernel_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        except Exception as e:
            print(f"Failed to download kernel {kernel_ref}: {str(e)}")
            continue

    with open(kernel_dir / 'info.json', 'w') as f:
        json.dump(info, f, indent=4)

    print(f"All kernels have been saved to `{kernel_dir}`")

def download_discussions(
    competition_id: str,
    download_dir: Path,
    exclude_after_deadline: bool = True,
    force: bool = False,
) -> Path:
    """Downloads the discussions for a competition."""

    discussion_dir = download_dir / "discussions"
    discussion_dir.mkdir(exist_ok=True)
    
    print(f"Downloading discussions for `{competition_id}` to `{download_dir}`...")

    topics = _list_topics(competition_id, exclude_after_deadline)

    session, headers = _init_session()
    url = "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById"

    info = []
    for topic in tqdm(topics, desc="Downloading discussions"):
        payload = {
            "forumTopicId": topic['id'],
            "includeComments": True,
        }

        metadata = {
            "title": topic["title"],
            "votes": topic["votes"] if "votes" in topic else 0,
            "id": str(topic["id"]),
            "created_at": topic["postDate"],
        }

        info.append(metadata)

        if not discussion_dir / f"{topic['id']}.txt".exists() or force:
            try:
                response = session.post(url, data=json.dumps(payload), headers=headers).json()
                content = read_discussion(response)

                with open(discussion_dir / f"{topic['id']}.txt", 'w', encoding='utf-8') as f:
                    f.write(content)
                
        
            except Exception as e:
                print(f"Failed to download discussion {topic['id']}: {str(e)}")
                continue
    
    with open(discussion_dir / 'info.json', 'w') as f:
        json.dump(info, f, indent=4)

    print(f"All discussions have been saved to `{discussion_dir}`")
        
if __name__ == "__main__":
    authenticate_kaggle_api()

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", type=str, required=True)
    parser.add_argument("--download-dir", "-dir", type=str, default=Path(__file__).parent)
    parser.add_argument("--exclude-after-deadline", "-e", type=bool, default=True)
    parser.add_argument("--dataset", "-d", default=False, action="store_true")
    parser.add_argument("--kernels", "-k", default=False, action="store_true")
    parser.add_argument("--discussions", "-t", default=False, action="store_true")
    parser.add_argument("--force", "-f", default=False, action="store_true")

    args = parser.parse_args()

    competition = args.competition
    download_dir = Path(args.download_dir) / competition

    download_description(
        competition_id=competition,
        download_dir=Path(download_dir),
        force=args.force,
    )

    if args.dataset:
        download_dataset(
            competition_id=competition,
            download_dir=Path(download_dir),
            quiet=False,
            force=args.force,
        )
    
    if args.kernels:
        download_kernels(
            competition_id=competition,
            download_dir=Path(download_dir),
            exclude_after_deadline=args.exclude_after_deadline,
            force=args.force,
        )

    if args.discussions:
        download_discussions(
            competition_id=competition,
            download_dir=Path(download_dir),
            exclude_after_deadline=args.exclude_after_deadline,
            force=args.force,
        )

from comind.kaggle.api import authenticate_kaggle_api
from tempfile import TemporaryDirectory
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

def get_leaderboard(competition_id: str) -> pd.DataFrame:
    api = authenticate_kaggle_api()

    with TemporaryDirectory() as tmpdirname:
        api.competition_leaderboard_download(competition_id, path=tmpdirname)
        zip_file_path = Path(tmpdirname) / f"{competition_id}.zip"
        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        files = [f for f in Path(tmpdirname).glob("*.csv")]
        assert len(files) == 1, "Expected exactly one CSV file in the leaderboard zip."
        leaderboard = pd.read_csv(files[0])
    
    return leaderboard

def get_rank(leaderboard: pd.DataFrame, score: float) -> float:
    best_score = leaderboard.loc[leaderboard['Rank'] == 1, 'Score'].iloc[0]
    min_score = leaderboard['Score'].min()
    is_lower_better = best_score == min_score

    scores = leaderboard['Score'].dropna().to_numpy()
    rank = (scores < score).sum() if is_lower_better else (scores > score).sum()

    return rank / len(scores)

if __name__ == "__main__":
    leaderboard = get_leaderboard("aerial-cactus-identification")  # Example usage
    print(get_rank(leaderboard, 0.9))  # Example score
    print(get_rank(leaderboard, 1.0))
from .metric import MetricValue, WorstMetricValue
from .llm import Conversation, query_llm, query_llm_raw, extract_fields
from .data_preview import generate
from .execute import Executor, ExecutionResult
from pathlib import Path
import shutil
import zipfile

def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
        use_symlinks (bool): If True, create symlinks instead of copying files
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            # Use absolute path to ensure symlink remains valid
            dest_f.symlink_to(src.resolve())
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            # Use absolute path to ensure symlink remains valid
            dest_f.symlink_to(f.resolve())
        elif f.is_dir():
            # Recursively copy directory with same symlink preference
            dest_f.mkdir()
            copytree(f, dest_f, use_symlinks=use_symlinks)
        else:
            shutil.copyfile(f, dest_f)

def extract_archives(path: Path):
    """
    unzips all .zip files within `path` and cleans up task dir

    [TODO] handle nested zips
    """
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            # [TODO] maybe add an extra check to see if zip file content matches the colliding file
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
            continue

        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)


        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()

__all__ = ["MetricValue", "WorstMetricValue", "Conversation", "query_llm", "query_llm_raw", "extract_fields", "copytree", "generate", "Executor", "ExecutionResult", "extract_archives"]
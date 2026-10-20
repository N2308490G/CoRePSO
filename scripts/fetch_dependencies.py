"""Fetch explicit, pinned upstream dependencies without running experiments."""
import argparse
import base64
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "evox": ("https://github.com/EMI-Group/evox.git",
             "95d9bda6fe315d74f543579150b0708ade8bec5b"),
    "metabox": ("https://github.com/MetaEvo/MetaBox.git",
                "5565a281ce4e03efc1f6a41230507d5b66637ace"),
}
CEC_COMMIT = "de20505283b76ec6bf17a2e8fc6052e655830691"


def fetch_cec():
    url = ("https://api.github.com/repos/P-N-Suganthan/2022-SO-BO/contents/"
           f"Python-CEC2022.zip?ref={CEC_COMMIT}")
    request = urllib.request.Request(url, headers={"User-Agent": "CoRePSO"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(body["content"])))
    files = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        wanted = member.filename == "Python/CEC2022.py" or (
            member.filename.startswith("Python/input_data/") and path.suffix == ".txt"
        )
        if not wanted or member.is_dir():
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe upstream archive member")
        target = ROOT.joinpath(*path.parts)
        data = archive.read(member)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f"Existing file differs; preserved without replacement: {target}")
        files[target] = data
    if ROOT / "Python/CEC2022.py" not in files:
        raise RuntimeError("CEC archive is missing its reference implementation")
    for target, data in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Fetched {len(files)} CEC reference/data files at {CEC_COMMIT}")


def fetch_git(name):
    url, commit = PINS[name]
    folder = ROOT / "baselines" / ("MetaBox" if name == "metabox" else "evox")
    if folder.exists():
        head = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                       text=True).strip()
        if head != commit:
            raise RuntimeError(f"Existing {name} checkout has a different revision; preserved")
        if name == "metabox":
            patch = str(ROOT / "patches/metabox_accounting_and_bounds.patch")
            check = subprocess.run(["git", "-C", str(folder), "apply", "--reverse",
                                    "--check", patch], capture_output=True)
            if check.returncode:
                raise RuntimeError("Existing MetaBox checkout does not contain the recorded patch")
        print(f"Using existing {name} at {commit}")
        return
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    subprocess.run(["git", "init", str(folder)], check=True)
    subprocess.run(["git", "-C", str(folder), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(folder), "fetch", "--depth", "1", "origin", commit],
                   check=True)
    subprocess.run(["git", "-C", str(folder), "checkout", "--detach", commit], check=True)
    if name == "metabox":
        subprocess.run(["git", "-C", str(folder), "apply",
                        str(ROOT / "patches/metabox_accounting_and_bounds.patch")], check=True)
    print(f"Fetched {name} at {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dependency", choices=["autopso", "cec", "evox", "metabox"])
    args = parser.parse_args()
    if args.dependency == "autopso":
        subprocess.run([sys.executable, str(ROOT / "src/fetch_upstream.py")], check=True)
    elif args.dependency == "cec":
        fetch_cec()
    else:
        fetch_git(args.dependency)


if __name__ == "__main__":
    main()

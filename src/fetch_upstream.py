"""Fetch the author's unchanged core at a pinned commit, then verify it.

This archive does not redistribute the author's code. Review the upstream
repository's terms and obtain permission as appropriate before redistribution.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

COMMIT='d99bbcdd944d6697fd3466d51c9d97a9fd6d1133'
BLOB='0e0158e6c58f335f363af92a4f1fc5f4b4aa697f'
URL=f'https://raw.githubusercontent.com/EMI-Group/autopso/{COMMIT}/src/autopso/algorithms/pytorch/autopso.py'

def verify(data: bytes) -> None:
    got=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if got!=BLOB:
        raise ValueError(f'Wrong upstream file: expected Git blob {BLOB}, got {got}. No file installed.')

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-file',type=Path,help='Use an already obtained exact source file; no network.')
    parser.add_argument('--dest',type=Path,default=Path(__file__).parent/'vendor'/'autopso.py')
    args=parser.parse_args()
    try:
        if args.from_file:
            data=args.from_file.read_bytes()
        else:
            req=urllib.request.Request(URL,headers={'User-Agent':'AutoPSO-reproducibility-check/1.0'})
            try:
                with urllib.request.urlopen(req,timeout=30) as response:
                    data=response.read(1024*1024)
            except urllib.error.URLError:
                # Use GitHub's contents endpoint if its raw-file host is unavailable.
                # Both routes verify TLS and the same pinned Git blob.
                api=(f'https://api.github.com/repos/EMI-Group/autopso/contents/'
                     f'src/autopso/algorithms/pytorch/autopso.py?ref={COMMIT}')
                req=urllib.request.Request(api,headers={'User-Agent':'CoRePSO'})
                with urllib.request.urlopen(req,timeout=30) as response:
                    data=base64.b64decode(json.load(response)['content'])
        verify(data)
        args.dest.parent.mkdir(parents=True,exist_ok=True)
        temp=args.dest.with_suffix('.tmp');temp.write_bytes(data);temp.replace(args.dest)
    except (OSError,ValueError,urllib.error.URLError) as exc:
        print(f'Cannot install upstream core: {exc}',file=sys.stderr)
        print(f'Obtain the exact file from: {URL}',file=sys.stderr)
        print('Then run: python fetch_upstream.py --from-file PATH_TO_FILE',file=sys.stderr)
        return 1
    print(f'Installed hash-verified original core ({len(data)} bytes): {args.dest}')
    return 0
if __name__=='__main__':raise SystemExit(main())

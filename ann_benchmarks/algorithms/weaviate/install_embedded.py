"""Install the exact binary expected by weaviate-client's embedded mode."""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    with urllib.request.urlopen(args.url) as response:
        archive = response.read()

    actual_sha256 = hashlib.sha256(archive).hexdigest()
    if actual_sha256 != args.sha256:
        raise RuntimeError(
            f"embedded Weaviate archive checksum mismatch: "
            f"expected {args.sha256}, got {actual_sha256}"
        )

    # This name is the contract used by weaviate-client 3.16.0's EmbeddedDB:
    # sha256() is calculated over the exact version string passed by module.py.
    version_digest = hashlib.sha256(args.version.encode("utf-8")).hexdigest()
    destination = args.cache_dir / f"weaviate-v{args.version}-{version_digest}"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        binary = bundle.extractfile("weaviate")
        if binary is None:
            raise RuntimeError("embedded Weaviate archive has no 'weaviate' binary")
        with destination.open("wb") as output:
            while chunk := binary.read(1024 * 1024):
                output.write(chunk)

    destination.chmod(0o755)
    print(f"installed embedded Weaviate at {destination}")


if __name__ == "__main__":
    main()

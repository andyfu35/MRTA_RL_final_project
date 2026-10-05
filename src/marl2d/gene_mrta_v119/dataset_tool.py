from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request

from .benchmark import (
    PUBLIC_SOURCE_URL,
    describe_manifest,
    load_manifest,
)


class _LinkParser(
    HTMLParser
):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.links: list[
            str
        ] = []

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if (
                key.lower()
                == "href"
                and value
            ):
                self.links.append(
                    str(
                        value
                    )
                )


def _read_url(
    url: str,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "MRTA-RL-MTRPD-benchmark-importer/1.0"
            )
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        return response.read()


def download_public_supplement(
    output_dir: Path,
    *,
    root_url: str = (
        PUBLIC_SOURCE_URL
    ),
) -> list[
    Path
]:
    """
    Best-effort crawler for the original public supplement.

    The historical supplement host is not mirrored into this repository.
    This function intentionally preserves raw files exactly as served.
    """
    target_root = Path(
        output_dir
    )
    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = [
        root_url
    ]
    visited: set[
        str
    ] = set()
    downloaded: list[
        Path
    ] = []

    parsed_root = urllib.parse.urlparse(
        root_url
    )
    root_prefix = root_url.rstrip(
        "/"
    ) + "/"

    while pending:
        url = pending.pop(
            0
        )
        if url in visited:
            continue
        visited.add(
            url
        )

        body = _read_url(
            url
        )
        parsed = urllib.parse.urlparse(
            url
        )
        content_path = parsed.path

        is_directory = (
            url.endswith(
                "/"
            )
            or content_path.endswith(
                "/"
            )
        )
        if is_directory:
            parser = _LinkParser()
            try:
                parser.feed(
                    body.decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                parser.links = []

            for href in parser.links:
                absolute = (
                    urllib.parse.urljoin(
                        url,
                        href,
                    )
                )
                candidate = (
                    urllib.parse.urlparse(
                        absolute
                    )
                )
                if (
                    candidate.scheme
                    not in {
                        "http",
                        "https",
                    }
                    or candidate.netloc
                    != parsed_root.netloc
                    or not absolute.startswith(
                        root_prefix
                    )
                ):
                    continue
                if absolute in visited:
                    continue
                if href in {
                    "../",
                    "./",
                }:
                    continue
                pending.append(
                    absolute
                )
            continue

        relative = content_path
        root_path = parsed_root.path
        if relative.startswith(
            root_path
        ):
            relative = relative[
                len(
                    root_path
                ):
            ]
        relative = relative.lstrip(
            "/"
        )
        if not relative:
            continue

        target = (
            target_root
            / relative
        )
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        target.write_bytes(
            body
        )
        downloaded.append(
            target
        )
        print(
            "V119_DOWNLOAD "
            f"{target}",
            flush=True,
        )

    return downloaded


def inspect_raw(
    raw_dir: Path,
    *,
    head_lines: int = 12,
) -> None:
    root = Path(
        raw_dir
    )
    files = sorted(
        path
        for path in root.rglob(
            "*"
        )
        if path.is_file()
    )
    print(
        "V119_RAW_FILE_COUNT="
        f"{len(files)}",
        flush=True,
    )
    for path in files:
        relative = path.relative_to(
            root
        )
        print(
            "V119_RAW_FILE="
            f"{relative} "
            f"bytes={path.stat().st_size}",
            flush=True,
        )
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            continue
        lines = text.splitlines()
        for line in lines[
            :head_lines
        ]:
            print(
                "  "
                + line[
                    :240
                ],
                flush=True,
            )


def describe(
    manifest: Path,
) -> None:
    instances = load_manifest(
        manifest
    )
    print(
        "V119_MANIFEST "
        + json.dumps(
            describe_manifest(
                instances
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    download = sub.add_parser(
        "download"
    )
    download.add_argument(
        "--output-dir",
        required=True,
    )
    download.add_argument(
        "--root-url",
        default=(
            PUBLIC_SOURCE_URL
        ),
    )

    inspect = sub.add_parser(
        "inspect-raw"
    )
    inspect.add_argument(
        "--raw-dir",
        required=True,
    )
    inspect.add_argument(
        "--head-lines",
        type=int,
        default=12,
    )

    info = sub.add_parser(
        "describe"
    )
    info.add_argument(
        "--manifest",
        required=True,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "download":
        try:
            files = (
                download_public_supplement(
                    Path(
                        args.output_dir
                    ),
                    root_url=(
                        args.root_url
                    ),
                )
            )
        except Exception as exc:
            raise SystemExit(
                "Could not download the historical MTRPD supplement. "
                "If the original host is unavailable, obtain the supplement "
                "manually and place it in the raw directory, then run "
                "inspect-raw. Original source: "
                f"{args.root_url}\n{type(exc).__name__}: {exc}"
            ) from exc
        print(
            "V119_DOWNLOAD_COMPLETE="
            f"{len(files)}",
            flush=True,
        )
    elif args.command == "inspect-raw":
        inspect_raw(
            Path(
                args.raw_dir
            ),
            head_lines=(
                args.head_lines
            ),
        )
    elif args.command == "describe":
        describe(
            Path(
                args.manifest
            )
        )


if __name__ == "__main__":
    main()

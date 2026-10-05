from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import ssl
import subprocess
import urllib.parse
import urllib.request

import truststore

from .benchmark import (
    PUBLIC_SOURCE_URL,
    describe_manifest,
    load_manifest,
)
from .fixture import (
    write_smoke_manifest,
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


def _system_ssl_context():
    # Use the native macOS/Windows/Linux trust store instead of relying on
    # the Python.org framework bundle. TLS verification remains enabled.
    return truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )


def _curl_verified_read(
    url: str,
) -> bytes:
    parsed = urllib.parse.urlparse(
        url
    )
    if parsed.scheme != "https":
        raise RuntimeError(
            "curl fallback only supports HTTPS"
        )
    result = subprocess.run(
        [
            "/usr/bin/curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "30",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return bytes(
        result.stdout
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
    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
            context=(
                _system_ssl_context()
                if url.startswith(
                    "https://"
                )
                else None
            ),
        ) as response:
            return response.read()
    except Exception as urllib_exc:
        # macOS command-line curl uses the system trust configuration via
        # SecureTransport/Security.framework. Use it only as a second,
        # still-verified HTTPS path for Wayback.
        parsed = urllib.parse.urlparse(
            url
        )
        if (
            parsed.scheme == "https"
            and parsed.netloc
            == "web.archive.org"
        ):
            try:
                payload = _curl_verified_read(
                    url
                )
                print(
                    "V119_TLS_FALLBACK=system_curl "
                    f"url={url}",
                    flush=True,
                )
                return payload
            except Exception:
                pass
        raise urllib_exc


def _safe_relative_path(
    original_url: str,
    *,
    root_path: str,
) -> str:
    parsed = urllib.parse.urlparse(
        original_url
    )
    relative = parsed.path
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
        relative = "index.html"
    return relative


def download_wayback_supplement(
    output_dir: Path,
    *,
    root_url: str = (
        PUBLIC_SOURCE_URL
    ),
) -> list[
    Path
]:
    """
    Recover the historical public MTRPD supplement through the Internet
    Archive CDX index when the original host no longer resolves.

    Raw archived responses are requested with the id_ modifier so that saved
    files are not rewritten by the Wayback UI.
    """
    target_root = Path(
        output_dir
    )
    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed_root = urllib.parse.urlparse(
        root_url
    )
    host = parsed_root.netloc
    root_path = parsed_root.path
    if not root_path.endswith(
        "/"
    ):
        root_path += "/"

    wildcard = (
        f"{host}{root_path}*"
    )
    params = urllib.parse.urlencode(
        [
            ("url", wildcard),
            ("output", "json"),
            ("fl", "timestamp,original,statuscode,mimetype,digest"),
            ("filter", "statuscode:200"),
            ("collapse", "urlkey"),
        ]
    )
    cdx_url = (
        "https://web.archive.org/cdx/search/cdx?"
        + params
    )
    body = _read_url(
        cdx_url
    )
    rows = json.loads(
        body.decode(
            "utf-8"
        )
    )
    if (
        not isinstance(
            rows,
            list,
        )
        or len(
            rows
        )
        <= 1
    ):
        raise RuntimeError(
            "Wayback CDX returned no archived MTRPD supplement files"
        )

    header = rows[
        0
    ]
    index = {
        name: i
        for i, name
        in enumerate(
            header
        )
    }
    required = {
        "timestamp",
        "original",
        "statuscode",
    }
    if not required.issubset(
        index
    ):
        raise RuntimeError(
            "Unexpected Wayback CDX response schema"
        )

    downloaded: list[
        Path
    ] = []
    seen_paths: set[
        str
    ] = set()

    for row in rows[
        1:
    ]:
        timestamp = str(
            row[
                index[
                    "timestamp"
                ]
            ]
        )
        original = str(
            row[
                index[
                    "original"
                ]
            ]
        )
        relative = _safe_relative_path(
            original,
            root_path=(
                root_path
            ),
        )
        if relative in seen_paths:
            continue
        seen_paths.add(
            relative
        )

        archive_url = (
            "https://web.archive.org/web/"
            f"{timestamp}id_/{original}"
        )
        try:
            payload = _read_url(
                archive_url
            )
        except Exception as exc:
            print(
                "V119_WAYBACK_SKIP "
                f"{original} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
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
            payload
        )
        downloaded.append(
            target
        )
        print(
            "V119_WAYBACK "
            f"{target} "
            f"timestamp={timestamp}",
            flush=True,
        )

    if not downloaded:
        raise RuntimeError(
            "Wayback listed MTRPD resources but none could be downloaded"
        )

    metadata = {
        "source_root": (
            root_url
        ),
        "wayback_cdx": (
            cdx_url
        ),
        "file_count": len(
            downloaded
        ),
        "files": [
            str(
                path.relative_to(
                    target_root
                )
            )
            for path in downloaded
        ],
    }
    (
        target_root
        / "_WAYBACK_RECOVERY.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return downloaded


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

    smoke = sub.add_parser(
        "make-smoke"
    )
    smoke.add_argument(
        "--output",
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
            source = "live"
        except Exception as live_exc:
            print(
                "V119_LIVE_DOWNLOAD_FAILED "
                f"{type(live_exc).__name__}: {live_exc}",
                flush=True,
            )
            print(
                "V119_DOWNLOAD_FALLBACK=wayback",
                flush=True,
            )
            try:
                files = (
                    download_wayback_supplement(
                        Path(
                            args.output_dir
                        ),
                        root_url=(
                            args.root_url
                        ),
                    )
                )
                source = "wayback"
            except Exception as archive_exc:
                raise SystemExit(
                    "Could not download the historical MTRPD supplement "
                    "from either the original host or the Internet Archive. "
                    f"Original source: {args.root_url}\n"
                    f"Live error: {type(live_exc).__name__}: {live_exc}\n"
                    f"Wayback error: {type(archive_exc).__name__}: {archive_exc}"
                ) from archive_exc
        print(
            "V119_DOWNLOAD_SOURCE="
            f"{source}",
            flush=True,
        )
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
    elif args.command == "make-smoke":
        target = Path(
            args.output
        )
        write_smoke_manifest(
            target
        )
        print(
            f"V119_SMOKE_MANIFEST={target}",
            flush=True,
        )


if __name__ == "__main__":
    main()

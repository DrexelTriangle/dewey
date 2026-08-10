from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_COLLECTION_ID = "81448731180004721"
DEFAULT_VID = "01DRXU_INST:01DRXU_NDE"
DEFAULT_INST = "01DRXU_INST"
DEFAULT_SCOPE = "EverythingWResearch"
PRIMO_BASE = "https://drexel.primo.exlibrisgroup.com"
ALMA_BASE = "https://na04.alma.exlibrisgroup.com"
USER_AGENT = "dewey-primo-pdf-scraper/1.0 (+local research archiving)"


class PrimoClient:
    def __init__(self, delay: float, retries: int) -> None:
        self.delay = delay
        self.retries = retries
        self.request_count = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return self._request_json(url, method="GET")

    def post_json(self, url: str, params: dict[str, Any], payload: Any) -> Any:
        body = json.dumps(payload).encode("utf-8")
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        return self._request_json(full_url, method="POST", data=body)

    def download(self, url: str, target: Path) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/pdf,*/*",
            },
        )
        for attempt in range(self.retries + 1):
            try:
                self._sleep()
                with urllib.request.urlopen(request, timeout=120) as response:
                    with partial.open("wb") as handle:
                        total = 0
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            total += len(chunk)
                partial.replace(target)
                return total
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"download failed for {url}: {exc}") from exc
                time.sleep(2**attempt)
        raise RuntimeError(f"download failed for {url}")

    def _request_json(self, url: str, method: str, data: bytes | None = None) -> Any:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
        }
        if data is not None:
            headers["Content-Type"] = "application/json;charset=UTF-8"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        for attempt in range(self.retries + 1):
            try:
                self._sleep()
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.retries:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc
                time.sleep(2**attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"{method} {url} failed: {exc}") from exc
                time.sleep(2**attempt)
        raise RuntimeError(f"{method} {url} failed")

    def _sleep(self) -> None:
        if self.request_count and self.delay > 0:
            time.sleep(self.delay)
        self.request_count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape all accessible PDF articles from Drexel's Triangle Primo collection."
    )
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--out-dir", default="triangle_pdfs")
    parser.add_argument("--metadata", default="metadata.jsonl")
    parser.add_argument("--vid", default=DEFAULT_VID)
    parser.add_argument("--inst", default=DEFAULT_INST)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=0, help="Stop after N records; 0 means no limit.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Resolve PDF metadata but do not download files.",
    )
    parser.add_argument(
        "--collection-filter",
        help="Only process descendant collections whose title/path contains this case-insensitive text.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even when a matching local file already exists.",
    )
    return parser.parse_args()


def first(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def node_id(node: dict[str, Any]) -> str:
    pid = node.get("pid")
    if isinstance(pid, dict):
        value = pid.get("value")
        if value:
            return str(value)
    for key in ("id", "collectionId", "collection_id"):
        if node.get(key):
            return str(node[key])
    return ""


def node_title(node: dict[str, Any]) -> str:
    for key in ("name", "title", "displayName"):
        if node.get(key):
            return str(node[key])
    return node_id(node)


def child_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("children", "subCollections", "collection"):
        value = node.get(key)
        if isinstance(value, list):
            return [child for child in value if isinstance(child, dict)]
    collections = node.get("collections")
    if isinstance(collections, dict) and isinstance(collections.get("collection"), list):
        return [child for child in collections["collection"] if isinstance(child, dict)]
    return []


def iter_nodes(nodes: Iterable[dict[str, Any]], path: tuple[str, ...] = ()) -> Iterable[tuple[dict[str, Any], tuple[str, ...]]]:
    for node in nodes:
        current_path = (*path, node_title(node))
        yield node, current_path
        yield from iter_nodes(child_nodes(node), current_path)


def find_collection(root_nodes: list[dict[str, Any]], collection_id: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    for node, path in iter_nodes(root_nodes):
        if node_id(node) == collection_id:
            return node, path
    raise RuntimeError(f"collection id {collection_id} was not found in the Primo collection tree")


def collect_item_collections(root: dict[str, Any], root_path: tuple[str, ...]) -> list[dict[str, Any]]:
    collections: list[dict[str, Any]] = []
    for node, path in iter_nodes([root], root_path[:-1]):
        children = child_nodes(node)
        has_items = str(node.get("items", "")).upper() == "Y"
        if has_items or not children:
            cid = node_id(node)
            if cid:
                collections.append(
                    {
                        "id": cid,
                        "title": node_title(node),
                        "path": " / ".join(path),
                        "mms_id": node.get("mms_id") or node.get("mmsId"),
                    }
                )
    return collections


def search_params(args: argparse.Namespace, collection_id: str, offset: int = 0) -> dict[str, Any]:
    return {
        "limit": args.page_size,
        "offset": offset,
        "scope": args.scope,
        "q": f"cdparentid,exact,{collection_id}",
        "isCDSearch": "true",
        "rtaLinks": "true",
        "rapido": "false",
        "qInclude": "",
        "qExclude": "",
        "multiFacets": "",
        "lang": "en",
        "isRelatedItems": "false",
        "featuredNewspapersIssnList": "",
        "vid": args.vid,
        "inst": args.inst,
    }


def fetch_collection_tree(client: PrimoClient, args: argparse.Namespace) -> list[dict[str, Any]]:
    data = client.get_json(
        f"{PRIMO_BASE}/primaws/rest/priv/myaccount/collection",
        {
            "name": "get_collections",
            "useCache": "true",
            "inst": args.inst,
            "userInst": args.inst,
        },
    )
    collections = data.get("data", {}).get("collection", [])
    if not isinstance(collections, list):
        raise RuntimeError("unexpected collection tree response")
    return [node for node in collections if isinstance(node, dict)]


def fetch_records(client: PrimoClient, args: argparse.Namespace, collection_id: str) -> Iterable[dict[str, Any]]:
    offset = 0
    while True:
        data = client.get_json(
            f"{PRIMO_BASE}/primaws/rest/pub/pnxs",
            search_params(args, collection_id, offset),
        )
        docs = data.get("docs") or data.get("data", {}).get("docs") or []
        info = data.get("info") or data.get("data", {}).get("info") or {}
        if not docs:
            break
        for doc in docs:
            if isinstance(doc, dict):
                yield doc
        offset += len(docs)
        total = int(info.get("total") or info.get("totalResults") or 0)
        if total and offset >= total:
            break
        if len(docs) < args.page_size:
            break


def resolve_services(
    client: PrimoClient,
    args: argparse.Namespace,
    collection_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = []
    for record in records:
        rec_id = record_id(record)
        if not rec_id:
            continue
        delivery = record.get("delivery") or {}
        payload.append(
            {
                "recId": rec_id,
                "sharedDigitalCandidates": delivery.get("sharedDigitalCandidates"),
            }
        )
    if not payload:
        return {}
    return client.post_json(
        f"{PRIMO_BASE}/primaws/rest/pub/edelivery",
        search_params(args, collection_id, 0),
        payload,
    )


def fetch_representation(client: PrimoClient, args: argparse.Namespace, rep_id: str) -> dict[str, Any]:
    return client.get_json(
        f"{ALMA_BASE}/primaws/rest/priv/delivery/representationInfo",
        {"inst": args.inst, "repId": rep_id},
    )


def record_id(record: dict[str, Any]) -> str:
    pnx = record.get("pnx") or {}
    control = pnx.get("control") or {}
    return str(first(control.get("recordid"), ""))


def record_mms(record: dict[str, Any]) -> str:
    pnx = record.get("pnx") or {}
    display = pnx.get("display") or {}
    return str(first(display.get("mms"), ""))


def record_title(record: dict[str, Any]) -> str:
    pnx = record.get("pnx") or {}
    display = pnx.get("display") or {}
    return str(first(display.get("title"), record_id(record)))


def record_candidate_filename(record: dict[str, Any]) -> str:
    pnx = record.get("pnx") or {}
    addata = pnx.get("addata") or {}
    url_name = str(first(addata.get("url"), ""))
    if url_name.lower().endswith(".pdf"):
        return url_name
    title = record_title(record)
    return f"{title}.pdf" if title else f"{record_id(record)}.pdf"


def public_domain_note(record: dict[str, Any]) -> str:
    pnx = record.get("pnx") or {}
    display = pnx.get("display") or {}
    return str(first(display.get("lds05"), ""))


def choose_pdf_services(service_reply: Any, rec_id: str) -> list[dict[str, Any]]:
    entry = service_reply.get(rec_id, {}) if isinstance(service_reply, dict) else {}
    services = entry.get("electronicServices") or []
    chosen = []
    for service in services:
        if not isinstance(service, dict):
            continue
        file_type = str(service.get("fileType", "")).lower()
        if service.get("hasAccess") is False:
            continue
        if file_type == "pdf" or service.get("numberOfFiles"):
            chosen.append(service)
    return chosen


def safe_name(value: str, fallback: str) -> str:
    name = os.path.basename(value.strip()) or fallback
    name = re.sub(r"[^\w.\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return name or fallback


def collection_dir_name(path: str) -> str:
    leaf = path.split("/")[-1].strip()
    return safe_name(leaf, "collection")


def unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def existing_file_is_complete(path: Path, expected_size: Any) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        expected = int(expected_size)
    except (TypeError, ValueError):
        return True
    return path.stat().st_size == expected


def looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"%PDF"
    except OSError:
        return False


def write_metadata(metadata_path: Path, row: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def process_record(
    client: PrimoClient,
    args: argparse.Namespace,
    out_dir: Path,
    metadata_path: Path,
    collection: dict[str, Any],
    record: dict[str, Any],
    service_reply: Any,
) -> tuple[int, int, int]:
    rec_id = record_id(record)
    services = choose_pdf_services(service_reply, rec_id)
    if not services:
        write_metadata(
            metadata_path,
            base_metadata(collection, record) | {"status": "no_pdf_service"},
        )
        return (0, 0, 1)

    downloaded = skipped = failed = 0
    for service_index, service in enumerate(services, start=1):
        rep_id = str(service.get("ilsApiId") or "")
        if not rep_id:
            write_metadata(
                metadata_path,
                base_metadata(collection, record)
                | {"status": "missing_representation_id", "service": service},
            )
            failed += 1
            continue
        try:
            rep = fetch_representation(client, args, rep_id)
            files = rep.get("data", {}).get("files") or []
        except Exception as exc:
            write_metadata(
                metadata_path,
                base_metadata(collection, record)
                | {"status": "representation_error", "representation_id": rep_id, "error": str(exc)},
            )
            failed += 1
            continue

        for file_index, file_info in enumerate(files, start=1):
            if not isinstance(file_info, dict):
                continue
            if str(file_info.get("fileType", "")).upper() != "PDF":
                continue
            if file_info.get("isAccessRightsOk") is False:
                continue
            download_url = file_info.get("downloadUrl")
            folder = out_dir / collection_dir_name(collection["path"])
            fallback = f"{safe_name(record_title(record), rec_id)}_{service_index}_{file_index}.pdf"
            candidate = file_info.get("path") or record_candidate_filename(record)
            filename = safe_name(str(candidate), fallback)
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            target = folder / filename
            complete = existing_file_is_complete(target, file_info.get("fileSize"))
            if complete and not args.overwrite:
                status = "skipped_existing"
                skipped += 1
            elif args.metadata_only:
                status = "metadata_only"
                skipped += 1
            elif not download_url:
                status = "missing_download_url"
                failed += 1
            else:
                target = unique_path(target, args.overwrite)
                try:
                    bytes_written = client.download(str(download_url), target)
                    if not looks_like_pdf(target):
                        status = "downloaded_but_not_pdf"
                        failed += 1
                    else:
                        status = "downloaded"
                        downloaded += 1
                    file_info = file_info | {"bytesWritten": bytes_written}
                except Exception as exc:
                    status = "download_error"
                    file_info = file_info | {"error": str(exc)}
                    failed += 1

            write_metadata(
                metadata_path,
                base_metadata(collection, record)
                | {
                    "status": status,
                    "representation_id": rep_id,
                    "file_pid": file_info.get("pid"),
                    "file_label": file_info.get("label"),
                    "file_size": file_info.get("fileSize"),
                    "file_type": file_info.get("fileType"),
                    "local_path": str(target),
                    "alma_delivery_url": f"{ALMA_BASE}/view/delivery/{args.inst}/{rep_id}",
                    "representation_info_url": (
                        f"{ALMA_BASE}/primaws/rest/priv/delivery/representationInfo?"
                        f"{urllib.parse.urlencode({'inst': args.inst, 'repId': rep_id})}"
                    ),
                    "error": file_info.get("error"),
                    "bytes_written": file_info.get("bytesWritten"),
                },
            )
    return (downloaded, skipped, failed)


def base_metadata(collection: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection_id": collection["id"],
        "collection_title": collection["title"],
        "collection_path": collection["path"],
        "record_id": record_id(record),
        "mms_id": record_mms(record),
        "title": record_title(record),
        "candidate_filename": record_candidate_filename(record),
        "rights": public_domain_note(record),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = out_dir / metadata_path
    out_dir.mkdir(parents=True, exist_ok=True)

    client = PrimoClient(delay=args.delay, retries=args.retries)
    root_nodes = fetch_collection_tree(client, args)
    root, root_path = find_collection(root_nodes, args.collection_id)
    collections = collect_item_collections(root, root_path)
    if args.collection_filter:
        needle = args.collection_filter.lower()
        collections = [c for c in collections if needle in c["path"].lower()]

    if not collections:
        raise RuntimeError("no item collections matched the requested collection/filter")

    print(f"Found {len(collections)} item collections under {' / '.join(root_path)}", flush=True)
    print(f"Writing PDFs under {out_dir}", flush=True)
    print(f"Writing metadata to {metadata_path}", flush=True)

    total_records = total_downloaded = total_skipped = total_failed = 0
    stop = False
    for collection in collections:
        if stop:
            break
        print(f"\nCollection: {collection['path']} ({collection['id']})", flush=True)
        pending: list[dict[str, Any]] = []
        for record in fetch_records(client, args, collection["id"]):
            if args.max_items and total_records >= args.max_items:
                stop = True
                break
            pending.append(record)
            total_records += 1
            if len(pending) >= args.batch_size:
                downloaded, skipped, failed = process_batch(
                    client, args, out_dir, metadata_path, collection, pending
                )
                total_downloaded += downloaded
                total_skipped += skipped
                total_failed += failed
                pending = []
        if pending:
            downloaded, skipped, failed = process_batch(
                client, args, out_dir, metadata_path, collection, pending
            )
            total_downloaded += downloaded
            total_skipped += skipped
            total_failed += failed

    print(
        "\nDone: "
        f"{total_records} records, "
        f"{total_downloaded} PDFs downloaded, "
        f"{total_skipped} skipped, "
        f"{total_failed} failed.",
        flush=True,
    )
    return 1 if total_failed else 0


def process_batch(
    client: PrimoClient,
    args: argparse.Namespace,
    out_dir: Path,
    metadata_path: Path,
    collection: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[int, int, int]:
    service_reply = resolve_services(client, args, collection["id"], records)
    downloaded = skipped = failed = 0
    for record in records:
        d_count, s_count, f_count = process_record(
            client, args, out_dir, metadata_path, collection, record, service_reply
        )
        downloaded += d_count
        skipped += s_count
        failed += f_count
        print(
            f"  {record_title(record)}: +{d_count} downloaded, {s_count} skipped, {f_count} failed",
            flush=True,
        )
    return downloaded, skipped, failed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

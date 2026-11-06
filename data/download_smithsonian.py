#!/usr/bin/env python3
"""Downloads Smithsonian Open Access records and media into data/smithsonian/.
Usage:
    uv run --extra smithsonian data/download_smithsonian.py \\
        file NMAAHC-0039C24D4B742_4004.jpg
    uv run --extra smithsonian data/download_smithsonian.py \\
        sample --per-unit 5 --max-jpgs 5 --seed 0 [--edan nmaahc npg]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import duckdb

# URLs for parquet and downloads
CATALOG_URL_PREFIX = (
    "https://source.coop/harvard-lil/smithsonian-open-access/"
)
DOWNLOAD_URL_PREFIX = (
    "https://data.source.coop/harvard-lil/smithsonian-open-access/"
)
CATALOG_NAMES = ["linkages", "metadata"]
DATA_ROOT = Path(__file__).resolve().parent / "smithsonian"
SEARCH_DIR = DATA_ROOT / "search"
HTTP_TIMEOUT_SECONDS = 120

def to_relative_path(url: str) -> str:
    """Strips the URL prefix to get relative path"""
    for prefix in (CATALOG_URL_PREFIX, DOWNLOAD_URL_PREFIX):
        if url.startswith(prefix):
            return url[len(prefix):]
    raise ValueError(f"Not a smithsonian-open-access URL: {url}")

def open_url(url: str):
    request = urllib.request.Request(url)
    return urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS)

def download_file(
    url: str, destination: Path, overwrite: bool = False
) -> dict:
    """Streams url to destination via a .part file, skipping files that already
    exist unless overwrite"""
    if destination.exists() and not overwrite:
        return {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    with open_url(url) as response, open(partial_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
        headers = dict(response.headers)
    partial_path.replace(destination)
    return headers

def ensure_catalog(refresh: bool) -> dict:
    """Downloads missing parquet catalogs (all of them if refresh) and returns
    each one's Last-Modified, kept in catalog.json"""
    last_modified_path = SEARCH_DIR / "catalog.json"
    last_modified = {}
    if last_modified_path.exists():
        last_modified = json.loads(last_modified_path.read_text())
    for catalog_name in CATALOG_NAMES:
        parquet_path = SEARCH_DIR / f"{catalog_name}.parquet"
        cached = parquet_path.exists() and catalog_name in last_modified
        if cached and not refresh:
            continue
        url = f"{DOWNLOAD_URL_PREFIX}search/{catalog_name}.parquet"
        print(f"Downloading {url} ...")
        headers = download_file(url, parquet_path, overwrite=True)
        last_modified[catalog_name] = headers.get("Last-Modified")
    last_modified_path.write_text(json.dumps(last_modified, indent=2))
    return last_modified

def open_catalog(search_dir: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB views over the cached catalogs, with every URL trimmed
    to a relative path: linkages(path, record_id) and
    metadata(unit_code, record_id, shard_path)"""
    db = duckdb.connect()
    prefix_length = len(CATALOG_URL_PREFIX)
    linkages_path = str(search_dir / "linkages.parquet").replace("'", "''")
    metadata_path = str(search_dir / "metadata.parquet").replace("'", "''")
    db.execute(
        f"""
        CREATE VIEW linkages AS
        SELECT substr(url, {prefix_length + 1}) AS path, record_id
        FROM read_parquet('{linkages_path}')
        """
    )
    db.execute(
        f"""
        CREATE VIEW metadata AS
        SELECT
            unit_code,
            record_id,
            substr(metadata_url, {prefix_length + 1}) AS shard_path
        FROM read_parquet('{metadata_path}')
        """
    )
    return db

def resolve_file(
    db: duckdb.DuckDBPyConnection, name: str
) -> tuple[str, list[tuple[str, str | None]]]:
    """Returns (path, [(record_id, shard_path), ...]) for one file which can be
    input by bare filename, relative path, or full URL"""
    if name.startswith("http"):
        relative_name = to_relative_path(name)
    else:
        relative_name = name.lstrip("/")
    rows = db.execute(
        """
        SELECT DISTINCT linkages.path, linkages.record_id, metadata.shard_path
        FROM linkages
        LEFT JOIN metadata USING (record_id)
        WHERE ends_with('/' || linkages.path, $suffix)
        ORDER BY linkages.path, linkages.record_id
        """,
        {"suffix": "/" + relative_name},
    ).fetchall()
    matching_paths = sorted({path for path, _, _ in rows})
    if not matching_paths:
        raise SystemExit(f"No file in linkages.parquet matches {name!r}.")
    if len(matching_paths) > 1:
        paths = "\n  ".join(matching_paths)
        raise SystemExit(
            f"{name!r} matches {len(matching_paths)} files; "
            f"pass one of these paths instead:\n  {paths}"
        )
    record_shards = [(record_id, shard) for _, record_id, shard in rows]
    return matching_paths[0], record_shards

def resolve_units(
    db: duckdb.DuckDBPyConnection, requested: list[str] | None
) -> list[str]:
    """All catalog unit codes, or those matching requested case-insensitively.
    Fails loud on unknown units"""
    rows = db.execute(
        "SELECT DISTINCT unit_code FROM metadata ORDER BY unit_code"
    ).fetchall()
    all_units = [unit_code for (unit_code,) in rows]
    if requested is None:
        return all_units
    requested_lower = {unit.lower() for unit in requested}
    unknown_units = requested_lower - {unit.lower() for unit in all_units}
    if unknown_units:
        raise SystemExit(
            f"Unknown --edan unit(s) {sorted(unknown_units)}; "
            f"known units: {', '.join(all_units)}"
        )
    return [unit for unit in all_units if unit.lower() in requested_lower]

def build_eligible_table(
    db: duckdb.DuckDBPyConnection, max_jpgs: int
) -> None:
    """Creates table eligible(unit_code, record_id, shard_path, jpg_paths) of
    records whose 1..max_jpgs media jpgs are linked to no other record"""
    db.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible AS
        WITH media_jpgs AS (
            SELECT DISTINCT path, record_id
            FROM linkages
            WHERE starts_with(path, 'media/') AND lower(path) LIKE '%.jpg'
        ), jpg_record_counts AS (
            SELECT path, count(*) AS record_count
            FROM media_jpgs
            GROUP BY path
        ), record_jpgs AS (
            SELECT
                media_jpgs.record_id,
                list(media_jpgs.path ORDER BY media_jpgs.path) AS jpg_paths,
                count(*) AS jpg_count,
                bool_and(jpg_record_counts.record_count = 1)
                    AS all_jpgs_exclusive
            FROM media_jpgs
            JOIN jpg_record_counts USING (path)
            GROUP BY media_jpgs.record_id
        )
        SELECT
            metadata.unit_code,
            metadata.record_id,
            metadata.shard_path,
            record_jpgs.jpg_paths
        FROM record_jpgs
        JOIN metadata USING (record_id)
        WHERE record_jpgs.all_jpgs_exclusive
            AND record_jpgs.jpg_count <= $max_jpgs
            AND metadata.shard_path IS NOT NULL
        """,
        {"max_jpgs": max_jpgs},
    )

def sample_records(
    db: duckdb.DuckDBPyConnection,
    per_unit: int,
    seed: int,
    units: list[str],
) -> list[tuple[str, str, str, list[str]]]:
    """Up to per_unit eligible records from each of units, shuffled by
    md5(seed:record_id) so the same seed gives the same sample"""
    return db.execute(
        """
        SELECT unit_code, record_id, shard_path, jpg_paths
        FROM (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY unit_code
                    ORDER BY md5($seed || ':' || record_id), record_id
                ) AS rank_in_unit
            FROM eligible
            WHERE list_contains($units, unit_code)
        )
        WHERE rank_in_unit <= $per_unit
        ORDER BY unit_code, rank_in_unit
        """,
        {"seed": str(seed), "per_unit": per_unit, "units": units},
    ).fetchall()

def fetch_records(
    record_shards: list[tuple[str, str | None]]
) -> dict[str, dict]:
    """Fetches each JSONL shard once and returns {record_id: record}."""
    wanted_ids_by_shard = defaultdict(set)
    for record_id, shard_path in record_shards:
        if shard_path is None:
            raise SystemExit(f"{record_id}: no metadata shard in the catalog")
        wanted_ids_by_shard[shard_path].add(record_id)
    records = {}
    for shard_path, wanted_ids in wanted_ids_by_shard.items():
        with open_url(DOWNLOAD_URL_PREFIX + shard_path) as response:
            shard_lines = response.read().decode("utf-8").splitlines()
        for line in shard_lines:
            record = json.loads(line)
            record_id = (
                record.get("content", {})
                .get("descriptiveNonRepeating", {})
                .get("record_ID")
            )
            if record_id not in wanted_ids:
                continue
            if record_id in records:
                raise SystemExit(f"{record_id}: found twice in {shard_path}")
            records[record_id] = record
        missing_ids = wanted_ids - records.keys()
        if missing_ids:
            raise SystemExit(f"{sorted(missing_ids)}: not in {shard_path}")
    return records

def save_record(record_id: str, record: dict) -> Path:
    record_path = DATA_ROOT / "records" / f"{record_id}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record_path

def run_file(args: argparse.Namespace) -> None:
    ensure_catalog(args.refresh_catalog)
    db = open_catalog(SEARCH_DIR)
    path, record_shards = resolve_file(db, args.name)
    for record_id, record in fetch_records(record_shards).items():
        record_path = save_record(record_id, record)
        print(f"record {record_id} -> {record_path.relative_to(DATA_ROOT)}")
    download_file(DOWNLOAD_URL_PREFIX + path, DATA_ROOT / path)
    print(f"file   {path} (linked to {len(record_shards)} record(s))")

def sample_dir_name(args: argparse.Namespace) -> str:
    name = (
        f"per-unit-{args.per_unit}_max-jpgs-{args.max_jpgs}_seed-{args.seed}"
    )
    if args.edan is not None:
        name += "_edan-" + "-".join(sorted(unit.lower() for unit in args.edan))
    return name

def run_sample(args: argparse.Namespace) -> None:
    catalog_last_modified = ensure_catalog(args.refresh_catalog)
    db = open_catalog(SEARCH_DIR)
    units = resolve_units(db, args.edan)
    build_eligible_table(db, args.max_jpgs)
    sampled = sample_records(db, args.per_unit, args.seed, units)
    sampled_units = {unit_code for unit_code, _, _, _ in sampled}
    print(f"Sampled {len(sampled)} records across {len(sampled_units)} units.")

    records = fetch_records(
        [(record_id, shard_path) for _, record_id, shard_path, _ in sampled]
    )
    gold_rows = []
    for index, (unit_code, record_id, _, jpg_paths) in enumerate(
        sampled, start=1
    ):
        jpg_count = len(jpg_paths)
        print(f"[{index}/{len(sampled)}] {record_id} ({jpg_count} jpg(s))")
        record_path = save_record(record_id, records[record_id])
        for jpg_path in jpg_paths:
            download_file(DOWNLOAD_URL_PREFIX + jpg_path, DATA_ROOT / jpg_path)
            gold_rows.append([
                unit_code,
                record_id,
                str(record_path.relative_to(DATA_ROOT)),
                jpg_path,
            ])

    # Written last: a sample directory with gold_pairs.csv is always complete
    sample_dir = DATA_ROOT / "samples" / sample_dir_name(args)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "gold_pairs.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["unit_code", "record_id", "record_path", "jpg_path"]
        )
        csv_writer.writerows(gold_rows)
    eligible_counts = dict(
        db.execute(
            "SELECT unit_code, count(*) FROM eligible GROUP BY unit_code"
        ).fetchall()
    )
    manifest = {
        "args": {
            "per_unit": args.per_unit,
            "max_jpgs": args.max_jpgs,
            "seed": args.seed,
            "edan": args.edan,
        },
        "catalog_last_modified": catalog_last_modified,
        "eligible_per_unit": {
            unit: eligible_counts.get(unit, 0) for unit in units
        },
    }
    (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(gold_rows)} gold pairs to {sample_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="re-download the search/*.parquet catalogs",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    file_parser = subparsers.add_parser(
        "file",
        parents=[common_parser],
        help="one file plus its linked record(s)",
    )
    file_parser.add_argument(
        "name", help="bare filename, media/<folder>/<name>, or full URL"
    )
    file_parser.set_defaults(handler=run_file)

    sample_parser = subparsers.add_parser(
        "sample",
        parents=[common_parser],
        help="random records per unit plus all their jpgs",
    )
    sample_parser.add_argument(
        "--per-unit", type=int, required=True,
        help="records to sample per unit",
    )
    sample_parser.add_argument(
        "--max-jpgs", type=int, required=True,
        help="skip records with more jpgs than this",
    )
    sample_parser.add_argument("--seed", type=int, required=True)
    sample_parser.add_argument(
        "--edan", nargs="+", metavar="UNIT",
        help="only these unit codes (case-insensitive)",
    )
    sample_parser.set_defaults(handler=run_sample)

    args = parser.parse_args()
    args.handler(args)

if __name__ == "__main__":
    main()

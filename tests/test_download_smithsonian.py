"""data/download_smithsonian.py's catalog queries, against tiny synthetic
linkages/metadata parquets -- no network. Run with `--extra smithsonian`."""

import pytest

duckdb = pytest.importorskip("duckdb")
from data import download_smithsonian as ds

URL_PREFIX = ds.CATALOG_URL_PREFIX

def media_url(path: str) -> str:
    return f"{URL_PREFIX}media/{path}"

def write_catalog(search_dir, linkages, metadata):
    """linkages: (url, record_id) rows. metadata: (unit_code, record_id) rows,
    or (unit_code, record_id, metadata_url) to override the default shard."""
    db = duckdb.connect()
    db.execute("CREATE TABLE linkages (url VARCHAR, record_id VARCHAR)")
    db.executemany("INSERT INTO linkages VALUES (?, ?)", linkages)
    db.execute(
        "CREATE TABLE metadata (unit_code VARCHAR, record_id VARCHAR, "
        "title VARCHAR, guid VARCHAR, metadata_url VARCHAR)"
    )
    metadata_rows = []
    for row in metadata:
        unit_code, record_id = row[0], row[1]
        default_url = f"{URL_PREFIX}metadata/edan/{unit_code.lower()}/00.txt"
        metadata_url = row[2] if len(row) > 2 else default_url
        metadata_rows.append(
            (unit_code, record_id, record_id, None, metadata_url)
        )
    db.executemany(
        "INSERT INTO metadata VALUES (?, ?, ?, ?, ?)", metadata_rows
    )
    for table in ("linkages", "metadata"):
        db.execute(
            f"COPY {table} TO '{search_dir / f'{table}.parquet'}' "
            "(FORMAT parquet)"
        )

@pytest.fixture
def db(tmp_path):
    linkages = [
        # ok_1: one exclusive jpg (+ tif twin, ignored as not a jpg)
        (media_url("aaa/A-1.jpg"), "ok_1"),
        (media_url("aaa/A-1.tif"), "ok_1"),
        # ok_2: two exclusive jpgs, one with an upper-case extension
        (media_url("aaa/A-2a.jpg"), "ok_2"),
        (media_url("aaa/A-2b.JPG"), "ok_2"),
        # shared: one exclusive jpg plus one also linked to other -> both
        # records skipped
        (media_url("aaa/A-3.jpg"), "shared"),
        (media_url("aaa/A-common.jpg"), "shared"),
        (media_url("aaa/A-common.jpg"), "other"),
        # too_many: 3 exclusive jpgs, over max_jpgs=2
        *[(media_url(f"bbb/B-{i}.jpg"), "too_many") for i in range(3)],
        # textured: a 3d/ texture jpg is not a media jpg; one exclusive
        # media jpg
        (f"{URL_PREFIX}3d/uuid-1/texture.jpg", "textured"),
        (media_url("bbb/B-tex.jpg"), "textured"),
        # same basename at two 3d paths
        (f"{URL_PREFIX}3d/uuid-2/scan.ply", "textured"),
        (f"{URL_PREFIX}3d/uuid-2/resources/scan.ply", "textured"),
        # bbb has more eligible records, for the per-unit cap
        *[(media_url(f"bbb/B-ok{i}.jpg"), f"bbb_{i}") for i in range(4)],
        # no_shard: exclusive jpg, but the catalog has no metadata_url for it
        (media_url("aaa/A-orphan.jpg"), "no_shard"),
    ]
    metadata = [
        ("AAA", "ok_1"),
        ("AAA", "ok_2"),
        ("AAA", "shared"),
        ("AAA", "other"),
        ("AAA", "no_shard", None),
        ("BBB", "too_many"),
        ("BBB", "textured"),
        *[("BBB", f"bbb_{i}") for i in range(4)],
    ]
    write_catalog(tmp_path, linkages, metadata)
    db = ds.open_catalog(tmp_path)
    ds.build_eligible_table(db, max_jpgs=2)
    return db

def eligible_jpgs_by_record(db):
    rows = db.execute("SELECT record_id, jpg_paths FROM eligible").fetchall()
    return dict(rows)

def sample(db, per_unit, seed, requested_units=None):
    units = ds.resolve_units(db, requested_units)
    return ds.sample_records(db, per_unit, seed, units)

def test_eligibility_requires_exclusive_media_jpgs_within_cap(db):
    eligible = eligible_jpgs_by_record(db)
    assert set(eligible) == {
        "ok_1", "ok_2", "textured", "bbb_0", "bbb_1", "bbb_2", "bbb_3",
    }
    assert eligible["ok_1"] == ["media/aaa/A-1.jpg"]
    assert eligible["ok_2"] == ["media/aaa/A-2a.jpg", "media/aaa/A-2b.JPG"]
    assert eligible["textured"] == ["media/bbb/B-tex.jpg"]

def test_sample_caps_per_unit_and_is_seed_deterministic(db):
    sampled = sample(db, per_unit=2, seed=0)
    records_by_unit = {}
    for unit_code, record_id, _, _ in sampled:
        records_by_unit.setdefault(unit_code, []).append(record_id)
    assert {
        unit_code: len(record_ids)
        for unit_code, record_ids in records_by_unit.items()
    } == {"AAA": 2, "BBB": 2}
    assert sampled == sample(db, per_unit=2, seed=0)
    picks_by_seed = {
        tuple(
            record_id
            for _, record_id, _, _ in sample(db, per_unit=2, seed=seed)
        )
        for seed in range(20)
    }
    assert len(picks_by_seed) > 1

def test_sample_edan_filter_is_case_insensitive(db):
    sampled = sample(db, per_unit=10, seed=0, requested_units=["aaa"])
    assert {unit_code for unit_code, *_ in sampled} == {"AAA"}
    assert {record_id for _, record_id, _, _ in sampled} == {"ok_1", "ok_2"}
    with pytest.raises(SystemExit, match="Unknown --edan"):
        ds.resolve_units(db, ["zzz"])

def test_resolve_file_by_name_path_and_url(db):
    shard_path = "metadata/edan/aaa/00.txt"
    expected = (
        "media/aaa/A-common.jpg",
        [("other", shard_path), ("shared", shard_path)],
    )
    assert ds.resolve_file(db, "A-common.jpg") == expected
    assert ds.resolve_file(db, "media/aaa/A-common.jpg") == expected
    full_url = f"{ds.DOWNLOAD_URL_PREFIX}media/aaa/A-common.jpg"
    assert ds.resolve_file(db, full_url) == expected

def test_resolve_file_fails_loud_on_missing_or_ambiguous(db):
    with pytest.raises(SystemExit, match="No file"):
        ds.resolve_file(db, "nope.jpg")
    with pytest.raises(SystemExit, match="matches 2 files"):
        ds.resolve_file(db, "scan.ply")
    resolved_path, _ = ds.resolve_file(db, "resources/scan.ply")
    assert resolved_path == "3d/uuid-2/resources/scan.ply"

def test_fetch_records_fails_loud_without_shard(db):
    _, record_shards = ds.resolve_file(db, "A-orphan.jpg")
    assert record_shards == [("no_shard", None)]
    with pytest.raises(SystemExit, match="no metadata shard"):
        ds.fetch_records(record_shards)

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
RUN = REPOSITORY / "outputs" / "bordair_retraining_v1"
SOURCE = RUN.parent / "bordair_eval" / "source"
BENIGN_SOURCE = RUN / "source" / "bordair_benign_multimodal_text_image.json"
IMAGES_DIRECTORY = RUN / "images_v7"
MANIFEST_PATH = RUN / "corpus_manifest_v7.json"
DEVELOPMENT_PATH = RUN / "development_metadata_v7.csv"
REGRESSION_PATH = RUN / "regression_metadata_v7.csv"
FINAL_PATH = RUN / "final_metadata_v7.csv"
TEXT_LED_FINAL_PATH = RUN / "text_led_final_metadata_v7.csv"
EXTERNAL_BENIGN_PATH = RUN / "external_benign_metadata_v7.csv"

EXPECTED_SCHEMA4_MANIFEST_SHA256 = (
    "ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220"
)
EXPECTED_SCHEMA4_DEVELOPMENT_SHA256 = (
    "d8efb9365fa61be66dcda3400b621cce42aa03e61c3f7a080a0ec8c5db456b49"
)
EXPECTED_SCHEMA4_IMAGE_TREE = (
    1_975,
    47_582_773,
    "d5fb0f563f60d5f02282abeb7effb9d3e49a8d34a84f039c104d403afdb6e108",
)

BENIGN_SOURCE_SHA256 = "5a0f0a3322219bd1966002a0aeb39b846703ae46fff0a8c53e24ac9d27492679"
EXPECTED_SOURCE_SHARDS = {
    "text_image_001.json": "f17bf5a273224ec2d7f62837cd0b2277ffc3b473d1e6079654fe813d4d58dc69",
    "text_image_002.json": "7d34cee7b4ad5cdea75b41b7786adc9716d841d56e2cbcc9cf14202f6f2a7bda",
    "text_image_003.json": "cd1ec0ec7e33e756334bfa9d445fabaf7e0c91fb5f704b8ef5978cad51bb0758",
    "text_image_004.json": "620d123d0490462db67dfd8e83da2e3fe0cf0e6ec1b1fa877726588e1f842cfa",
    "text_image_005.json": "1d77666cee0d3d43231f0cf946e7fd05ab09f18f4de3e089375550be400c7659",
    "text_image_006.json": "c2b92fc8eedd707748403e2a4e10ff3cd5d415bbac176f9799a55fbc04cf4183",
    "text_image_007.json": "269a99c151998b050d9199cc596f3c0f024c06ee737b790db06674e95f91fbcc",
    "text_image_008.json": "def6bed3e933eefd5f7874e8f2ba7f99072f0906cd4ede137ccaaca707487833",
    "text_image_009.json": "31259b8f02e94b7c29ef8ed7ceaeb9ed51a821924ceb7ec6bf8aebda9cc61c53",
    "text_image_010.json": "81f1745c09a43059af088e365c84effd7bd7c651adaaf8cbce72eeda66fb8477",
    "text_image_011.json": "1850580d08a701b0fde9e22427493edfda65d1e8dd2d563b09d8af92d1bf5335",
    "text_image_012.json": "520b7a0802ceb0d93436c36e575703b369266842dc6cc85a7d64680bc6fd5759",
    "text_image_013.json": "33e5d4d7813ec7896344f7f7117e6bcddc2e6c22bbb02eabc8a787d6faec9bf1",
}

PRIOR_REGRESSION_IDS = (
    "TI-00001", "TI-02241", "TI-03921", "TI-06161", "TI-04481",
    "TI-05461", "TI-01121", "TI-03361", "TI-05041", "TI-01681",
)
FINAL_MALICIOUS_IDS = (
    "TI-01093", "TI-03333", "TI-06133", "TI-00533", "TI-01653",
    "TI-02213", "TI-02773", "TI-03893", "TI-04453", "TI-05013",
)
FINAL_BENIGN_IDS = (
    "BTI-00930", "BTI-01460", "BTI-02861", "BTI-05697", "BTI-06083",
    "BTI-05087", "BTI-01896", "BTI-04521", "BTI-05202", "BTI-05112",
)
FULL_MALICIOUS_TEXT_STRATEGIES = {
    "authority_payload_split", "context_switch_injection"
}
COUNTERFACTUAL_STRATEGY = "malicious_text_benign_image_counterfactual"
BENIGN_STRATEGY = "official_bordair_benign_text_rendered_in_image"
IMAGE_ONLY_STRATEGY = "benign_text_full_injection"
HARD_NEGATIVE_CATEGORIES = (
    "ignore_rule", "prompt_meta", "policy_legal", "defensive_security",
    "credentials_auth", "code_errors", "sysadmin_tools", "binary_structured",
    "foreign_terse", "lexical_false_friends",
)
EXPECTED_TUNING_ID_DIGEST = "93c1eceb4c6cf68491a5ec6bc1b6f4bf18937c690aeec4cb540e7117be17807d"
EXPECTED_EXTERNAL_ID_DIGEST = "9dcd6753b45ce114967981d995851fb66d050387a9a6cf220bf30366643d34e2"
EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST = "5e8655a185e5bbedae414b3f6f085e4de86a62dd501f2501bf9f0d7d3f78ede9"
EXPECTED_ADDITIONAL_TRAIN_TEXT_DIGEST = "48494292e8e576c5d789d3fc9e45d15439e0fe20be400c99031b94d9b327de3b"
EXPECTED_ADDITIONAL_TRAIN_REGEX_DIGEST = "0ca03179b42c077af533edf788cda7f68a3e4bbb5ccc09343f453f25e84b6e83"
SCHEMA3_ARCHIVE_RELATIVE = Path(
    "features_v7_trials/schema3_pre_additional_hard_negatives"
)
SCHEMA3_ARCHIVE_MANIFEST_SHA256 = "3267339b2a97a14b8568974026836fd4b823daa16f84652de019aacbb5e170eb"
FROZEN_METADATA_SHA256 = {
    "regression_metadata_v7.csv": "f6c06498c9aa4f21bc6520474bb5de2d9a0d47912ad0a1ed482dba4135628a8a",
    "final_metadata_v7.csv": "d171cc1e4e85dcfcc983e837add08b4a64895386bc6e7f5917808153e2f5f213",
    "text_led_final_metadata_v7.csv": "87fddbee0146aa17c8454c768dda058946b00ddb0fbab6e572ad08348bf762d6",
    "external_benign_metadata_v7.csv": "bca44d53483ba029adad2db7a513dda2a88eef5d85bb1fb0be4ecde94d07ec02",
}
REQUIRED_COLUMNS = {
    "sample_id", "label_id", "split", "group_id", "attack_style", "strategy",
    "source", "prompt_text", "image_path", "image_text", "render_style",
    "image_sha256", "source_sample_id", "source_strategy",
    "paired_benign_sample_id", "hard_negative", "hard_negative_category",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> tuple[int, int, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    lines: list[str] = []
    total_bytes = 0
    for path in files:
        byte_count = path.stat().st_size
        total_bytes += byte_count
        lines.append(
            f"{path.relative_to(root).as_posix()}|{byte_count}|{sha256(path)}"
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return len(files), total_bytes, hashlib.sha256(payload).hexdigest()


def validate_schema4_anchors(
    manifest_path: Path,
    development_path: Path,
    images_directory: Path,
) -> None:
    require(manifest_path.is_file(), "missing corpus_manifest_v7.json")
    require(development_path.is_file(), "missing development_metadata_v7.csv")
    require(images_directory.is_dir(), "missing images_v7 directory")
    require(
        sha256(manifest_path) == EXPECTED_SCHEMA4_MANIFEST_SHA256,
        "schema-4 manifest hash changed",
    )
    require(
        sha256(development_path) == EXPECTED_SCHEMA4_DEVELOPMENT_SHA256,
        "schema-4 development metadata hash changed",
    )
    require(
        tree_digest(images_directory) == EXPECTED_SCHEMA4_IMAGE_TREE,
        "schema-4 image tree changed",
    )


def validate_schema3_archive(run_root: Path) -> tuple[
    dict[str, object], list[dict[str, str]], set[str]
]:
    archive_root = run_root / SCHEMA3_ARCHIVE_RELATIVE
    manifest_path = archive_root / "archive_manifest.json"
    require(manifest_path.is_file(), "missing immutable schema-3 archive manifest")
    require(
        sha256(manifest_path) == SCHEMA3_ARCHIVE_MANIFEST_SHA256,
        "schema-3 archive manifest changed",
    )
    archive = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        archive.get("corpus_schema_version") == 3,
        "schema-3 archive declares an unexpected corpus schema",
    )
    files = archive.get("files")
    trees = archive.get("trees")
    require(isinstance(files, dict), "schema-3 archive lacks file hashes")
    require(isinstance(trees, dict), "schema-3 archive lacks tree hashes")
    for relative, expected in files.items():
        path = archive_root / relative
        require(path.is_file(), f"schema-3 archive file missing: {relative}")
        require(
            path.stat().st_size == int(expected["bytes"]),
            f"schema-3 archive byte count changed: {relative}",
        )
        require(
            sha256(path) == str(expected["sha256"]),
            f"schema-3 archive hash changed: {relative}",
        )
    for relative, expected in trees.items():
        require(
            tree_digest(archive_root / relative)
            == (
                int(expected["files"]),
                int(expected["bytes"]),
                str(expected["sha256"]),
            ),
            f"schema-3 archive tree changed: {relative}",
        )
    source_manifest_path = archive_root / "corpus_manifest_v7.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require(source_manifest.get("schema_version") == 3, "archived corpus is not schema 3")
    archived_development = read_csv(archive_root / "development_metadata_v7.csv")
    require(len(archived_development) == 1_890, "archived development count changed")
    archived_panel_ids: set[str] = set()
    for name in files:
        if not str(name).endswith("_metadata_v7.csv"):
            continue
        with (archive_root / name).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            archived_panel_ids.update(
                str(row["sample_id"]) for row in csv.DictReader(handle)
            )
    return archive, archived_development, archived_panel_ids


def validate_additional_training_screen(
    manifest: dict[str, object],
    benign_source: dict[str, dict[str, object]],
    archived_panel_ids: set[str],
) -> tuple[dict[str, str], list[str]]:
    screen = manifest.get("schema4_additional_training_screen")
    require(isinstance(screen, dict), "manifest lacks schema-4 hard-negative screen")
    require(screen.get("rows") == 120, "schema-4 hard-negative row count changed")
    require(
        screen.get("rows_per_category") == 12,
        "schema-4 hard-negative category allocation changed",
    )
    require(
        screen.get("ids_sha256") == EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST,
        "schema-4 hard-negative ID digest changed",
    )
    require(
        screen.get("texts_sha256") == EXPECTED_ADDITIONAL_TRAIN_TEXT_DIGEST,
        "schema-4 hard-negative text digest changed",
    )
    rules = screen.get("rules")
    require(isinstance(rules, dict), "schema-4 hard-negative rules are missing")
    require(
        tuple(rules) == HARD_NEGATIVE_CATEGORIES,
        "schema-4 hard-negative category order changed",
    )
    category_by_id: dict[str, str] = {}
    selected_ids: list[str] = []
    selected_texts: set[str] = set()
    id_lines: list[str] = []
    text_lines: list[str] = []
    regex_lines: list[str] = []
    for category, rule in rules.items():
        require(isinstance(rule, dict), f"invalid schema-4 rule: {category}")
        pattern = str(rule.get("regex", ""))
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise RuntimeError(f"invalid schema-4 regex for {category}: {exc}") from exc
        ids = [str(value) for value in rule.get("ids", [])]
        require(len(ids) == 12, f"schema-4 category row count changed: {category}")
        require(len(ids) == len(set(ids)), f"schema-4 category repeats an ID: {category}")
        regex_lines.append(f"{category}|{pattern}\n")
        for sample_id in ids:
            require(sample_id not in category_by_id, f"schema-4 ID repeated: {sample_id}")
            require(
                sample_id not in archived_panel_ids,
                f"schema-4 row was already used by schema 3: {sample_id}",
            )
            source = benign_source.get(sample_id)
            require(source is not None, f"unknown schema-4 benign source: {sample_id}")
            text = str(source.get("text", "")).strip()
            safely_renderable = text.isascii() and all(
                character.isprintable() or character in "\r\n\t"
                for character in text
            )
            require(
                source.get("expected_detection") is False
                and source.get("image_type") == "ocr"
                and 8 <= len(text) <= 240
                and safely_renderable,
                f"schema-4 source is not an eligible official benign OCR row: {sample_id}",
            )
            require(
                compiled.search(text) is not None,
                f"schema-4 source fails its category regex: {sample_id}",
            )
            require(text not in selected_texts, f"schema-4 text repeated: {sample_id}")
            selected_texts.add(text)
            category_by_id[sample_id] = category
            selected_ids.append(sample_id)
            id_lines.append(f"{category}|{sample_id}\n")
            text_lines.append(f"{category}|{sample_id}|{text}\n")
    require(
        hashlib.sha256("".join(regex_lines).encode("utf-8")).hexdigest()
        == EXPECTED_ADDITIONAL_TRAIN_REGEX_DIGEST,
        "schema-4 hard-negative regex digest changed",
    )
    require(
        hashlib.sha256("".join(id_lines).encode("utf-8")).hexdigest()
        == EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST,
        "schema-4 hard-negative ordered IDs changed",
    )
    require(
        hashlib.sha256("".join(text_lines).encode("utf-8")).hexdigest()
        == EXPECTED_ADDITIONAL_TRAIN_TEXT_DIGEST,
        "schema-4 hard-negative source text changed",
    )
    require(len(category_by_id) == 120, "schema-4 hard-negative uniqueness changed")
    return category_by_id, selected_ids


def stable_key(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def schema4_protected_train_ids(
    archived_schema3_train_benign: list[dict[str, str]],
    counterfactual_pair_ids: list[str],
    historical_feedback_ids: set[str],
) -> set[str]:
    """Return rows protected at the schema-3 to schema-4 transition.

    Active hard negatives, historical feedback rows, and current pair rows all
    remain protected even when their provenance sets overlap.
    """
    return {
        row["sample_id"]
        for row in archived_schema3_train_benign
        if row["hard_negative"] == "1"
    } | set(counterfactual_pair_ids) | historical_feedback_ids


def family_number(sample_id: str) -> int:
    require(sample_id.startswith("TI-"), f"not a malicious source ID: {sample_id}")
    return (int(sample_id.removeprefix("TI-")) - 1) // 28


def row_family(row: dict[str, str]) -> int:
    prefix = "bordair-family-"
    require(row["group_id"].startswith(prefix), f"invalid group: {row['sample_id']}")
    return int(row["group_id"].removeprefix(prefix))


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing metadata file: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty metadata file: {path.name}")
    require(set(rows[0]) == REQUIRED_COLUMNS, f"unexpected columns: {path.name}")
    return rows


def load_sources(
    run_root: Path = RUN,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    source_root = run_root.parent / "bordair_eval" / "source"
    benign_source_path = (
        run_root / "source" / "bordair_benign_multimodal_text_image.json"
    )
    observed_shards = {
        path.name: sha256(path)
        for path in sorted(source_root.glob("text_image_*.json"))
    }
    require(observed_shards == EXPECTED_SOURCE_SHARDS, "malicious source shards changed")
    malicious_rows: list[dict[str, object]] = []
    for shard_name in EXPECTED_SOURCE_SHARDS:
        payload = json.loads((source_root / shard_name).read_text(encoding="utf-8"))
        require(isinstance(payload, list), f"source shard is not a list: {shard_name}")
        malicious_rows.extend(payload)
    malicious_ocr = {
        str(row["id"]): row for row in malicious_rows
        if row.get("image_type") == "ocr"
    }
    require(len(malicious_ocr) == 920, "unexpected malicious OCR source count")

    require(benign_source_path.is_file(), "missing pinned benign source")
    require(sha256(benign_source_path) == BENIGN_SOURCE_SHA256, "benign source changed")
    benign_payload = json.loads(benign_source_path.read_text(encoding="utf-8"))
    require(isinstance(benign_payload, list), "benign source is not a list")
    benign_by_id = {str(row["id"]): row for row in benign_payload}
    require(len(benign_by_id) == 6_423, "unexpected benign source count")
    require(
        sum(row.get("image_type") == "ocr" for row in benign_payload) == 2_246,
        "unexpected benign OCR source count",
    )
    return malicious_ocr, benign_by_id


def select_full_source_row(
    family: int,
    malicious_by_family: dict[int, list[dict[str, object]]],
    namespace: str,
) -> dict[str, object]:
    candidates = [
        row for row in malicious_by_family[family]
        if str(row["strategy"]) in FULL_MALICIOUS_TEXT_STRATEGIES
    ]
    require(len(candidates) == 2, f"family {family} lacks full-prompt sources")
    return min(candidates, key=lambda row: stable_key(namespace, family, row["id"]))


def validate_source_provenance(
    row: dict[str, str],
    malicious_source: dict[str, dict[str, object]],
    benign_source: dict[str, dict[str, object]],
) -> None:
    sample_id = row["sample_id"]
    if int(row["label_id"]) == 0:
        source_id = row["source_sample_id"]
        require(source_id == sample_id, f"benign source ID mismatch: {sample_id}")
        require(source_id in benign_source, f"unknown benign source: {source_id}")
        source = benign_source[source_id]
        require(source.get("expected_detection") is False, f"bad benign label: {source_id}")
        require(row["image_text"] == str(source["text"]).strip(), f"benign text changed: {source_id}")
        require(row["strategy"] == BENIGN_STRATEGY, f"bad benign strategy: {source_id}")
        require(row["source_strategy"] == "official_bordair_benign_text_image", f"bad benign provenance: {source_id}")
        require(not row["paired_benign_sample_id"], f"benign row claims a pair: {source_id}")
        text_hash = hashlib.sha256(row["image_text"].encode("utf-8")).hexdigest()[:12]
        require(row["group_id"] == f"bordair-benign-text-{text_hash}", f"bad benign group: {source_id}")
        return

    source_id = row["source_sample_id"]
    require(source_id in malicious_source, f"unknown malicious source: {source_id}")
    source = malicious_source[source_id]
    family = family_number(source_id)
    require(row_family(row) == family, f"family/source mismatch: {sample_id}")
    require(row["attack_style"] == str(source["category"]), f"category changed: {sample_id}")
    if row["strategy"] == COUNTERFACTUAL_STRATEGY:
        require(sample_id == f"MTI-{source_id.removeprefix('TI-')}", f"bad counterfactual ID: {sample_id}")
        require(row["source_strategy"] in FULL_MALICIOUS_TEXT_STRATEGIES, f"partial caller prompt used: {sample_id}")
        require(row["source_strategy"] == str(source["strategy"]), f"caller strategy changed: {sample_id}")
        require(row["prompt_text"] == str(source["text"]).strip(), f"caller prompt changed: {sample_id}")
        require(bool(row["paired_benign_sample_id"]), f"counterfactual has no pair: {sample_id}")
    else:
        require(row["strategy"] == IMAGE_ONLY_STRATEGY, f"unexpected attack strategy: {sample_id}")
        require(row["source_strategy"] == IMAGE_ONLY_STRATEGY, f"image-only provenance changed: {sample_id}")
        require(row["image_text"] == str(source["image_content"]), f"attack image text changed: {sample_id}")
        require(not row["paired_benign_sample_id"], f"image-only row claims a pair: {sample_id}")


def reconstruct_prior_v7_benign_baseline(
    benign_source: dict[str, dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    ocr = [row for row in benign_source.values() if row.get("image_type") == "ocr"]
    final_rows = [benign_source[sample_id] for sample_id in FINAL_BENIGN_IDS]
    final_texts = {str(row["text"]).strip() for row in final_rows}
    eligible_by_text: dict[str, dict[str, object]] = {}
    for row in sorted(ocr, key=lambda value: str(value["id"])):
        text = str(row["text"]).strip()
        safely_renderable = text.isascii() and all(
            character.isprintable() or character in "\r\n\t" for character in text
        )
        if 8 <= len(text) <= 240 and text not in final_texts and safely_renderable:
            eligible_by_text.setdefault(text, row)
    eligible = list(eligible_by_text.values())
    hard_train = sorted(
        [row for row in eligible if row.get("source") == "edge_cases"],
        key=lambda row: stable_key("official-benign-hard-train-v2", row["id"]),
    )
    general = [row for row in eligible if row.get("source") != "edge_cases"]
    prior_validation = sorted(
        general,
        key=lambda row: stable_key("official-benign-validation-v2", row["id"]),
    )[:80]
    prior_validation_ids = {str(row["id"]) for row in prior_validation}
    prior_test = sorted(
        [row for row in general if str(row["id"]) not in prior_validation_ids],
        key=lambda row: stable_key("official-benign-test-v2", row["id"]),
    )[:80]
    prior_feedback_ids = prior_validation_ids | {
        str(row["id"]) for row in prior_test
    }
    prior_train_general = sorted(
        [row for row in general if str(row["id"]) not in prior_feedback_ids],
        key=lambda row: stable_key("official-benign-train-v2", row["id"]),
    )[: 680 - len(hard_train)]
    prior_selected_ids = prior_feedback_ids | {
        str(row["id"]) for row in hard_train + prior_train_general
    }
    fresh_candidates = [
        row for row in general if str(row["id"]) not in prior_selected_ids
    ]
    validation_rows = sorted(
        fresh_candidates,
        key=lambda row: stable_key("official-benign-validation-v5", row["id"]),
    )[:80]
    validation_ids = {str(row["id"]) for row in validation_rows}
    prior_v5_test = sorted(
        [row for row in fresh_candidates if str(row["id"]) not in validation_ids],
        key=lambda row: stable_key("official-benign-test-v5", row["id"]),
    )[:80]
    prior_v5_test_ids = {str(row["id"]) for row in prior_v5_test}
    test_rows = sorted(
        [
            row for row in fresh_candidates
            if str(row["id"]) not in (validation_ids | prior_v5_test_ids)
        ],
        key=lambda row: stable_key("official-benign-test-v6", row["id"]),
    )[:80]
    feedback_train = hard_train + prior_validation + prior_test + prior_v5_test
    train_general = sorted(
        prior_train_general,
        key=lambda row: stable_key("official-benign-retained-train-v6", row["id"]),
    )[: 680 - len(feedback_train)]
    baseline = {
        "train": feedback_train + train_general,
        "validation": validation_rows,
        "test": test_rows,
        "final": final_rows,
    }
    historical_feedback = {
        str(row["id"])
        for row in hard_train + prior_validation + prior_test + prior_v5_test
    }
    return baseline, historical_feedback


def screened_allocations(
    manifest: dict[str, object],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    screen = manifest["hard_negative_screen"]
    allocations = screen["category_allocations"]
    require(tuple(allocations) == HARD_NEGATIVE_CATEGORIES, "hard-negative categories changed")
    by_split: dict[str, list[str]] = {
        "train": [], "validation": [], "test": [], "external_benign": []
    }
    category_by_id: dict[str, str] = {}
    tuning_in_screening_order: list[str] = []
    external_in_screening_order: list[str] = []
    for category in HARD_NEGATIVE_CATEGORIES:
        allocation = allocations[category]
        require(
            {split: len(allocation[split]) for split in by_split}
            == {"train": 6, "validation": 1, "test": 1, "external_benign": 2},
            f"hard-negative allocation count changed: {category}",
        )
        for split in by_split:
            ids = [str(value) for value in allocation[split]]
            by_split[split].extend(ids)
            for sample_id in ids:
                require(sample_id not in category_by_id, f"duplicate screened ID: {sample_id}")
                category_by_id[sample_id] = category
        tuning_in_screening_order.extend(
            str(value)
            for split in ("train", "validation", "test")
            for value in allocation[split]
        )
        external_in_screening_order.extend(
            str(value) for value in allocation["external_benign"]
        )
    tuning_digest = hashlib.sha256(
        ("\n".join(tuning_in_screening_order) + "\n").encode("utf-8")
    ).hexdigest()
    external_digest = hashlib.sha256(
        ("\n".join(external_in_screening_order) + "\n").encode("utf-8")
    ).hexdigest()
    require(tuning_digest == EXPECTED_TUNING_ID_DIGEST, "screened tuning IDs changed")
    require(external_digest == EXPECTED_EXTERNAL_ID_DIGEST, "screened external IDs changed")
    require(screen["tuning_ids_sha256"] == tuning_digest, "manifest tuning digest changed")
    require(screen["external_ids_sha256"] == external_digest, "manifest external digest changed")
    return category_by_id, {split: tuple(ids) for split, ids in by_split.items()}


def reconstruct_malicious_family_partitions(
    malicious_source: dict[str, dict[str, object]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    regression = {family_number(sample_id) for sample_id in PRIOR_REGRESSION_IDS}
    final = {family_number(sample_id) for sample_id in FINAL_MALICIOUS_IDS}
    category_by_family = {
        family_number(sample_id): str(row["category"])
        for sample_id, row in malicious_source.items()
    }
    available = sorted(set(category_by_family) - regression - final)
    by_category: dict[str, list[int]] = defaultdict(list)
    for family in available:
        by_category[category_by_family[family]].append(family)
    assignments: dict[int, str] = {}
    leftovers: list[int] = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda family: stable_key(
                "bordair-family-split-v1", category, family
            ),
        )
        assignments[ordered[0]] = "validation"
        assignments[ordered[1]] = "test"
        leftovers.extend(ordered[2:])
    remaining = sorted(
        leftovers,
        key=lambda family: stable_key("bordair-family-remainder-v1", family),
    )
    validation_needed = 20 - sum(value == "validation" for value in assignments.values())
    test_needed = 20 - sum(value == "test" for value in assignments.values())
    for family in remaining[:validation_needed]:
        assignments[family] = "validation"
    cursor = validation_needed
    for family in remaining[cursor : cursor + test_needed]:
        assignments[family] = "test"
    cursor += test_needed
    for family in remaining[cursor:]:
        assignments[family] = "train"
    prior = {
        split: sorted(family for family, value in assignments.items() if value == split)
        for split in ("train", "validation", "test")
    }
    prior["regression"] = sorted(regression)
    prior["final"] = sorted(final)

    train_by_category: dict[str, list[int]] = defaultdict(list)
    for family in prior["train"]:
        train_by_category[category_by_family[family]].append(family)
    refreshed_validation: set[int] = set()
    refreshed_test: set[int] = set()
    refresh_leftovers: list[int] = []
    for category in sorted(train_by_category):
        ordered = sorted(
            train_by_category[category],
            key=lambda family: stable_key(
                "rolling-malicious-holdout-v7", category, family
            ),
        )
        refreshed_validation.add(ordered[0])
        refreshed_test.add(ordered[1])
        refresh_leftovers.extend(ordered[2:])
    refresh_remaining = sorted(
        refresh_leftovers,
        key=lambda family: stable_key("rolling-malicious-remainder-v7", family),
    )
    validation_needed = 20 - len(refreshed_validation)
    test_needed = 20 - len(refreshed_test)
    refreshed_validation.update(refresh_remaining[:validation_needed])
    cursor = validation_needed
    refreshed_test.update(refresh_remaining[cursor : cursor + test_needed])
    refreshed_train = (
        set(prior["train"]) - refreshed_validation - refreshed_test
    ) | set(prior["validation"]) | set(prior["test"])
    refreshed = {
        "train": sorted(refreshed_train),
        "validation": sorted(refreshed_validation),
        "test": sorted(refreshed_test),
        "regression": prior["regression"],
        "final": prior["final"],
    }
    return prior, refreshed


def validate_run(run_root: Path = RUN) -> dict[str, object]:
    run_root = run_root.resolve()
    MANIFEST_PATH = run_root / "corpus_manifest_v7.json"
    DEVELOPMENT_PATH = run_root / "development_metadata_v7.csv"
    REGRESSION_PATH = run_root / "regression_metadata_v7.csv"
    FINAL_PATH = run_root / "final_metadata_v7.csv"
    TEXT_LED_FINAL_PATH = run_root / "text_led_final_metadata_v7.csv"
    EXTERNAL_BENIGN_PATH = run_root / "external_benign_metadata_v7.csv"
    IMAGES_DIRECTORY = run_root / "images_v7"
    validate_schema4_anchors(
        MANIFEST_PATH,
        DEVELOPMENT_PATH,
        IMAGES_DIRECTORY,
    )
    archive, archived_development, archived_panel_ids = validate_schema3_archive(
        run_root
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 4, "unexpected manifest schema")
    require(manifest.get("corpus_version") == "v7", "unexpected corpus version")
    require(manifest.get("dataset") == "Bordair/bordair-multimodal", "dataset spelling or name changed")
    require("BordAIR" not in json.dumps(manifest), "manifest contains stale BordAIR spelling")

    development = read_csv(DEVELOPMENT_PATH)
    regression = read_csv(REGRESSION_PATH)
    final = read_csv(FINAL_PATH)
    text_led_final = read_csv(TEXT_LED_FINAL_PATH)
    external_benign = read_csv(EXTERNAL_BENIGN_PATH)
    panels = {
        "development": development, "regression": regression,
        "final": final, "text_led_final": text_led_final,
        "external_benign": external_benign,
    }
    panel_directories = {
        "development": "development", "regression": "regression",
        "final": "final", "text_led_final": "text_led_final",
        "external_benign": "external_benign",
    }
    expected_panel_counts = {
        "development": 1_915, "regression": 10,
        "final": 20, "text_led_final": 10, "external_benign": 20,
    }
    require({name: len(rows) for name, rows in panels.items()} == expected_panel_counts, "panel row counts changed")
    require(
        {
            path.name: sha256(path)
            for path in (
                REGRESSION_PATH,
                FINAL_PATH,
                TEXT_LED_FINAL_PATH,
                EXTERNAL_BENIGN_PATH,
            )
        }
        == FROZEN_METADATA_SHA256,
        "schema-4 generation changed a declared frozen metadata panel",
    )

    malicious_source, benign_source = load_sources(run_root)
    screened_category_by_id, screened_ids_by_split = screened_allocations(manifest)
    additional_category_by_id, additional_training_ids = (
        validate_additional_training_screen(
            manifest,
            benign_source,
            archived_panel_ids,
        )
    )
    baseline_benign, historical_feedback_ids = (
        reconstruct_prior_v7_benign_baseline(benign_source)
    )
    expected_hard_category_by_id = {
        **screened_category_by_id,
        **additional_category_by_id,
        **{
            str(row["id"]): "prior_v7_validation_feedback"
            for row in baseline_benign["validation"]
        },
        **{
            str(row["id"]): "prior_v7_test_feedback"
            for row in baseline_benign["test"]
        },
    }
    malicious_by_family: dict[int, list[dict[str, object]]] = defaultdict(list)
    for source_row in malicious_source.values():
        malicious_by_family[family_number(str(source_row["id"]))].append(source_row)

    all_ids: list[str] = []
    all_paths: list[Path] = []
    rows_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    images_root = IMAGES_DIRECTORY.resolve()
    for panel, panel_rows in panels.items():
        ids = [row["sample_id"] for row in panel_rows]
        require(len(ids) == len(set(ids)), f"duplicate sample ID in {panel}")
        all_ids.extend(ids)
        required_prefix = f"images_v7/{panel_directories[panel]}/"
        for row in panel_rows:
            require(row["image_path"].startswith(required_prefix), f"unversioned or misplaced image: {row['sample_id']}")
            image_path = (run_root / row["image_path"]).resolve()
            require(image_path.is_relative_to(images_root), f"image escapes images_v7: {row['sample_id']}")
            require(image_path.is_file(), f"missing image: {row['image_path']}")
            require(sha256(image_path) == row["image_sha256"], f"image hash mismatch: {row['sample_id']}")
            require(row["split"] in {"train", "validation", "test", "regression", "final", "external_benign"}, f"unexpected split: {row['sample_id']}")
            require(int(row["label_id"]) in {0, 1}, f"unexpected label: {row['sample_id']}")
            require("BordAIR" not in row["source"], f"stale BordAIR spelling: {row['sample_id']}")
            validate_source_provenance(row, malicious_source, benign_source)
            expected_category = (
                expected_hard_category_by_id.get(row["sample_id"], "")
                if int(row["label_id"]) == 0
                else ""
            )
            require(
                row["hard_negative"] == str(int(bool(expected_category))),
                f"hard-negative marker changed: {row['sample_id']}",
            )
            require(
                row["hard_negative_category"] == expected_category,
                f"hard-negative category changed: {row['sample_id']}",
            )
            all_paths.append(image_path)
            rows_by_hash[row["image_sha256"]].append(row)
    require(len(all_ids) == len(set(all_ids)), "sample IDs overlap across panels")
    require(len(all_paths) == len(set(all_paths)), "image paths overlap across panels")

    directory_counts = {
        panel: len(list((IMAGES_DIRECTORY / directory).glob("*.png")))
        for panel, directory in panel_directories.items()
    }
    require(directory_counts == expected_panel_counts, "versioned image directory counts changed")
    require(len(list(IMAGES_DIRECTORY.rglob("*.png"))) == 1_975, "unexpected total v7 image count")

    development_counts = Counter(
        (row["split"], int(row["label_id"]), row["strategy"])
        for row in development
    )
    expected_development_counts = Counter({
        ("train", 0, BENIGN_STRATEGY): 705,
        ("train", 1, IMAGE_ONLY_STRATEGY): 680,
        ("train", 1, COUNTERFACTUAL_STRATEGY): 170,
        ("validation", 0, BENIGN_STRATEGY): 80,
        ("validation", 1, IMAGE_ONLY_STRATEGY): 80,
        ("validation", 1, COUNTERFACTUAL_STRATEGY): 20,
        ("test", 0, BENIGN_STRATEGY): 80,
        ("test", 1, IMAGE_ONLY_STRATEGY): 80,
        ("test", 1, COUNTERFACTUAL_STRATEGY): 20,
    })
    require(development_counts == expected_development_counts, "development class/strategy counts changed")
    require(all(row["split"] == "regression" for row in regression), "bad regression split")
    require(all(row["split"] == "final" for row in final + text_led_final), "bad final split")
    require(all(row["split"] == "external_benign" for row in external_benign), "bad external benign split")
    require(all(int(row["label_id"]) == 1 for row in regression + text_led_final), "non-malicious row in attack-only panel")
    require(all(int(row["label_id"]) == 0 for row in external_benign), "non-benign row in external panel")
    require(all(row["strategy"] == COUNTERFACTUAL_STRATEGY for row in text_led_final), "bad text-led final strategy")

    family_sets = {
        name: set(values) for name, values in manifest["partition_families"].items()
    }
    require(set(family_sets) == {"train", "validation", "test", "regression", "final"}, "unexpected family partitions")
    require({name: len(values) for name, values in family_sets.items()} == {"train": 170, "validation": 20, "test": 20, "regression": 10, "final": 10}, "family counts changed")
    prior_family_partitions, refreshed_family_partitions = (
        reconstruct_malicious_family_partitions(malicious_source)
    )
    require(
        manifest["partition_families"] == refreshed_family_partitions,
        "refreshed malicious family partitions changed",
    )
    require(
        set(prior_family_partitions["validation"])
        | set(prior_family_partitions["test"])
        <= family_sets["train"],
        "inspected malicious holdout family was not moved to train",
    )
    malicious_revision = manifest["malicious_partition_revision"]
    require(malicious_revision["prior_train_families"] == prior_family_partitions["train"], "prior malicious train families changed")
    require(malicious_revision["prior_validation_families"] == prior_family_partitions["validation"], "prior malicious validation families changed")
    require(malicious_revision["prior_test_families"] == prior_family_partitions["test"], "prior malicious test families changed")
    require(malicious_revision["feedback_families_moved_to_train"] == sorted(prior_family_partitions["validation"] + prior_family_partitions["test"]), "malicious feedback family list changed")
    require(malicious_revision["train_families_moved_to_validation"] == refreshed_family_partitions["validation"], "malicious validation replacements changed")
    require(malicious_revision["train_families_moved_to_test"] == refreshed_family_partitions["test"], "malicious test replacements changed")
    require(malicious_revision["retained_prior_train_families"] == sorted(set(prior_family_partitions["train"]) & set(refreshed_family_partitions["train"])), "retained malicious train families changed")
    require(
        malicious_revision["validation_replacements"]
        == [
            {"prior_family": prior, "replacement_family": replacement}
            for prior, replacement in zip(
                prior_family_partitions["validation"],
                refreshed_family_partitions["validation"],
                strict=True,
            )
        ],
        "malicious validation replacement mapping changed",
    )
    require(
        malicious_revision["test_replacements"]
        == [
            {"prior_family": prior, "replacement_family": replacement}
            for prior, replacement in zip(
                prior_family_partitions["test"],
                refreshed_family_partitions["test"],
                strict=True,
            )
        ],
        "malicious test replacement mapping changed",
    )
    family_names = sorted(family_sets)
    for index, left in enumerate(family_names):
        for right in family_names[index + 1:]:
            require(not (family_sets[left] & family_sets[right]), f"family leakage: {left}/{right}")

    development_malicious_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in development:
        if int(row["label_id"]) == 1:
            require(row_family(row) in family_sets[row["split"]], f"development family leakage: {row['sample_id']}")
            development_malicious_by_split[row["split"]].append(row)
    for split in ("train", "validation", "test"):
        split_rows = development_malicious_by_split[split]
        image_only = [row for row in split_rows if row["strategy"] == IMAGE_ONLY_STRATEGY]
        text_led = [row for row in split_rows if row["strategy"] == COUNTERFACTUAL_STRATEGY]
        image_only_family_counts = Counter(row_family(row) for row in image_only)
        text_led_family_counts = Counter(row_family(row) for row in text_led)
        require(set(image_only_family_counts) == family_sets[split], f"missing image-only family in {split}")
        require(set(image_only_family_counts.values()) == {4}, f"image-only family does not have four styles in {split}")
        require(text_led_family_counts == Counter({family: 1 for family in family_sets[split]}), f"text-led family count changed in {split}")
        for family in family_sets[split]:
            family_rows = sorted(
                [row for row in image_only if row_family(row) == family],
                key=lambda row: row["sample_id"],
            )
            expected_source_ids = sorted(
                str(source["id"]) for source in malicious_by_family[family]
            )
            require(
                [row["sample_id"] for row in family_rows] == expected_source_ids,
                f"image-only source variants changed for family {family}",
            )
            require(
                [int(row["render_style"]) for row in family_rows] == [0, 1, 2, 3],
                f"render-style coverage changed for family {family}",
            )
        for row in text_led:
            expected_source = select_full_source_row(row_family(row), malicious_by_family, "development-text-led-v7")
            require(row["source_sample_id"] == str(expected_source["id"]), f"non-deterministic text-led source: {row['sample_id']}")

    benign_rows = [
        row
        for row in development + final + external_benign
        if int(row["label_id"]) == 0
    ]
    benign_by_id = {row["sample_id"]: row for row in benign_rows}
    require(len(benign_by_id) == 895, "benign controls overlap")
    benign_ids_by_split: dict[str, set[str]] = defaultdict(set)
    benign_text_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    benign_image_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    normalized_benign_texts_by_split: dict[str, set[str]] = defaultdict(set)
    for row in benign_rows:
        benign_ids_by_split[row["split"]].add(row["sample_id"])
        benign_text_hashes_by_split[row["split"]].add(
            hashlib.sha256(row["image_text"].encode("utf-8")).hexdigest()
        )
        benign_image_hashes_by_split[row["split"]].add(row["image_sha256"])
        normalized_benign_texts_by_split[row["split"]].add(
            " ".join(row["image_text"].split()).casefold()
        )
    for mapping, label in (
        (benign_ids_by_split, "benign source ID"),
        (benign_text_hashes_by_split, "benign text"),
        (benign_image_hashes_by_split, "benign image"),
        (normalized_benign_texts_by_split, "normalized benign text"),
    ):
        names = sorted(mapping)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                require(not (mapping[left] & mapping[right]), f"{label} leakage: {left}/{right}")

    baseline_ids_by_split = {
        split: {str(row["id"]) for row in rows}
        for split, rows in baseline_benign.items()
    }
    current_ids_by_split = {
        split: {row["sample_id"] for row in benign_rows if row["split"] == split}
        for split in ("train", "validation", "test", "final", "external_benign")
    }
    require(
        baseline_ids_by_split["validation"] <= current_ids_by_split["train"],
        "not all prior-v7 validation rows moved into train",
    )
    require(
        baseline_ids_by_split["test"] <= current_ids_by_split["train"],
        "not all prior-v7 internal-test rows moved into train",
    )
    require(
        current_ids_by_split["validation"].isdisjoint(
            set().union(*baseline_ids_by_split.values())
        ),
        "new validation membership is not completely fresh",
    )
    require(
        current_ids_by_split["test"].isdisjoint(
            set().union(*baseline_ids_by_split.values())
        ),
        "new internal-test membership is not completely fresh",
    )
    require(
        [row["sample_id"] for row in external_benign]
        == list(screened_ids_by_split["external_benign"]),
        "external benign IDs/order changed",
    )

    baseline_pair_ids = [
        str(row["id"])
        for row in sorted(
            baseline_benign["train"],
            key=lambda row: stable_key(
                "text-led-benign-pair-v7", "train", row["id"]
            ),
        )[:170]
    ]
    prior_pair_by_family = dict(
        zip(
            prior_family_partitions["train"],
            baseline_pair_ids,
            strict=True,
        )
    )
    retained_malicious_train_families = set(
        prior_family_partitions["train"]
    ) & set(refreshed_family_partitions["train"])
    freed_pair_ids = [
        prior_pair_by_family[family]
        for family in prior_family_partitions["train"]
        if family not in retained_malicious_train_families
    ]
    incoming_malicious_feedback_families = sorted(
        prior_family_partitions["validation"]
        + prior_family_partitions["test"]
    )
    refreshed_pair_by_family = {
        family: prior_pair_by_family[family]
        for family in retained_malicious_train_families
    }
    refreshed_pair_by_family.update(
        dict(
            zip(
                incoming_malicious_feedback_families,
                freed_pair_ids,
                strict=True,
            )
        )
    )
    expected_refreshed_train_pair_ids = [
        refreshed_pair_by_family[family]
        for family in refreshed_family_partitions["train"]
    ]
    protected_train_ids = historical_feedback_ids | set(baseline_pair_ids)
    evictable_train = [
        row
        for row in baseline_benign["train"]
        if str(row["id"]) not in protected_train_ids
    ]
    expected_evicted = sorted(
        evictable_train,
        key=lambda row: stable_key("rolling-feedback-train-eviction-v7", row["id"]),
    )[:220]
    expected_evicted_ids = {str(row["id"]) for row in expected_evicted}
    expected_schema3_train_ids = (
        baseline_ids_by_split["train"] - expected_evicted_ids
    ) | baseline_ids_by_split["validation"] | baseline_ids_by_split["test"] | set(
        screened_ids_by_split["train"]
    )
    archived_by_id = {row["sample_id"]: row for row in archived_development}
    archived_schema3_train_benign = [
        row
        for row in archived_development
        if row["split"] == "train" and int(row["label_id"]) == 0
    ]
    require(
        {row["sample_id"] for row in archived_schema3_train_benign}
        == expected_schema3_train_ids,
        "archived schema-3 train membership changed",
    )
    schema3_unprotected_ordinary = [
        row
        for row in archived_schema3_train_benign
        if row["hard_negative"] == "0"
        and row["sample_id"] not in set(expected_refreshed_train_pair_ids)
        and row["sample_id"] not in historical_feedback_ids
    ]
    require(
        len(schema3_unprotected_ordinary) == 95,
        "schema-3 unprotected ordinary train count changed",
    )
    schema4_evicted = sorted(
        schema3_unprotected_ordinary,
        key=lambda row: stable_key(
            "schema4-additional-hard-negative-eviction-v1", row["sample_id"]
        ),
    )
    schema4_evicted_ids = {row["sample_id"] for row in schema4_evicted}
    require(len(schema4_evicted_ids) == 95, "schema-4 eviction count changed")
    require(
        all(row["hard_negative"] == "0" for row in schema4_evicted),
        "schema-4 eviction touched a prior hard negative",
    )
    require(
        schema4_evicted_ids.isdisjoint(expected_refreshed_train_pair_ids),
        "schema-4 eviction touched a counterfactual benign pair",
    )
    expected_train_ids = (
        expected_schema3_train_ids - schema4_evicted_ids
    ) | set(additional_training_ids)
    require(
        current_ids_by_split["train"] == expected_train_ids,
        "schema-4 training membership changed",
    )
    protected_schema4_ids = schema4_protected_train_ids(
        archived_schema3_train_benign,
        expected_refreshed_train_pair_ids,
        historical_feedback_ids,
    )
    require(len(historical_feedback_ids) == 247, "historical feedback count changed")
    require(len(protected_schema4_ids) == 585, "schema-4 protected union changed")
    require(
        protected_schema4_ids <= current_ids_by_split["train"],
        "protected schema-3 train rows were evicted",
    )

    current_development_by_id = {row["sample_id"]: row for row in development}
    shared_schema3_ids = set(archived_by_id) & set(current_development_by_id)
    require(len(shared_schema3_ids) == 1_795, "schema-4 unchanged-row count changed")
    require(
        set(archived_by_id) - set(current_development_by_id) == schema4_evicted_ids,
        "schema-4 removed rows differ from the deterministic eviction set",
    )
    require(
        set(current_development_by_id) - set(archived_by_id)
        == set(additional_training_ids),
        "schema-4 added rows differ from the pinned hard-negative slate",
    )
    for sample_id in shared_schema3_ids:
        require(
            current_development_by_id[sample_id] == archived_by_id[sample_id],
            f"schema-4 changed an unchanged development row: {sample_id}",
        )

    carrier_pool = sorted(
        {
            str(row["text"])
            for row in malicious_source.values()
            if row.get("strategy") == IMAGE_ONLY_STRATEGY
        }
    )
    require(len(carrier_pool) >= 20, "too few neutral Bordair carrier prompts")
    for sample_id in additional_training_ids:
        added_row = current_development_by_id[sample_id]
        expected_prompt = carrier_pool[
            int(stable_key("benign-carrier-v2", "train", sample_id)[:8], 16)
            % len(carrier_pool)
        ]
        require(
            added_row["prompt_text"] == expected_prompt,
            f"schema-4 added-row carrier prompt changed: {sample_id}",
        )

    final_texts = {str(row["text"]).strip() for row in baseline_benign["final"]}
    eligible_by_text: dict[str, dict[str, object]] = {}
    for source_row in sorted(benign_source.values(), key=lambda row: str(row["id"])):
        if source_row.get("image_type") != "ocr":
            continue
        text_value = str(source_row["text"]).strip()
        safely_renderable = text_value.isascii() and all(
            character.isprintable() or character in "\r\n\t"
            for character in text_value
        )
        if 8 <= len(text_value) <= 240 and text_value not in final_texts and safely_renderable:
            eligible_by_text.setdefault(text_value, source_row)
    screened_all_ids = set(screened_category_by_id)
    baseline_all_ids = set().union(*baseline_ids_by_split.values())
    fresh_general = [
        row
        for row in eligible_by_text.values()
        if row.get("source") != "edge_cases"
        and str(row["id"]) not in (baseline_all_ids | screened_all_ids)
    ]
    expected_validation_general = sorted(
        fresh_general,
        key=lambda row: stable_key(
            "rolling-feedback-validation-general-v7", row["id"]
        ),
    )[:70]
    validation_general_ids = {str(row["id"]) for row in expected_validation_general}
    expected_test_general = sorted(
        [row for row in fresh_general if str(row["id"]) not in validation_general_ids],
        key=lambda row: stable_key("rolling-feedback-test-general-v7", row["id"]),
    )[:70]
    require(
        current_ids_by_split["validation"]
        == set(screened_ids_by_split["validation"])
        | {str(row["id"]) for row in expected_validation_general},
        "fresh validation selection changed",
    )
    require(
        current_ids_by_split["test"]
        == set(screened_ids_by_split["test"])
        | {str(row["id"]) for row in expected_test_general},
        "fresh internal-test selection changed",
    )

    development_pairs = [
        row for row in development if row["strategy"] == COUNTERFACTUAL_STRATEGY
    ]
    require(len({row["paired_benign_sample_id"] for row in development_pairs}) == 210, "development benign pair reused")
    expected_development_pairs: dict[tuple[str, int], str] = {}
    for split in ("train", "validation", "test"):
        selected_pair_ids = manifest["counterfactual_benign_pair_ids"][split]
        if split == "train":
            require(
                selected_pair_ids == expected_refreshed_train_pair_ids,
                "refreshed train counterfactual assignments changed",
            )
        else:
            expected_fresh_pairs = [
                row["sample_id"]
                for row in sorted(
                    [row for row in benign_rows if row["split"] == split],
                    key=lambda row: stable_key(
                        "text-led-benign-pair-v7", split, row["sample_id"]
                    ),
                )[:20]
            ]
            require(selected_pair_ids == expected_fresh_pairs, f"fresh {split} counterfactual pairs changed")
        expected_development_pairs.update(
            {
                (split, family): pair_id
                for family, pair_id in zip(
                    manifest["partition_families"][split],
                    selected_pair_ids,
                    strict=True,
                )
            }
        )
    for row in development_pairs:
        pair_id = row["paired_benign_sample_id"]
        require(
            pair_id == expected_development_pairs[(row["split"], row_family(row))],
            f"non-deterministic benign pair: {row['sample_id']}",
        )
        require(pair_id in benign_by_id, f"unknown development benign pair: {row['sample_id']}")
        paired = benign_by_id[pair_id]
        require(paired["split"] == row["split"], f"cross-split development pair: {row['sample_id']}")
        require(paired["image_sha256"] == row["image_sha256"], f"counterfactual bytes differ: {row['sample_id']}")
        require(paired["image_text"] == row["image_text"], f"counterfactual image text differs: {row['sample_id']}")
        require(paired["render_style"] == row["render_style"], f"counterfactual render style differs: {row['sample_id']}")

    require([row["sample_id"] for row in regression] == list(PRIOR_REGRESSION_IDS), "regression panel IDs/order changed")
    require([row["sample_id"] for row in final] == list(FINAL_MALICIOUS_IDS + FINAL_BENIGN_IDS), "fixed 10/10 final panel IDs/order changed")
    require([int(row["label_id"]) for row in final] == [1] * 10 + [0] * 10, "fixed final labels changed")
    for row in regression:
        require(row_family(row) in family_sets["regression"], f"regression family mismatch: {row['sample_id']}")
    for row in final[:10]:
        require(row_family(row) in family_sets["final"], f"final family mismatch: {row['sample_id']}")

    expected_text_led_final: list[tuple[str, str, str, int]] = []
    for image_only_id, paired_benign_id in zip(
        FINAL_MALICIOUS_IDS, FINAL_BENIGN_IDS, strict=True
    ):
        family = family_number(image_only_id)
        source = select_full_source_row(family, malicious_by_family, "final-text-led-v7")
        source_id = str(source["id"])
        expected_text_led_final.append((
            f"MTI-{source_id.removeprefix('TI-')}", source_id,
            paired_benign_id, family,
        ))
    observed_text_led_final = [
        (row["sample_id"], row["source_sample_id"],
         row["paired_benign_sample_id"], row_family(row))
        for row in text_led_final
    ]
    require(observed_text_led_final == expected_text_led_final, "text-led final IDs, sources, pairs, or order changed")
    for row in text_led_final:
        paired = benign_by_id[row["paired_benign_sample_id"]]
        require(paired["split"] == "final", f"text-led final pair not held out: {row['sample_id']}")
        require(row_family(row) in family_sets["final"], f"text-led final family mismatch: {row['sample_id']}")
        require(paired["image_sha256"] == row["image_sha256"], f"text-led final bytes differ: {row['sample_id']}")
        require(paired["image_text"] == row["image_text"], f"text-led final image text differs: {row['sample_id']}")

    duplicate_hash_groups = [rows for rows in rows_by_hash.values() if len(rows) > 1]
    require(len(duplicate_hash_groups) == 220, "unexpected number of intentional image-byte pairs")
    require(len(rows_by_hash) == 1_755, "unexpected unique v7 image hash count")
    for shared_rows in duplicate_hash_groups:
        require(len(shared_rows) == 2, "image bytes used by more than one counterfactual pair")
        require({int(row["label_id"]) for row in shared_rows} == {0, 1}, "shared bytes do not form benign/malicious pair")
        counterfactual = next(row for row in shared_rows if int(row["label_id"]) == 1)
        benign = next(row for row in shared_rows if int(row["label_id"]) == 0)
        require(counterfactual["strategy"] == COUNTERFACTUAL_STRATEGY, "non-counterfactual attack shares benign bytes")
        require(counterfactual["paired_benign_sample_id"] == benign["sample_id"], "shared-byte pair ID mismatch")
        require(counterfactual["split"] == benign["split"], "shared image bytes cross partitions")
        require(counterfactual["image_text"] == benign["image_text"], "shared bytes have different image text")

    image_only_text_splits: dict[str, set[str]] = defaultdict(set)
    for row in development + regression + final:
        if int(row["label_id"]) == 1 and row["strategy"] == IMAGE_ONLY_STRATEGY:
            image_only_text_splits[row["image_text"]].add(row["split"])
    require(not {
        text: splits for text, splits in image_only_text_splits.items()
        if len(splits) > 1
    }, "malicious image text leaks across partitions")

    benign_groups = {
        split: set(values)
        for split, values in manifest["benign_text_groups"].items()
    }
    require(set(benign_groups) == {"train", "validation", "test", "final", "external_benign"}, "unexpected benign manifest partitions")
    require({split: len(values) for split, values in benign_groups.items()} == {"train": 705, "validation": 80, "test": 80, "final": 10, "external_benign": 20}, "benign manifest counts changed")
    for split, hashes in benign_text_hashes_by_split.items():
        require(hashes == benign_groups[split], f"benign manifest hashes differ in {split}")

    require(manifest["development"]["rows"] == 1_915, "manifest development count changed")
    require(manifest["development"]["image_only_malicious_rows"] == 840, "manifest image-led count changed")
    require(manifest["development"]["text_led_malicious_rows"] == 210, "manifest text-led count changed")
    require(manifest["development"]["benign_rows"] == 865, "manifest benign count changed")
    require(manifest["development"]["hard_negative_benign_rows"] == 360, "manifest hard-negative count changed")
    require(
        manifest["development"]["hard_negative_benign_per_split"]
        == {"train": 340, "validation": 10, "test": 10},
        "manifest split hard-negative counts changed",
    )
    require(manifest["final_evaluation"]["rows"] == 20, "manifest final count changed")
    require(manifest["final_evaluation"]["malicious_rows"] == 10, "manifest final malicious count changed")
    require(manifest["final_evaluation"]["benign_rows"] == 10, "manifest final benign count changed")
    require(manifest["final_evaluation"]["malicious_ids"] == list(FINAL_MALICIOUS_IDS), "manifest final malicious IDs changed")
    require(manifest["final_evaluation"]["benign_ids"] == list(FINAL_BENIGN_IDS), "manifest final benign IDs changed")
    require(manifest["text_led_final_evaluation"]["rows"] == 10, "manifest text-led final count changed")
    require(manifest["external_benign_evaluation"]["rows"] == 20, "manifest external benign count changed")
    require(manifest["external_benign_evaluation"]["ids"] == list(screened_ids_by_split["external_benign"]), "manifest external benign IDs changed")
    require(
        manifest["external_benign_evaluation"]["category_counts"]
        == {category: 2 for category in HARD_NEGATIVE_CATEGORIES},
        "manifest external category counts changed",
    )

    rolling = manifest["rolling_feedback"]
    require(rolling["baseline_train_ids"] == sorted(baseline_ids_by_split["train"]), "manifest baseline train IDs changed")
    require(rolling["baseline_validation_ids"] == [str(row["id"]) for row in baseline_benign["validation"]], "manifest baseline validation IDs changed")
    require(rolling["baseline_test_ids"] == [str(row["id"]) for row in baseline_benign["test"]], "manifest baseline test IDs changed")
    require(rolling["historical_feedback_ids"] == sorted(historical_feedback_ids), "manifest historical feedback IDs changed")
    require(rolling["protected_train_pair_ids"] == baseline_pair_ids, "manifest protected pair IDs changed")
    require(rolling["evicted_train_ids"] == [str(row["id"]) for row in expected_evicted], "manifest evicted train IDs changed")
    require(rolling["retained_baseline_train_rows"] == 460, "manifest retained train count changed")
    require(rolling["moved_validation_feedback_rows"] == 80, "manifest moved validation count changed")
    require(rolling["moved_test_feedback_rows"] == 80, "manifest moved test count changed")
    require(rolling["mined_train_rows"] == 60, "manifest mined train count changed")
    require(rolling["fresh_validation_general_ids"] == [str(row["id"]) for row in expected_validation_general], "manifest fresh validation IDs changed")
    require(rolling["fresh_test_general_ids"] == [str(row["id"]) for row in expected_test_general], "manifest fresh test IDs changed")
    require(rolling["schema3_training_rows"] == 680, "manifest schema-3 train count changed")
    require(
        rolling["schema4_additional_hard_negative_ids"]
        == additional_training_ids,
        "manifest schema-4 additional IDs changed",
    )
    require(
        rolling["schema4_evicted_ordinary_train_ids"]
        == [row["sample_id"] for row in schema4_evicted],
        "manifest schema-4 eviction IDs changed",
    )
    require(
        rolling["schema4_protected_train_pair_rows"] == 170,
        "manifest schema-4 protected pair count changed",
    )
    require(
        rolling["schema4_protected_prior_hard_negative_rows"] == 220,
        "manifest schema-4 protected hard-negative count changed",
    )
    require(
        rolling["schema4_protected_historical_feedback_rows"] == 247,
        "manifest schema-4 protected historical-feedback count changed",
    )
    require(
        rolling["schema4_protected_train_union_rows"] == 585,
        "manifest schema-4 protected union count changed",
    )
    require(
        rolling["schema4_unprotected_ordinary_train_rows"] == 95,
        "manifest schema-4 unprotected ordinary count changed",
    )
    require(
        rolling["schema4_train_growth_rows"] == 25,
        "manifest schema-4 train growth changed",
    )

    archive_lineage = manifest.get("schema3_archive")
    require(isinstance(archive_lineage, dict), "manifest lacks schema-3 archive lineage")
    require(
        archive_lineage.get("path") == SCHEMA3_ARCHIVE_RELATIVE.as_posix(),
        "manifest schema-3 archive path changed",
    )
    require(
        archive_lineage.get("archive_manifest_sha256")
        == SCHEMA3_ARCHIVE_MANIFEST_SHA256,
        "manifest schema-3 archive hash changed",
    )
    require(
        archive_lineage.get("source_manifest_sha256")
        == archive["files"]["corpus_manifest_v7.json"]["sha256"],
        "manifest schema-3 source-manifest lineage changed",
    )
    require(
        archive_lineage.get("source_development_metadata_sha256")
        == archive["files"]["development_metadata_v7.csv"]["sha256"],
        "manifest schema-3 development lineage changed",
    )
    require(
        archive_lineage.get("source_images_tree_sha256")
        == archive["trees"]["images_v7"]["sha256"],
        "manifest schema-3 image lineage changed",
    )
    require(
        archive_lineage.get("source_llava_feature_tree_sha256")
        == archive["trees"]["llava05b"]["sha256"],
        "manifest schema-3 LLaVA-feature lineage changed",
    )
    require(
        archive_lineage.get("unchanged_development_rows") == 1_795
        and archive_lineage.get("replaced_training_benign_rows") == 95
        and archive_lineage.get("added_training_benign_rows") == 120
        and archive_lineage.get("training_growth_rows") == 25,
        "manifest schema-4 transition counts changed",
    )

    metadata_entries = (
        ("development", DEVELOPMENT_PATH),
        ("regression", REGRESSION_PATH),
        ("final_evaluation", FINAL_PATH),
        ("text_led_final_evaluation", TEXT_LED_FINAL_PATH),
        ("external_benign_evaluation", EXTERNAL_BENIGN_PATH),
    )
    for key, path in metadata_entries:
        require(manifest[key]["metadata"] == path.name, f"manifest points to wrong {key} metadata")
        require(manifest[key]["metadata_sha256"] == sha256(path), f"manifest {key} metadata hash changed")
    require(manifest["benign_source"]["path"] == "source/bordair_benign_multimodal_text_image.json", "manifest benign source path changed")
    require(manifest["benign_source"]["sha256"] == BENIGN_SOURCE_SHA256, "manifest benign source hash changed")
    require({entry["path"]: entry["sha256"] for entry in manifest["source_shards"]} == EXPECTED_SOURCE_SHARDS, "manifest source hashes changed")

    result = {
        "ok": True,
        "corpus_version": "v7",
        "corpus_schema_version": 4,
        "image_paths": len(all_paths),
        "unique_image_hashes": len(rows_by_hash),
        "intentional_counterfactual_image_pairs": len(duplicate_hash_groups),
        "development_rows": len(development),
        "development_image_only_malicious": 840,
        "development_text_led_malicious": 210,
        "development_benign": 865,
        "regression_rows": len(regression),
        "final_rows": len(final),
        "text_led_final_rows": len(text_led_final),
        "external_benign_rows": len(external_benign),
        "development_hard_negative_rows": sum(
            int(row["hard_negative"])
            for row in development
            if int(row["label_id"]) == 0
        ),
        "old_development_benign_rows_reused": len(
            set().union(
                baseline_ids_by_split["train"],
                baseline_ids_by_split["validation"],
                baseline_ids_by_split["test"],
            )
            & set().union(
                current_ids_by_split["train"],
                current_ids_by_split["validation"],
                current_ids_by_split["test"],
            )
        ),
        "old_train_rows_retained_in_train": len(
            baseline_ids_by_split["train"] & current_ids_by_split["train"]
        ),
        "old_evaluation_rows_moved_to_train": len(
            (baseline_ids_by_split["validation"] | baseline_ids_by_split["test"])
            & current_ids_by_split["train"]
        ),
        "old_train_rows_evicted_during_schema3_feedback": len(expected_evicted_ids),
        "schema3_development_rows_reused": len(shared_schema3_ids),
        "schema4_train_rows_replaced": len(schema4_evicted_ids),
        "schema4_additional_hard_negative_rows": len(additional_training_ids),
        "schema4_additional_ids_sha256": EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST,
        "fresh_development_benign_rows_added_since_baseline": 340,
        "family_counts": {name: len(values) for name, values in family_sets.items()},
        "image_directory_counts": directory_counts,
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN,
        help="Bordair run root containing corpus_manifest_v7.json.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(validate_run(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

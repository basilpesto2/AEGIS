from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REPOSITORY = Path(__file__).resolve().parents[3]
RUN = REPOSITORY / "outputs" / "bordair_retraining_v1"
SOURCE = REPOSITORY / "outputs" / "bordair_eval" / "source"
PRIOR_CASES = REPOSITORY / "outputs" / "bordair_eval" / "cases.json"
BENIGN_SOURCE = RUN / "source" / "bordair_benign_multimodal_text_image.json"
DATASET_NAME = "Bordair/bordair-multimodal"
DATASET_COMMIT = "0e7dbe76ff8730312136019400becbe26d74e4ee"
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

CANVAS_SIZE = (768, 768)
MARGIN = 64
MAX_FONT_SIZE = 48
MIN_FONT_SIZE = 10
LINE_SPACING = 14
FONT_CANDIDATES = (
    (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ),
    (
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ),
    (
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ),
)

IMAGES_DIRECTORY = Path("images_v7")
DEVELOPMENT_METADATA = RUN / "development_metadata_v7.csv"
REGRESSION_METADATA = RUN / "regression_metadata_v7.csv"
FINAL_METADATA = RUN / "final_metadata_v7.csv"
TEXT_LED_FINAL_METADATA = RUN / "text_led_final_metadata_v7.csv"
EXTERNAL_BENIGN_METADATA = RUN / "external_benign_metadata_v7.csv"
CORPUS_MANIFEST = RUN / "corpus_manifest_v7.json"
FULL_MALICIOUS_TEXT_STRATEGIES = (
    "authority_payload_split",
    "context_switch_injection",
)
COUNTERFACTUAL_STRATEGY = "malicious_text_benign_image_counterfactual"

HARD_NEGATIVE_ALLOCATIONS = {
    "ignore_rule": {
        "train": ("BTI-04302", "BTI-05123", "BTI-01285", "BTI-04286", "BTI-04982", "BTI-01020"),
        "validation": ("BTI-00367",), "test": ("BTI-00353",),
        "external_benign": ("BTI-03114", "BTI-05032"),
    },
    "prompt_meta": {
        "train": ("BTI-01955", "BTI-05797", "BTI-05385", "BTI-04782", "BTI-01268", "BTI-04405"),
        "validation": ("BTI-00188",), "test": ("BTI-03442",),
        "external_benign": ("BTI-03782", "BTI-05745"),
    },
    "policy_legal": {
        "train": ("BTI-05750", "BTI-02811", "BTI-03839", "BTI-02628", "BTI-02994", "BTI-01172"),
        "validation": ("BTI-03085",), "test": ("BTI-04143",),
        "external_benign": ("BTI-05472", "BTI-03542"),
    },
    "defensive_security": {
        "train": ("BTI-01571", "BTI-06324", "BTI-03128", "BTI-04424", "BTI-00724", "BTI-02992"),
        "validation": ("BTI-02064",), "test": ("BTI-03616",),
        "external_benign": ("BTI-06440", "BTI-04994"),
    },
    "credentials_auth": {
        "train": ("BTI-03570", "BTI-01770", "BTI-04571", "BTI-00258", "BTI-04170", "BTI-03143"),
        "validation": ("BTI-04021",), "test": ("BTI-01296",),
        "external_benign": ("BTI-03541", "BTI-02067"),
    },
    "code_errors": {
        "train": ("BTI-05325", "BTI-04467", "BTI-05426", "BTI-04612", "BTI-03158", "BTI-01884"),
        "validation": ("BTI-01793",), "test": ("BTI-06350",),
        "external_benign": ("BTI-03823", "BTI-00012"),
    },
    "sysadmin_tools": {
        "train": ("BTI-04963", "BTI-02848", "BTI-04180", "BTI-05481", "BTI-06002", "BTI-00932"),
        "validation": ("BTI-03675",), "test": ("BTI-03196",),
        "external_benign": ("BTI-02784", "BTI-03910"),
    },
    "binary_structured": {
        "train": ("BTI-01673", "BTI-03577", "BTI-04409", "BTI-03762", "BTI-03099", "BTI-03661"),
        "validation": ("BTI-06241",), "test": ("BTI-00325",),
        "external_benign": ("BTI-04766", "BTI-06024"),
    },
    "foreign_terse": {
        "train": ("BTI-04378", "BTI-02133", "BTI-04764", "BTI-03917", "BTI-05234", "BTI-04648"),
        "validation": ("BTI-00278",), "test": ("BTI-02699",),
        "external_benign": ("BTI-04340", "BTI-01913"),
    },
    "lexical_false_friends": {
        "train": ("BTI-01908", "BTI-02862", "BTI-05757", "BTI-05669", "BTI-04278", "BTI-06351"),
        "validation": ("BTI-01246",), "test": ("BTI-05293",),
        "external_benign": ("BTI-00764", "BTI-04308"),
    },
}


def allocated_ids(split: str) -> tuple[str, ...]:
    return tuple(
        sample_id
        for allocation in HARD_NEGATIVE_ALLOCATIONS.values()
        for sample_id in allocation[split]
    )


SAFE_HARD_NEGATIVE_IDS = allocated_ids("train")
VALIDATION_HARD_NEGATIVE_IDS = allocated_ids("validation")
TEST_HARD_NEGATIVE_IDS = allocated_ids("test")
EXTERNAL_BENIGN_IDS = allocated_ids("external_benign")
MINED_HARD_NEGATIVE_CATEGORY_BY_ID = {
    sample_id: category
    for category, allocation in HARD_NEGATIVE_ALLOCATIONS.items()
    for ids in allocation.values()
    for sample_id in ids
}
EXPECTED_TUNING_ID_DIGEST = "93c1eceb4c6cf68491a5ec6bc1b6f4bf18937c690aeec4cb540e7117be17807d"
EXPECTED_EXTERNAL_ID_DIGEST = "9dcd6753b45ce114967981d995851fb66d050387a9a6cf220bf30366643d34e2"

# A development-only expansion mined from official benign OCR rows that were
# unused by schema 3. Every entry was manually reviewed as benign. The regex is
# retained as explicit, independently checkable category provenance; the pinned
# IDs and text digest prevent a broad expression from silently changing the set.
ADDITIONAL_TRAIN_HARD_NEGATIVE_RULES = {
    "ignore_rule": {
        "regex": r"(?i)\b(ignore|override|bypass|disregard|circumvent|skip|omit\w*|exclude|leave|do not|don['’]t|without|instead|replace\w*|contrary|opposite|remove\w*)\b",
        "description": "Benign editing or everyday negation/override wording.",
        "ids": (
            "BTI-04888", "BTI-04564", "BTI-02421", "BTI-01645",
            "BTI-05597", "BTI-02997", "BTI-00533", "BTI-03410",
            "BTI-00939", "BTI-00164", "BTI-03770", "BTI-01483",
        ),
    },
    "prompt_meta": {
        "regex": r"(?i)\b(prompt\w*|instruction\w*|chatgpt|gpt(?:-?\d(?:\.\d)?)?|assistant)\b",
        "description": "Benign discussion of prompts, instructions, or assistants.",
        "ids": (
            "BTI-05992", "BTI-04907", "BTI-02773", "BTI-05778",
            "BTI-03229", "BTI-06137", "BTI-05951", "BTI-00465",
            "BTI-06335", "BTI-00862", "BTI-00490", "BTI-01123",
        ),
    },
    "policy_legal": {
        "regex": r"(?i)\b(policy|law|rights?|court|government\w*|guidelines?|censor\w*|constitution\w*|ethical|elections?|congress\w*|public schools?)\b",
        "description": "Benign legal, public-policy, rights, or ethics discussion.",
        "ids": (
            "BTI-04155", "BTI-02650", "BTI-00329", "BTI-04158",
            "BTI-05317", "BTI-00276", "BTI-00403", "BTI-03756",
            "BTI-04783", "BTI-05581", "BTI-06052", "BTI-06159",
        ),
    },
    "defensive_security": {
        "regex": r"(?i)\b(security|safe|protection|protect|threats?|risk|prevent|dangers?|healthy|health)\b",
        "description": "Benign safety, protection, health, or risk-reduction wording.",
        "ids": (
            "BTI-01569", "BTI-03840", "BTI-02591", "BTI-04700",
            "BTI-06001", "BTI-02519", "BTI-03348", "BTI-05934",
            "BTI-01565", "BTI-00890", "BTI-00538", "BTI-04451",
        ),
    },
    "credentials_auth": {
        "regex": r"(?i)\b(accounts?|access|username|email|login|customer portal)\b",
        "description": "Benign account, identity, email, login, or access wording.",
        "ids": (
            "BTI-05583", "BTI-02627", "BTI-01128", "BTI-02870",
            "BTI-04291", "BTI-01314", "BTI-02142", "BTI-02308",
            "BTI-04188", "BTI-06195", "BTI-06264", "BTI-02685",
        ),
    },
    "code_errors": {
        "regex": r"(?i)\b(problem|fix|invalid|cannot|unable|error|missing)\b",
        "description": "Benign troubleshooting or error-adjacent wording.",
        "ids": (
            "BTI-05639", "BTI-04304", "BTI-03260", "BTI-00995",
            "BTI-00906", "BTI-04380", "BTI-03946", "BTI-05895",
            "BTI-04636", "BTI-03237", "BTI-06329", "BTI-00337",
        ),
    },
    "sysadmin_tools": {
        "regex": r"(?i)\b(linux|python|java|c\+\+|c programming|esp32|code|algorithm|operating system|windows|output file)\b",
        "description": "Benign development, operating-system, or tooling requests.",
        "ids": (
            "BTI-00302", "BTI-00768", "BTI-02481", "BTI-02770",
            "BTI-02901", "BTI-03140", "BTI-03463", "BTI-04525",
            "BTI-05091", "BTI-05254", "BTI-05428", "BTI-02212",
        ),
    },
    "binary_structured": {
        "regex": r"(?i)\b(data|algorithm|string|matlab|xml|integer|linked list|hexadecimal|array)\b",
        "description": "Benign structured-data, encoding, and algorithm language.",
        "ids": (
            "BTI-06331", "BTI-04527", "BTI-01122", "BTI-05858",
            "BTI-00961", "BTI-01501", "BTI-03572", "BTI-03303",
            "BTI-00149", "BTI-00173", "BTI-01942", "BTI-06394",
        ),
    },
    "foreign_terse": {
        "regex": r"(?i)\b(chancellor|atomic|europe|mallorca|germany|corona|atomausstieg|wochenende|trump|labor market|hobby|world)\b",
        "description": "Benign terse or multilingual queries from the pinned source.",
        "ids": (
            "BTI-00722", "BTI-00833", "BTI-00948", "BTI-01225",
            "BTI-01332", "BTI-01829", "BTI-02071", "BTI-02384",
            "BTI-03112", "BTI-04494", "BTI-05833", "BTI-06048",
        ),
    },
    "lexical_false_friends": {
        "regex": r"(?i)\b(war|fight|crime|theft|dead|attack|delete|remove|harm|burglary|death|viruses?)\b",
        "description": "Benign uses of lexically alarming words.",
        "ids": (
            "BTI-00328", "BTI-01450", "BTI-02459", "BTI-03817",
            "BTI-03847", "BTI-04691", "BTI-04885", "BTI-05007",
            "BTI-05509", "BTI-05984", "BTI-06172", "BTI-01392",
        ),
    },
}
EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST = "5e8655a185e5bbedae414b3f6f085e4de86a62dd501f2501bf9f0d7d3f78ede9"
EXPECTED_ADDITIONAL_TRAIN_TEXT_DIGEST = "48494292e8e576c5d789d3fc9e45d15439e0fe20be400c99031b94d9b327de3b"
SCHEMA3_ARCHIVE = RUN / "features_v7_trials" / "schema3_pre_additional_hard_negatives"
SCHEMA3_ARCHIVE_MANIFEST_SHA256 = "3267339b2a97a14b8568974026836fd4b823daa16f84652de019aacbb5e170eb"
FROZEN_METADATA_SHA256 = {
    "regression_metadata_v7.csv": "f6c06498c9aa4f21bc6520474bb5de2d9a0d47912ad0a1ed482dba4135628a8a",
    "final_metadata_v7.csv": "d171cc1e4e85dcfcc983e837add08b4a64895386bc6e7f5917808153e2f5f213",
    "text_led_final_metadata_v7.csv": "87fddbee0146aa17c8454c768dda058946b00ddb0fbab6e572ad08348bf762d6",
    "external_benign_metadata_v7.csv": "bca44d53483ba029adad2db7a513dda2a88eef5d85bb1fb0be4ecde94d07ec02",
}

# These ten families contain the already-observed failures and remain outside
# every fit, validation, and internal-test partition.
PRIOR_REGRESSION_IDS = (
    "TI-00001",
    "TI-02241",
    "TI-03921",
    "TI-06161",
    "TI-04481",
    "TI-05461",
    "TI-01121",
    "TI-03361",
    "TI-05041",
    "TI-01681",
)

# Fixed before retraining. Each row is the image-only attack form: the caller
# text is benign and the malicious instruction is contained in the image.
FINAL_MALICIOUS_IDS = (
    "TI-01093",  # exfiltration
    "TI-03333",  # social engineering
    "TI-06133",  # hypothetical
    "TI-00533",  # direct override
    "TI-01653",  # DAN jailbreak
    "TI-02213",  # template injection
    "TI-02773",  # authority impersonation
    "TI-03893",  # encoding obfuscation
    "TI-04453",  # context switching
    "TI-05013",  # compliance forcing
)

# Frozen before detector adaptation. These official Bordair benign OCR rows
# remain outside every fit, validation, and internal-test partition.
FINAL_BENIGN_IDS = (
    "BTI-00930",  # benign use of "ignore"
    "BTI-01460",  # benign use of "prompt system"
    "BTI-02861",  # benign software-rule wording
    "BTI-05697",  # benign use of "exploit"
    "BTI-06083",  # benign use of "ignore all"
    "BTI-05087",  # additional fixed benign control
    "BTI-01896",  # additional fixed benign control
    "BTI-04521",  # additional fixed benign control
    "BTI-05202",  # additional fixed benign control
    "BTI-05112",  # additional fixed benign control
)


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
        relative = path.relative_to(root).as_posix()
        lines.append(f"{relative}|{byte_count}|{sha256(path)}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return len(files), total_bytes, hashlib.sha256(payload).hexdigest()


def validate_schema3_archive() -> dict[str, object]:
    archive_manifest_path = SCHEMA3_ARCHIVE / "archive_manifest.json"
    if not archive_manifest_path.is_file():
        raise RuntimeError("missing immutable schema-3 archive manifest")
    if sha256(archive_manifest_path) != SCHEMA3_ARCHIVE_MANIFEST_SHA256:
        raise RuntimeError("schema-3 archive manifest changed")
    archive = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if archive.get("corpus_schema_version") != 3:
        raise RuntimeError("schema-3 archive declares an unexpected corpus schema")
    files = archive.get("files")
    trees = archive.get("trees")
    if not isinstance(files, dict) or not isinstance(trees, dict):
        raise RuntimeError("schema-3 archive manifest is incomplete")
    for relative, expected in files.items():
        path = SCHEMA3_ARCHIVE / relative
        if not path.is_file():
            raise RuntimeError(f"schema-3 archive file is missing: {relative}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"schema-3 archive byte count changed: {relative}")
        if sha256(path) != str(expected["sha256"]):
            raise RuntimeError(f"schema-3 archive hash changed: {relative}")
    for relative, expected in trees.items():
        observed = tree_digest(SCHEMA3_ARCHIVE / relative)
        wanted = (
            int(expected["files"]),
            int(expected["bytes"]),
            str(expected["sha256"]),
        )
        if observed != wanted:
            raise RuntimeError(f"schema-3 archive tree changed: {relative}")
    return archive


def resolve_fonts() -> tuple[Path, Path, Path]:
    selected: list[Path] = []
    failures: list[str] = []
    for role_index, candidates in enumerate(FONT_CANDIDATES, start=1):
        chosen: Path | None = None
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                ImageFont.truetype(str(candidate), 12)
            except OSError as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            chosen = candidate.resolve()
            break
        if chosen is None:
            candidate_list = ", ".join(str(path) for path in candidates)
            suffix = f"; load failures: {'; '.join(failures)}" if failures else ""
            raise RuntimeError(
                f"no compatible TrueType font found for rendering role {role_index}; "
                f"checked {candidate_list}{suffix}"
            )
        selected.append(chosen)
    return selected[0], selected[1], selected[2]


def family_number(sample_id: str) -> int:
    return (int(sample_id.removeprefix("TI-")) - 1) // 28


def stable_key(*values: object) -> str:
    text = "|".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows() -> tuple[list[dict[str, object]], list[Path]]:
    shards = sorted(SOURCE.glob("text_image_*.json"))
    if len(shards) != 13:
        raise RuntimeError(f"expected 13 source shards, found {len(shards)}")
    observed_hashes = {shard.name: sha256(shard) for shard in shards}
    if observed_hashes != EXPECTED_SOURCE_SHARDS:
        raise RuntimeError("pinned Bordair malicious source shard hashes changed")
    rows: list[dict[str, object]] = []
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"expected a list in {shard}")
        rows.extend(payload)
    if len(rows) != 6_440:
        raise RuntimeError(f"expected 6,440 rows, found {len(rows)}")
    return rows, shards


def validate_additional_training_hard_negatives(
    all_by_id: dict[str, dict[str, object]],
    already_selected_ids: set[str],
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    if tuple(ADDITIONAL_TRAIN_HARD_NEGATIVE_RULES) != tuple(
        HARD_NEGATIVE_ALLOCATIONS
    ):
        raise RuntimeError("additional hard-negative category order changed")
    category_by_id: dict[str, str] = {}
    selected_rows: list[dict[str, object]] = []
    id_lines: list[str] = []
    text_lines: list[str] = []
    text_to_id: dict[str, str] = {}
    already_selected_texts = {
        str(all_by_id[sample_id].get("text", "")).strip()
        for sample_id in already_selected_ids
    }
    rule_manifest: dict[str, object] = {}
    for category, rule in ADDITIONAL_TRAIN_HARD_NEGATIVE_RULES.items():
        pattern = str(rule["regex"])
        ids = tuple(str(value) for value in rule["ids"])
        if len(ids) != 12 or len(ids) != len(set(ids)):
            raise RuntimeError(
                f"additional hard-negative category must pin 12 unique IDs: {category}"
            )
        compiled = re.compile(pattern)
        for sample_id in ids:
            if sample_id in category_by_id:
                raise RuntimeError(f"additional hard-negative ID repeated: {sample_id}")
            if sample_id in already_selected_ids:
                raise RuntimeError(
                    f"additional hard negative was not unused in schema 3: {sample_id}"
                )
            row = all_by_id.get(sample_id)
            if row is None:
                raise RuntimeError(
                    f"additional hard-negative source ID is missing: {sample_id}"
                )
            text = str(row.get("text", "")).strip()
            safely_renderable = text.isascii() and all(
                character.isprintable() or character in "\r\n\t"
                for character in text
            )
            if (
                row.get("expected_detection") is not False
                or row.get("image_type") != "ocr"
                or not 8 <= len(text) <= 240
                or not safely_renderable
            ):
                raise RuntimeError(
                    f"additional hard negative is not an eligible official benign OCR row: "
                    f"{sample_id}"
                )
            if compiled.search(text) is None:
                raise RuntimeError(
                    f"additional hard negative does not match its pinned regex: {sample_id}"
                )
            if text in already_selected_texts:
                raise RuntimeError(
                    f"additional hard-negative text was already used in schema 3: {sample_id}"
                )
            prior_id = text_to_id.setdefault(text, sample_id)
            if prior_id != sample_id:
                raise RuntimeError(
                    f"additional hard-negative text is duplicated: {prior_id}, {sample_id}"
                )
            category_by_id[sample_id] = category
            selected_rows.append(row)
            id_lines.append(f"{category}|{sample_id}\n")
            text_lines.append(f"{category}|{sample_id}|{text}\n")
        rule_manifest[category] = {
            "regex": pattern,
            "description": str(rule["description"]),
            "ids": list(ids),
            "rows": len(ids),
            "manual_review": "confirmed benign; no harmful instruction or requested harm",
        }
    if len(selected_rows) != 120 or len(category_by_id) != 120:
        raise RuntimeError("additional hard-negative selection must contain 120 rows")
    id_digest = hashlib.sha256("".join(id_lines).encode("utf-8")).hexdigest()
    text_digest = hashlib.sha256("".join(text_lines).encode("utf-8")).hexdigest()
    if id_digest != EXPECTED_ADDITIONAL_TRAIN_ID_DIGEST:
        raise RuntimeError("additional hard-negative ID digest changed")
    if text_digest != EXPECTED_ADDITIONAL_TRAIN_TEXT_DIGEST:
        raise RuntimeError("additional hard-negative text digest changed")
    provenance = {
        "source_population": "2,246 pinned official benign OCR rows",
        "selection_scope": "unused schema-3 rows after mechanically excluding all declared panels",
        "selection_method": (
            "Pinned one-to-one ID/text slate; each row must match its category regex, "
            "retain the official benign label, and pass manual benign review."
        ),
        "rows": len(selected_rows),
        "rows_per_category": 12,
        "ids_sha256": id_digest,
        "texts_sha256": text_digest,
        "rules": rule_manifest,
    }
    return selected_rows, category_by_id, provenance


def load_and_partition_benign_ocr() -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, dict[str, object]],
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, object],
]:
    if sha256(BENIGN_SOURCE) != BENIGN_SOURCE_SHA256:
        raise RuntimeError("official Bordair benign source hash changed")
    payload = json.loads(BENIGN_SOURCE.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 6_423:
        raise RuntimeError("unexpected official Bordair benign text-image row count")
    if any(row.get("expected_detection") is not False for row in payload):
        raise RuntimeError("official Bordair benign source contains a non-benign label")
    all_by_id = {str(row["id"]): row for row in payload}
    if len(all_by_id) != len(payload):
        raise RuntimeError("official benign source sample IDs are not unique")
    ocr = [row for row in payload if row.get("image_type") == "ocr"]
    if len(ocr) != 2_246:
        raise RuntimeError(f"expected 2,246 official benign OCR rows, found {len(ocr)}")
    ocr_by_id = {str(row["id"]): row for row in ocr}
    if len(ocr_by_id) != len(ocr):
        raise RuntimeError("official benign OCR sample IDs are not unique")
    if any(not str(row.get("text", "")).strip() for row in ocr):
        raise RuntimeError("official benign OCR source contains an empty caller prompt")
    if any(not str(row.get("image_content", "")).strip() for row in ocr):
        raise RuntimeError("official benign OCR source contains empty image content")

    final_rows = [ocr_by_id[sample_id] for sample_id in FINAL_BENIGN_IDS]
    final_texts = {str(row["text"]).strip() for row in final_rows}
    if len(final_rows) != 10 or len(final_texts) != 10:
        raise RuntimeError("final benign rows must reserve ten distinct source texts")
    if any(
        not text.isascii()
        or any(not character.isprintable() and character not in "\r\n\t" for character in text)
        for text in final_texts
    ):
        raise RuntimeError("final benign texts must be safely renderable as ASCII")

    eligible_by_text: dict[str, dict[str, object]] = {}
    for row in sorted(ocr, key=lambda value: str(value["id"])):
        text = str(row["text"]).strip()
        safely_renderable = text.isascii() and all(
            character.isprintable() or character in "\r\n\t" for character in text
        )
        if not 8 <= len(text) <= 240 or text in final_texts or not safely_renderable:
            continue
        eligible_by_text.setdefault(text, row)
    eligible = list(eligible_by_text.values())
    if len(eligible) < 1_000:
        raise RuntimeError("too few concise unique official benign texts")

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
    prior_test_candidates = [
        row for row in general if str(row["id"]) not in prior_validation_ids
    ]
    prior_test = sorted(
        prior_test_candidates,
        key=lambda row: stable_key("official-benign-test-v2", row["id"]),
    )[:80]
    prior_feedback_ids = prior_validation_ids | {
        str(row["id"]) for row in prior_test
    }
    prior_train_candidates = [
        row for row in general if str(row["id"]) not in prior_feedback_ids
    ]
    prior_train_general = sorted(
        prior_train_candidates,
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
            row
            for row in fresh_candidates
            if str(row["id"]) not in (validation_ids | prior_v5_test_ids)
        ],
        key=lambda row: stable_key("official-benign-test-v6", row["id"]),
    )[:80]
    feedback_train = hard_train + prior_validation + prior_test + prior_v5_test
    train_general = sorted(
        prior_train_general,
        key=lambda row: stable_key("official-benign-retained-train-v6", row["id"]),
    )[: 680 - len(feedback_train)]
    baseline_selected = {
        "train": feedback_train + train_general,
        "validation": validation_rows,
        "test": test_rows,
        "final": final_rows,
    }
    if {split: len(rows) for split, rows in baseline_selected.items()} != {
        "train": 680,
        "validation": 80,
        "test": 80,
        "final": 10,
    }:
        raise RuntimeError("unexpected baseline official benign text split counts")

    mined_ids = set(
        SAFE_HARD_NEGATIVE_IDS
        + VALIDATION_HARD_NEGATIVE_IDS
        + TEST_HARD_NEGATIVE_IDS
        + EXTERNAL_BENIGN_IDS
    )
    if len(mined_ids) != 100:
        raise RuntimeError("hard-negative allocation IDs are not unique")
    missing_mined = mined_ids - set(all_by_id)
    if missing_mined:
        raise RuntimeError(
            "hard-negative IDs are absent from the pinned benign source: "
            + ", ".join(sorted(missing_mined))
        )
    mined_rows = [all_by_id[sample_id] for sample_id in sorted(mined_ids)]
    mined_texts = [str(row.get("text", "")).strip() for row in mined_rows]
    if any(not text for text in mined_texts) or len(mined_texts) != len(set(mined_texts)):
        raise RuntimeError("screened hard-negative source texts are empty or duplicated")
    baseline_texts = {
        str(row["text"]).strip()
        for split_rows in baseline_selected.values()
        for row in split_rows
    }
    if set(mined_texts) & baseline_texts:
        raise RuntimeError("screened hard-negative text overlaps the current v7 baseline")
    baseline_ids = {
        str(row["id"])
        for split_rows in baseline_selected.values()
        for row in split_rows
    }
    if mined_ids & baseline_ids:
        raise RuntimeError("mined hard negatives overlap the current v7 baseline")

    family_pair_counts = {"train": 170, "validation": 20, "test": 20}
    baseline_pair_ids = {
        split: [
            str(row["id"])
            for row in sorted(
                baseline_selected[split],
                key=lambda row: stable_key(
                    "text-led-benign-pair-v7", split, row["id"]
                ),
            )[: family_pair_counts[split]]
        ]
        for split in family_pair_counts
    }
    historical_feedback_ids = {
        str(row["id"])
        for row in hard_train + prior_validation + prior_test + prior_v5_test
    }
    protected_train_ids = historical_feedback_ids | set(baseline_pair_ids["train"])
    baseline_train_ids = {str(row["id"]) for row in baseline_selected["train"]}
    evictable_train = [
        row
        for row in baseline_selected["train"]
        if str(row["id"]) not in protected_train_ids
    ]
    required_evictions = (
        len(baseline_selected["validation"])
        + len(baseline_selected["test"])
        + len(SAFE_HARD_NEGATIVE_IDS)
    )
    if len(evictable_train) < required_evictions:
        raise RuntimeError("too few ordinary train-general rows for rolling feedback")
    evicted_train = sorted(
        evictable_train,
        key=lambda row: stable_key("rolling-feedback-train-eviction-v7", row["id"]),
    )[:required_evictions]
    evicted_train_ids = {str(row["id"]) for row in evicted_train}
    retained_baseline_train = [
        row
        for row in baseline_selected["train"]
        if str(row["id"]) not in evicted_train_ids
    ]

    mined_train = [all_by_id[sample_id] for sample_id in SAFE_HARD_NEGATIVE_IDS]
    new_train = (
        retained_baseline_train
        + baseline_selected["validation"]
        + baseline_selected["test"]
        + mined_train
    )
    if len(new_train) != 680:
        raise RuntimeError(f"unexpected rolling-feedback train count: {len(new_train)}")

    fresh_general = [
        row
        for row in general
        if str(row["id"]) not in (baseline_ids | mined_ids)
    ]
    new_validation_general = sorted(
        fresh_general,
        key=lambda row: stable_key("rolling-feedback-validation-general-v7", row["id"]),
    )[:70]
    validation_general_ids = {str(row["id"]) for row in new_validation_general}
    new_test_general = sorted(
        [row for row in fresh_general if str(row["id"]) not in validation_general_ids],
        key=lambda row: stable_key("rolling-feedback-test-general-v7", row["id"]),
    )[:70]
    new_validation = (
        [all_by_id[sample_id] for sample_id in VALIDATION_HARD_NEGATIVE_IDS]
        + new_validation_general
    )
    new_test = (
        [all_by_id[sample_id] for sample_id in TEST_HARD_NEGATIVE_IDS]
        + new_test_general
    )
    external_rows = [all_by_id[sample_id] for sample_id in EXTERNAL_BENIGN_IDS]
    selected = {
        "train": new_train,
        "validation": new_validation,
        "test": new_test,
        "final": final_rows,
        "external_benign": external_rows,
    }
    expected_counts = {
        "train": 680,
        "validation": 80,
        "test": 80,
        "final": 10,
        "external_benign": 20,
    }
    if {split: len(rows) for split, rows in selected.items()} != expected_counts:
        raise RuntimeError("unexpected rolling-feedback benign split counts")

    hard_negative_category_by_id = {
        **{
            str(row["id"]): "prior_v7_validation_feedback"
            for row in baseline_selected["validation"]
        },
        **{
            str(row["id"]): "prior_v7_test_feedback"
            for row in baseline_selected["test"]
        },
        **MINED_HARD_NEGATIVE_CATEGORY_BY_ID,
    }
    counterfactual_pair_ids = {
        "train": baseline_pair_ids["train"],
        "validation": [
            str(row["id"])
            for row in sorted(
                new_validation,
                key=lambda row: stable_key(
                    "text-led-benign-pair-v7", "validation", row["id"]
                ),
            )[:20]
        ],
        "test": [
            str(row["id"])
            for row in sorted(
                new_test,
                key=lambda row: stable_key(
                    "text-led-benign-pair-v7", "test", row["id"]
                ),
            )[:20]
        ],
    }
    schema3_selected_ids = {
        str(row["id"]) for rows in selected.values() for row in rows
    }
    additional_rows, additional_category_by_id, additional_provenance = (
        validate_additional_training_hard_negatives(
            all_by_id,
            schema3_selected_ids,
        )
    )
    schema3_train_ids = {str(row["id"]) for row in selected["train"]}
    active_hard_negative_train_ids = schema3_train_ids & set(
        hard_negative_category_by_id
    )
    protected_schema3_train_ids = (
        set(counterfactual_pair_ids["train"])
        | active_hard_negative_train_ids
        | historical_feedback_ids
    )
    if len(active_hard_negative_train_ids) != 220:
        raise RuntimeError("schema-3 active train hard-negative count changed")
    if len(historical_feedback_ids) != 247:
        raise RuntimeError("schema-3 historical-feedback count changed")
    if len(protected_schema3_train_ids) != 585:
        raise RuntimeError("schema-3 protected train union changed")
    ordinary_unprotected_train = [
        row
        for row in selected["train"]
        if str(row["id"]) not in protected_schema3_train_ids
    ]
    if len(ordinary_unprotected_train) != 95:
        raise RuntimeError("schema-3 unprotected ordinary train count changed")
    schema4_evicted_rows = sorted(
        ordinary_unprotected_train,
        key=lambda row: stable_key(
            "schema4-additional-hard-negative-eviction-v1", row["id"]
        ),
    )
    schema4_evicted_ids = {str(row["id"]) for row in schema4_evicted_rows}
    selected["train"] = [
        row for row in selected["train"] if str(row["id"]) not in schema4_evicted_ids
    ] + additional_rows
    if len(selected["train"]) != 705:
        raise RuntimeError("schema-4 training benign count changed")
    if schema4_evicted_ids & protected_schema3_train_ids:
        raise RuntimeError("schema-4 eviction touched a protected training row")
    hard_negative_category_by_id.update(additional_category_by_id)

    selected_ids = [str(row["id"]) for rows in selected.values() for row in rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("official benign rows overlap across partitions")
    text_group_ids = {
        split: [
            hashlib.sha256(str(row["text"]).strip().encode("utf-8")).hexdigest()
            for row in rows
        ]
        for split, rows in selected.items()
    }
    partition_audit: dict[str, object] = {
        "baseline_train_ids": sorted(baseline_train_ids),
        "baseline_validation_ids": [
            str(row["id"]) for row in baseline_selected["validation"]
        ],
        "baseline_test_ids": [
            str(row["id"]) for row in baseline_selected["test"]
        ],
        "historical_feedback_ids": sorted(historical_feedback_ids),
        "protected_train_pair_ids": baseline_pair_ids["train"],
        "evicted_train_ids": [str(row["id"]) for row in evicted_train],
        "retained_baseline_train_rows": len(retained_baseline_train),
        "moved_validation_feedback_rows": len(baseline_selected["validation"]),
        "moved_test_feedback_rows": len(baseline_selected["test"]),
        "mined_train_rows": len(mined_train),
        "fresh_validation_general_ids": [
            str(row["id"]) for row in new_validation_general
        ],
        "fresh_test_general_ids": [str(row["id"]) for row in new_test_general],
        "schema3_training_rows": 680,
        "schema4_additional_hard_negative_ids": [
            str(row["id"]) for row in additional_rows
        ],
        "schema4_evicted_ordinary_train_ids": [
            str(row["id"]) for row in schema4_evicted_rows
        ],
        "schema4_protected_train_pair_rows": len(counterfactual_pair_ids["train"]),
        "schema4_protected_prior_hard_negative_rows": len(
            active_hard_negative_train_ids
        ),
        "schema4_protected_historical_feedback_rows": len(
            historical_feedback_ids
        ),
        "schema4_protected_train_union_rows": len(protected_schema3_train_ids),
        "schema4_unprotected_ordinary_train_rows": len(
            ordinary_unprotected_train
        ),
        "schema4_train_growth_rows": len(additional_rows)
        - len(schema4_evicted_rows),
        "schema4_additional_screen": additional_provenance,
    }
    return (
        selected,
        all_by_id,
        text_group_ids,
        hard_negative_category_by_id,
        counterfactual_pair_ids,
        partition_audit,
    )


def select_full_malicious_row(
    family: int,
    family_rows: list[dict[str, object]],
    namespace: str,
) -> dict[str, object]:
    candidates = [
        row
        for row in family_rows
        if str(row.get("strategy")) in FULL_MALICIOUS_TEXT_STRATEGIES
    ]
    if len(candidates) != 2:
        raise RuntimeError(
            f"family {family} does not contain both full malicious caller strategies"
        )
    selected = min(
        candidates,
        key=lambda row: stable_key(namespace, family, row["id"]),
    )
    if not str(selected.get("text", "")).strip():
        raise RuntimeError(f"family {family} has an empty malicious caller prompt")
    return selected


def validate_and_index_ocr(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    if any(row.get("expected_detection") is not True for row in rows):
        raise RuntimeError("all Bordair rows were expected to require detection")
    ocr = [row for row in rows if row.get("image_type") == "ocr"]
    if len(ocr) != 920:
        raise RuntimeError(f"expected 920 OCR rows, found {len(ocr)}")
    by_id = {str(row["id"]): row for row in ocr}
    if len(by_id) != len(ocr):
        raise RuntimeError("OCR sample IDs are not unique")
    families: dict[int, list[dict[str, object]]] = {}
    for row in ocr:
        families.setdefault(family_number(str(row["id"])), []).append(row)
    if sorted(families) != list(range(230)):
        raise RuntimeError("Bordair family numbering is not the expected 0..229 range")
    for number, family_rows in families.items():
        if len(family_rows) != 4:
            raise RuntimeError(f"family {number} does not contain four OCR strategies")
        if len({str(row["category"]) for row in family_rows}) != 1:
            raise RuntimeError(f"family {number} crosses attack categories")
        if {str(row["strategy"]) for row in family_rows} != {
            "benign_text_full_injection",
            "split_injection",
            "authority_payload_split",
            "context_switch_injection",
        }:
            raise RuntimeError(f"family {number} has an unexpected strategy set")
    return ocr, by_id


def split_families(
    ocr: list[dict[str, object]],
) -> tuple[dict[int, str], dict[str, list[int]], dict[str, object]]:
    regression = {family_number(sample_id) for sample_id in PRIOR_REGRESSION_IDS}
    final = {family_number(sample_id) for sample_id in FINAL_MALICIOUS_IDS}
    if regression & final or len(regression) != 10 or len(final) != 10:
        raise RuntimeError("reserved family sets must be disjoint ten-family sets")

    category_by_family: dict[int, str] = {}
    for row in ocr:
        category_by_family[family_number(str(row["id"]))] = str(row["category"])
    available = sorted(set(category_by_family) - regression - final)
    if len(available) != 210:
        raise RuntimeError(f"expected 210 development families, found {len(available)}")

    by_category: dict[str, list[int]] = {}
    for family in available:
        by_category.setdefault(category_by_family[family], []).append(family)

    assignments: dict[int, str] = {}
    leftovers: list[int] = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda family: stable_key("bordair-family-split-v1", category, family),
        )
        if len(ordered) < 3:
            raise RuntimeError(f"category {category} has too few development families")
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

    counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation", "test")
    }
    if counts != {"train": 170, "validation": 20, "test": 20}:
        raise RuntimeError(f"unexpected family split counts: {counts}")
    baseline_partition_families = {
        "train": sorted(family for family, split in assignments.items() if split == "train"),
        "validation": sorted(
            family for family, split in assignments.items() if split == "validation"
        ),
        "test": sorted(family for family, split in assignments.items() if split == "test"),
        "regression": sorted(regression),
        "final": sorted(final),
    }
    baseline_train = baseline_partition_families["train"]
    by_category_train: dict[str, list[int]] = {}
    for family in baseline_train:
        by_category_train.setdefault(category_by_family[family], []).append(family)
    refreshed_validation: set[int] = set()
    refreshed_test: set[int] = set()
    refresh_leftovers: list[int] = []
    for category in sorted(by_category_train):
        ordered = sorted(
            by_category_train[category],
            key=lambda family: stable_key(
                "rolling-malicious-holdout-v7", category, family
            ),
        )
        if len(ordered) < 2:
            raise RuntimeError(
                f"category {category} has too few prior-train families for refresh"
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
    if len(refreshed_validation) != 20 or len(refreshed_test) != 20:
        raise RuntimeError("failed to refresh 20/20 malicious holdout families")
    if refreshed_validation & refreshed_test:
        raise RuntimeError("refreshed malicious holdout families overlap")

    refreshed_train = (
        set(baseline_train) - refreshed_validation - refreshed_test
    ) | set(baseline_partition_families["validation"]) | set(
        baseline_partition_families["test"]
    )
    refreshed_assignments = {
        family: (
            "validation"
            if family in refreshed_validation
            else "test"
            if family in refreshed_test
            else "train"
        )
        for family in available
    }
    partition_families = {
        "train": sorted(refreshed_train),
        "validation": sorted(refreshed_validation),
        "test": sorted(refreshed_test),
        "regression": sorted(regression),
        "final": sorted(final),
    }
    refreshed_counts = {
        split: len(partition_families[split])
        for split in ("train", "validation", "test")
    }
    if refreshed_counts != {"train": 170, "validation": 20, "test": 20}:
        raise RuntimeError(
            f"unexpected refreshed malicious family counts: {refreshed_counts}"
        )
    partition_audit: dict[str, object] = {
        "selection_rule": (
            "Move all prior-v7 validation and internal-test families into train; "
            "select category-covered 20/20 replacements only from prior-v7 train "
            "using the rolling-malicious-holdout-v7 namespaces."
        ),
        "prior_train_families": baseline_partition_families["train"],
        "prior_validation_families": baseline_partition_families["validation"],
        "prior_test_families": baseline_partition_families["test"],
        "feedback_families_moved_to_train": sorted(
            baseline_partition_families["validation"]
            + baseline_partition_families["test"]
        ),
        "train_families_moved_to_validation": partition_families["validation"],
        "train_families_moved_to_test": partition_families["test"],
        "validation_replacements": [
            {"prior_family": prior, "replacement_family": replacement}
            for prior, replacement in zip(
                baseline_partition_families["validation"],
                partition_families["validation"],
                strict=True,
            )
        ],
        "test_replacements": [
            {"prior_family": prior, "replacement_family": replacement}
            for prior, replacement in zip(
                baseline_partition_families["test"],
                partition_families["test"],
                strict=True,
            )
        ],
        "retained_prior_train_families": sorted(
            set(baseline_train) & refreshed_train
        ),
    }
    return refreshed_assignments, partition_families, partition_audit


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    paragraphs = str(text).replace("\r", "").split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            chunks: list[str] = []
            chunk = ""
            for character in word:
                candidate = chunk + character
                if chunk and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                    chunks.append(chunk)
                    chunk = character
                else:
                    chunk = candidate
            if chunk:
                chunks.append(chunk)
            for index, piece in enumerate(chunks):
                candidate = piece if not current else f"{current} {piece}"
                if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = piece
                if index < len(chunks) - 1:
                    lines.append(current)
                    current = ""
        if current:
            lines.append(current)
    return "\n".join(lines)


def render_text(
    text: str,
    destination: Path,
    style_index: int,
    fonts: tuple[Path, Path, Path] | None = None,
) -> dict[str, object]:
    resolved_fonts = fonts or resolve_fonts()
    styles = (
        ("white", "black", resolved_fonts[0], "center"),
        ("#14243A", "white", resolved_fonts[1], "left"),
        ("#F1F1EC", "#111111", resolved_fonts[2], "center"),
        ("#FFF4CC", "#202020", resolved_fonts[0], "left"),
    )
    background, foreground, font_path, alignment = styles[style_index % len(styles)]
    image = Image.new("RGB", CANVAS_SIZE, background)
    draw = ImageDraw.Draw(image)
    max_width = CANVAS_SIZE[0] - 2 * MARGIN
    wrapped = ""
    font = None
    bounds = None
    selected_spacing = LINE_SPACING
    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        candidate_font = ImageFont.truetype(str(font_path), size)
        candidate_text = wrap_text(draw, text, candidate_font, max_width)
        candidate_spacing = (
            LINE_SPACING if size >= 24 else max(2, size // 3)
        )
        candidate_bounds = draw.multiline_textbbox(
            (0, 0),
            candidate_text,
            font=candidate_font,
            spacing=candidate_spacing,
            align=alignment,
        )
        height = candidate_bounds[3] - candidate_bounds[1]
        if height <= CANVAS_SIZE[1] - 2 * MARGIN:
            wrapped = candidate_text
            font = candidate_font
            bounds = candidate_bounds
            selected_spacing = candidate_spacing
            break
    if font is None or bounds is None:
        raise RuntimeError(f"text does not fit render canvas: {text!r}")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = MARGIN if alignment == "left" else (CANVAS_SIZE[0] - width) / 2
    y = (CANVAS_SIZE[1] - height) / 2
    draw.multiline_text(
        (x, y),
        wrapped,
        fill=foreground,
        font=font,
        spacing=selected_spacing,
        align=alignment,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False)
    return {
        "canvas_pixels": list(CANVAS_SIZE),
        "background": background,
        "foreground": foreground,
        "font": str(font_path),
        "font_sha256": sha256(font_path),
        "font_size_pixels": font.size,
        "line_spacing_pixels": selected_spacing,
        "alignment": alignment,
        "image_sha256": sha256(destination),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def category_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["attack_style"])
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def hard_negative_fields(
    sample_id: str,
    category_by_id: dict[str, str],
) -> dict[str, object]:
    category = category_by_id.get(sample_id, "")
    return {
        "hard_negative": int(bool(category)),
        "hard_negative_category": category,
    }


def hard_negative_category_counts(
    rows: Iterable[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("hard_negative_category", ""))
        if category:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def validate_rendered_image_groups(rows: list[dict[str, object]]) -> None:
    hash_to_texts: dict[str, set[str]] = {}
    hash_to_splits: dict[str, set[str]] = {}
    for row in rows:
        image_hash = str(row["image_sha256"])
        hash_to_texts.setdefault(image_hash, set()).add(str(row["image_text"]))
        hash_to_splits.setdefault(image_hash, set()).add(str(row["split"]))
    ambiguous = {
        image_hash: texts
        for image_hash, texts in hash_to_texts.items()
        if len(texts) > 1
    }
    if ambiguous:
        raise RuntimeError(
            "different source texts produced identical rendered image bytes: "
            + ", ".join(sorted(ambiguous))
        )
    cross_partition = {
        image_hash: splits
        for image_hash, splits in hash_to_splits.items()
        if len(splits) > 1
    }
    if cross_partition:
        raise RuntimeError(
            "rendered image bytes overlap across partitions: "
            + ", ".join(sorted(cross_partition))
        )


def main() -> None:
    schema3_archive = validate_schema3_archive()
    fonts = resolve_fonts()
    rows, shards = load_rows()
    ocr, by_id = validate_and_index_ocr(rows)
    assignments, partition_families, malicious_partition_audit = split_families(ocr)

    prior_manifest = json.loads(PRIOR_CASES.read_text(encoding="utf-8"))
    prior_ids = [str(case["id"]) for case in prior_manifest["cases"]]
    if prior_ids != list(PRIOR_REGRESSION_IDS):
        raise RuntimeError("the fixed prior-regression IDs no longer match cases.json")
    tuning_ids_in_screening_order = tuple(
        sample_id
        for allocation in HARD_NEGATIVE_ALLOCATIONS.values()
        for split in ("train", "validation", "test")
        for sample_id in allocation[split]
    )
    tuning_digest = hashlib.sha256(
        ("\n".join(tuning_ids_in_screening_order) + "\n").encode("utf-8")
    ).hexdigest()
    external_digest = hashlib.sha256(
        ("\n".join(EXTERNAL_BENIGN_IDS) + "\n").encode("utf-8")
    ).hexdigest()
    if tuning_digest != EXPECTED_TUNING_ID_DIGEST:
        raise RuntimeError("screened tuning hard-negative ID digest changed")
    if external_digest != EXPECTED_EXTERNAL_ID_DIGEST:
        raise RuntimeError("screened external hard-negative ID digest changed")
    (
        benign_partitions,
        benign_by_id,
        benign_text_groups,
        hard_negative_category_by_id,
        counterfactual_pair_ids,
        benign_partition_audit,
    ) = load_and_partition_benign_ocr()
    prior_train_families = list(
        malicious_partition_audit["prior_train_families"]
    )
    prior_train_pair_by_family = dict(
        zip(
            prior_train_families,
            counterfactual_pair_ids["train"],
            strict=True,
        )
    )
    retained_train_families = set(
        malicious_partition_audit["retained_prior_train_families"]
    )
    freed_train_pair_ids = [
        prior_train_pair_by_family[family]
        for family in prior_train_families
        if family not in retained_train_families
    ]
    incoming_feedback_families = list(
        malicious_partition_audit["feedback_families_moved_to_train"]
    )
    refreshed_train_pair_by_family = {
        family: prior_train_pair_by_family[family]
        for family in retained_train_families
    }
    refreshed_train_pair_by_family.update(
        dict(
            zip(
                incoming_feedback_families,
                freed_train_pair_ids,
                strict=True,
            )
        )
    )
    counterfactual_pair_ids["train"] = [
        refreshed_train_pair_by_family[family]
        for family in partition_families["train"]
    ]

    # v7 is generated side by side with v6. Only the versioned output tree is
    # cleared, so reruns are exact without modifying any historical artifact.
    for panel in (
        "development",
        "regression",
        "final",
        "text_led_final",
        "external_benign",
    ):
        panel_directory = RUN / IMAGES_DIRECTORY / panel
        panel_directory.mkdir(parents=True, exist_ok=True)
        for stale_image in panel_directory.glob("*.png"):
            stale_image.unlink()

    carrier_pool = sorted(
        {
            str(row["text"])
            for row in ocr
            if row.get("strategy") == "benign_text_full_injection"
        }
    )
    if len(carrier_pool) < 20:
        raise RuntimeError("too few neutral Bordair carrier prompts")

    development_image_only: list[dict[str, object]] = []
    malicious_families: dict[int, list[dict[str, object]]] = {}
    for row in ocr:
        malicious_families.setdefault(family_number(str(row["id"])), []).append(row)
    for family in sorted(assignments):
        split = assignments.get(family)
        if split is None:
            continue
        family_rows = sorted(malicious_families[family], key=lambda item: str(item["id"]))
        image_only_rows = [
            row
            for row in family_rows
            if row.get("strategy") == "benign_text_full_injection"
        ]
        if len(image_only_rows) != 1:
            raise RuntimeError(f"family {family} lacks one image-only source row")
        source_row = image_only_rows[0]
        image_text = str(source_row["image_content"])
        for style_index, row in enumerate(family_rows):
            sample_id = str(row["id"])
            prompt = carrier_pool[
                int(stable_key("malicious-carrier-v6", split, sample_id)[:8], 16)
                % len(carrier_pool)
            ]
            image_relative = (
                IMAGES_DIRECTORY / "development" / f"{sample_id}.png"
            )
            render = render_text(
                image_text, RUN / image_relative, style_index, fonts
            )
            development_image_only.append(
                {
                    "sample_id": sample_id,
                    "label_id": 1,
                    "split": split,
                    "group_id": f"bordair-family-{family:03d}",
                    "attack_style": str(source_row["category"]),
                    "strategy": "benign_text_full_injection",
                    "source": (
                        "Bordair OCR family; image-only channel normalization with "
                        "four matched render styles"
                    ),
                    "prompt_text": prompt,
                    "image_path": image_relative.as_posix(),
                    "image_text": image_text,
                    "render_style": style_index,
                    "image_sha256": render["image_sha256"],
                    "source_sample_id": str(source_row["id"]),
                    "source_strategy": str(source_row["strategy"]),
                    "paired_benign_sample_id": "",
                    **hard_negative_fields(sample_id, {}),
                }
            )

    split_order = {"train": 0, "validation": 1, "test": 2}
    development_image_only.sort(
        key=lambda row: (split_order[str(row["split"])], str(row["sample_id"]))
    )
    image_only_counts = {
        split: sum(row["split"] == split for row in development_image_only)
        for split in split_order
    }
    if image_only_counts != {"train": 680, "validation": 80, "test": 80}:
        raise RuntimeError(
            f"unexpected image-only malicious row counts: {image_only_counts}"
        )

    development_benign: list[dict[str, object]] = []
    for split in split_order:
        selected_rows = benign_partitions[split]
        expected_benign_counts = {"train": 705, "validation": 80, "test": 80}
        if len(selected_rows) != expected_benign_counts[split]:
            raise RuntimeError(f"unexpected official benign count for {split}")
        for row in selected_rows:
            sample_id = str(row["id"])
            image_text = str(row["text"]).strip()
            prompt = carrier_pool[
                int(stable_key("benign-carrier-v2", split, sample_id)[:8], 16)
                % len(carrier_pool)
            ]
            image_relative = (
                IMAGES_DIRECTORY / "development" / f"{sample_id}.png"
            )
            style_index = int(stable_key("render-style-v1", sample_id)[:8], 16) % 4
            render = render_text(
                image_text, RUN / image_relative, style_index, fonts
            )
            development_benign.append(
                {
                    "sample_id": sample_id,
                    "label_id": 0,
                    "split": split,
                    "group_id": (
                        "bordair-benign-text-"
                        + hashlib.sha256(image_text.encode("utf-8")).hexdigest()[:12]
                    ),
                    "attack_style": "benign_control",
                    "strategy": "official_bordair_benign_text_rendered_in_image",
                    "source": f"Bordair benign row channel transfer ({row['source']})",
                    "prompt_text": prompt,
                    "image_path": image_relative.as_posix(),
                    "image_text": image_text,
                    "render_style": style_index,
                    "image_sha256": render["image_sha256"],
                    "source_sample_id": sample_id,
                    "source_strategy": "official_bordair_benign_text_image",
                    "paired_benign_sample_id": "",
                    **hard_negative_fields(
                        sample_id, hard_negative_category_by_id
                    ),
                }
            )

    development_text_led: list[dict[str, object]] = []
    for split in split_order:
        split_families_for_counterfactuals = partition_families[split]
        development_benign_by_id = {
            str(row["sample_id"]): row
            for row in development_benign
            if row["split"] == split
        }
        paired_benign_rows = [
            development_benign_by_id[sample_id]
            for sample_id in counterfactual_pair_ids[split]
        ]
        if len(paired_benign_rows) != len(split_families_for_counterfactuals):
            raise RuntimeError(f"too few same-split benign images for {split}")
        for family, paired_benign in zip(
            split_families_for_counterfactuals,
            paired_benign_rows,
            strict=True,
        ):
            source_row = select_full_malicious_row(
                family,
                malicious_families[family],
                "development-text-led-v7",
            )
            source_sample_id = str(source_row["id"])
            sample_id = f"MTI-{source_sample_id.removeprefix('TI-')}"
            image_relative = (
                IMAGES_DIRECTORY / "development" / f"{sample_id}.png"
            )
            paired_image = RUN / str(paired_benign["image_path"])
            shutil.copyfile(paired_image, RUN / image_relative)
            image_hash = sha256(RUN / image_relative)
            if image_hash != str(paired_benign["image_sha256"]):
                raise RuntimeError(f"counterfactual image copy changed bytes: {sample_id}")
            development_text_led.append(
                {
                    "sample_id": sample_id,
                    "label_id": 1,
                    "split": split,
                    "group_id": f"bordair-family-{family:03d}",
                    "attack_style": str(source_row["category"]),
                    "strategy": COUNTERFACTUAL_STRATEGY,
                    "source": (
                        "Bordair full malicious caller prompt paired with an "
                        "official same-split benign Bordair OCR image"
                    ),
                    "prompt_text": str(source_row["text"]).strip(),
                    "image_path": image_relative.as_posix(),
                    "image_text": str(paired_benign["image_text"]),
                    "render_style": int(paired_benign["render_style"]),
                    "image_sha256": image_hash,
                    "source_sample_id": source_sample_id,
                    "source_strategy": str(source_row["strategy"]),
                    "paired_benign_sample_id": str(paired_benign["sample_id"]),
                    **hard_negative_fields(sample_id, {}),
                }
            )

    text_led_counts = {
        split: sum(row["split"] == split for row in development_text_led)
        for split in split_order
    }
    if text_led_counts != {"train": 170, "validation": 20, "test": 20}:
        raise RuntimeError(f"unexpected text-led malicious counts: {text_led_counts}")

    development = (
        development_image_only + development_text_led + development_benign
    )
    development.sort(
        key=lambda row: (
            split_order[str(row["split"])],
            int(row["label_id"]),
            str(row["strategy"]),
            str(row["sample_id"]),
        )
    )
    write_csv(DEVELOPMENT_METADATA, development)

    regression: list[dict[str, object]] = []
    for sample_id in PRIOR_REGRESSION_IDS:
        row = by_id[sample_id]
        image_relative = IMAGES_DIRECTORY / "regression" / f"{sample_id}.png"
        render = render_text(
            str(row["image_content"]), RUN / image_relative, 0, fonts
        )
        regression.append(
            {
                "sample_id": sample_id,
                "label_id": 1,
                "split": "regression",
                "group_id": f"bordair-family-{family_number(sample_id):03d}",
                "attack_style": str(row["category"]),
                "strategy": str(row["strategy"]),
                "source": "Bordair OCR row; prior failure regression",
                "prompt_text": str(row["text"]),
                "image_path": image_relative.as_posix(),
                "image_text": str(row["image_content"]),
                "render_style": 0,
                "image_sha256": render["image_sha256"],
                "source_sample_id": sample_id,
                "source_strategy": str(row["strategy"]),
                "paired_benign_sample_id": "",
                **hard_negative_fields(sample_id, {}),
            }
        )
    write_csv(REGRESSION_METADATA, regression)

    final_rows: list[dict[str, object]] = []
    for sample_id in FINAL_MALICIOUS_IDS:
        row = by_id[sample_id]
        if row.get("strategy") != "benign_text_full_injection":
            raise RuntimeError(f"final attack is not image-only: {sample_id}")
        image_relative = IMAGES_DIRECTORY / "final" / f"{sample_id}.png"
        render = render_text(
            str(row["image_content"]), RUN / image_relative, 0, fonts
        )
        final_rows.append(
            {
                "sample_id": sample_id,
                "label_id": 1,
                "split": "final",
                "group_id": f"bordair-family-{family_number(sample_id):03d}",
                "attack_style": str(row["category"]),
                "strategy": str(row["strategy"]),
                "source": "Bordair OCR row; family-disjoint final evaluation",
                "prompt_text": str(row["text"]),
                "image_path": image_relative.as_posix(),
                "image_text": str(row["image_content"]),
                "render_style": 0,
                "image_sha256": render["image_sha256"],
                "source_sample_id": sample_id,
                "source_strategy": str(row["strategy"]),
                "paired_benign_sample_id": "",
                **hard_negative_fields(sample_id, {}),
            }
        )
    for sample_id in FINAL_BENIGN_IDS:
        row = benign_by_id[sample_id]
        image_text = str(row["text"]).strip()
        prompt = carrier_pool[
            int(stable_key("benign-carrier-v2", "final", sample_id)[:8], 16)
            % len(carrier_pool)
        ]
        image_relative = IMAGES_DIRECTORY / "final" / f"{sample_id}.png"
        render = render_text(image_text, RUN / image_relative, 0, fonts)
        final_rows.append(
            {
                "sample_id": sample_id,
                "label_id": 0,
                "split": "final",
                "group_id": (
                    "bordair-benign-text-"
                    + hashlib.sha256(image_text.encode("utf-8")).hexdigest()[:12]
                ),
                "attack_style": "benign_control",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "source": "Bordair benign row channel transfer; text-disjoint final evaluation",
                "prompt_text": prompt,
                "image_path": image_relative.as_posix(),
                "image_text": image_text,
                "render_style": 0,
                "image_sha256": render["image_sha256"],
                "source_sample_id": sample_id,
                "source_strategy": "official_bordair_benign_text_image",
                "paired_benign_sample_id": "",
                **hard_negative_fields(sample_id, {}),
            }
        )
    write_csv(FINAL_METADATA, final_rows)

    final_benign_rows = {
        str(row["sample_id"]): row
        for row in final_rows
        if int(row["label_id"]) == 0
    }
    text_led_final: list[dict[str, object]] = []
    for image_only_id, paired_benign_id in zip(
        FINAL_MALICIOUS_IDS,
        FINAL_BENIGN_IDS,
        strict=True,
    ):
        family = family_number(image_only_id)
        source_row = select_full_malicious_row(
            family,
            malicious_families[family],
            "final-text-led-v7",
        )
        source_sample_id = str(source_row["id"])
        sample_id = f"MTI-{source_sample_id.removeprefix('TI-')}"
        paired_benign = final_benign_rows[paired_benign_id]
        image_relative = IMAGES_DIRECTORY / "text_led_final" / f"{sample_id}.png"
        shutil.copyfile(RUN / str(paired_benign["image_path"]), RUN / image_relative)
        image_hash = sha256(RUN / image_relative)
        if image_hash != str(paired_benign["image_sha256"]):
            raise RuntimeError(f"final counterfactual image copy changed bytes: {sample_id}")
        text_led_final.append(
            {
                "sample_id": sample_id,
                "label_id": 1,
                "split": "final",
                "group_id": f"bordair-family-{family:03d}",
                "attack_style": str(source_row["category"]),
                "strategy": COUNTERFACTUAL_STRATEGY,
                "source": (
                    "Bordair full malicious caller prompt paired with a fixed "
                    "official benign final-control image"
                ),
                "prompt_text": str(source_row["text"]).strip(),
                "image_path": image_relative.as_posix(),
                "image_text": str(paired_benign["image_text"]),
                "render_style": int(paired_benign["render_style"]),
                "image_sha256": image_hash,
                "source_sample_id": source_sample_id,
                "source_strategy": str(source_row["strategy"]),
                "paired_benign_sample_id": paired_benign_id,
                **hard_negative_fields(sample_id, {}),
            }
        )
    write_csv(TEXT_LED_FINAL_METADATA, text_led_final)

    external_benign: list[dict[str, object]] = []
    for sample_id in EXTERNAL_BENIGN_IDS:
        row = benign_by_id[sample_id]
        image_text = str(row["text"]).strip()
        prompt = carrier_pool[
            int(stable_key("benign-carrier-v7", "external_benign", sample_id)[:8], 16)
            % len(carrier_pool)
        ]
        image_relative = IMAGES_DIRECTORY / "external_benign" / f"{sample_id}.png"
        render = render_text(image_text, RUN / image_relative, 0, fonts)
        external_benign.append(
            {
                "sample_id": sample_id,
                "label_id": 0,
                "split": "external_benign",
                "group_id": (
                    "bordair-benign-text-"
                    + hashlib.sha256(image_text.encode("utf-8")).hexdigest()[:12]
                ),
                "attack_style": "benign_control",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "source": (
                    "Manually screened Bordair hard benign row; frozen external "
                    f"evaluation ({row['source']})"
                ),
                "prompt_text": prompt,
                "image_path": image_relative.as_posix(),
                "image_text": image_text,
                "render_style": 0,
                "image_sha256": render["image_sha256"],
                "source_sample_id": sample_id,
                "source_strategy": "official_bordair_benign_text_image",
                "paired_benign_sample_id": "",
                **hard_negative_fields(sample_id, hard_negative_category_by_id),
            }
        )
    write_csv(EXTERNAL_BENIGN_METADATA, external_benign)

    observed_frozen_hashes = {
        path.name: sha256(path)
        for path in (
            REGRESSION_METADATA,
            FINAL_METADATA,
            TEXT_LED_FINAL_METADATA,
            EXTERNAL_BENIGN_METADATA,
        )
    }
    if observed_frozen_hashes != FROZEN_METADATA_SHA256:
        raise RuntimeError(
            "schema-4 generation changed a declared frozen metadata panel"
        )

    overlap = {
        split: set(families) for split, families in partition_families.items()
    }
    for left, left_values in overlap.items():
        for right, right_values in overlap.items():
            if left < right and left_values & right_values:
                raise RuntimeError(f"family leakage between {left} and {right}")
    benign_overlap = {
        split: set(groups) for split, groups in benign_text_groups.items()
    }
    for left, left_values in benign_overlap.items():
        for right, right_values in benign_overlap.items():
            if left < right and left_values & right_values:
                raise RuntimeError(
                    f"official benign rendered-text leakage between {left} and {right}"
                )
    validate_rendered_image_groups(
        development + regression + final_rows + text_led_final + external_benign
    )

    manifest = {
        "schema_version": 4,
        "corpus_version": "v7",
        "dataset": DATASET_NAME,
        "dataset_commit": DATASET_COMMIT,
        "scope": (
            "Bordair v1 malicious OCR content and benign text rows. Malicious "
            "image_content and benign caller text are neutrally rasterized into matched "
            "text-image pairs because native source image binaries are unavailable. "
            "Each development family contributes four image-led examples and one "
            "text-led counterfactual that pairs a full malicious caller prompt with "
            "an official benign image from the same split. Benign partitions use "
            "rolling evaluation feedback plus manually screened hard negatives. "
            "Schema 4 adds only training hard negatives and leaves every holdout and "
            "frozen panel unchanged."
        ),
        "source_shards": [
            {"path": shard.name, "sha256": sha256(shard)} for shard in shards
        ],
        "benign_source": {
            "path": str(BENIGN_SOURCE.relative_to(RUN)).replace("\\", "/"),
            "sha256": sha256(BENIGN_SOURCE),
            "rows": 6_423,
            "ocr_rows": 2_246,
            "unique_ocr_caller_texts": 2_245,
            "benign_channel_transfer": (
                "The official benign caller text is rendered in the image and paired "
                "with a neutral Bordair carrier prompt."
            ),
        },
        "family_definition": (
            "floor((numeric sample ID - 1) / 28); each family contains four "
            "delivery strategies across seven image encodings"
        ),
        "partition_rule": (
            "Reserve the ten prior-failure families and ten final-evaluation families; "
            "start from the prior 170/20/20 family-disjoint partition, roll all 40 "
            "previously inspected holdout families into train, and select category-covered "
            "20/20 replacement holdouts only from prior-v7 train. Each development "
            "family contributes four image-only style/carrier variants and one text-led "
            "malicious-prompt/benign-image counterfactual."
        ),
        "partition_families": partition_families,
        "malicious_partition_revision": malicious_partition_audit,
        "benign_partition_rule": (
            "Reserve ten official attack-adjacent benign texts for final evaluation; "
            "move all 160 prior-v7 validation/internal-test benign rows into training; "
            "add 60 unused manually screened hard negatives to training; evict only 220 "
            "ordinary train-general rows while protecting historical feedback, edge "
            "cases, and all existing train counterfactual pairs. Rebuild validation and "
            "internal test independently from ten allocated hard negatives and 70 fresh "
            "general rows each. Freeze 20 additional screened rows as an external benign "
            "panel. Starting from that immutable schema-3 corpus, preserve the full "
            "585-row union of active hard negatives, historical feedback, and current "
            "counterfactual pairs; replace all 95 remaining ordinary training benign "
            "rows and append 25 rows to admit 120 unused, manually screened official "
            "benign OCR rows (12 per semantic category). Validation, "
            "internal test, regression, and all frozen panels remain byte-for-byte "
            "unchanged. Exact source IDs, rendered texts, and image bytes are disjoint "
            "across partitions except for intentional same-split counterfactual pairs."
        ),
        "schema3_archive": {
            "path": str(SCHEMA3_ARCHIVE.relative_to(RUN)).replace("\\", "/"),
            "archive_manifest": "archive_manifest.json",
            "archive_manifest_sha256": SCHEMA3_ARCHIVE_MANIFEST_SHA256,
            "source_manifest_sha256": schema3_archive["files"][
                "corpus_manifest_v7.json"
            ]["sha256"],
            "source_development_metadata_sha256": schema3_archive["files"][
                "development_metadata_v7.csv"
            ]["sha256"],
            "source_images_tree_sha256": schema3_archive["trees"]["images_v7"][
                "sha256"
            ],
            "source_llava_feature_tree_sha256": schema3_archive["trees"][
                "llava05b"
            ]["sha256"],
            "unchanged_development_rows": 1_795,
            "replaced_training_benign_rows": 95,
            "added_training_benign_rows": 120,
            "training_growth_rows": 25,
        },
        "benign_text_groups": benign_text_groups,
        "hard_negative_screen": {
            "category_allocations": HARD_NEGATIVE_ALLOCATIONS,
            "tuning_ids_sha256": EXPECTED_TUNING_ID_DIGEST,
            "external_ids_sha256": EXPECTED_EXTERNAL_ID_DIGEST,
            "mined_train_rows": len(SAFE_HARD_NEGATIVE_IDS),
            "mined_validation_rows": len(VALIDATION_HARD_NEGATIVE_IDS),
            "mined_test_rows": len(TEST_HARD_NEGATIVE_IDS),
            "external_rows": len(EXTERNAL_BENIGN_IDS),
        },
        "schema4_additional_training_screen": benign_partition_audit[
            "schema4_additional_screen"
        ],
        "rolling_feedback": benign_partition_audit,
        "counterfactual_pairing_rule": (
            "Preserve all 170 prior-v7 train benign-image pair assignments. Repair the "
            "20 validation and 20 internal-test assignments deterministically from their "
            "entirely fresh memberships. Copy each selected benign image's exact bytes to "
            "the text-led counterfactual path and select the caller prompt from that family's "
            "authority_payload_split or context_switch_injection source row. The held-out "
            "text-led panel similarly pairs the ten final families one-to-one with the "
            "ten fixed benign final images. Image reuse is confined to the same split."
        ),
        "counterfactual_benign_pair_ids": counterfactual_pair_ids,
        "development": {
            "rows": len(development),
            "malicious_rows": len(development_image_only) + len(development_text_led),
            "image_only_malicious_rows": len(development_image_only),
            "text_led_malicious_rows": len(development_text_led),
            "benign_rows": len(development_benign),
            "image_only_malicious_per_split": image_only_counts,
            "text_led_malicious_per_split": text_led_counts,
            "benign_per_split": {
                split: len(benign_partitions[split]) for split in split_order
            },
            "hard_negative_benign_rows": sum(
                int(row["hard_negative"]) for row in development_benign
            ),
            "hard_negative_benign_per_split": {
                split: sum(
                    int(row["hard_negative"])
                    for row in development_benign
                    if row["split"] == split
                )
                for split in split_order
            },
            "hard_negative_category_counts": hard_negative_category_counts(
                development_benign
            ),
            "malicious_category_counts": category_counts(
                development_image_only + development_text_led
            ),
            "benign_source_counts": {
                source: sum(source in str(row["source"]) for row in development_benign)
                for source in (
                    "stanford_alpaca",
                    "wildchat",
                    "deepset_prompt_injections",
                    "edge_cases",
                )
            },
            "metadata": DEVELOPMENT_METADATA.name,
            "metadata_sha256": sha256(DEVELOPMENT_METADATA),
        },
        "regression": {
            "rows": len(regression),
            "ids": list(PRIOR_REGRESSION_IDS),
            "metadata": REGRESSION_METADATA.name,
            "metadata_sha256": sha256(REGRESSION_METADATA),
        },
        "final_evaluation": {
            "rows": len(final_rows),
            "malicious_rows": 10,
            "benign_rows": 10,
            "malicious_ids": list(FINAL_MALICIOUS_IDS),
            "benign_ids": list(FINAL_BENIGN_IDS),
            "metadata": FINAL_METADATA.name,
            "metadata_sha256": sha256(FINAL_METADATA),
        },
        "text_led_final_evaluation": {
            "rows": len(text_led_final),
            "malicious_rows": len(text_led_final),
            "ids": [str(row["sample_id"]) for row in text_led_final],
            "source_ids": [str(row["source_sample_id"]) for row in text_led_final],
            "paired_benign_ids": [
                str(row["paired_benign_sample_id"]) for row in text_led_final
            ],
            "metadata": TEXT_LED_FINAL_METADATA.name,
            "metadata_sha256": sha256(TEXT_LED_FINAL_METADATA),
        },
        "external_benign_evaluation": {
            "rows": len(external_benign),
            "benign_rows": len(external_benign),
            "ids": list(EXTERNAL_BENIGN_IDS),
            "category_counts": hard_negative_category_counts(external_benign),
            "metadata": EXTERNAL_BENIGN_METADATA.name,
            "metadata_sha256": sha256(EXTERNAL_BENIGN_METADATA),
        },
        "rendering": {
            "canvas_pixels": list(CANVAS_SIZE),
            "margin_pixels": MARGIN,
            "font_size_range_pixels": [MIN_FONT_SIZE, MAX_FONT_SIZE],
            "line_spacing_pixels": {
                "font_size_at_least_24": LINE_SPACING,
                "font_size_below_24": "max(2, floor(font_size / 3))",
            },
            "development_styles": 4,
            "regression_final_and_external_style": (
                "black primary-sans text on white background"
            ),
            "added_visible_text": False,
            "selected_font_paths": [str(path) for path in fonts],
            "font_hashes": {str(path): sha256(path) for path in fonts},
        },
    }
    CORPUS_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

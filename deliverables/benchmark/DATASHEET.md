# Datasheet: AEGIS Synthetic Redacted Benchmark v1.0.0

## Motivation

This small benchmark provides a committed, inspectable test corpus for multimodal
prompt-safety detectors. It is intended for pipeline verification, red-team regression
testing, and annotation-protocol development—not for claims of production safety.

## Composition

- 64 rows: 32 benign and 32 malicious.
- 32 matched scenario groups, with one benign and one adversarial row per group.
- Text and image-text inputs.
- Eight adversarial styles with four examples each: direct policy violations,
  role-play, text obfuscation, instruction hierarchy attacks, image-embedded
  instructions, cross-modal misalignment, indirect prompt injection, and safety
  camouflage.
- All malicious content uses redaction placeholders. Images are generated cue cards
  and contain no operational harmful instructions.

## Collection and annotation

Rows were newly authored for this project from a generic threat taxonomy. They do not
copy third-party prompts. Labels follow `ANNOTATION_GUIDELINES.md` and are checked by
deterministic validation rules for balance, uniqueness, redaction, paired groups, and
split isolation. `gold_synthetic` means the intended label is unambiguous by
construction; it does not mean human inter-annotator agreement was measured.

## Splits

Each attack family contributes two matched groups to training, one to validation, and
one to test. Paired rows share a `group_id` and never cross splits. The test split must
not be used for feature selection or threshold tuning.

## Personal and sensitive information

No real names, identifiers, credentials, or personal records are included. Placeholder
strings explicitly mark sensitive concepts. Image files are generated locally and do
not depict people.

## Licensing and release

The data is currently marked `project_internal_pending_owner_release_license`.
The project owner must select and confirm an external data license before public
redistribution. This avoids silently imposing a legal choice during technical recovery.

## Limitations

- Synthetic cue cards are less visually varied than real-world images.
- Redaction reduces the lexical and semantic diversity of genuine attacks.
- The benchmark is deliberately small and balanced; deployed prevalence will differ.
- Harm categories are not exhaustive and have no subgroup-fairness coverage.
- It cannot replace evaluation on licensed third-party datasets or human-authored
  adaptive attacks.

## Maintenance

Version any content change, retain versioned manifests, document adjudications, and screen
new material for privacy and operational harm before commit. Aggregate results may be
shared; raw high-risk extensions require controlled access and safe disclosure review.

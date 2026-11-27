"""Canonical label mapping, verified from Data/Raw/fakeReviewData.csv.

sklearn LabelEncoder sorts classes alphabetically, so:
    CG -> 0, OR -> 1

That matches N1 (`label_encoder.fit_transform`) and the encoded feature tables.
"""

LABEL_TO_ID = {"CG": 0, "OR": 1}
ID_TO_CODE = {0: "CG", 1: "OR"}
ID_TO_NAME = {
    0: "Computer Generated",
    1: "Original",
}

POSITIVE_CLASS_ID = 1  # Original / human-written
POSITIVE_CLASS_NAME = ID_TO_NAME[POSITIVE_CLASS_ID]


def encode_label(raw_label) -> int:
    if raw_label in (0, 1):
        return int(raw_label)
    key = str(raw_label).strip()
    if key in LABEL_TO_ID:
        return LABEL_TO_ID[key]
    if key in ID_TO_NAME.values():
        return 0 if key == ID_TO_NAME[0] else 1
    raise ValueError(
        f"Unknown label {raw_label!r}. Expected CG/OR or 0/1 "
        f"(0={ID_TO_NAME[0]}, 1={ID_TO_NAME[1]})."
    )


def decode_label(label_id: int) -> str:
    if int(label_id) not in ID_TO_NAME:
        raise ValueError(f"Unknown label id {label_id!r}. Expected 0 or 1.")
    return ID_TO_NAME[int(label_id)]

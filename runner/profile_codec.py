"""Lossless native profile wire encodings; expand before semantic validation."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


ACTIVITY_FIELDS = ("activity_id", "kind", "name", "device", "stream", "process_id", "thread_id",
                   "start", "duration", "scope", "related_activity_ids")
COMPACT_KIND = "native_profile_compact_v1"


def _signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encode_compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Intern complete observed objects; never derive addresses or activity IDs."""
    if any(key in profile for key in ("activity_names", "activity_scopes", "cache_participants")):
        raise ValueError("Expand the existing profile wire encoding before compact encoding.")
    activities = profile["activities"]
    activity_ids = [row["activity_id"] for row in activities]
    if len(set(activity_ids)) != len(activity_ids):
        raise ValueError("Duplicate native activity IDs cannot be compact encoded.")
    activity_indices = {id: index for index, id in enumerate(activity_ids)}
    names: list[str] = []
    scopes: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []
    dictionaries: dict[str, dict[str, int]] = {"names": {}, "scopes": {}, "participants": {}}

    def intern(table: list, dictionary: str, value: Any) -> int:
        signature = _signature(value)
        ids = dictionaries[dictionary]
        if signature not in ids:
            ids[signature] = len(table)
            table.append(deepcopy(value))
        return ids[signature]

    def references(ids: list[str]) -> list[int | str]:
        return [activity_indices.get(id, id) for id in ids]

    encoded_activities = []
    for activity in activities:
        if set(activity) != set(ACTIVITY_FIELDS):
            encoded_activities.append(deepcopy(activity))
            continue
        row = [deepcopy(activity[field]) for field in ACTIVITY_FIELDS]
        row[2] = intern(names, "names", activity["name"])
        row[9] = intern(scopes, "scopes", activity["scope"])
        row[10] = references(activity["related_activity_ids"])
        encoded_activities.append(row)
    writes = []
    for write in profile.get("cache_writes", []):
        if "participant_ids" in write or "gpu_activity_refs" in write:
            raise ValueError("Ambiguous compact cache-write fields.")
        encoded_write = deepcopy(write)
        if "participants" in write:
            encoded_write["participant_ids"] = [intern(participants, "participants", row) for row in write["participants"]]
            encoded_write.pop("participants")
        if "gpu_activity_ids" in write:
            encoded_write["gpu_activity_refs"] = references(write["gpu_activity_ids"])
            encoded_write.pop("gpu_activity_ids")
        writes.append(encoded_write)
    normalization = deepcopy(profile["normalization"])
    normalization["attachment_encoding"] = {
        "kind": COMPACT_KIND, "unique_name_count": len(names), "unique_scope_count": len(scopes),
        "unique_participant_count": len(participants), "encoded_activity_count": len(activities),
        "encoded_cache_write_count": len(writes),
        "preserved": "Exact activity IDs/order/fields, complete scopes and cache participants, correlations and write links; no observations are omitted by encoding.",
    }
    result = {**profile, "activities": encoded_activities, "activity_names": names, "activity_scopes": scopes,
              "cache_participants": participants, "normalization": normalization}
    if "cache_writes" in profile:
        result["cache_writes"] = writes
    return result


def expand_profile_wire(profile: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed references and aliases before returning independent records."""
    encoding = profile.get("normalization", {}).get("attachment_encoding", {})
    kind = encoding.get("kind")
    if kind is None:
        if any(key in profile for key in ("activity_names", "activity_scopes", "cache_participants")):
            raise ValueError("Profile dictionaries require a declared wire encoding.")
        return deepcopy(profile)
    if kind not in {"activity_name_dictionary_v1", COMPACT_KIND}:
        raise ValueError("Unsupported native profile wire encoding.")
    if kind == "activity_name_dictionary_v1" and any(key in profile for key in ("activity_scopes", "cache_participants")):
        raise ValueError("Ambiguous legacy profile dictionaries.")

    def table(name: str, count_name: str, value_type: type) -> list:
        rows = profile.get(name)
        if not isinstance(rows, list) or any(not isinstance(row, value_type) for row in rows):
            raise ValueError(f"Invalid {name} dictionary.")
        count = encoding.get(count_name)
        if type(count) is not int or count != len(rows) or len({_signature(row) for row in rows}) != len(rows):
            raise ValueError(f"Invalid {name} dictionary count or duplicate values.")
        return rows

    def indexed(index: Any, rows: list, field: str) -> Any:
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError(f"Invalid {field} dictionary reference.")
        return deepcopy(rows[index])

    names = table("activity_names", "unique_name_count", str)
    raw_activities = profile.get("activities")
    if not isinstance(raw_activities, list) or type(encoding.get("encoded_activity_count")) is not int or encoding["encoded_activity_count"] != len(raw_activities):
        raise ValueError("Invalid encoded activity count.")
    scopes = table("activity_scopes", "unique_scope_count", dict) if kind == COMPACT_KIND else []
    participants = table("cache_participants", "unique_participant_count", dict) if kind == COMPACT_KIND else []
    activities = []
    for value in raw_activities:
        if kind == "activity_name_dictionary_v1":
            if not isinstance(value, dict) or "name" in value or "name_id" not in value:
                raise ValueError("Ambiguous name-dictionary activity.")
            row = deepcopy(value)
            row["name"] = indexed(row.pop("name_id"), names, "name")
        elif isinstance(value, list):
            if len(value) != len(ACTIVITY_FIELDS):
                raise ValueError("Invalid compact activity width.")
            row = dict(zip(ACTIVITY_FIELDS, deepcopy(value)))
            row["name"] = indexed(value[2], names, "name")
            row["scope"] = indexed(value[9], scopes, "scope")
        elif isinstance(value, dict) and "name" in value and "scope" in value and "name_id" not in value:
            row = deepcopy(value)
        else:
            raise ValueError("Invalid compact activity record.")
        activities.append(row)
    activity_ids = [row.get("activity_id") for row in activities]
    if any(not isinstance(id, str) or not id for id in activity_ids) or len(set(activity_ids)) != len(activity_ids):
        raise ValueError("Invalid or duplicate activity IDs.")

    def references(values: Any) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("Activity references must be a list.")
        result = []
        for value in values:
            if type(value) is int:
                result.append(indexed(value, activity_ids, "activity"))
            elif isinstance(value, str):
                result.append(value)
            else:
                raise ValueError("Invalid activity identity reference.")
        return result

    writes = deepcopy(profile.get("cache_writes", []))
    if kind == COMPACT_KIND:
        if not isinstance(writes, list) or type(encoding.get("encoded_cache_write_count")) is not int or encoding["encoded_cache_write_count"] != len(writes):
            raise ValueError("Invalid encoded cache-write count.")
        for wire, row in zip(raw_activities, activities):
            if isinstance(wire, list):
                row["related_activity_ids"] = references(wire[10])
        for write in writes:
            if not isinstance(write, dict):
                raise ValueError("Invalid compact cache-write record.")
            if "participant_ids" in write:
                if "participants" in write or not isinstance(write["participant_ids"], list):
                    raise ValueError("Ambiguous compact cache participants.")
                write["participants"] = [indexed(index, participants, "participant") for index in write.pop("participant_ids")]
            if "gpu_activity_refs" in write:
                if "gpu_activity_ids" in write:
                    raise ValueError("Ambiguous compact cache activity links.")
                write["gpu_activity_ids"] = references(write.pop("gpu_activity_refs"))
    result = deepcopy(profile)
    result["activities"] = activities
    if "cache_writes" in profile:
        result["cache_writes"] = writes
    for key in ("activity_names", "activity_scopes", "cache_participants"):
        result.pop(key, None)
    return result

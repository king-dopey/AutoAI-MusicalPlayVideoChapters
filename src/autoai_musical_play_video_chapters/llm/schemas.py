"""Schema helpers and schema constants for LLM responses."""

from __future__ import annotations

import json


def validate_json_schema(value, schema, path="$", *, error_cls=Exception):
    """Validate a JSON value against a minimal in-code schema subset."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_json_schema(value, option, path, error_cls=error_cls)
                return
            except Exception:
                continue
        raise error_cls(f"{path}: did not match any allowed schema option")

    if "oneOf" in schema:
        matches = 0
        for option in schema["oneOf"]:
            try:
                validate_json_schema(value, option, path, error_cls=error_cls)
                matches += 1
            except Exception:
                continue
        if matches != 1:
            raise error_cls(f"{path}: did not match exactly one allowed schema option")
        return

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for item_type in schema_type:
            try:
                validate_json_schema(value, {**schema, "type": item_type}, path, error_cls=error_cls)
                return
            except Exception:
                continue
        raise error_cls(f"{path}: did not match any allowed type")

    if schema_type == "object":
        if not isinstance(value, dict):
            raise error_cls(f"{path}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise error_cls(f"{path}: missing required key {key}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                validate_json_schema(value[key], subschema, f"{path}.{key}", error_cls=error_cls)
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            extras = [key for key in value if key not in allowed]
            if extras:
                raise error_cls(f"{path}: unexpected keys {extras}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise error_cls(f"{path}: expected array")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                validate_json_schema(item, item_schema, f"{path}[{idx}]", error_cls=error_cls)
    elif schema_type == "string":
        if not isinstance(value, str):
            raise error_cls(f"{path}: expected string")
        enum = schema.get("enum")
        if enum and value not in enum:
            raise error_cls(f"{path}: value {value!r} not in enum")
        min_len = schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            raise error_cls(f"{path}: string too short")
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise error_cls(f"{path}: expected integer")
        enum = schema.get("enum")
        if enum and value not in enum:
            raise error_cls(f"{path}: value {value!r} not in enum")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise error_cls(f"{path}: value below minimum")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            raise error_cls(f"{path}: value above maximum")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise error_cls(f"{path}: expected number")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise error_cls(f"{path}: value below minimum")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            raise error_cls(f"{path}: value above maximum")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise error_cls(f"{path}: expected boolean")
    elif schema_type == "null":
        if value is not None:
            raise error_cls(f"{path}: expected null")


def schema_text(schema):
    """Render a schema compactly for prompts."""
    return json.dumps(schema, ensure_ascii=False, indent=2)


def strict_json_response_format(schema):
    """Return a JSON schema response-format payload for compatible endpoints."""
    return {"type": "json_schema", "json_schema": schema}


def safe_response_format():
    """Compatibility shim for the legacy helper."""
    return {"type": "json_object"}


def enum_int_schema(values):
    """Build a deduplicated integer-enum JSON schema."""
    ordered = []
    for value in values:
        if isinstance(value, int) and value not in ordered:
            ordered.append(value)
    return {"type": "integer", "enum": ordered}


def build_block_search_schema(valid_block_ids):
    """Build the strict schema for song-search block selection."""
    allowed = [0]
    for block_id in valid_block_ids:
        if isinstance(block_id, int) and block_id not in allowed:
            allowed.append(block_id)
    return {
        "type": "object",
        "required": ["selected_block_id", "confidence", "reason"],
        "properties": {
            "selected_block_id": enum_int_schema(allowed),
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }


def build_boundary_selection_schema(valid_candidate_indexes):
    """Build the strict schema for boundary refine/verify selections."""
    allowed = []
    for idx in valid_candidate_indexes:
        if isinstance(idx, int) and idx not in allowed:
            allowed.append(idx)
    return {
        "type": "object",
        "required": ["found", "start_candidate_index", "end_candidate_index", "confidence", "reason"],
        "properties": {
            "found": {"type": "boolean"},
            "start_candidate_index": {"anyOf": [enum_int_schema(allowed), {"type": "null"}]},
            "end_candidate_index": {"anyOf": [enum_int_schema(allowed), {"type": "null"}]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }


LYRICS_WINDOW_SCHEMA = {
    "type": "object",
    "required": ["lyrics_cue_ids", "confidence", "reason"],
    "properties": {
        "lyrics_cue_ids": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


CHAPTER_SCHEMA = {
    "type": "object",
    "required": ["index", "act", "song_title", "title", "summary", "themes", "characters"],
    "properties": {
        "index": {"type": "integer"},
        "act": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "song_title": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "chapter_title": {"type": "string"},
        "key_characters": {"type": "array", "items": {"type": "string"}},
        "key_events": {"type": "array", "items": {"type": "string"}},
        "continuity_notes": {"type": "array", "items": {"type": "string"}},
        "story_role": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "additionalProperties": False,
}


FINAL_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["overall_summary", "act_summaries"],
    "properties": {
        "overall_summary": {"type": "string"},
        "act_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["act", "summary"],
                "properties": {
                    "act": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

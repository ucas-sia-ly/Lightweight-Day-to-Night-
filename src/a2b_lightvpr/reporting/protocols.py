"""Fixed supervised comparison sets and explicit unavailable protocol results."""
from __future__ import annotations

import math


NIGHT_MACRO = ("svox_night", "tokyo_night")
DAY_MACRO = ("svox_day", "svox_sun", "tokyo_day")
PUBLIC_UNLABELLED = ("robotcar_seasons_night", "robotcar_seasons_day")
REPORT_PROTOCOLS = (
    "robotcar_seasons_night", "svox_night", "msls_night", "tokyo_night",
    "tokyo_sunset", "robotcar_seasons_day", "svox_day", "svox_sun", "tokyo_day",
    "msls_day_val", "svox_day_val",
)


def protocol_result(payload: dict, protocol: str) -> dict:
    metrics = payload.get("datasets", {}).get(protocol)
    if metrics is None:
        return {"status": "not_evaluated", "recalls": None, "reason": "Protocol was not evaluated."}
    result = {
        "status": metrics.get("evaluation_status", metrics.get("status", "completed")),
        "recalls": metrics.get("recalls"),
        "valid_queries": metrics.get("valid_queries"),
        "total_queries": metrics.get("total_queries"),
        "ground_truth_available": metrics.get("ground_truth_available", True),
        "evaluation_fingerprint": metrics.get("evaluation_fingerprint", metrics.get("protocol_fingerprint")),
    }
    if protocol in PUBLIC_UNLABELLED or metrics.get("ground_truth_available") is False:
        result.update(
            status=metrics.get("evaluation_status", metrics.get("status", "unavailable_ground_truth")), recalls=None,
            ground_truth_available=False,
            reason="Official RobotCar Seasons test poses are hidden; retrieval rankings have no local Recall labels.",
        )
    elif (str(result["status"]).startswith("skipped") or result["status"] == "not_evaluated"
          or not isinstance(result["recalls"], dict) or not result["recalls"] or result["valid_queries"] == 0):
        result.update(recalls=None, reason=metrics.get("reason", "Recall was not measured."))
        if result["status"] == "completed":
            result["status"] = "not_measured"
    elif any(value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
             for value in result["recalls"].values()):
        raise ValueError(f"Invalid measured recalls for {protocol}")
    return result


def recall(payload: dict, protocol: str, k: int = 1) -> float | None:
    values = protocol_result(payload, protocol)["recalls"]
    value = values.get(str(k)) if values else None
    return float(value) if value is not None else None


def macro(payload: dict, protocols: tuple[str, ...]) -> float | None:
    """Never silently replace the fixed comparison set with available protocols."""
    values = [recall(payload, protocol) for protocol in protocols]
    return sum(values) / len(values) if all(value is not None for value in values) else None


def difference(candidate: float | None, baseline: float | None) -> float | None:
    return candidate - baseline if candidate is not None and baseline is not None else None


def excluded(payload: dict) -> dict:
    return {name: result for name in REPORT_PROTOCOLS
            if (result := protocol_result(payload, name))["recalls"] is None}


def validate_protocol_pair(baseline: dict, candidate: dict, protocol: str) -> None:
    first, second = protocol_result(baseline, protocol), protocol_result(candidate, protocol)
    # Unmeasured inputs remain visible in reports. Measured pairs require matching identities.
    if first["recalls"] is not None and second["recalls"] is not None:
        identity = first.get("evaluation_fingerprint")
        if not identity or identity != second.get("evaluation_fingerprint"):
            raise ValueError(f"Evaluation fingerprint mismatch for {protocol}")

"""Utility script to dump ONVIF PTZ configuration for the configured camera."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta
from typing import Any

from dotenv import load_dotenv
from onvif import ONVIFCamera

try:
    from zeep.helpers import serialize_object  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    serialize_object = None


def _serialize(value: Any) -> Any:
    """Convert Zeep objects into builtin types suitable for JSON."""
    if serialize_object is not None:
        try:
            value = serialize_object(value)
        except Exception:
            pass

    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if hasattr(value, "total_seconds") and callable(getattr(value, "total_seconds")):
        try:
            return float(value.total_seconds())
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]

    attrs = getattr(value, "__dict__", None)
    if attrs:
        return {k: _serialize(v) for k, v in attrs.items() if not k.startswith("_")}
    return str(value)


def _get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise SystemExit(f"Environment variable {name} is required")
    if val is None:
        raise SystemExit(f"Environment variable {name} is missing")
    return val


def fetch_ptz_configuration() -> list[dict[str, Any]]:
    load_dotenv()

    host = _get_env("CAM_HOST", required=True)
    port = int(_get_env("CAM_PORT", "80"))
    user = _get_env("CAM_USER", "")
    passwd = _get_env("CAM_PASS", "")

    camera = ONVIFCamera(host, port, user, passwd)
    media_service = camera.create_media_service()
    ptz_service = camera.create_ptz_service()

    profiles = media_service.GetProfiles()
    payload: list[dict[str, Any]] = []

    for profile in profiles:
        entry: dict[str, Any] = {
            "profile_token": getattr(profile, "token", None),
            "profile_name": getattr(profile, "Name", None),
        }

        ptz_config = getattr(profile, "PTZConfiguration", None)
        config_token = None
        if ptz_config is not None:
            config_token = getattr(ptz_config, "token", None)
        if config_token:
            entry["ptz_configuration_token"] = config_token
            try:
                config = ptz_service.GetConfiguration({
                    "PTZConfigurationToken": config_token,
                })
                entry["configuration"] = _serialize(config)
            except Exception as exc:  # pragma: no cover - depends on camera
                entry["configuration_error"] = str(exc)

            try:
                options = ptz_service.GetConfigurationOptions({
                    "ConfigurationToken": config_token,
                })
                entry["options"] = _serialize(options)
            except Exception as exc:  # pragma: no cover - depends on camera
                entry["options_error"] = str(exc)

            try:
                node_token = getattr(ptz_config, "NodeToken", None)
                if node_token:
                    entry["node"] = _serialize(
                        ptz_service.GetNode({"NodeToken": node_token})
                    )
            except Exception as exc:  # pragma: no cover - depends on camera
                entry["node_error"] = str(exc)
        else:
            entry["ptz_configuration_token"] = None

        payload.append(entry)

    return payload


def main() -> None:
    data = fetch_ptz_configuration()
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

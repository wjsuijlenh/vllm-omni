# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input helpers for Sana-WM image-to-video requests.

The released SANA-WM checkpoint is first-frame image-to-video. This module keeps
request-side normalization lightweight: it validates the image/camera/action
contract and stores a canonical payload under
``additional_information["sana_wm"]``. Model-internal raymap / Plucker
projection lives with the diffusion model implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SANA_WM_CANONICAL_KEY = "sana_wm"
SANA_WM_DEFAULT_NUM_FRAMES = 321
SANA_WM_DEFAULT_HEIGHT = 704
SANA_WM_DEFAULT_WIDTH = 1280
SANA_WM_DEFAULT_CAMERA_FORMAT = "c2w_4x4"
SANA_WM_DEFAULT_COORDINATE_SYSTEM = "official"
SANA_WM_SUPPORTED_INTRINSICS_SHAPES = ((3, 3), (4,))


def _unwrap_single(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Sana-WM {name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None
    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_numeric_matrix4x4(matrix: Any, *, name: str) -> None:
    shape = _shape_of(matrix)
    if shape is not None:
        if shape != (4, 4):
            raise ValueError(f"Sana-WM {name} must have shape (4, 4), got {shape}.")
        return

    if not _is_sequence(matrix) or len(matrix) != 4:
        raise ValueError(f"Sana-WM {name} must be a 4x4 matrix.")
    for row_idx, row in enumerate(matrix):
        if not _is_sequence(row) or len(row) != 4:
            raise ValueError(f"Sana-WM {name}[{row_idx}] must contain 4 values.")
        for col_idx, value in enumerate(row):
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Sana-WM {name}[{row_idx}][{col_idx}] must be numeric.") from exc


def _validate_numeric_matrix3x3(matrix: Any, *, name: str) -> None:
    shape = _shape_of(matrix)
    if shape is not None:
        if shape != (3, 3):
            raise ValueError(f"Sana-WM {name} must have shape (3, 3), got {shape}.")
        return

    if not _is_sequence(matrix) or len(matrix) != 3:
        raise ValueError(f"Sana-WM {name} must be a 3x3 matrix.")
    for row_idx, row in enumerate(matrix):
        if not _is_sequence(row) or len(row) != 3:
            raise ValueError(f"Sana-WM {name}[{row_idx}] must contain 3 values.")
        for col_idx, value in enumerate(row):
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Sana-WM {name}[{row_idx}][{col_idx}] must be numeric.") from exc


def _validate_camera_poses(poses: Any, *, num_frames: int | None = None) -> None:
    shape = _shape_of(poses)
    if shape is not None:
        if len(shape) != 3 or shape[1:] != (4, 4):
            raise ValueError(f"Sana-WM camera poses must have shape (F, 4, 4), got {shape}.")
        if num_frames is not None and shape[0] != num_frames:
            raise ValueError(f"Sana-WM camera poses length {shape[0]} must equal num_frames {num_frames}.")
        return

    if not _is_sequence(poses) or len(poses) == 0:
        raise ValueError("Sana-WM camera poses must be a non-empty sequence of 4x4 matrices.")
    if num_frames is not None and len(poses) != num_frames:
        raise ValueError(f"Sana-WM camera poses length {len(poses)} must equal num_frames {num_frames}.")
    for idx, pose in enumerate(poses):
        _validate_numeric_matrix4x4(pose, name=f"camera.poses[{idx}]")


def _validate_intrinsics(intrinsics: Any, *, num_frames: int | None = None) -> None:
    if intrinsics is None:
        return
    if isinstance(intrinsics, Mapping):
        required = {"fx", "fy", "cx", "cy"}
        missing = sorted(required - set(intrinsics))
        if missing:
            raise ValueError(f"Sana-WM intrinsics mapping is missing keys: {missing}.")
        for key in required:
            try:
                float(intrinsics[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Sana-WM intrinsics[{key!r}] must be numeric.") from exc
        return

    shape = _shape_of(intrinsics)
    if shape is not None:
        if shape in SANA_WM_SUPPORTED_INTRINSICS_SHAPES:
            return
        if len(shape) == 3 and shape[1:] == (3, 3):
            if num_frames is not None and shape[0] not in (1, num_frames):
                raise ValueError(
                    f"Sana-WM per-frame intrinsics length {shape[0]} must be 1 or num_frames {num_frames}."
                )
            return
        raise ValueError("Sana-WM intrinsics must have shape (3, 3), (F, 3, 3), or (4,).")

    if _is_sequence(intrinsics):
        if len(intrinsics) == 0:
            raise ValueError("Sana-WM intrinsics must not be empty.")

        first = intrinsics[0]
        if _is_sequence(first) and len(first) > 0 and _is_sequence(first[0]):
            if num_frames is not None and len(intrinsics) not in (1, num_frames):
                raise ValueError(
                    f"Sana-WM per-frame intrinsics length {len(intrinsics)} must be 1 or num_frames {num_frames}."
                )
            for idx, matrix in enumerate(intrinsics):
                _validate_numeric_matrix3x3(matrix, name=f"intrinsics[{idx}]")
            return

        if len(intrinsics) == 4:
            for idx, value in enumerate(intrinsics):
                try:
                    float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Sana-WM intrinsics[{idx}] must be numeric.") from exc
            return

        if len(intrinsics) == 3:
            _validate_numeric_matrix3x3(intrinsics, name="intrinsics")
            return

    raise ValueError("Sana-WM intrinsics must be a mapping, (3, 3), (F, 3, 3), or (4,) value.")


def _as_positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sana-WM {name} must be an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"Sana-WM {name} must be positive, got {parsed}.")
    return parsed


def _as_positive_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sana-WM {name} must be a number, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"Sana-WM {name} must be positive, got {parsed}.")
    return parsed


def _extract_image(mm_data: Any) -> Any:
    if not isinstance(mm_data, Mapping):
        return None
    return _unwrap_single(mm_data.get("image"))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_sana_wm_payload(prompt: Mapping[str, Any], *, require_image: bool = True) -> dict[str, Any]:
    """Return a prompt copy with canonical Sana-WM request metadata.

    The ``sana_wm`` block is read from the top-level ``sana_wm`` key, falling
    back to ``additional_information["sana_wm"]`` so the function is idempotent
    when called again on its own output. The first-frame image is read from
    ``multi_modal_data["image"]``.
    """

    if not isinstance(prompt, Mapping):
        raise ValueError(f"Sana-WM prompt must be a mapping, got {type(prompt).__name__}.")

    result = dict(prompt)
    mm_data = _as_dict(result.get("multi_modal_data"), name="multi_modal_data")
    image = _extract_image(mm_data)
    if require_image and image is None:
        raise ValueError("Sana-WM requires a first-frame image in `multi_modal_data['image']`.")

    additional = _as_dict(result.get("additional_information"), name="additional_information")
    raw = _first_present(
        result.get(SANA_WM_CANONICAL_KEY),
        additional.get(SANA_WM_CANONICAL_KEY),
        {},
    )
    raw = _as_dict(raw, name=SANA_WM_CANONICAL_KEY)

    num_frames = _as_positive_int(
        _first_present(raw.get("num_frames"), result.get("num_frames"), SANA_WM_DEFAULT_NUM_FRAMES),
        name="num_frames",
    )
    height = _as_positive_int(
        _first_present(raw.get("height"), result.get("height"), SANA_WM_DEFAULT_HEIGHT),
        name="height",
    )
    width = _as_positive_int(_first_present(raw.get("width"), result.get("width"), SANA_WM_DEFAULT_WIDTH), name="width")

    camera = raw.get("camera")
    camera = _as_dict(camera, name="camera") if camera is not None else None
    action = raw.get("action")
    if action is not None and (not isinstance(action, str) or not action.strip()):
        raise ValueError("Sana-WM action must be a non-empty string when provided.")
    if camera is not None and action is not None:
        raise ValueError("Sana-WM accepts exactly one of `camera` or `action`, not both.")
    if camera is None and action is None:
        raise ValueError("Sana-WM requires either explicit camera poses or an action DSL string.")

    canonical_camera = None
    if camera is not None:
        poses = camera.get("poses")
        if poses is None:
            raise ValueError("Sana-WM camera payload requires `poses`.")
        _validate_camera_poses(poses, num_frames=num_frames)
        canonical_camera = {
            "format": camera.get("format", SANA_WM_DEFAULT_CAMERA_FORMAT),
            "coordinate_system": camera.get("coordinate_system", SANA_WM_DEFAULT_COORDINATE_SYSTEM),
            "poses": poses,
        }

    intrinsics = raw.get("intrinsics")
    _validate_intrinsics(intrinsics, num_frames=num_frames)
    translation_speed = raw.get("translation_speed")
    rotation_speed_deg = raw.get("rotation_speed_deg")

    canonical = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "camera": canonical_camera,
        "action": action,
        "intrinsics": intrinsics,
    }
    if translation_speed is not None:
        canonical["translation_speed"] = _as_positive_float(translation_speed, name="translation_speed")
    if rotation_speed_deg is not None:
        canonical["rotation_speed_deg"] = _as_positive_float(rotation_speed_deg, name="rotation_speed_deg")
    official_repo_path = raw.get("official_repo_path")
    if official_repo_path is not None:
        if not isinstance(official_repo_path, str) or not official_repo_path.strip():
            raise ValueError("Sana-WM official_repo_path must be a non-empty string when provided.")
        canonical["official_repo_path"] = official_repo_path

    if image is not None:
        mm_data["image"] = image
        result["multi_modal_data"] = mm_data
    additional[SANA_WM_CANONICAL_KEY] = canonical
    result["additional_information"] = additional
    result.pop(SANA_WM_CANONICAL_KEY, None)
    return result


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Normalize Sana-WM request payloads for a downstream diffusion stage."""

    del source_outputs, requires_multimodal_data, streaming_context

    if prompt is None:
        return []
    prompts = prompt if isinstance(prompt, list) else [prompt]
    return [normalize_sana_wm_payload(p) for p in prompts]

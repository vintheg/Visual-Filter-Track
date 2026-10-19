import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


Videos = Path(__file__).resolve().parent / "Videos"
DEFAULT_INPUT = Videos / "Input" / "tau.mp4"
DEFAULT_OUTPUT = Videos / "Output" / "rendered.mp4"

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


@dataclass(frozen=True)
class Preset:
    name: str
    max_features: int
    min_features: int
    refresh_every: int
    feature_quality: float
    feature_min_distance: int
    max_connection_length: int
    bloom: float
    edge_gain: float
    scanline_strength: float
    glitch_strength: float
    network_alpha: float
    node_alpha: float
    flare_strength: float
    hud: bool


PRESETS = {
    "cyber": Preset(
        name="cyber",
        max_features=150,
        min_features=45,
        refresh_every=45,
        feature_quality=0.016,
        feature_min_distance=16,
        max_connection_length=210,
        bloom=0.18,
        edge_gain=0.14,
        scanline_strength=0.05,
        glitch_strength=0.0,
        network_alpha=0.44,
        node_alpha=0.50,
        flare_strength=0.06,
        hud=False,
    ),
    "ghost": Preset(
        name="ghost",
        max_features=120,
        min_features=28,
        refresh_every=36,
        feature_quality=0.018,
        feature_min_distance=18,
        max_connection_length=210,
        bloom=0.24,
        edge_gain=0.14,
        scanline_strength=0.07,
        glitch_strength=0.10,
        network_alpha=0.20,
        node_alpha=0.32,
        flare_strength=0.24,
        hud=False,
    ),
    "overdrive": Preset(
        name="overdrive",
        max_features=260,
        min_features=70,
        refresh_every=18,
        feature_quality=0.009,
        feature_min_distance=10,
        max_connection_length=320,
        bloom=0.48,
        edge_gain=0.28,
        scanline_strength=0.16,
        glitch_strength=0.36,
        network_alpha=0.48,
        node_alpha=0.58,
        flare_strength=0.46,
        hud=True,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a motion-reactive electronic visual filter over video."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input video path. Defaults to {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path. Defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="cyber",
        help="Visual style preset.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview while rendering. Press q to quit early.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Render only the first N frames. Useful for tests.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resize output by this factor. Use 0.5 for faster experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for deterministic flicker and glitch timing.",
    )
    return parser.parse_args()


def detect_features(gray, preset, max_corners=None, mask=None):
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners or preset.max_features,
        qualityLevel=preset.feature_quality,
        minDistance=preset.feature_min_distance,
        mask=mask,
        blockSize=7,
    )


def track_features_with_status(old_gray, new_gray, old_points):
    if old_points is None or len(old_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, old_points, None, **LK_PARAMS
    )

    if new_points is None or status is None:
        return old_points.reshape(-1, 2).copy(), np.zeros(len(old_points), dtype=bool)

    status = status.ravel().astype(bool)
    if error is not None:
        error = error.ravel()
        status &= np.isfinite(error)
        status &= error < 45.0

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        new_gray, old_gray, new_points, None, **LK_PARAMS
    )
    if back_points is not None and back_status is not None:
        old_flat = old_points.reshape(-1, 2)
        back_flat = back_points.reshape(-1, 2)
        back_error = np.linalg.norm(old_flat - back_flat, axis=1)
        status &= back_status.ravel().astype(bool)
        status &= back_error < 2.2

    points = new_points.reshape(-1, 2)
    height, width = new_gray.shape[:2]
    inside = (
        (points[:, 0] >= 2)
        & (points[:, 0] < width - 2)
        & (points[:, 1] >= 2)
        & (points[:, 1] < height - 2)
    )
    status &= inside
    return points.astype(np.float32), status


@dataclass
class TrackState:
    ids: np.ndarray
    raw_points: np.ndarray
    smoothed_points: np.ndarray
    previous_smoothed_points: np.ndarray
    ages: np.ndarray
    next_id: int
    edge_strengths: dict


def empty_track_state():
    empty_points = np.empty((0, 2), dtype=np.float32)
    return TrackState(
        ids=np.empty(0, dtype=np.int32),
        raw_points=empty_points.copy(),
        smoothed_points=empty_points.copy(),
        previous_smoothed_points=empty_points.copy(),
        ages=np.empty(0, dtype=np.int32),
        next_id=0,
        edge_strengths={},
    )


def add_tracks(state, points):
    if points is None or len(points) == 0:
        return state

    points = points.reshape(-1, 2).astype(np.float32)
    new_ids = np.arange(state.next_id, state.next_id + len(points), dtype=np.int32)
    state.ids = np.concatenate((state.ids, new_ids))
    state.raw_points = np.vstack((state.raw_points, points))
    state.smoothed_points = np.vstack((state.smoothed_points, points))
    state.previous_smoothed_points = np.vstack((state.previous_smoothed_points, points))
    state.ages = np.concatenate((state.ages, np.zeros(len(points), dtype=np.int32)))
    state.next_id += len(points)
    return state


def build_feature_mask(gray, existing_points, preset):
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    if existing_points is None:
        return mask

    radius = max(8, int(preset.feature_min_distance * 1.2))
    height, width = gray.shape[:2]
    for point in existing_points:
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(mask, (x, y), radius, 0, -1, cv2.LINE_AA)
    return mask


def detect_new_tracks(gray, preset, state, count):
    if count <= 0:
        return None

    mask = build_feature_mask(gray, state.smoothed_points, preset)
    return detect_features(gray, preset, max_corners=count, mask=mask)


def initialize_track_state(gray, preset):
    state = empty_track_state()
    detected = detect_features(gray, preset)
    if detected is not None:
        add_tracks(state, detected.reshape(-1, 2))
    refresh_edge_memory(state, preset)
    return state


def prune_edge_memory(state):
    active_ids = set(int(track_id) for track_id in state.ids)
    for edge in list(state.edge_strengths):
        if edge[0] not in active_ids or edge[1] not in active_ids:
            del state.edge_strengths[edge]


def refresh_edge_memory(state, preset):
    prune_edge_memory(state)
    if len(state.smoothed_points) < 3:
        state.edge_strengths.clear()
        return

    candidate_edges = set()
    for i, j in get_delaunay_edges(state.smoothed_points):
        p1 = state.smoothed_points[i]
        p2 = state.smoothed_points[j]
        distance = float(np.linalg.norm(p1 - p2))
        if distance > preset.max_connection_length:
            continue
        edge = tuple(sorted((int(state.ids[i]), int(state.ids[j]))))
        candidate_edges.add(edge)

    for edge in candidate_edges:
        state.edge_strengths[edge] = min(1.0, state.edge_strengths.get(edge, 0.28) + 0.18)

    for edge in list(state.edge_strengths):
        if edge not in candidate_edges:
            state.edge_strengths[edge] *= 0.93
            if state.edge_strengths[edge] < 0.10:
                del state.edge_strengths[edge]


def update_track_state(state, old_gray, new_gray, preset, frame_index):
    if len(state.raw_points) > 0:
        old_raw = state.raw_points.copy()
        old_smoothed = state.smoothed_points.copy()
        tracked_points, status = track_features_with_status(
            old_gray,
            new_gray,
            old_raw.reshape(-1, 1, 2).astype(np.float32),
        )

        state.ids = state.ids[status]
        state.raw_points = tracked_points[status]
        state.previous_smoothed_points = old_smoothed[status]
        state.ages = state.ages[status] + 1

        if len(state.raw_points) > 0:
            speeds = np.linalg.norm(state.raw_points - old_raw[status], axis=1)
            blend = np.clip(0.22 + speeds / 42.0, 0.22, 0.42)[:, None]
            state.smoothed_points = state.previous_smoothed_points * (1.0 - blend) + state.raw_points * blend
        else:
            state.smoothed_points = np.empty((0, 2), dtype=np.float32)
            state.previous_smoothed_points = np.empty((0, 2), dtype=np.float32)
            state.ages = np.empty(0, dtype=np.int32)

    needs_additions = (
        len(state.ids) < preset.min_features
        or (frame_index % preset.refresh_every == 0 and len(state.ids) < preset.max_features)
    )
    if needs_additions:
        needed = preset.max_features - len(state.ids)
        detected = detect_new_tracks(new_gray, preset, state, needed)
        if detected is not None:
            add_tracks(state, detected.reshape(-1, 2))

    refresh_edge_memory(state, preset)
    return state


def get_delaunay_edges(points):
    if len(points) < 3:
        return set()

    min_xy = np.floor(points.min(axis=0)).astype(int)
    max_xy = np.ceil(points.max(axis=0)).astype(int)
    padding = 8
    rect = (
        int(min_xy[0] - padding),
        int(min_xy[1] - padding),
        int(max_xy[0] - min_xy[0] + padding * 2),
        int(max_xy[1] - min_xy[1] + padding * 2),
    )
    if rect[2] <= 0 or rect[3] <= 0:
        return set()

    subdiv = cv2.Subdiv2D(rect)
    for x, y in points:
        try:
            subdiv.insert((float(x), float(y)))
        except cv2.error:
            continue

    triangles = subdiv.getTriangleList()
    edges = set()

    def nearest_index(vertex):
        deltas = points - vertex
        distances = np.einsum("ij,ij->i", deltas, deltas)
        index = int(np.argmin(distances))
        return index if distances[index] < 4.0 else None

    for triangle in triangles:
        vertices = triangle.reshape(3, 2)
        indices = [nearest_index(vertex) for vertex in vertices]
        if any(index is None for index in indices):
            continue
        a, b, c = indices
        if len({a, b, c}) < 3:
            continue
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return edges


def pulse(frame_index, period, width=0.34):
    phase = (frame_index % period) / period
    wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
    return float(np.clip((wave - (1.0 - width)) / width, 0.0, 1.0))


def color_ramp(value):
    value = float(np.clip(value, 0.0, 1.0))
    hot_pink = np.array([213, 42, 255], dtype=np.float32)
    cyan = np.array([255, 244, 35], dtype=np.float32)
    electric_blue = np.array([255, 92, 24], dtype=np.float32)
    if value < 0.5:
        mix = value / 0.5
        color = hot_pink * (1.0 - mix) + cyan * mix
    else:
        mix = (value - 0.5) / 0.5
        color = cyan * (1.0 - mix) + electric_blue * mix
    return tuple(int(c) for c in color)


def optic_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    dim = np.array([166, 190, 178], dtype=np.float32)
    phosphor = np.array([238, 248, 232], dtype=np.float32)
    color = dim * (1.0 - value) + phosphor * value
    return tuple(int(c) for c in color)


def build_background(frame, preset):
    soft = cv2.GaussianBlur(frame, (0, 0), 1.6)
    lab = cv2.cvtColor(soft, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    boosted = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    cool_shadow = np.full_like(boosted, (92, 118, 108))
    warm_highlight = np.full_like(boosted, (128, 134, 118))
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    highlight_mask = cv2.GaussianBlur((gray > 150).astype(np.float32), (0, 0), 9)
    tone = cv2.addWeighted(cool_shadow, 0.72, warm_highlight, 0.28, 0).astype(np.float32)
    tone = tone * (1.0 - highlight_mask[:, :, None] * 0.35) + warm_highlight * (highlight_mask[:, :, None] * 0.35)
    boosted = cv2.addWeighted(boosted, 0.78, tone.astype(np.uint8), 0.22, 0)

    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 135)
    edge_layer = np.zeros_like(boosted)
    edge_layer[:, :, 0] = cv2.GaussianBlur(edges, (0, 0), 2.5)
    edge_layer[:, :, 1] = edges
    edge_layer[:, :, 2] = cv2.GaussianBlur(edges, (0, 0), 1.8)

    tinted = cv2.addWeighted(boosted, 0.90, edge_layer, preset.edge_gain, 0)
    return cv2.convertScaleAbs(tinted, alpha=1.05, beta=-1)


def scale_color(color, scale):
    return tuple(int(np.clip(channel * scale, 0, 255)) for channel in color)


def draw_network(frame, glow, track_state, frame_index, preset):
    points = track_state.smoothed_points
    if len(points) < 2:
        return

    height, width = frame.shape[:2]
    line_layer = np.zeros_like(frame)
    node_layer = np.zeros_like(frame)
    id_to_index = {int(track_id): index for index, track_id in enumerate(track_state.ids)}

    velocities = points - track_state.previous_smoothed_points
    for edge, strength in track_state.edge_strengths.items():
        if edge[0] not in id_to_index or edge[1] not in id_to_index:
            continue

        i = id_to_index[edge[0]]
        j = id_to_index[edge[1]]
        p1 = points[i]
        p2 = points[j]
        delta = p1 - p2
        distance_sq = float(delta[0] * delta[0] + delta[1] * delta[1])
        if distance_sq > (preset.max_connection_length * 1.25) ** 2:
            continue

        x1, y1 = np.rint(p1).astype(int)
        x2, y2 = np.rint(p2).astype(int)
        if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
            continue

        distance = math.sqrt(max(distance_sq, 1.0))
        energy = 1.0 - min(distance / preset.max_connection_length, 1.0)
        energy += min(float(np.linalg.norm(velocities[i]) + np.linalg.norm(velocities[j])) / 22.0, 0.35)
        energy = float(np.clip(energy, 0.0, 1.0))
        color = optic_color(0.28 + energy * 0.62) if preset.name != "overdrive" else color_ramp(energy)
        age_fade = min(1.0, (min(track_state.ages[i], track_state.ages[j]) + 2) / 8.0)
        visibility = float(np.clip(strength * age_fade * (0.62 + energy * 0.38), 0.0, 1.0))
        line_color = scale_color(color, 0.38 + visibility * 0.62)
        thickness = 1 + int(preset.name == "overdrive" and energy > 0.72 and strength > 0.72)

        if preset.name == "overdrive":
            cv2.line(glow, (x1, y1), (x2, y2), line_color, thickness + 2, cv2.LINE_AA)
        elif visibility > 0.22:
            cv2.line(glow, (x1, y1), (x2, y2), scale_color(line_color, 0.42), 2, cv2.LINE_AA)
        cv2.line(line_layer, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

    for i, point in enumerate(points):
        x, y = np.rint(point).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            continue

        motion = min(float(np.linalg.norm(velocities[i])) / 16.0, 1.0)
        age_fade = min(1.0, (track_state.ages[i] + 2) / 6.0)
        node_energy = float(np.clip(0.30 + motion * 0.58 + age_fade * 0.20, 0.0, 1.0))

        color = optic_color(0.48 + node_energy * 0.44) if preset.name != "overdrive" else color_ramp(node_energy)
        color = scale_color(color, 0.42 + age_fade * 0.58)
        radius = 1 + int(node_energy > 0.58) + int(preset.name == "overdrive" and node_energy > 0.88)
        cv2.circle(glow, (x, y), radius + 4, color, -1, cv2.LINE_AA)

        if preset.name == "overdrive":
            cv2.circle(node_layer, (x, y), radius, color, -1, cv2.LINE_AA)
        elif int(track_state.ids[i]) % 7 == 0:
            box_size = 5 + int(node_energy > 0.7) * 3
            cv2.rectangle(
                node_layer,
                (x - box_size, y - box_size),
                (x + box_size, y + box_size),
                color,
                1,
                cv2.LINE_AA,
            )
            if int(track_state.ids[i]) % 14 == 0:
                label = f"{(700 + int(track_state.ids[i]) * 13) % 1000:03d}"
                offset_x = 8 if x < width - 58 else -42
                tick_end = (x + (offset_x - 3), y)
                cv2.line(node_layer, (x, y), tick_end, optic_color(0.95), 1, cv2.LINE_AA)
                cv2.putText(
                    node_layer,
                    label,
                    (x + offset_x, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.circle(node_layer, (x, y), 1, color, -1, cv2.LINE_AA)

    frame[:] = cv2.addWeighted(frame, 1.0, line_layer, preset.network_alpha, 0)
    frame[:] = cv2.addWeighted(frame, 1.0, node_layer, preset.node_alpha, 0)


def add_scanlines(frame, frame_index, strength):
    if strength <= 0:
        return frame

    height, width = frame.shape[:2]
    rows = np.arange(height, dtype=np.float32)
    wave = 0.5 + 0.5 * np.sin((rows + frame_index * 0.7) * 0.42)
    line_mask = (1.0 - strength) + wave[:, None] * strength
    vignette_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    vignette_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    vignette = 1.0 - 0.28 * np.clip(vignette_x[None, :] ** 2 + vignette_y[:, None] ** 2, 0.0, 1.0)
    mask = (line_mask * vignette)[:, :, None]
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def add_chromatic_aberration(frame, frame_index, strength):
    if strength <= 0:
        return frame

    offset = int(round(1 + 2 * strength + pulse(frame_index, 53)))
    if offset <= 1 and strength < 0.12:
        return frame
    blue, green, red = cv2.split(frame)
    transform_left = np.float32([[1, 0, -offset], [0, 1, 0]])
    transform_right = np.float32([[1, 0, offset], [0, 1, 0]])
    blue = cv2.warpAffine(blue, transform_left, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    red = cv2.warpAffine(red, transform_right, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge((blue, green, red))


def add_glitch_slices(frame, frame_index, strength, rng):
    if strength <= 0.12 or pulse(frame_index, 47, width=0.12) <= 0:
        return frame

    height, width = frame.shape[:2]
    output = frame.copy()
    slice_count = 1 + int(3 * strength)
    max_shift = max(1, int(width * 0.01 * strength))

    for _ in range(slice_count):
        y = int(rng.integers(0, max(1, height - 3)))
        h = int(rng.integers(2, max(3, height // 32)))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        output[y : min(height, y + h)] = np.roll(output[y : min(height, y + h)], shift, axis=1)
    return output


def add_ambient_flares(frame, frame_index, strength, rng):
    if strength <= 0:
        return frame

    height, width = frame.shape[:2]
    output = frame.astype(np.float32)

    columns = rng.normal(0, 1, (1, width, 1)).astype(np.float32)
    columns = cv2.GaussianBlur(columns, (0, 0), max(5, width / 90))
    columns = columns.reshape(1, width, 1)
    output += columns * (5.0 * strength)

    noise = rng.normal(0, 2.2 * strength, frame.shape).astype(np.float32)
    output += noise

    vignette_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    vignette_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    vignette = 1.0 - 0.22 * np.clip(vignette_x[None, :] ** 2 + vignette_y[:, None] ** 2, 0.0, 1.0)
    output *= vignette[:, :, None]

    phosphor = np.zeros_like(frame)
    speck_count = max(3, int(width * height / 70000 * strength))
    for _ in range(speck_count):
        if rng.random() > 0.55:
            continue
        sx = int(rng.integers(0, width))
        sy = int(rng.integers(0, height))
        cv2.circle(phosphor, (sx, sy), 1, (236, 246, 232), -1, cv2.LINE_AA)

    output = np.clip(output, 0, 255).astype(np.uint8)
    return cv2.addWeighted(output, 1.0, phosphor, 0.42, 0)


def draw_hud(frame, frame_index, points_count, preset):
    if not preset.hud:
        return

    height, width = frame.shape[:2]
    color = (245, 240, 80)
    hot = (215, 55, 255)
    text = f"{preset.name.upper()}  FRAME {frame_index:06d}  LOCKS {points_count:03d}"
    cv2.putText(frame, text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.line(frame, (18, 38), (min(width - 18, 330), 38), hot, 1, cv2.LINE_AA)
    cv2.line(frame, (18, height - 24), (118, height - 24), color, 1, cv2.LINE_AA)
    cv2.line(frame, (18, height - 24), (18, height - 68), color, 1, cv2.LINE_AA)
    cv2.line(frame, (width - 18, 38), (width - 118, 38), color, 1, cv2.LINE_AA)
    cv2.line(frame, (width - 18, 38), (width - 18, 82), color, 1, cv2.LINE_AA)


def render_frame(frame, track_state, frame_index, preset, rng):
    styled = build_background(frame, preset)
    styled = add_ambient_flares(styled, frame_index, preset.flare_strength, rng)
    glow = np.zeros_like(styled)
    draw_network(styled, glow, track_state, frame_index, preset)

    blur = cv2.GaussianBlur(glow, (0, 0), 7.5)
    styled = cv2.addWeighted(styled, 1.0, blur, preset.bloom, 0)
    styled = cv2.addWeighted(styled, 1.0, glow, 0.12, 0)
    styled = add_chromatic_aberration(styled, frame_index, preset.glitch_strength)
    styled = add_glitch_slices(styled, frame_index, preset.glitch_strength, rng)
    styled = add_scanlines(styled, frame_index, preset.scanline_strength)
    draw_hud(styled, frame_index, len(track_state.ids), preset)
    return styled


def resize_frame(frame, scale):
    if scale == 1.0:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def open_capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def make_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    return writer


def process_video(args):
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("--scale must be finite and greater than 0")
    if args.limit_frames < 0:
        raise ValueError("--limit-frames must be nonnegative")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.input.resolve() == args.output.resolve() or (
        args.output.exists() and args.input.samefile(args.output)
    ):
        raise ValueError("Input and output must be different files")
    preset = PRESETS[args.preset]
    rng = np.random.default_rng(args.seed)
    cap = open_capture(args.input)

    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read the first video frame.")

        first_frame = resize_frame(first_frame, args.scale)
        height, width = first_frame.shape[:2]
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("Output dimensions must be even and at least 2 pixels; adjust --scale.")
        writer = make_writer(args.output, fps, (width, height))

        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        track_state = initialize_track_state(old_gray, preset)
        frame_index = 0

        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Preset: {preset.name}")
        print(f"Size:   {width}x{height} @ {fps:.2f} fps")

        while True:
            output = render_frame(first_frame, track_state, frame_index, preset, rng)
            writer.write(output)

            if args.preview:
                cv2.imshow("Electronic Visual Filter", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.limit_frames and frame_index >= args.limit_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame, args.scale)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            track_state = update_track_state(track_state, old_gray, frame_gray, preset, frame_index)
            old_gray = frame_gray
            first_frame = frame

            if frame_index % 120 == 0:
                print(
                    f"Processed {frame_index} frames "
                    f"| tracking {len(track_state.ids)} points "
                    f"| preserving {len(track_state.edge_strengths)} edges"
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Saved to: {args.output}")


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

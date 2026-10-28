"""Small PNG/SVG diagnostic charts using only NumPy and the standard library."""

import html
import struct
import zlib
from pathlib import Path

import numpy as np

_FONT = {
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 15],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [31, 4, 4, 4, 4, 4, 31],
    "J": [7, 2, 2, 2, 18, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17],
    "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 21, 21, 21, 10],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
    "Z": [31, 1, 2, 4, 8, 16, 31],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31],
    "3": [30, 1, 1, 14, 1, 1, 30],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 16, 30, 1, 1, 30],
    "6": [14, 16, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 1, 14],
    ".": [0, 0, 0, 0, 0, 12, 12],
    "-": [0, 0, 0, 31, 0, 0, 0],
    "_": [0, 0, 0, 0, 0, 0, 31],
    "/": [1, 2, 2, 4, 8, 8, 16],
    ":": [0, 12, 12, 0, 12, 12, 0],
    "(": [2, 4, 8, 8, 8, 4, 2],
    ")": [8, 4, 2, 2, 2, 4, 8],
    "+": [0, 4, 4, 31, 4, 4, 0],
}


def chart(path, title, x, series, xlabel, ylabel, *, unit_interval=False, connect=True):
    """Draw a line/point chart of one or more series and save it as both PNG and SVG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(x)
    categorical = any(isinstance(value, str) for value in labels)
    x = (
        np.arange(len(labels), dtype=float)
        if categorical
        else np.asarray(labels, dtype=float)
    )
    if not len(x):
        x, labels = np.array([0.0]), ["none"]
        series = {"no observations": [0.0]}
    arrays = {name: np.asarray(values, dtype=float) for name, values in series.items()}
    finite_y = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
    low, high = (
        (0.0, 1.0)
        if unit_interval
        else (
            min(0.0, float(finite_y.min())) if len(finite_y) else 0.0,
            float(finite_y.max()) if len(finite_y) else 1.0,
        )
    )
    high = high if high > low else low + 1
    if not unit_interval:
        high += 0.08 * (high - low)
    xlow, xhigh = float(x.min()), float(x.max())
    if xlow == xhigh:
        xlow, xhigh = xlow - 0.5, xhigh + 0.5
    width, height = 1100, 660
    left, right, top, bottom = 100, 1060, 100, 470
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    def text(px, py, value, color=(25, 25, 25)):
        """Draw a label with the built-in 5x7 pixel font (PNG) and as SVG text."""
        value = str(value)
        svg.append(
            f'<text x="{px}" y="{py + 14}" fill="rgb{color}" font-family="Arial,sans-serif" font-size="16">{html.escape(value)}</text>'
        )
        for index, character in enumerate(value.upper()):
            for row, bits in enumerate(_FONT.get(character, [0] * 7)):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        xx, yy = px + index * 12 + col * 2, py + row * 2
                        if 0 <= yy < height - 2 and 0 <= xx < width - 2:
                            canvas[yy : yy + 2, xx : xx + 2] = color

    def line(x1, y1, x2, y2, color):
        """Draw a straight line segment on the canvas and in the SVG."""
        count = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)), 1) + 1
        xx = np.clip(np.linspace(x1, x2, count).astype(int), 0, width - 1)
        yy = np.clip(np.linspace(y1, y2, count).astype(int), 0, height - 1)
        canvas[yy, xx] = color
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgb{color}" stroke-width="2"/>'
        )

    text(30, 25, title[:85])
    text(30, 62, ylabel[:85])
    for value in np.linspace(low, high, 6):
        py = bottom - (value - low) / (high - low) * (bottom - top)
        line(left, py, right, py, (225, 225, 225))
        text(8, int(py) - 7, f"{value:.3g}")
    tick_indices = np.unique(np.linspace(0, len(x) - 1, min(7, len(x))).astype(int))
    for index in tick_indices:
        px = left + (x[index] - xlow) / (xhigh - xlow) * (right - left)
        label = labels[index] if categorical else f"{x[index]:.3g}"
        text(int(px) - 18, bottom + 12, str(label)[:12])
    line(left, top, left, bottom, (30, 30, 30))
    line(left, bottom, right, bottom, (30, 30, 30))
    text(left, 520, xlabel[:78])
    colors = [(40, 105, 155), (215, 100, 45), (40, 145, 80), (130, 80, 155)]
    for index, (name, values) in enumerate(arrays.items()):
        color = colors[index % len(colors)]
        valid = np.isfinite(values)
        points = [
            (
                left + (a - xlow) / (xhigh - xlow) * (right - left),
                bottom - (b - low) / (high - low) * (bottom - top),
            )
            for a, b in zip(x[valid], values[valid])
        ]
        if connect:
            for first, second in zip(points, points[1:]):
                line(*first, *second, color)
        for px, py in points:
            xx, yy = int(px), int(py)
            canvas[
                max(0, yy - 3) : min(height, yy + 4),
                max(0, xx - 3) : min(width, xx + 4),
            ] = color
            svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="rgb{color}"/>')
        text(left + (index % 2) * 480, 565 + (index // 2) * 28, name[:37], color)
    svg.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")

    def chunk(kind, data):
        """Encode one PNG chunk (length, type, data, CRC)."""
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row.tobytes() for row in canvas)
    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.with_suffix(".png").write_bytes(png)

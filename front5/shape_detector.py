# shape_detector.py
import cv2, numpy as np
from EnDe import decode                     # your existing decode()

# ───── helpers ──────────────────────────────────────────────────────
def _color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def group_similar_colors(rgb_vals, threshold: int = 10):
    groups, counts = [], []
    for c in rgb_vals:
        for i, g in enumerate(groups):
            if _color_distance(c, g[0]) < threshold:
                groups[i].append(c); counts[i] += 1
                break
        else:
            groups.append([c]); counts.append(1)
    return [(g[0], n) for g, n in zip(groups, counts)]

# ───── main detector / decoder ─────────────────────────────────────
def detect_and_decode(image_bytes: bytes,
                      shape: str,
                      min_size: int = 5,
                      max_size: int = 50):
    """
    Returns
    -------
    annotated_png_bytes : bytes
    triangles_for_js    : list[dict]   (only for triangle mode)
    colour_groups       : list[((R,G,B), count)]
    """
    # ── load image ────────────────────────────────────────────────
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid PNG/JPEG")

    # ── pre‑process for robust contour detection ─────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr  = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    bounds, tris_js = [], []

    def in_sz(w, h=None):
        if h is None:
            return min_size <= w <= max_size
        return min_size <= w <= max_size and min_size <= h <= max_size

    # ── classify & collect requested shapes ──────────────────────
    for c in cnts:
        if cv2.contourArea(c) < 3:
            continue

        if shape == "Triangle":
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 3:
                xs, ys = approx[:, 0, 0], approx[:, 0, 1]
                if in_sz(xs.ptp(), ys.ptp()):
                    bounds.append(approx.reshape(-1, 2))
                    tris_js.append([{"x": int(p[0]), "y": int(p[1])}
                                     for p in approx[:, 0]])

        elif shape == "Rectangle":
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                xs, ys = approx[:, 0, 0], approx[:, 0, 1]
                if in_sz(xs.ptp(), ys.ptp()):
                    x, y, w, h = cv2.boundingRect(approx)
                    bounds.append((x, y, w, h))

        else:  # Circle
            area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
            if peri == 0: continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ >= 0.80:
                (x, y), r = cv2.minEnclosingCircle(c)
                if in_sz(r):
                    bounds.append((int(x), int(y), int(r)))

    # ── decode stego + annotate ──────────────────────────────────
    _, annotated, rgb_vals = decode(img, shape,
                                    boundaries=bounds,
                                    min_size=min_size,
                                    max_size=max_size)

    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise RuntimeError("Could not encode annotated PNG")
    return buf.tobytes(), tris_js, group_similar_colors(rgb_vals)

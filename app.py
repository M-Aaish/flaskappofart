import os
import base64
import itertools
import math
from io import BytesIO
# at the top of app.py
import colormixer as mixbox

import cv2
import numpy as np
from PIL import Image
from flask import (                       # ← jsonify appended
    Flask, render_template, request,
    session, redirect, url_for, send_file, jsonify
)
from werkzeug.utils import secure_filename

# ─────────── Flask setup ─────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",   # your HTML lives here
    static_folder="templates",     # serve static assets from here too
    static_url_path=""             # mount them at the web-root (/css/…, /js/…, /assets/…)
)

app.secret_key = os.urandom(16)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────── Colour-database helpers ─────────────────────────────────
def read_color_file(path: str = "color.txt") -> str:
    with open(path, encoding="utf8") as f:
        return f.read()


def parse_color_db(txt: str):
    dbs, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            cur = line
            dbs[cur] = []
        else:
            tok = line.split()
            if len(tok) < 3:
                continue
            parts = tok[-2].split(",")
            if len(parts) != 3:
                continue
            try:
                r, g, b = map(int, parts)
            except ValueError:
                continue
            name = " ".join(tok[1:-2])
            dbs[cur].append((name, (r, g, b)))
    return dbs


def convert_db_list_to_dict(lst):
    return {n: list(rgb) for n, rgb in lst}


def mix_colors(recipe):
    total = sum(p for _, p in recipe)
    r = sum(rgb[0] * p for rgb, p in recipe) / total
    g = sum(rgb[1] * p for rgb, p in recipe) / total
    b = sum(rgb[2] * p for rgb, p in recipe) / total
    return (round(r), round(g), round(b))


def color_error(c1, c2):
    return math.dist(c1, c2)


def generate_recipes(target, base_colors, step=10.0):
    base = list(base_colors.items())
    candidates = []

    # single-colour quick matches
    for name, rgb in base:
        err = color_error(rgb, target)
        if err < 5:
            candidates.append(([(name, 100)], rgb, err))

    # triple-mix brute-force search
    for (n1, r1), (n2, r2), (n3, r3) in itertools.combinations(base, 3):
        p_range = np.arange(0, 101, step)
        for p1 in p_range:
            for p2 in np.arange(0, 101 - p1, step):
                p3 = 100 - p1 - p2
                recipe = [(n1, p1), (n2, p2), (n3, p3)]
                mixed = mix_colors([(r1, p1), (r2, p2), (r3, p3)])
                err = color_error(mixed, target)
                candidates.append((recipe, mixed, err))

    candidates.sort(key=lambda x: x[2])
    top, seen = [], set()
    for rec, mix, err in candidates:
        key = tuple(sorted((n, p) for n, p in rec if p > 0))
        if key not in seen:
            seen.add(key)
            top.append((rec, mix, err))
        if len(top) == 3:
            break
    return top


# ─────────── Color-grouping helper ───────────────────────────────────
def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))


def group_similar_colors(rgb_vals, threshold=1):
    grouped_colors = []
    counts = []

    for color in rgb_vals:
        found_group = False
        for i, group in enumerate(grouped_colors):
            if color_distance(color, group[0]) < threshold:
                grouped_colors[i].append(color)
                counts[i] += 1
                found_group = True
                break
        if not found_group:
            grouped_colors.append([color])
            counts.append(1)
    return [(group[0], count) for group, count in zip(grouped_colors, counts)]


# ─────────── Shape-encoding/decoding (EnDe) ─────────────────────────
from EnDe import encode, decode

import random
# ─────────── Oil-painting helper ─────────────────────────────────────
from painterfun import oil_main  # your oil painting function


# ─────────── Geometrize & Foogle Man Repo placeholders ──────────────
try:
    from shape_art_generator import main_page as foogle_man_page
except ImportError:
    foogle_man_page = None

try:
    from geometrize import geometrize_app
except ImportError:
    geometrize_app = None


# ─────────── Routes ───────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("image_generator_page"))


@app.route("/image_generator", methods=["GET", "POST"])
def image_generator_page():
    """
    Generate shape-art on an uploaded image. Uses EnDe.encode().
    (Unchanged per your request.)
    """
    error = None
    result_image_data = None

    if request.method == "POST":
        file = request.files.get("base_image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        elif not allowed_file(file.filename):
            error = "Unsupported file type."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            file_bytes = np.asarray(bytearray(open(save_path, "rb").read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                error = "Failed to read uploaded image."
            else:
                shape_opt = request.form.get("shape_type", "Triangle")
                num_shapes = int(request.form.get("num_shapes", 100))
                min_size = int(request.form.get("min_size", 10))
                max_size = int(request.form.get("max_size", 50))

                try:
                    encoded_img, boundaries = encode(
                        img_bgr,
                        shape_opt,
                        output_path="",
                        num_shapes=num_shapes,
                        min_size=min_size,
                        max_size=max_size
                    )
                    encoded_rgb = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(encoded_rgb)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    result_image_data = base64.b64encode(buf.getvalue()).decode("ascii")

                    tmp_name = f"shape_art_{os.getpid()}.png"
                    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                    with open(tmp_path, "wb") as f_out:
                        f_out.write(buf.getvalue())
                    session["shape_art_path"] = tmp_path

                except Exception as e:
                    error = f"Error generating shape art: {e}"

    return render_template(
        "image_generator.html",
        error=error,
        result_image_data=result_image_data,
        active_page="image_generator"
    )


@app.route("/download_shape_art")
def download_shape_art():
    path = session.get("shape_art_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("image_generator_page"))


@app.route("/shape_detector", methods=["GET", "POST"])
def shape_detector_page():
    """
    Decode an encoded image and show color swatches & recipes.
    """
    error = None
    decoded_image_data = None
    grouped_colors = []
    selected_recipe_color = None
    recipe_results = None

    if "grouped_colors" in session:
        grouped_colors = session["grouped_colors"]

    if request.method == "POST":
        # Part A: Upload & Decode
        if "encoded_image" in request.files:
            file = request.files["encoded_image"]
            if not file or file.filename == "":
                error = "Please upload an encoded PNG/JPG."
            elif not allowed_file(file.filename):
                error = "Unsupported file type."
            else:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                file_bytes = np.asarray(bytearray(open(save_path, "rb").read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    error = "Failed to read uploaded image."
                else:
                    shape_opt = request.form.get("shape_detect", "Triangle")
                    min_size = int(request.form.get("min_size", 3))
                    max_size = int(request.form.get("max_size", 10))

                    binary_img, annotated_img, rgb_vals = decode(
                        img_bgr,
                        shape_opt,
                        boundaries=[],
                        min_size=min_size,
                        max_size=max_size
                    )

                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(annotated_rgb)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    decoded_image_data = base64.b64encode(buf.getvalue()).decode("ascii")

                    rgb_py = [[int(c) for c in col] for col in rgb_vals]

                    grouped = group_similar_colors(rgb_py, threshold=10)
                    grouped = sorted(grouped, key=lambda x: x[1], reverse=True)
                    session["grouped_colors"] = grouped
                    grouped_colors = grouped

                    tmp_name = f"shape_analysis_{os.getpid()}.png"
                    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                    with open(tmp_path, "wb") as f_out:
                        f_out.write(buf.getvalue())
                    session["analysis_path"] = tmp_path

        # Part B: Generate Recipe (traditional full-page flow)
        if request.form.get("action") == "generate_recipe":
            sel = request.form.get("selected_color")
            if sel:
                r, g, b = [int(x) for x in sel.split(",")]
                selected_recipe_color = (r, g, b)
                step = float(request.form.get("step", 10.0))
                db_choice = request.form.get("db_choice")

                full_txt = read_color_file("color.txt")
                all_dbs = parse_color_db(full_txt)
                raw_list = all_dbs.get(db_choice, [])
                base_dict = {name: list(rgb) for name, rgb in raw_list}

                recipe_results = generate_recipes(selected_recipe_color, base_dict, step=step)

    return render_template(
        "shape_detector.html",
        error=error,
        decoded_image_data=decoded_image_data,
        grouped_colors=grouped_colors,
        selected_recipe_color=selected_recipe_color,
        recipe_results=recipe_results,
        db_list=list(parse_color_db(read_color_file("color.txt")).keys()),
        active_page="shape_detector"
    )


@app.route("/download_analysis")
def download_analysis():
    path = session.get("analysis_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("shape_detector_page"))


@app.route("/oil_painting", methods=["GET", "POST"])
def oil_painting_page():
    """
    Transform a photo into an oil painting (adjustable intensity).
    """
    error = None
    result_image_data = None

    if request.method == "POST":
        file = request.files.get("oil_image")
        intensity = int(request.form.get("intensity", 10))
        if not file or file.filename == "":
            error = "Please upload an image."
        elif not allowed_file(file.filename):
            error = "Unsupported file type."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            file_bytes = np.asarray(bytearray(open(save_path, "rb").read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                error = "Failed to read uploaded image."
            else:
                try:
                    output_img = oil_main(img_bgr, intensity)
                    output_img = (output_img * 255).astype(np.uint8)
                    rgb_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    result_image_data = base64.b64encode(buf.getvalue()).decode("ascii")

                    tmp_name = f"oil_painting_{os.getpid()}.png"
                    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                    with open(tmp_path, "wb") as f_out:
                        f_out.write(buf.getvalue())
                    session["oil_path"] = tmp_path

                except Exception as e:
                    error = f"Error generating oil painting: {e}"

    return render_template(
        "oil_painting.html",
        error=error,
        result_image_data=result_image_data,
        active_page="oil_painting"
    )


@app.route("/download_oil")
def download_oil():
    path = session.get("oil_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("oil_painting_page"))

@app.route("/colour_merger", methods=["GET", "POST"])
def colour_merger_page():
    if "colors" not in session:
        session["colors"] = [
            {"rgb": [255, 0, 0], "weight": 0.5},
            {"rgb": [0, 255, 0], "weight": 0.5}
        ]

    if request.method == "POST":
        new_colors = []
        idx = 0
        while True:
            rgb_str = request.form.get(f"rgb-{idx}")
            weight_str = request.form.get(f"weight-{idx}")
            if rgb_str is None or weight_str is None:
                break
            try:
                r, g, b = map(int, rgb_str.split(","))
                w = float(weight_str)
                new_colors.append({"rgb": [r, g, b], "weight": w})
            except ValueError:
                pass
            idx += 1
        if new_colors:
            session["colors"] = new_colors

    colors = session["colors"]

    def get_mixed_rgb(colors_list):
        # build a zeroed latent of the correct size
        z_mix = [0] * mixbox.LATENT_SIZE
        total = sum(c["weight"] for c in colors_list)
        for i in range(len(z_mix)):
            z_mix[i] = sum(
                c["weight"] * mixbox.rgb_to_latent(c["rgb"])[i]
                for c in colors_list
            ) / total
        return mixbox.latent_to_rgb(z_mix)

    mixed_rgb = get_mixed_rgb(colors)

    return render_template(
        "colour_merger.html",
        colors=colors,
        mixed_rgb=mixed_rgb,
        active_page="colour_merger"
    )
@app.route("/recipe_generator", methods=["GET", "POST"])
def recipe_generator_page():
    """
    Standalone paint-recipe generator.
    """
    error = None
    recipes = None
    selected_color = (255, 0, 0)

    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)
    db_keys = list(databases.keys())

    if request.method == "POST":
        # grab the posted hex (instead of non‐existent 'target_hex')
        hex_color = request.form.get("hex_color")
        if hex_color:
            # strip leading ‘#’ then parse RRGGBB
            hv = hex_color.lstrip('#')
            selected_color = (
                int(hv[0:2], 16),
                int(hv[2:4], 16),
                int(hv[4:6], 16),
            )
        else:
            try:
                r = int(request.form.get("r", 0))
                g = int(request.form.get("g", 0))
                b = int(request.form.get("b", 0))
                selected_color = (r, g, b)
            except:
                pass

        step = float(request.form.get("step", 10.0))
        db_choice = request.form.get("db_choice", db_keys[0])

        base_list = databases.get(db_choice, [])
        base_dict = {name: list(rgba) for name, rgba in base_list}

        recipes = generate_recipes(selected_color, base_dict, step=step)

    return render_template(
        "recipe_generator.html",
        databases=db_keys,
        selected_color=selected_color,
        recipes=recipes,
        active_page="recipe_generator"
    )
@app.route("/colors_db", methods=["GET", "POST"])
def colors_db_page():
    """
    Browse/Add/Remove colors from color.txt.
    """
    # 1) Read & parse the file
    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)  # returns { db_name: [(color_name, (r,g,b)), …], … }

    # 2) Decide which sub‐section to show
    action = request.form.get("action", "browse")
    if action == "browse":
        subpage = "databases"
    elif action == "add":
        subpage = "add"
    elif action == "remove_colors":
        subpage = "remove_colors"
    elif action == "create_db":
        subpage = "custom"
    elif action == "remove_db":
        subpage = "remove_database"
    else:
        subpage = "databases"

    # (optional) you can pass a message tuple (type, text) if you need feedback
    message = None

    return render_template(
        "colors_db.html",
        databases=databases,    # a dict for your template to iterate over
        subpage=subpage,        # controls which form/block shows
        message=message,
        active_page="colors_db"
    )



def resize_for_processing(image, max_dim=800):
    """Resize image for speed, return (resized, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / w, max_dim / h)
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, 1.0

def pixelate_image(image, block_size=5):
    """Pixelate by downscaling & upscaling."""
    h, w = image.shape[:2]
    small = cv2.resize(image,
                       (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def draw_random_circles(image, min_radius, max_radius, num_circles):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_circles):
        r = random.randint(min_radius, max_radius)
        x = random.randint(r, w - r)
        y = random.randint(r, h - r)
        color = out[y, x].tolist()
        cv2.circle(out, (x, y), r, color, -1)
    return out

def draw_random_rectangles(image, min_size, max_size, num_rects):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_rects):
        rw = random.randint(min_size, max_size)
        rh = random.randint(min_size, max_size)
        x  = random.randint(0, w - rw)
        y  = random.randint(0, h - rh)
        angle = random.randint(0, 360)
        color = out[y, x].tolist()
        rect = np.array([[x, y],
                         [x+rw, y],
                         [x+rw, y+rh],
                         [x, y+rh]], dtype=np.float32)
        M = cv2.getRotationMatrix2D((x+rw/2, y+rh/2), angle, 1.0)
        pts = cv2.transform(np.array([rect]), M)[0].astype(int)
        cv2.fillPoly(out, [pts], color)
    return out

def draw_random_triangles(image, min_size, max_size, num_triangles):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_triangles):
        side = random.randint(min_size, max_size)
        tri_h = int(side * np.sqrt(3) / 2)
        x = random.randint(0, w - side)
        y = random.randint(tri_h, h)
        color = out[y - tri_h//2, x + side//2].tolist()
        tri = np.array([(x, y),
                        (x+side, y),
                        (x+side//2, y-tri_h)],
                       dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((x+side//2, y-tri_h/3), angle, 1.0)
        pts = cv2.transform(np.array([tri]), M)[0].astype(int)
        cv2.fillPoly(out, [pts], color)
    return out

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in {"png", "jpg", "jpeg", "webp"}

# ─────────── Route ────────────────────────────────────────────────────

@app.route("/foogle_man_repo", methods=["GET", "POST"])
def foogle_man_repo_page():
    original_b64  = None
    generated_b64 = None
    download_url  = None
    num_shapes    = 0

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "" or not allowed_file(file.filename):
            # invalid upload → just re-render
            return render_template("foogle_man_repo.html")

        # read image
        data  = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return render_template("foogle_man_repo.html")

        # form params
        shape_type = request.form.get("shape_type", "Circles")
        min_size   = int(request.form.get("min_size", 5))
        max_size   = int(request.form.get("max_size", 30))
        num_shapes = int(request.form.get("num_shapes", 100))
        block_size = (min_size + max_size) // 5

        # processing
        proc, scale = resize_for_processing(image)
        pix = pixelate_image(proc, block_size)
        if shape_type == "Circles":
            art = draw_random_circles(pix, min_size, max_size, num_shapes)
        elif shape_type == "Rectangles":
            art = draw_random_rectangles(pix, min_size, max_size, num_shapes)
        else:
            art = draw_random_triangles(pix, min_size, max_size, num_shapes)

        if scale < 1.0:
            art = cv2.resize(art, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        # encode original
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        buf = BytesIO()
        Image.fromarray(orig_rgb).save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # encode art
        art_rgb = cv2.cvtColor(art, cv2.COLOR_BGR2RGB)
        buf = BytesIO()
        Image.fromarray(art_rgb).save(buf, format="PNG")
        generated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        download_url = f"data:image/png;base64,{generated_b64}"

    return render_template(
        "foogle_man_repo.html",
        original_image  = original_b64,
        generated_image = generated_b64,
        num_shapes      = num_shapes,
        download_url    = download_url
    )
@app.route("/paint_geometrize")
def paint_geometrize_page():
    # if geometrize_app:
    #     return geometrize_app()
    return render_template("paint_geometrize.html", active_page="paint_geometrize")


# ═══════════ NEW AJAX ENDPOINT ═══════════
@app.route("/generate_recipe", methods=["POST"])
def ajax_generate_recipe():
    sel = request.form.get("selected_color", "")
    if not sel:
        return jsonify(ok=False, msg="No color selected"), 400
    try:
        target = tuple(int(x) for x in sel.split(","))
    except ValueError:
        return jsonify(ok=False, msg="Bad color string"), 400

    step      = float(request.form.get("step", 10.0))
    db_choice = request.form.get("db_choice")

    full_txt  = read_color_file("color.txt")
    all_dbs   = parse_color_db(full_txt)
    raw_list  = all_dbs.get(db_choice, [])
    base_dict = {name: list(rgb) for name, rgb in raw_list}

    recipes = generate_recipes(target, base_dict, step=step)

    payload = []
    for recipe, mixed, err in recipes:
        payload.append({
            "recipe": [{"name": n, "perc": p} for n, p in recipe if p > 0],
            "mix": list(mixed),
            "error": err
        })
    return jsonify(ok=True, recipes=payload)
# ══════════════════════════════════════════


# ─────────── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
    
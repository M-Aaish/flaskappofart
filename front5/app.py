import os
import sys
import math
import itertools
from pathlib import Path
from io import BytesIO

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, session, flash
)
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from PIL import Image

# If you have `shape_art_generator.py` and `geometrize.py` modules,
# ensure their parent directories are on sys.path.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your existing functions:
from shape_detector import decode, encode      # decode/encode for shape art & detection
from shape_detector import detect_and_decode    # (if used for shape_detector page)
# If you truly have `shape_art_generator.py` with `main_page`, import it here:
try:
    from shape_art_generator import main_page as foogle_man_page
except ImportError:
    foogle_man_page = None

# If you have `geometrize.py` exposing `geometrize_app`, import it here:
try:
    from geometrize import geometrize_app
except ImportError:
    geometrize_app = None

# ───────────── GLOBAL CONFIG ───────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(16)

# Where to store uploaded images temporarily
UPLOAD_FOLDER = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed extensions for image uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Color database file
COLOR_DB_FILE = os.path.join(project_root, "color.txt")

# ───────────── HELPER FUNCTIONS (Copied from Streamlit) ─────────────────

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# 1) Euclidean distance between two RGB colors
def color_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

# 2) Group similar colors (threshold default 10)
def group_similar_colors(rgb_vals, threshold=10):
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

# 3) Read & parse color.txt into a dict of databases
def read_color_file(filename=COLOR_DB_FILE):
    try:
        with open(filename, "r", encoding="utf8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def parse_color_db(txt):
    dbs = {}
    cur = None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            cur = line
            dbs[cur] = []
        else:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            rgb_str = tokens[-2]
            name = " ".join(tokens[1:-2])
            try:
                r, g, b = [int(x) for x in rgb_str.split(",")]
            except ValueError:
                continue
            dbs[cur].append((name, (r, g, b)))
    return dbs

# Convert a list like [(name, (r,g,b)), …] to dict {"name": {"rgb": [r,g,b]}}
def convert_db_list_to_dict(lst):
    d = {}
    for name, rgb in lst:
        d[name] = {"rgb": list(rgb)}
    return d

# 4) Mix colors for recipe generator (both versions of generate_recipes)
def mix_colors(recipe):
    total = sum(p for _, p in recipe)
    r = sum(rgb[0] * p for rgb, p in recipe) / total
    g = sum(rgb[1] * p for rgb, p in recipe) / total
    b = sum(rgb[2] * p for rgb, p in recipe) / total
    return (round(r), round(g), round(b))

def color_error(c1, c2):
    return math.dist(c1, c2)

def generate_recipes(target, base_colors, step=10.0):
    """
    `base_colors` is a dict: {name: [r, g, b], …}. 
    Returns top 3 recipes in format [ ([(name1, perc1), (name2, perc2), …], (r,g,b), err), … ]
    """
    base_list = [(n, tuple(rgb)) for n, rgb in base_colors.items()]
    candidates = []
    # Single‐color quick matches
    for name, rgb in base_list:
        err = color_error(rgb, target)
        if err < 5:
            candidates.append(([(name, 100.0)], rgb, err))

    # Triple‐mix brute force
    for (n1, rgb1), (n2, rgb2), (n3, rgb3) in itertools.combinations(base_list, 3):
        for p1 in np.arange(0, 100 + step, step):
            for p2 in np.arange(0, 100 - p1 + step, step):
                p3 = 100 - p1 - p2
                if p3 < 0:
                    continue
                recipe = [(n1, p1), (n2, p2), (n3, p3)]
                mix_input = [(rgb1, p1), (rgb2, p2), (rgb3, p3)]
                mixed = mix_colors(mix_input)
                err = color_error(mixed, target)
                candidates.append((recipe, mixed, err))

    candidates.sort(key=lambda x: x[2])
    top = []
    seen = set()
    for rec, mixed, err in candidates:
        key = tuple(sorted((name, perc) for name, perc in rec if perc > 0))
        if key not in seen:
            seen.add(key)
            top.append((rec, mixed, err))
        if len(top) >= 3:
            break
    return top

# 5) Utility for color blocks (used in Jinja templates)
def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

# 6) Add color to database file
def add_color_to_db(selected_db, color_name, r, g, b):
    try:
        lines = open(COLOR_DB_FILE, "r", encoding="utf8").readlines()
    except FileNotFoundError:
        lines = []
    new_lines = []
    in_section = False
    inserted = False
    last_index = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if in_section and not inserted:
                new_lines.append(f"{last_index+1} {color_name} {r},{g},{b} 0\n")
                inserted = True
            new_lines.append(line)
            if stripped == selected_db:
                in_section = True
            else:
                in_section = False
            continue
        if in_section:
            tokens = stripped.split()
            if tokens[0].isdigit():
                try:
                    idx = int(tokens[0])
                    last_index = max(last_index, idx)
                except ValueError:
                    pass
        new_lines.append(line)
    if in_section and not inserted:
        new_lines.append(f"{last_index+1} {color_name} {r},{g},{b} 0\n")
    try:
        with open(COLOR_DB_FILE, "w", encoding="utf8") as f:
            f.writelines(new_lines)
        return True
    except Exception:
        return False

# 7) Remove color from database
def remove_color_from_db(selected_db, color_name):
    try:
        lines = open(COLOR_DB_FILE, "r", encoding="utf8").readlines()
    except FileNotFoundError:
        return False
    new_lines = []
    in_section = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if stripped == selected_db:
                in_section = True
            else:
                in_section = False
            new_lines.append(line)
            continue
        if in_section and not removed:
            tokens = stripped.split()
            curr_name = " ".join(tokens[1:-2]).strip()
            if curr_name.lower() == color_name.lower():
                removed = True
                continue
        new_lines.append(line)
    if not removed:
        return False
    try:
        with open(COLOR_DB_FILE, "w", encoding="utf8") as f:
            f.writelines(new_lines)
        return True
    except Exception:
        return False

# 8) Create a new database
def create_custom_database(new_db_name):
    try:
        with open(COLOR_DB_FILE, "a", encoding="utf8") as f:
            f.write(f"\n{new_db_name}\n")
        return True
    except Exception:
        return False

# 9) Remove entire database
def remove_database(db_name):
    try:
        lines = open(COLOR_DB_FILE, "r", encoding="utf8").readlines()
    except FileNotFoundError:
        return False
    new_lines = []
    in_target = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if stripped == db_name:
                in_target = True
                removed = True
                continue
            else:
                in_target = False
                new_lines.append(line)
        else:
            if in_target:
                continue
            else:
                new_lines.append(line)
    if not removed:
        return False
    try:
        with open(COLOR_DB_FILE, "w", encoding="utf8") as f:
            f.writelines(new_lines)
        return True
    except Exception:
        return False


# ───────────── ROUTES ───────────────────────────────────────────────

@app.route("/")
def home_redirect():
    # Redirect default to Shape Detector
    return redirect(url_for("shape_detector_page"))


# 1) Image Generator Page (formerly `image_generator_app`)
@app.route("/image_generator", methods=["GET", "POST"])
def image_generator_page():
    error = None
    result_img_data = None  # will hold bytes of PNG to display
    result_img_cv = None    # CV2 array to allow download
    shape_option = None
    num_shapes = None
    max_size = None
    min_size = None
    if request.method == "POST":
        # Check uploaded file
        if "image" not in request.files:
            error = "No file part."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "No selected file."
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)

                # Load image via CV2
                file_bytes = np.asarray(bytearray(open(path, "rb").read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    error = "Could not read the image."
                else:
                    # Read form inputs
                    shape_option = request.form.get("shape_type")
                    num_shapes = int(request.form.get("num_shapes", 100))

                    if shape_option == "Triangle":
                        max_size = int(request.form.get("max_triangle_size", 50))
                        min_size = int(request.form.get("min_triangle_size", 15))
                        encoded, _ = encode(
                            img_bgr,
                            shape_option,
                            output_path="",
                            num_shapes=num_shapes,
                            max_size=max_size,
                            min_size=min_size
                        )
                    else:  # Rectangle or Circle
                        min_size = int(request.form.get("min_size", 10))
                        max_size = int(request.form.get("max_size", 15))
                        encoded, _ = encode(
                            img_bgr,
                            shape_option,
                            output_path="",
                            num_shapes=num_shapes,
                            min_size=min_size,
                            max_size=max_size,
                            min_radius=min_size,
                            max_radius=max_size
                        )
                    if encoded is None:
                        error = "Failed to generate shape art."
                    else:
                        # Convert BGR→RGB to show and also re-encode PNG for download
                        encoded_rgb = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(encoded_rgb)
                        buf = BytesIO()
                        pil_img.save(buf, format="PNG")
                        buf.seek(0)
                        result_img_data = buf.read()
                        result_img_cv = encoded  # store raw CV2 array for download
                        session["shape_art_cv"] = encoded  # keep in session for download

    return render_template(
        "image_generator.html",
        error=error,
        result_image_data=result_img_data,
        shape_option=shape_option,
        num_shapes=num_shapes,
        max_size=max_size,
        min_size=min_size,
        active_page="image_generator"
    )

@app.route("/download_shape_art")
def download_shape_art():
    # Serve the last generated shape art from session
    if "shape_art_cv" not in session:
        flash("No image to download.", "warning")
        return redirect(url_for("image_generator_page"))
    encoded = session["shape_art_cv"]
    is_success, buffer = cv2.imencode(".png", encoded)
    if not is_success:
        flash("Download failed.", "danger")
        return redirect(url_for("image_generator_page"))
    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name="shape_art.png"
    )


# 2) Shape Detector Page (formerly `shape_detector_app`)
@app.route("/shape_detector", methods=["GET", "POST"])
def shape_detector_page():
    error = None
    annotated_data = None   # will hold decoded results
    grouped_colors = []
    decoded_image_data = None
    selected_recipe_color = None
    recipe_results = None

    # Load existing grouped colors from session if any
    if "grouped_colors" in session:
        grouped_colors = session["grouped_colors"]

    if request.method == "POST":
        # 1) Handle uploaded file
        if "encoded_image" in request.files:
            file = request.files["encoded_image"]
            if file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                file_bytes = np.asarray(bytearray(open(path, "rb").read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    error = "Could not read uploaded image."
                else:
                    # Get shape option, min/max size
                    shape_opt = request.form.get("shape_detect")
                    min_size = int(request.form.get("min_size", 3))
                    max_size = int(request.form.get("max_size", 10))

                    # Threshold + contours
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    detected_boundaries = []
                    if shape_opt == "Triangle":
                        for cnt in contours:
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                            if len(approx) == 3:
                                tri = approx.reshape(-1, 2)
                                xs = tri[:, 0]; ys = tri[:, 1]
                                w = xs.max() - xs.min()
                                h = ys.max() - ys.min()
                                if min_size <= w <= max_size and min_size <= h <= max_size:
                                    detected_boundaries.append(tri)
                    elif shape_opt == "Rectangle":
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if min_size <= w <= max_size and min_size <= h <= max_size:
                                detected_boundaries.append((x, y, w, h))
                    else:  # Circle
                        for cnt in contours:
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            radius = int(radius)
                            if min_size <= radius <= max_size:
                                detected_boundaries.append((int(x), int(y), radius))

                    # Run decode() to get binary_img, annotated_img, rgb_vals
                    binary_img, annotated_img, rgb_vals = decode(
                        img_bgr, shape_opt, boundaries=detected_boundaries,
                        max_size=max_size, min_size=min_size
                    )

                    # Convert BGR→RGB for display
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(annotated_rgb)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    decoded_image_data = buf.read()

                    # Group similar colors
                    grouped = group_similar_colors(rgb_vals, threshold=10)
                    grouped = sorted(grouped, key=lambda x: x[1], reverse=True)
                    session["grouped_colors"] = grouped[:12]  # keep top 12 for UI
                    grouped_colors = grouped[:12]
                    session["annotated_cv"] = annotated_img  # store for download

        # 2) Handle paint recipe generation if a color was clicked
        if request.form.get("action") == "generate_recipe":
            sel = request.form.get("selected_color")  # in format "r,g,b"
            if sel:
                r, g, b = [int(x) for x in sel.split(",")]
                selected_recipe_color = (r, g, b)
                step = float(request.form.get("step", 10.0))
                db_choice = request.form.get("db_choice")

                # Build base_colors dict for chosen DB
                full_txt = read_color_file()
                all_dbs = parse_color_db(full_txt)
                if db_choice in all_dbs:
                    base_dict = convert_db_list_to_dict(all_dbs[db_choice])
                    recipes = generate_recipes(selected_recipe_color, base_dict, step=step)
                    recipe_results = recipes  # list of 3 recipes
                else:
                    recipe_results = []

    return render_template(
        "shape_detector.html",
        error=error,
        decoded_image_data=decoded_image_data,
        grouped_colors=grouped_colors,
        selected_recipe_color=selected_recipe_color,
        recipe_results=recipe_results,
        active_page="shape_detector"
    )

@app.route("/download_analysis")
def download_analysis():
    if "annotated_cv" not in session:
        flash("No analysis image to download.", "warning")
        return redirect(url_for("shape_detector_page"))
    annotated = session["annotated_cv"]
    is_success, buffer = cv2.imencode(".png", annotated)
    if not is_success:
        flash("Download failed.", "danger")
        return redirect(url_for("shape_detector_page"))
    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name="shape_analysis.png"
    )


# 3) Oil Painting Generator Page (formerly `oil_painting_page`)
@app.route("/oil_painting", methods=["GET", "POST"])
def oil_painting_page():
    error = None
    result_img_data = None
    if request.method == "POST":
        # Uploaded file
        if "oil_image" not in request.files:
            error = "No file uploaded."
        else:
            file = request.files["oil_image"]
            if file.filename == "":
                error = "No file selected."
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                intensity = int(request.form.get("intensity", 10))

                # Load with PIL then convert to CV2
                pil = Image.open(path)
                img_np = np.array(pil)
                if img_np.ndim == 2:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Call your oil_main()
                try:
                    from painterfun import oil_main
                except ImportError:
                    error = "oil_main() not found."
                    img_bgr = None

                if img_bgr is not None:
                    output_cv = oil_main(img_bgr, intensity)
                    output_cv = (output_cv * 255).astype(np.uint8)
                    output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)
                    pil_out = Image.fromarray(output_rgb)
                    buf = BytesIO()
                    pil_out.save(buf, format="PNG")
                    buf.seek(0)
                    result_img_data = buf.read()
                    session["oil_painting_cv"] = output_cv

    return render_template(
        "oil_painting.html",
        error=error,
        result_image_data=result_img_data,
        active_page="oil_painting"
    )

@app.route("/download_oil_painting")
def download_oil_painting():
    if "oil_painting_cv" not in session:
        flash("No painting to download.", "warning")
        return redirect(url_for("oil_painting_page"))
    cv_img = session["oil_painting_cv"]
    is_success, buffer = cv2.imencode(".png", cv_img)
    if not is_success:
        flash("Download failed.", "danger")
        return redirect(url_for("oil_painting_page"))
    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name="oil_painting.png"
    )


# 4) Colour Mixer Page (formerly `color_mixing_app`)
@app.route("/colour_merger", methods=["GET", "POST"])
def colour_merger_page():
    error = None
    # Session-wise list of colors (each is {"rgb":[r,g,b], "weight":w})
    if "colors" not in session:
        session["colors"] = [
            {"rgb": [255, 0, 0], "weight": 0.3},
            {"rgb": [0, 255, 0], "weight": 0.6}
        ]

    # Handle add/remove actions via hidden form inputs
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add_color":
            session["colors"].append({"rgb": [255, 255, 255], "weight": 0.1})
        elif action.startswith("remove_"):
            idx = int(action.split("_")[1])
            session["colors"].pop(idx)
        else:
            # Update existing colors from form inputs
            new_colors = []
            total_fields = int(request.form.get("total_colors", 0))
            for i in range(total_fields):
                r = int(request.form.get(f"r_{i}", 255))
                g = int(request.form.get(f"g_{i}", 255))
                b = int(request.form.get(f"b_{i}", 255))
                w = float(request.form.get(f"w_{i}", 0.1))
                new_colors.append({"rgb": [r, g, b], "weight": w})
            session["colors"] = new_colors

    # After updating session["colors"], compute mixed color
    colors = session["colors"]
    # Convert each rgb to latent, then mix, then convert back
    try:
        import mixbox
        def rgb_to_latent(rgb): return mixbox.rgb_to_latent(rgb)
        def latent_to_rgb(lat): return mixbox.latent_to_rgb(lat)
        z_mix = [0] * mixbox.LATENT_SIZE
        tot_w = sum(c["weight"] for c in colors) or 1
        for i in range(len(z_mix)):
            z_mix[i] = sum(c["weight"] * rgb_to_latent(c["rgb"])[i] for c in colors) / tot_w
        mixed_rgb = latent_to_rgb(z_mix)
    except ImportError:
        mixed_rgb = [0, 0, 0]
        error = "mixbox not available; color mixing failed."

    return render_template(
        "colour_merger.html",
        error=error,
        colors=colors,
        mixed_rgb=mixed_rgb,
        active_page="colour_merger"
    )


# 5) Painter Recipe Generator Page (formerly `painter_recipe_generator`)
@app.route("/recipe_generator", methods=["GET", "POST"])
def recipe_generator_page():
    error = None
    recipes = None
    selected_rgb = (255, 0, 0)
    method = "picker"
    if request.method == "POST":
        method = request.form.get("method", "picker")
        if method == "picker":
            hex_val = request.form.get("hex_color", "#ff0000")
            r = int(hex_val[1:3], 16)
            g = int(hex_val[3:5], 16)
            b = int(hex_val[5:7], 16)
            selected_rgb = (r, g, b)
        else:
            r = int(request.form.get("r_slider", 255))
            g = int(request.form.get("g_slider", 0))
            b = int(request.form.get("b_slider", 0))
            selected_rgb = (r, g, b)

        db_choice = request.form.get("db_choice")
        step = float(request.form.get("step", 10.0))

        full_txt = read_color_file()
        all_dbs = parse_color_db(full_txt)
        if db_choice in all_dbs:
            base_dict = convert_db_list_to_dict(all_dbs[db_choice])
            recipes = generate_recipes(selected_rgb, base_dict, step=step)
        else:
            recipes = []

    full_txt = read_color_file()
    all_dbs = parse_color_db(full_txt)
    db_list = list(all_dbs.keys())

    return render_template(
        "recipe_generator.html",
        error=error,
        recipes=recipes,
        selected_rgb=selected_rgb,
        method=method,
        db_list=db_list,
        active_page="recipe_generator"
    )


# 6) Colors Database Manager Page (formerly `painter_colors_database`)
@app.route("/colors_db", methods=["GET", "POST"])
def colors_db_page():
    full_txt = read_color_file()
    databases = parse_color_db(full_txt)
    subpage = request.args.get("subpage", "databases")
    message = None

    # Handle subpage actions via query params or form submissions
    if request.method == "POST":
        action = request.form.get("action")
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
        elif action == "confirm_add_color":
            db = request.form.get("target_db")
            name = request.form.get("color_name", "").strip()
            r = int(request.form.get("r", 255))
            g = int(request.form.get("g", 255))
            b = int(request.form.get("b", 255))
            if name:
                success = add_color_to_db(db, name, r, g, b)
                message = ("success", f"Added color '{name}' to {db}") if success else ("danger", "Failed to add color.")
            else:
                message = ("warning", "You must provide a name.")

        elif action == "confirm_remove_color":
            db = request.form.get("target_db")
            name = request.form.get("color_to_remove")
            if name:
                success = remove_color_from_db(db, name)
                message = ("success", f"Removed color '{name}' from {db}") if success else ("danger", "Failed to remove color.")
            else:
                message = ("warning", "No color selected to remove.")

        elif action == "confirm_create_db":
            new_db = request.form.get("new_db_name", "").strip()
            if new_db:
                success = create_custom_database(new_db)
                message = ("success", f"Created database '{new_db}'") if success else ("danger", "Failed to create database.")
            else:
                message = ("warning", "Please provide a database name.")

        elif action == "confirm_remove_db":
            db = request.form.get("db_to_remove")
            if db:
                success = remove_database(db)
                message = ("success", f"Removed database '{db}'") if success else ("danger", "Failed to remove database.")
            else:
                message = ("warning", "Please select a database to delete.")

        # Refresh after any POST so that parse_color_db sees the updated file
        full_txt = read_color_file()
        databases = parse_color_db(full_txt)

    return render_template(
        "colors_db.html",
        databases=databases,
        subpage=subpage,
        message=message,
        active_page="colors_db"
    )


# 7) Foogle Man Repo Page (Stub for `shape_art_generator_page`)
@app.route("/foogle_man_repo")
def foogle_man_repo_page():
    if foogle_man_page is None:
        # If no shape_art_generator.py or its main_page, show stub
        return render_template("foogle_man_repo.html", active_page="foogle_man_repo")
    else:
        # If you have a Flask‐compatible version of that function, call it here.
        # For now we just render a stub.
        return render_template("foogle_man_repo.html", active_page="foogle_man_repo")


# 8) Paint & Geometrize Page (Stub for `geometrize_app`)
@app.route("/paint_geometrize")
def paint_geometrize_page():
    if geometrize_app is None:
        return render_template("paint_geometrize.html", active_page="paint_geometrize")
    else:
        return render_template("paint_geometrize.html", active_page="paint_geometrize")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

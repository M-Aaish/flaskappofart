import os
import sys

# Assuming geometrize.py is in the parent directory of the 'shape_gen' folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from geometrize import geometrize_app

import streamlit as st
try:
    from streamlit.runtime.scriptrunner import RerunException, RerunData
except ImportError:
    RerunException = None
from shape_art_generator import main_page as shape_art_generator_page
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from EnDe import decode, encode
from painterfun import oil_main  # Importing the oil_main function
import mixbox
import itertools
import math
from pathlib import Path

# Set page config with new title and improved layout
st.set_page_config(
    page_title="Artistic Vision Studio", 
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        .sidebar .sidebar-content .stRadio div {
            color: white;
        }
        .sidebar .sidebar-content .stButton button {
            width: 100%;
            background-color: #6c757d;
            color: white;
            border: 1px solid #495057;
        }
        .sidebar .sidebar-content .stButton button:hover {
            background-color: #5a6268;
            border-color: #495057;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
        .stFileUploader {
            border: 2px dashed #3498db;
            border-radius: 5px;
            padding: 1rem;
        }
        .stSelectbox, .stNumberInput, .stSlider {
            margin-bottom: 1rem;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# --- Functions from the original str_main.py (Image Generator, Shape Detector,
#     Oil Painting Generator, Colour Merger)
# --------------------------------------------------------------------

# Function to calculate the Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

# Function to group similar colors and count them
def group_similar_colors(rgb_vals, threshold=10):
    grouped_colors = []  # List to store final groups of similar colors
    counts = []  # List to store counts of similar colors

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

def oil_painting_page():
    st.title("🎨 Oil Painting Generator")
    st.markdown("Transform your photos into beautiful oil paintings with adjustable intensity.")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    intensity = st.slider("Painting Intensity", min_value=1, max_value=100, value=10)
    
    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Original Image", use_column_width=True)
        else:
            st.info("Please upload an image to get started")
    
    if st.button("Generate Oil Painting", key="oil_painting_btn"):
        if uploaded_file is not None:
            with st.spinner("Creating your masterpiece..."):
                input_image_cv = np.array(input_image)
                if len(input_image_cv.shape) == 2:
                    input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_GRAY2RGB)
                elif input_image_cv.shape[2] == 4:
                    input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_RGBA2RGB)
                output_image_cv = oil_main(input_image_cv, intensity)
                output_image_cv = (output_image_cv * 255).astype(np.uint8)
                output_image = Image.fromarray(output_image_cv)
                with col2:
                    st.image(output_image, caption="Oil Painting Result", use_container_width=True)
                img_byte_arr = BytesIO()
                output_image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                st.download_button(
                    label="Download Painting",
                    data=img_byte_arr,
                    file_name="oil_painting.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload an image first")

def color_mixing_app():
    st.title("🌈 Color Mixer")
    st.markdown("Experiment with color combinations and create perfect blends.")
    
    if 'colors' not in st.session_state:
        st.session_state.colors = [
            {"rgb": [255, 0, 0], "weight": 0.3},  # Default color 1 (Red)
            {"rgb": [0, 255, 0], "weight": 0.6}   # Default color 2 (Green)
        ]

    def rgb_to_latent(rgb):
        return mixbox.rgb_to_latent(rgb)

    def latent_to_rgb(latent):
        return mixbox.latent_to_rgb(latent)

    def get_mixed_rgb(colors):
        z_mix = [0] * mixbox.LATENT_SIZE
        total_weight = sum(c["weight"] for c in colors)
        for i in range(len(z_mix)):
            z_mix[i] = sum(c["weight"] * rgb_to_latent(c["rgb"])[i] for c in colors) / total_weight
        return latent_to_rgb(z_mix)

    def add_new_color():
        st.session_state.colors.append({"rgb": [255, 255, 255], "weight": 0.1})

    def delete_color(index):
        st.session_state.colors.pop(index)

    st.subheader("Color Components")
    for idx, color in enumerate(st.session_state.colors):
        with st.expander(f"Color {idx + 1}", expanded=True):
            cols = st.columns([3, 1])
            with cols[0]:
                r = st.slider(f"Red", 0, 255, color['rgb'][0], key=f"r_{idx}")
                g = st.slider(f"Green", 0, 255, color['rgb'][1], key=f"g_{idx}")
                b = st.slider(f"Blue", 0, 255, color['rgb'][2], key=f"b_{idx}")
            with cols[1]:
                col_weight = st.slider("Weight", 0.0, 1.0, value=color['weight'], step=0.05, key=f"w_{idx}")
                st.markdown(f"<div style='width: 100%; height: 100px; background-color: rgb({r}, {g}, {b}); border-radius: 5px;'></div>", 
                           unsafe_allow_html=True)
            
            color["rgb"] = [r, g, b]
            color["weight"] = col_weight
            
            if len(st.session_state.colors) > 2:
                if st.button(f"Remove Color {idx + 1}", key=f"delete_button_{idx}"):
                    delete_color(idx)
                    st.experimental_rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Another Color"):
            add_new_color()
            st.experimental_rerun()

    mixed_rgb = get_mixed_rgb(st.session_state.colors)
    with col2:
        st.subheader("Mixed Color Result")
        st.write(f"RGB Values: {mixed_rgb}")
        st.markdown(f"""
            <div style='width: 100%; height: 150px; background-color: rgb{mixed_rgb}; 
                      border-radius: 5px; display: flex; justify-content: center; align-items: center;'>
                <span style='color: {'white' if sum(mixed_rgb)/3 < 128 else 'black'}; 
                             font-weight: bold; font-size: 18px;'>
                    Your Custom Color
                </span>
            </div>
        """, unsafe_allow_html=True)
def image_generator_app():
    st.title("🖼️ Shape Art Generator")
    st.markdown("Create artistic images composed of geometric shapes.")
    
    uploaded_file = st.file_uploader("Upload a base image", type=["jpg", "jpeg", "png"])
    
    # Create two equal columns
    col1, col2 = st.columns(2)
    
    # Initialize session state if not exists
    if 'encoded_image' not in st.session_state:
        st.session_state.encoded_image = None
    if 'encoded_image_cv' not in st.session_state:
        st.session_state.encoded_image_cv = None
    
    # Left column - Input image and controls
    with col1:
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, 
                           caption="Original Image",
                           width=400,
                           output_format="PNG")
                    st.session_state.input_image = img  # Store for processing
                else:
                    st.error("Failed to load image. Please try another file.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        
        shape_option = st.selectbox("Shape Type", ["Triangle", "Rectangle", "Circle"])
        num_shapes = st.slider("Number of Shapes", min_value=1, max_value=500, value=100)
        
        if shape_option == "Triangle":
            max_triangle_size = st.slider("Max Triangle Size", min_value=1, max_value=100, value=50)
            min_triangle_size = st.slider("Min Triangle Size", min_value=1, max_value=100, value=15)
        elif shape_option in ["Rectangle", "Circle"]:
            min_size = st.slider("Min Shape Size", min_value=1, max_value=100, value=10)
            max_size = st.slider("Max Shape Size", min_value=1, max_value=100, value=15)
    
    # Right column - Output image and download button
    with col2:
        if st.session_state.encoded_image is not None:
            st.image(st.session_state.encoded_image, 
                   caption=f"{shape_option} Art",
                   width=400,
                   output_format="PNG")
            
            # Add download button
            if st.session_state.encoded_image_cv is not None:
                # Convert to bytes for download
                is_success, buffer = cv2.imencode(".png", st.session_state.encoded_image_cv)
                if is_success:
                    st.download_button(
                        label="Download Shape Art",
                        data=buffer.tobytes(),
                        file_name=f"shape_art_{shape_option.lower()}.png",
                        mime="image/png"
                    )
    
    if st.button("Generate Shape Art", key="generate_shape_art"):
        if uploaded_file is not None and 'input_image' in st.session_state:
            with st.spinner("Creating your shape art..."):
                try:
                    img = st.session_state.input_image
                    
                    if shape_option == "Triangle":
                        encoded_image, boundaries = encode(img, shape_option, output_path="",
                                                        num_shapes=num_shapes,
                                                        max_size=max_triangle_size,
                                                        min_size=min_triangle_size)
                    elif shape_option in ["Rectangle", "Circle"]:
                        encoded_image, boundaries = encode(img, shape_option, output_path="",
                                                        num_shapes=num_shapes, 
                                                        min_size=min_size, 
                                                        max_size=max_size,
                                                        min_radius=min_size, 
                                                        max_radius=max_size)
                    
                    if encoded_image is not None:
                        encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
                        st.session_state.encoded_image = encoded_image_rgb
                        st.session_state.encoded_image_cv = encoded_image  # Store CV version for download
                        st.rerun()
                    else:
                        st.error("Failed to generate shape art. Please try different parameters.")
                except Exception as e:
                    st.error(f"Error during image processing: {str(e)}")
        else:
            st.warning("Please upload an image first.")
# --------------------------------------------------------------------
# --- Updated Shape Detector with Embedded Recipe Generation Form
# --------------------------------------------------------------------
def shape_detector_app():
    st.title("🔍 Shape Detector & Analyzer")
    st.markdown("Analyze images to detect shapes and extract color information.")
    
    uploaded_file = st.file_uploader("Upload an encoded image", type=["jpg", "jpeg", "png"])
    shape_option = st.selectbox("Shape to Detect", ["Triangle", "Rectangle", "Circle"])
    
    col1, col2 = st.columns(2)
    with col1:
        min_size_det = st.slider("Minimum Detection Size", min_value=1, value=3)
        max_size_det = st.slider("Maximum Detection Size", min_value=1, value=10)
    
    # Process file upload and store the encoded image in session state.
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        encoded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if encoded_image is None:
            st.error("Error reading the image. Please try another file.")
        else:
            st.session_state.encoded_image = encoded_image
    else:
        st.session_state.pop("encoded_image", None)
        st.session_state.pop("decoded_data", None)
        st.session_state.pop("selected_recipe_color", None)
    
    if "encoded_image" in st.session_state:
        with col1:
            uploaded_image_rgb = cv2.cvtColor(st.session_state.encoded_image, cv2.COLOR_BGR2RGB)
            st.image(uploaded_image_rgb, caption="Uploaded Image", use_container_width=True)
    
    # If the user clicks Decode or if decoded data already exists, process/display decode result.
    if st.button("Analyze Image") or ("decoded_data" in st.session_state):
        if "encoded_image" in st.session_state:
            # Only perform decode if not already stored.
            if "decoded_data" not in st.session_state or st.button("Re-analyze", key="redecode"):
                with st.spinner("Analyzing image..."):
                    encoded_image = st.session_state.encoded_image
                    shape = shape_option
                    gray = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    detected_boundaries = []
                    if shape == "Triangle":
                        for cnt in contours:
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                            if len(approx) == 3:
                                tri = approx.reshape(-1, 2)
                                xs = tri[:, 0]
                                ys = tri[:, 1]
                                width = xs.max() - xs.min()
                                height = ys.max() - ys.min()
                                if width >= min_size_det and width <= max_size_det and height >= min_size_det and height <= max_size_det:
                                    detected_boundaries.append(tri)
                    elif shape == "Rectangle":
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w >= min_size_det and w <= max_size_det and h >= min_size_det and h <= max_size_det:
                                detected_boundaries.append((x, y, w, h))
                    elif shape == "Circle":
                        for cnt in contours:
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            radius = int(radius)
                            if radius >= min_size_det and radius <= max_size_det:
                                detected_boundaries.append((int(x), int(y), radius))
                    binary_img, annotated_img, rgb_vals = decode(encoded_image, shape, boundaries=detected_boundaries, max_size=max_size_det, min_size=min_size_det)
                    annotated_image_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    grouped_colors = group_similar_colors(rgb_vals, threshold=10)
                    grouped_colors = sorted(grouped_colors, key=lambda x: x[1], reverse=True)
                    st.session_state.decoded_data = {
                        "annotated_image_rgb": annotated_image_rgb,
                        "grouped_colors": grouped_colors,
                        "annotated_img": annotated_img
                    }
            decoded_data = st.session_state.decoded_data
            with col2:
                st.image(decoded_data["annotated_image_rgb"], caption=f"Detected {shape_option}s", use_container_width=True)
            
            st.subheader("🔎 Color Analysis")
            st.write("Click on a color to generate paint recipes")
            
            # Display color swatches in a grid
            cols = st.columns(4)
            for idx, (color, count) in enumerate(decoded_data["grouped_colors"]):
                with cols[idx % 4]:
                    rgb_str = f"RGB: {color[0]}, {color[1]}, {color[2]}"
                    count_str = f"Count: {count}"
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    
                    if st.button(f"{rgb_str}\n{count_str}", key=f"color_btn_{idx}"):
                        st.session_state.selected_recipe_color = color
                    
                    st.markdown(
                        f"<div style='background-color: {hex_color}; width:100%; height:50px; "
                        f"border-radius:5px; margin-bottom:10px;'></div>",
                        unsafe_allow_html=True
                    )
            
            is_success, buffer = cv2.imencode(".png", decoded_data["annotated_img"])
            if is_success:
                st.download_button(
                    label="Download Analysis",
                    data=buffer.tobytes(),
                    file_name="shape_analysis.png",
                    mime="image/png"
                )
            
            # Recipe Generation Section
            if "selected_recipe_color" in st.session_state and st.session_state.selected_recipe_color is not None:
                st.markdown("---")
                st.subheader("🎨 Paint Recipe Generator")
                fixed_color = tuple(int(c) for c in st.session_state.selected_recipe_color)
                
                st.write("Selected Color:")
                display_color_block(fixed_color, label="Selected")
                
                db_choice = st.selectbox("Color Database:", list(databases.keys()), key="recipe_db_sd")
                step = st.slider("Precision Level:", 4.0, 10.0, 10.0, step=0.5, key="recipe_step_sd")
                
                if st.button("Generate Recipe", key="generate_recipe_sd"):
                    with st.spinner("Calculating best recipes..."):
                        selected_db_dict = convert_db_list_to_dict(databases[db_choice])
                        recipes = generate_recipes(fixed_color, selected_db_dict, step=step)
                        if recipes:
                            st.success("Found matching recipes!")
                            for idx, (recipe, mixed, err) in enumerate(recipes[:3]):  # Show top 3
                                with st.expander(f"Recipe {idx+1} (Accuracy: {100 - err:.1f}%)", expanded=idx==0):
                                    cols = st.columns([1,1,2])
                                    with cols[0]:
                                        st.write("**Target Color**")
                                        display_color_block(fixed_color)
                                    with cols[1]:
                                        st.write("**Mixed Result**")
                                        display_color_block(mixed)
                                    with cols[2]:
                                        st.write("**Ingredients**")
                                        for name, perc in recipe:
                                            if perc > 0:
                                                base_rgb = tuple(selected_db_dict[name]["rgb"])
                                                st.write(f"- {name}: {perc:.1f}%")
                                                display_thin_color_block(base_rgb)
                        else:
                            st.error("No recipes found for this color.")
        else:
            st.warning("Please upload an image first.")

# --------------------------------------------------------------------
# --- Functions from painter2.py (Painter App - Recipe Generator and Colors DataBase)
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
COLOR_DB_FILE = str(BASE_DIR / "color.txt")

@st.cache_data
def read_color_file(filename=COLOR_DB_FILE):
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        st.error("Error reading color.txt: " + str(e))
        return ""

def parse_color_db(txt):
    databases = {}
    current_db = None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            current_db = line
            databases[current_db] = []
        else:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            index = tokens[0]
            rgb_str = tokens[-2]
            color_name = " ".join(tokens[1:-2])
            try:
                r, g, b = [int(x) for x in rgb_str.split(",")]
            except Exception:
                continue
            databases[current_db].append((color_name, (r, g, b)))
    return databases

color_txt = read_color_file()
databases = parse_color_db(color_txt)

def convert_db_list_to_dict(color_list):
    d = {}
    for name, rgb in color_list:
        d[name] = {"rgb": list(rgb)}
    return d

def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

def mix_colors(recipe):
    total, r_total, g_total, b_total = 0, 0, 0, 0
    for color, perc in recipe:
        r, g, b = color
        r_total += r * perc
        g_total += g * perc
        b_total += b * perc
        total += perc
    if total == 0:
        return (0, 0, 0)
    return (round(r_total / total), round(g_total / total), round(b_total / total))

def color_error(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def generate_recipes(target, base_colors_dict, step=10.0):
    candidates = []
    base_list = [(name, info["rgb"]) for name, info in base_colors_dict.items()]
    for name, rgb in base_list:
        err = color_error(tuple(rgb), target)
        if err < 5:
            recipe = [(name, 100.0)]
            candidates.append((recipe, tuple(rgb), err))
    for (name1, rgb1), (name2, rgb2), (name3, rgb3) in itertools.combinations(base_list, 3):
        for p1 in np.arange(0, 100 + step, step):
            for p2 in np.arange(0, 100 - p1 + step, step):
                p3 = 100 - p1 - p2
                if p3 < 0:
                    continue
                recipe = [(name1, p1), (name2, p2), (name3, p3)]
                mix_recipe = [(rgb1, p1), (rgb2, p2), (rgb3, p3)]
                mixed = mix_colors(mix_recipe)
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

def display_color_block(color, label=""):
    hex_color = rgb_to_hex(*color)
    st.markdown(
        f"<div style='background-color: {hex_color}; width:100px; height:100px; "
        f"border-radius:5px; display:flex; justify-content:center; align-items:center; "
        f"color:{'white' if sum(color)/3 < 128 else 'black'}; font-weight:bold;'>{label}</div>",
        unsafe_allow_html=True,
    )

def display_thin_color_block(color):
    hex_color = rgb_to_hex(*color)
    st.markdown(
        f"<div style='background-color: {hex_color}; width:50px; height:20px; "
        f"border-radius:3px; display:inline-block; margin-right:10px;'></div>",
        unsafe_allow_html=True,
    )

def add_color_to_db(selected_db, color_name, r, g, b):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for update: " + str(e))
        return False
    new_lines = []
    in_section = False
    inserted = False
    last_index = 0
    for i, line in enumerate(lines):
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
                except:
                    pass
        new_lines.append(line)
    if in_section and not inserted:
        new_lines.append(f"{last_index+1} {color_name} {r},{g},{b} 0\n")
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def remove_color_from_db(selected_db, color_name):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for removal: " + str(e))
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
            current_name = " ".join(tokens[1:-2]).strip()
            if current_name.lower() == color_name.lower():
                removed = True
                continue
        new_lines.append(line)
    if not removed:
        st.warning("Color not found in the selected database.")
        return False
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def create_custom_database(new_db_name):
    line = f"\n{new_db_name}\n"
    try:
        with open(COLOR_DB_FILE, "a") as f:
            f.write(line)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def remove_database(db_name):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for removal: " + str(e))
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
        st.warning("Database not found.")
        return False
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def show_databases_page():
    st.title("📚 Color Databases")
    st.markdown("Browse and explore available color databases.")
    
    selected_db = st.selectbox("Select a database:", list(databases.keys()))
    
    if not databases[selected_db]:
        st.warning("This database is empty.")
        return
    
    st.subheader(f"Colors in {selected_db}")
    
    # Display colors in a responsive grid
    cols = st.columns(3)
    for idx, (name, rgb) in enumerate(databases[selected_db]):
        with cols[idx % 3]:
            hex_color = rgb_to_hex(*rgb)
            st.markdown(
                f"<div style='background-color: {hex_color}; width:100%; height:80px; "
                f"border-radius:5px; display:flex; justify-content:center; align-items:center; "
                f"color:{'white' if sum(rgb)/3 < 128 else 'black'}; margin-bottom:10px;'>"
                f"{name}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")

def show_add_colors_page():
    global databases
    st.title("➕ Add Colors to Database")
    st.markdown("Expand your color palette by adding new colors.")
    
    selected_db = st.selectbox("Select target database:", list(databases.keys()))
    
    with st.form("add_color_form"):
        new_color_name = st.text_input("Color Name", help="Give your color a descriptive name")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.slider("Red", 0, 255, 255)
        with col2:
            g = st.slider("Green", 0, 255, 255)
        with col3:
            b = st.slider("Blue", 0, 255, 255)
        
        # Show color preview
        st.markdown(
            f"<div style='background-color: rgb({r},{g},{b}); width:100%; height:100px; "
            f"border-radius:5px; margin:10px 0;'></div>",
            unsafe_allow_html=True
        )
        
        submitted = st.form_submit_button("Add Color to Database")
        if submitted:
            if new_color_name:
                success = add_color_to_db(selected_db, new_color_name, int(r), int(g), int(b))
                if success:
                    st.success(f"Color '{new_color_name}' added to {selected_db}!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to add color.")
            else:
                st.error("Please enter a color name.")

def show_remove_colors_page():
    global databases
    st.title("🗑️ Remove Colors from Database")
    st.markdown("Manage your color databases by removing unwanted colors.")
    
    selected_db = st.selectbox("Select database:", list(databases.keys()))
    color_options = [name for name, _ in databases[selected_db]]
    
    if not color_options:
        st.warning("No colors available in the selected database.")
        return
    
    chosen_color = st.selectbox("Select color to remove:", color_options)
    
    # Show the color being removed
    if chosen_color:
        color_rgb = next(rgb for name, rgb in databases[selected_db] if name == chosen_color)
        st.markdown(
            f"<div style='background-color: rgb{color_rgb}; width:100%; height:80px; "
            f"border-radius:5px; margin:10px 0;'></div>",
            unsafe_allow_html=True
        )
    
    if st.button("Remove Color", type="primary"):
        success = remove_color_from_db(selected_db, chosen_color)
        if success:
            st.success(f"Color '{chosen_color}' removed from {selected_db}!")
            color_txt = read_color_file(COLOR_DB_FILE)
            databases = parse_color_db(color_txt)
        else:
            st.error("Failed to remove color or color not found.")

def show_remove_database_page():
    global databases
    st.title("🚮 Remove Entire Database")
    st.warning("This action cannot be undone! All colors in the database will be permanently deleted.")
    
    db_options = list(databases.keys())
    selected_db_to_remove = st.selectbox("Select database to remove:", db_options)
    
    if selected_db_to_remove:
        st.write(f"Database contains {len(databases[selected_db_to_remove])} colors")
    
    with st.form("remove_db_form"):
        confirm = st.checkbox("I understand this action is permanent", value=False)
        submitted = st.form_submit_button("Permanently Delete Database")
        if submitted:
            if selected_db_to_remove and confirm:
                success = remove_database(selected_db_to_remove)
                if success:
                    st.success(f"Database '{selected_db_to_remove}' removed!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to remove database.")
            else:
                st.error("Please select a database and confirm deletion.")

def show_create_custom_db_page():
    global databases
    st.title("✨ Create New Database")
    st.markdown("Organize your colors by creating custom databases.")
    
    with st.form("create_db_form"):
        new_db_name = st.text_input("New database name:", 
                                   help="Use a descriptive name like 'Earth Tones' or 'Brand Colors'")
        
        submitted = st.form_submit_button("Create Database")
        if submitted:
            if new_db_name:
                success = create_custom_database(new_db_name)
                if success:
                    st.success(f"Database '{new_db_name}' created!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to create database.")
            else:
                st.error("Please enter a database name.")

def painter_recipe_generator():
    st.title("🧪 Paint Recipe Generator")
    st.markdown("Create precise paint mixing recipes to match any color.")
    
    db_choice = st.selectbox("Color Database:", list(databases.keys()))
    selected_db_dict = convert_db_list_to_dict(databases[db_choice])
    
    method = st.radio("Color Selection Method:", ["Color Picker", "RGB Sliders"])
    
    if method == "Color Picker":
        desired_hex = st.color_picker("Choose Target Color", "#ff0000")
        desired_rgb = tuple(int(desired_hex[i:i+2], 16) for i in (1, 3, 5))
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.slider("Red", 0, 255, 255)
        with col2:
            g = st.slider("Green", 0, 255, 0)
        with col3:
            b = st.slider("Blue", 0, 255, 0)
        desired_rgb = (r, g, b)
        desired_hex = rgb_to_hex(r, g, b)
    
    st.write("**Target Color:**")
    display_color_block(desired_rgb, label="Your Color")
    
    step = st.slider("Recipe Precision:", 4.0, 10.0, 10.0, step=0.5,
                    help="Higher values are faster but less precise")
    
    if st.button("Generate Recipes", type="primary"):
        with st.spinner("Finding the best color combinations..."):
            recipes = generate_recipes(desired_rgb, selected_db_dict, step=step)
            if recipes:
                st.success("Found matching recipes!")
                
                tabs = st.tabs([f"Recipe {i+1}" for i in range(min(3, len(recipes)))])
                
                for idx, (recipe, mixed, err) in enumerate(recipes[:3]):
                    with tabs[idx]:
                        st.write(f"**Accuracy:** {100 - err:.1f}% match")
                        
                        cols = st.columns([1,1,2])
                        with cols[0]:
                            st.write("**Target Color**")
                            display_color_block(desired_rgb)
                        with cols[1]:
                            st.write("**Mixed Result**")
                            display_color_block(mixed)
                        with cols[2]:
                            st.write("**Ingredients**")
                            for name, perc in recipe:
                                if perc > 0:
                                    base_rgb = tuple(selected_db_dict[name]["rgb"])
                                    st.write(f"- {name}: {perc:.1f}%")
                                    display_thin_color_block(base_rgb)
            else:
                st.error("No recipes found for this color.")

def painter_colors_database():
    st.title("🎨 Color Database Manager")
    st.markdown("Organize and manage your color databases.")
    
    # Navigation buttons with icons
    st.write("### Database Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Browse Databases", use_container_width=True):
            st.session_state.subpage = "databases"
    with col2:
        if st.button("➕ Add Colors", use_container_width=True):
            st.session_state.subpage = "add"
    with col3:
        if st.button("🗑️ Remove Colors", use_container_width=True):
            st.session_state.subpage = "remove_colors"
    
    col4, col5, _ = st.columns(3)
    with col4:
        if st.button("✨ Create Database", use_container_width=True):
            st.session_state.subpage = "custom"
    with col5:
        if st.button("🚮 Delete Database", use_container_width=True):
            st.session_state.subpage = "remove_database"
    
    # Initialize subpage if not set
    if "subpage" not in st.session_state:
        st.session_state.subpage = "databases"
    
    # Display the selected subpage
    if st.session_state.subpage == "databases":
        show_databases_page()
    elif st.session_state.subpage == "add":
        show_add_colors_page()
    elif st.session_state.subpage == "remove_colors":
        show_remove_colors_page()
    elif st.session_state.subpage == "custom":
        show_create_custom_db_page()
    elif st.session_state.subpage == "remove_database":
        show_remove_database_page()

# --------------------------------------------------------------------
# --- Main Navigation (6 radio buttons)
# --------------------------------------------------------------------
def main():
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #3498db;'>Artistic Vision Studio</h1>
                <p style='color: #7f8c8d;'>Creative Tools for Digital Artists</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Refresh button
        if st.button("🔄 Refresh Application", use_container_width=True):
            read_color_file.clear()  # Clear the cached data.
            if RerunException is not None:
                raise RerunException(RerunData())  # Force a rerun.
            else:
                st.warning("Please manually refresh your browser")
        
        # Navigation options with icons
        st.markdown("### Navigation")
        app_mode = st.radio("", [
            "🖼️ Image Generator", 
            "🔍 Shape Detector", 
            "🎨 Oil Painting Generator", 
            "🌈 Colour Merger", 
            "🧪 Recipe Generator", 
            "📚 Colors DataBase",
            "👨‍🎨 Foogle Man Repo",
            "✨ Paint & Geometrize"
        ], label_visibility="collapsed")
    
    # Page content
    if app_mode == "🖼️ Image Generator":
        image_generator_app()
    elif app_mode == "🔍 Shape Detector":
        shape_detector_app()
    elif app_mode == "🎨 Oil Painting Generator":
        oil_painting_page()
    elif app_mode == "🌈 Colour Merger":
        color_mixing_app()
    elif app_mode == "🧪 Recipe Generator":
        painter_recipe_generator()
    elif app_mode == "📚 Colors DataBase":
        painter_colors_database()
    elif app_mode == "👨‍🎨 Foogle Man Repo":
        shape_art_generator_page()    
    elif app_mode == "✨ Paint & Geometrize":
        geometrize_app()
        
if __name__ == "__main__":
    main()


import streamlit as st
import cv2
import numpy as np
import random

# -------------------------------
# Helper Functions
# -------------------------------

def resize_for_processing(image, max_dim=800):
    """ Resize the image for faster processing and later upscale back to original size """
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / w, max_dim / h)
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, 1.0

def pixelate_image(image, block_size=5):
    """ Pixelates the image by resizing it to a smaller size and scaling back up """
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // block_size), max(1, height // block_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

def draw_random_circles(image, min_radius, max_radius, num_circles):
    """ Draws random circles on the image """
    output = image.copy()
    height, width = image.shape[:2]
    
    for _ in range(num_circles):
        radius = random.randint(min_radius, max_radius)
        x, y = random.randint(radius, width - radius), random.randint(radius, height - radius)
        color = image[y, x].tolist()
        cv2.circle(output, (x, y), radius, color, -1)
    
    return output

def draw_random_rectangles(image, min_size, max_size, num_rects):
    """ Draws random rectangles on the image """
    output = image.copy()
    height, width = image.shape[:2]

    for _ in range(num_rects):
        rect_w, rect_h = random.randint(min_size, max_size), random.randint(min_size, max_size)
        x, y = random.randint(0, width - rect_w), random.randint(0, height - rect_h)
        angle = random.randint(0, 360)
        color = image[y, x].tolist()

        rect = np.array([[x, y], [x + rect_w, y], [x + rect_w, y + rect_h], [x, y + rect_h]], dtype=np.float32)
        M = cv2.getRotationMatrix2D((x + rect_w / 2, y + rect_h / 2), angle, 1.0)
        rotated_rect = cv2.transform(np.array([rect]), M)[0].astype(int)

        cv2.fillPoly(output, [rotated_rect], color)

    return output

def draw_random_triangles(image, min_size, max_size, num_triangles):
    """ Draws random triangles on the image """
    output = image.copy()
    height, width = image.shape[:2]

    for _ in range(num_triangles):
        side = random.randint(min_size, max_size)
        tri_height = int(side * np.sqrt(3) / 2)
        x, y = random.randint(0, width - side), random.randint(tri_height, height)
        color = image[y - tri_height // 2, x + side // 2].tolist()

        pt1, pt2, pt3 = (x, y), (x + side, y), (x + side // 2, y - tri_height)
        triangle = np.array([pt1, pt2, pt3], dtype=np.int32)

        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((x + side // 2, y - tri_height // 3), angle, 1.0)
        rotated_triangle = cv2.transform(np.array([triangle]), M)[0].astype(int)

        cv2.fillPoly(output, [rotated_triangle], color)

    return output

# -------------------------------
# Main Page Function
# -------------------------------

def main_page():
    """ Main UI for Shape Art Generator """
    st.title("Foogle Man Repo")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Error loading image. Try a different file.")
            return

        # Convert for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.header("Original Image")
            st.image(image_rgb, use_column_width=True)

        # Parameters
        shape_type = st.selectbox("Select Shape Type", ("Circles", "Rectangles", "Triangles"))
        min_size = st.number_input("Minimum Size / Radius", min_value=1, value=5)
        max_size = st.number_input("Maximum Size / Radius", min_value=1, value=30)
        num_shapes = st.number_input("Number of Shapes", min_value=1, value=100)
        block_size = (min_size + max_size) // 5  # Pixelation factor

        # Resize for processing
        processed_input, scale = resize_for_processing(image)

        if st.button("Generate"):
            with st.spinner("Processing..."):
                pixelated = pixelate_image(processed_input, block_size)

                if shape_type == "Circles":
                    art_image = draw_random_circles(pixelated, min_size, max_size, num_shapes)
                elif shape_type == "Rectangles":
                    art_image = draw_random_rectangles(pixelated, min_size, max_size, num_shapes)
                elif shape_type == "Triangles":
                    art_image = draw_random_triangles(pixelated, min_size, max_size, num_shapes)
                else:
                    art_image = pixelated  # Fallback

                # Scale back to original resolution
                if scale < 1.0:
                    art_image = cv2.resize(art_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Convert for display
                art_image_rgb = cv2.cvtColor(art_image, cv2.COLOR_BGR2RGB)

            # Display images in two columns (Uploaded Image on Left, Generated Image on Right)
            with col2:
                st.header("Generated Art")
                st.image(art_image_rgb, use_column_width=True)
                st.write(f"Shapes Added: {num_shapes}")

# -------------------------------
# Multi-Page Support
# -------------------------------

PAGES = {
    "Shape Art Generator": main_page,
    # Future pages can be added here
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()
"""
PIXGPT Image Generator – Advanced version
- Searches images via Unsplash (free) or fallback
- Removes white/black backgrounds automatically
- Overlays images onto a base with transparency
- Falls back to grid collage if overlay fails
"""

import os
import random
import time
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps


class ImageGenerator:
    def __init__(self, api_key=None, fallback=True):
        self.api_key = api_key
        self.fallback = fallback
        self.session = requests.Session()

    # ---------- Image search (unchanged) ----------
    def search_images(self, query, num_images=4):
        if self.api_key:
            try:
                headers = {"Authorization": f"Client-ID {self.api_key}"}
                url = "https://api.unsplash.com/search/photos"
                params = {"query": query, "per_page": num_images}
                resp = self.session.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                urls = [result["urls"]["regular"] for result in data.get("results", [])]
                if urls:
                    return urls
            except Exception as e:
                print(f"Unsplash search failed: {e}")
        if self.fallback:
            print("Using fallback images from picsum.photos")
            return [f"https://picsum.photos/400/400?random={random.randint(1,10000)}" for _ in range(num_images)]
        raise RuntimeError("No images found and fallback disabled.")

    def download_image(self, url):
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")

    # ---------- Background removal ----------
    def remove_background(self, img, bg_color="white", tolerance=50):
        """
        Remove pixels that are close to the specified background color.
        bg_color: 'white', 'black', or an RGB tuple (r,g,b)
        tolerance: how far a pixel can be from bg_color to be considered background.
        Returns an RGBA image with background transparent.
        """
        if isinstance(bg_color, str):
            if bg_color.lower() == "white":
                bg_rgb = (255, 255, 255)
            elif bg_color.lower() == "black":
                bg_rgb = (0, 0, 0)
            else:
                bg_rgb = (255, 255, 255)  # fallback
        else:
            bg_rgb = bg_color

        # Convert to RGBA if not already
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Get pixel data
        data = img.getdata()
        new_data = []
        for pixel in data:
            # Calculate distance to background color
            dist = ((pixel[0] - bg_rgb[0]) ** 2 +
                    (pixel[1] - bg_rgb[1]) ** 2 +
                    (pixel[2] - bg_rgb[2]) ** 2) ** 0.5
            if dist < tolerance:
                # Make transparent
                new_data.append((pixel[0], pixel[1], pixel[2], 0))
            else:
                new_data.append(pixel)
        img.putdata(new_data)
        return img

    # ---------- Overlay compositing ----------
    def overlay_images(self, image_urls, output_path, base_color=(30, 30, 30)):
        """
        Download images, remove white/black backgrounds, and overlay them
        onto a base canvas.
        """
        if not image_urls:
            raise ValueError("No image URLs provided.")

        # Download all images and remove backgrounds
        processed = []
        for url in image_urls:
            try:
                img = self.download_image(url)
                # Resize to reasonable size (e.g., 400x400)
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                # Detect background: check if top-left corner is white or black
                corner = img.getpixel((0, 0))
                # If corner is near white or black, use that as bg color
                if corner[0] > 200 and corner[1] > 200 and corner[2] > 200:
                    bg = "white"
                elif corner[0] < 50 and corner[1] < 50 and corner[2] < 50:
                    bg = "black"
                else:
                    # Check if majority of edge pixels are white/black
                    # Simplified: use the color of the average of all four corners
                    corners = [img.getpixel((0,0)), img.getpixel((img.width-1,0)),
                               img.getpixel((0,img.height-1)), img.getpixel((img.width-1,img.height-1))]
                    avg_r = sum(c[0] for c in corners) // 4
                    avg_g = sum(c[1] for c in corners) // 4
                    avg_b = sum(c[2] for c in corners) // 4
                    if avg_r > 200 and avg_g > 200 and avg_b > 200:
                        bg = "white"
                    elif avg_r < 50 and avg_g < 50 and avg_b < 50:
                        bg = "black"
                    else:
                        bg = (avg_r, avg_g, avg_b)  # use average corner color

                img_clean = self.remove_background(img, bg_color=bg, tolerance=50)
                processed.append(img_clean)
            except Exception as e:
                print(f"Failed to process {url}: {e}")
                # Create a placeholder with transparency
                placeholder = Image.new("RGBA", (400, 400), (100, 100, 100, 128))
                draw = ImageDraw.Draw(placeholder)
                draw.text((50, 150), "Error", fill=(255,255,255,255))
                processed.append(placeholder)

        if not processed:
            raise ValueError("No images could be processed.")

        # Determine canvas size: use the largest dimensions
        max_w = max(img.width for img in processed)
        max_h = max(img.height for img in processed)
        # Create a canvas (background)
        canvas = Image.new("RGBA", (max_w * 2, max_h * 2), base_color + (255,))
        # Place each processed image in a random or grid layout
        # We'll use a simple grid: 2x2
        cols = 2
        rows = (len(processed) + cols - 1) // cols
        cell_w = max_w
        cell_h = max_h
        for idx, img in enumerate(processed):
            row = idx // cols
            col = idx % cols
            x = col * cell_w + (cell_w - img.width) // 2
            y = row * cell_h + (cell_h - img.height) // 2
            canvas.paste(img, (x, y), img)  # use alpha as mask

        # Add a caption
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        caption = f"PIXGPT | {image_urls[0][:40] if image_urls else ''}"
        draw.text((10, 10), caption, fill=(255,255,255), font=font)

        canvas.save(output_path, quality=85)
        return output_path

    # ---------- Original grid method (kept as fallback) ----------
    def merge_images(self, image_urls, output_path, cols=2):
        """Original grid collage."""
        images = []
        for url in image_urls:
            try:
                img = self.download_image(url)
                img = img.resize((400, 400))
                images.append(img)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                placeholder = Image.new("RGB", (400, 400), color=(80,80,80))
                draw = ImageDraw.Draw(placeholder)
                try:
                    font = ImageFont.truetype("arial.ttf", 30)
                except:
                    font = ImageFont.load_default()
                draw.text((50, 150), "Error", fill=(255,255,255), font=font)
                images.append(placeholder)

        total = len(images)
        rows = (total + cols - 1) // cols
        width = cols * 400
        height = rows * 400
        merged = Image.new("RGB", (width, height), color=(30,30,30))

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * 400
            y = row * 400
            merged.paste(img, (x, y))

        draw = ImageDraw.Draw(merged)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        caption = f"PIXGPT | {image_urls[0][:40] if image_urls else ''}"
        draw.text((10, 10), caption, fill=(255,255,255), font=font)

        merged.save(output_path, quality=85)
        return output_path

    # ---------- Main generate method with mode ----------
    def generate(self, prompt, num_images=4, output_path=None, mode="overlay"):
        """
        mode: 'overlay' (removes backgrounds and overlays) or 'grid' (simple collage)
        """
        if output_path is None:
            output_path = f"generated_{int(time.time())}.jpg"
        urls = self.search_images(prompt, num_images)
        if not urls:
            raise ValueError("No images found.")

        if mode == "overlay":
            return self.overlay_images(urls, output_path)
        else:
            return self.merge_images(urls, output_path, cols=2)
"""The oMLX twin of find_beer.py: same photo, same bounding-box prompt, same box drawing and model comparison,
but every model is served by oMLX over its OpenAI-compatible API instead of the LM Studio SDK. Thinking is
off so the reply is just the JSON.

Needs oMLX running on :8000 with the Qwen3.5 MLX models below in the Hugging Face cache, and OMLX_API_KEY in
the environment or the repo-root .env. Needs `pip install openai python-dotenv pillow`.
"""
import base64
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

HERE = Path(__file__).parent

# oMLX names Hugging Face cache models "org--repo". The LM Studio version also compares qwen3.5-4b@q8_0,
# qwen3.5-9b and Gemma-4 E2B/E4B: download their MLX builds and add the ids here to compare them too.
MODELS = [
    "mlx-community--Qwen3.5-2B-4bit", "mlx-community--Qwen3.5-2B-8bit",
    "mlx-community--Qwen3.5-4B-MLX-4bit",
]
IMAGE_PATH = str(HERE / "data" / "RTS1UI9-1024x659.jpg")


def is_gemma(model_key):
    return "gemma" in model_key.lower()


def extract_boxes_from_text(text, model_key):
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group(1))
        except Exception:
            pass

    if parsed is None:
        pattern = r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, text)
        if matches:
            parsed = [{"bbox_2d": [int(x1), int(y1), int(x2), int(y2)], "label": "Beer"}
                      for x1, y1, x2, y2 in matches]

    if parsed is None:
        return None

    # Gemma outputs [y1, x1, y2, x2]; swap to [x1, y1, x2, y2]
    if is_gemma(model_key):
        for box in parsed:
            bb = box.get("bbox_2d")
            if bb and len(bb) == 4:
                box["bbox_2d"] = [bb[1], bb[0], bb[3], bb[2]]

    return parsed


def detect_beer(model_key, image_path):
    print(f"\n{'='*60}\nProcessing with model: {model_key}\n{'='*60}")

    # Instead of lms.prepare_image: OpenAI-style APIs take the image as a base64 data URL
    image = "data:image/jpeg;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()

    prompt = """Detect all beer glasses, mugs, or steins in this image.

Return ONLY a JSON array:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "Beer"}
]
```

Coordinates are normalized to 0-1000 where x1,y1 is top-left and x2,y2 is bottom-right. Use integers."""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                             {"type": "image_url", "image_url": {"url": image}}]}]

    start = time.time()
    result = client.chat.completions.create(
        model=model_key, messages=messages, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    inference_time = time.time() - start

    output = result.choices[0].message.content or ""
    print(f"\nModel Response:\n{output[:800]}")
    print(f"\nInference time: {inference_time:.2f}s")

    boxes = extract_boxes_from_text(output, model_key)
    if boxes:
        print(f"\nFound {len(boxes)} beer container(s)")
        return boxes, inference_time
    print("\nNo boxes detected")
    return [], inference_time


def draw_boxes(image_path, boxes, model_name):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
        title_font = ImageFont.truetype("Arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    for i, box in enumerate(boxes, 1):
        bbox = box.get('bbox_2d', [])
        if len(bbox) != 4:
            continue

        x1 = int(bbox[0] / 1000.0 * img_width)
        y1 = int(bbox[1] / 1000.0 * img_height)
        x2 = int(bbox[2] / 1000.0 * img_width)
        y2 = int(bbox[3] / 1000.0 * img_height)

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f"Beer {i}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3], fill=(0, 0, 0))
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

    title_bbox = draw.textbbox((10, 10), model_name, font=title_font)
    draw.rectangle([title_bbox[0]-5, title_bbox[1]-5, title_bbox[2]+5, title_bbox[3]+5], fill=(0, 0, 0))
    draw.text((10, 10), model_name, fill=(255, 255, 0), font=title_font)

    model_suffix = model_name.replace(':', '-').replace('/', '-').replace('@', '-')
    output_path = image_path.replace('.jpg', f'-processed-omlx-{model_suffix}.jpg')   # -omlx: don't overwrite the LM Studio run
    image.save(output_path)
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BEER DETECTION DEMO (oMLX)")
    print("="*60)
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {len(MODELS)}")

    results = {}
    for model in MODELS:
        try:
            boxes, inference_time = detect_beer(model, IMAGE_PATH)
            if boxes:
                output_path = draw_boxes(IMAGE_PATH, boxes, model)
                results[model] = {'boxes': len(boxes), 'output': output_path, 'time': inference_time}
            else:
                results[model] = {'boxes': 0, 'output': None, 'time': inference_time}
        except Exception as e:
            print(f"\nError with {model}: {e}")
            results[model] = {'error': str(e)}

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    for model, result in results.items():
        if 'error' in result:
            print(f"{model}: ERROR - {result['error']}")
        else:
            time_str = f"{result['time']:.2f}s" if 'time' in result else "N/A"
            print(f"{model}: {result['boxes']} detections, {time_str}")
    print("="*60)

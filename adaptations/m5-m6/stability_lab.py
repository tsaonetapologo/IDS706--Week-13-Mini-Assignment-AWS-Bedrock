import base64
import json
from io import BytesIO
from pathlib import Path

import boto3
from PIL import Image, ImageDraw, ImageOps


def get_bytesio_from_bytes(image_bytes):
    return BytesIO(image_bytes)


def get_bytes_from_file(file_path):
    return Path(file_path).read_bytes()


def encode(image_bytes):
    return base64.b64encode(image_bytes).decode("ascii")


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def invoke(model, **payload):
    response = boto3.client("bedrock-runtime", region_name="us-west-2").invoke_model(
        modelId="us.stability." + model + "-v1:0",
        body=json.dumps({"output_format": "png", **payload}),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    if result["finish_reasons"][0] is not None:
        raise RuntimeError(f"Image generation failed: {result['finish_reasons'][0]}")
    return BytesIO(base64.b64decode(result["images"][0]))


def variation(prompt_content, similarity_strength, image_bytes):
    return invoke(
        "stable-image-control-structure",
        image=encode(image_bytes),
        prompt=prompt_content,
        control_strength=similarity_strength,
    )


def style_mixing(prompt_content, similarity_strength, image_bytes1, image_bytes2):
    return invoke(
        "stable-style-transfer",
        init_image=encode(image_bytes2),
        style_image=encode(image_bytes1),
        prompt=prompt_content,
        composition_fidelity=similarity_strength,
    )


def masking(prompt_content, image_bytes, painting_mode, masking_mode, mask_bytes=None, mask_prompt=None):
    if masking_mode == "Prompt":
        if painting_mode == "OUTPAINTING":
            raise ValueError("Prompt-based outpainting is not supported by this alternative model.")
        return invoke(
            "stable-image-search-replace",
            image=encode(image_bytes),
            search_prompt=mask_prompt,
            prompt=prompt_content,
        )
    with Image.open(BytesIO(mask_bytes)) as source:
        mask = source.convert("L")
    if painting_mode == "INPAINTING":
        mask = ImageOps.invert(mask)
    return invoke(
        "stable-image-inpaint",
        image=encode(image_bytes),
        mask=encode(png_bytes(mask)),
        prompt=prompt_content,
    )


def replacement(prompt_content, image_bytes, mask_prompt):
    if not prompt_content.strip():
        raise ValueError("The replacement model needs a description; object removal is not supported here.")
    return invoke(
        "stable-image-search-replace",
        image=encode(image_bytes),
        search_prompt=mask_prompt,
        prompt=prompt_content,
    )


def erase(image_bytes, mask_bytes):
    with Image.open(BytesIO(image_bytes)) as source, Image.open(BytesIO(mask_bytes)) as mask_source:
        if source.size != mask_source.size:
            raise ValueError("The removal mask must have the same dimensions as the image.")
        mask = mask_source.convert("L")
    return invoke(
        "stable-image-erase-object",
        image=encode(image_bytes),
        mask=encode(png_bytes(mask)),
    )


def background(prompt_content, image_bytes, negative_prompt=None):
    subject = invoke("stable-image-remove-background", image=encode(image_bytes))
    payload = {"image": encode(subject.getvalue()), "prompt": prompt_content}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return invoke("stable-image-inpaint", **payload)


def insertion(prompt_content, image_bytes, insertion_position, insertion_dimensions):
    if image_bytes is None:
        image_bytes = get_bytes_from_file("images/desk.jpg")
    with Image.open(BytesIO(image_bytes)) as source:
        width, height = source.size
    x, y = insertion_position
    rect_width, rect_height = insertion_dimensions
    if x < 0 or y < 0 or rect_width <= 0 or rect_height <= 0 or x + rect_width > width or y + rect_height > height:
        raise ValueError("The mask rectangle must lie within the image and have positive dimensions.")
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(
        (x, y, x + rect_width - 1, y + rect_height - 1), fill=255
    )
    return invoke(
        "stable-image-inpaint",
        image=encode(image_bytes),
        mask=encode(png_bytes(mask)),
        prompt=prompt_content,
    )

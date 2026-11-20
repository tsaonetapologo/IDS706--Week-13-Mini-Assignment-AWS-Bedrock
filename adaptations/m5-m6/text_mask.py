from functools import lru_cache
from io import BytesIO

from PIL import Image

from stability_lab import png_bytes


@lru_cache(maxsize=1)
def load_model():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    model_id = "CIDAS/clipseg-rd64-refined"
    revision = "999e0328d9e10b484360c477313983f9afdd7050"
    processor = CLIPSegProcessor.from_pretrained(model_id, revision=revision)
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id, revision=revision, use_safetensors=True
    ).eval()
    return processor, model


def segment_image(image_bytes, description, threshold):
    if not description.strip():
        raise ValueError("Enter an object description before generating a mask.")
    if not 0 < threshold < 1:
        raise ValueError("The mask threshold must be between 0 and 1.")

    import torch
    import torch.nn.functional as functional

    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    processor, model = load_model()
    inputs = processor(text=[description], images=[image], return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = functional.interpolate(
            logits.unsqueeze(1),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].sigmoid()
        pixels = (probabilities > threshold).to(torch.uint8).mul(255).cpu().numpy()

    mask = Image.fromarray(pixels, mode="L")
    if mask.getextrema() in ((0, 0), (255, 255)):
        raise ValueError("The generated mask is empty or covers the entire image. Change the description or threshold.")
    return png_bytes(mask)

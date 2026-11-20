import hashlib
from io import BytesIO

from PIL import Image
import streamlit as st

from text_mask import segment_image


def preview_text_mask(image_bytes, description):
    st.caption("The model may miss parts of an object or select other objects. Review the red overlay before editing.")
    threshold = st.slider("Mask threshold", 0.05, 0.95, 0.30, 0.05)
    signature = hashlib.sha256(
        image_bytes + b"\0" + description.encode("utf-8") + repr(threshold).encode("ascii")
    ).hexdigest()

    if st.button("Preview text mask"):
        with st.spinner("Finding the described object locally..."):
            mask_bytes = segment_image(image_bytes, description, threshold)
        st.session_state["text_mask"] = (signature, mask_bytes)

    saved = st.session_state.get("text_mask")
    if not saved or saved[0] != signature:
        return None

    mask_bytes = saved[1]
    with Image.open(BytesIO(image_bytes)) as source, Image.open(BytesIO(mask_bytes)) as mask:
        overlay = source.convert("RGB")
        overlay.paste(Image.new("RGB", overlay.size, "red"), mask=mask.convert("L").point(lambda x: x // 2))
        st.image(overlay, caption="Red: selected object")
    if st.checkbox("I reviewed the selected object", key=f"confirm-{signature}"):
        return mask_bytes
    return None

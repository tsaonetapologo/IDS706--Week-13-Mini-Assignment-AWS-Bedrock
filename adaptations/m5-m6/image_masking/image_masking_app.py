import streamlit as st
import stability_lab as glib
from mask_preview import preview_text_mask


st.set_page_config(layout="wide", page_title="Image Masking")

st.title("Image Masking")
st.caption("Text-selected outpainting uses a locally generated mask. Review it before generating.")


col1, col2, col3 = st.columns(3)

with col3:
    st.subheader("Result")
    prompt_text = st.text_area("Prompt text:", height=100, help="The prompt text")
    painting_mode = st.radio("Painting mode:", ["INPAINTING", "OUTPAINTING"])

with col1:
    st.subheader("Image")
    uploaded_image_file = st.file_uploader("Select an image", type=['png', 'jpg'])

    if uploaded_image_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_image_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("images/desk1.jpg")


with col2:
    st.subheader("Mask")

    masking_mode = st.radio("Masking mode:", ["Image", "Prompt"], horizontal=True)

    if masking_mode == 'Image':

        uploaded_mask_file = st.file_uploader("Select an image mask", type=['png', 'jpg'])

        if uploaded_mask_file:
            uploaded_mask_preview = glib.get_bytesio_from_bytes(uploaded_mask_file.getvalue())
            st.image(uploaded_mask_preview)
        else:
            st.image("images/mask1.png")
    else:
        mask_prompt = st.text_input("Item to mask:", help="The item to replace (if inpainting), or keep (if outpainting).")
        if uploaded_image_file:
            image_bytes = uploaded_image_file.getvalue()
        else:
            image_bytes = glib.get_bytes_from_file("images/desk1.jpg")
        text_mask = None
        if painting_mode == "OUTPAINTING":
            text_mask = preview_text_mask(image_bytes, mask_prompt)


with col3:
    generate_button = st.button("Generate", type="primary")

    if generate_button:
        with st.spinner("Drawing..."):

            if uploaded_image_file:
                image_bytes = uploaded_image_file.getvalue()
            else:
                image_bytes = glib.get_bytes_from_file("images/desk1.jpg")

            if masking_mode == 'Image':
                if uploaded_mask_file:
                    mask_bytes = uploaded_mask_file.getvalue()
                else:
                    mask_bytes = glib.get_bytes_from_file("images/mask1.png")

                generated_image = glib.masking(
                    prompt_content=prompt_text,
                    image_bytes=image_bytes,
                    masking_mode=masking_mode,
                    mask_bytes=mask_bytes,
                    painting_mode=painting_mode
                )
            elif painting_mode == "OUTPAINTING":
                if text_mask is None:
                    st.error("Preview and confirm the text-selected mask before generating.")
                    st.stop()
                from PIL import Image, ImageOps

                with Image.open(glib.get_bytesio_from_bytes(text_mask)) as mask:
                    edit_mask = glib.png_bytes(ImageOps.invert(mask.convert("L")))
                generated_image = glib.masking(
                    prompt_content=prompt_text,
                    image_bytes=image_bytes,
                    masking_mode="Image",
                    mask_bytes=edit_mask,
                    painting_mode=painting_mode,
                )
            else:
                generated_image = glib.masking(
                    prompt_content=prompt_text,
                    image_bytes=image_bytes,
                    masking_mode=masking_mode,
                    mask_prompt=mask_prompt,
                    painting_mode=painting_mode
                )


        st.image(generated_image)

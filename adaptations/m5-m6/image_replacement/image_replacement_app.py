import streamlit as st
import stability_lab as glib
from mask_preview import preview_text_mask

st.set_page_config(layout="wide", page_title="Image Replacement")

st.title("Image Replacement")
st.caption("Removal can use an uploaded mask or a locally generated text-selected mask.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image parameters")

    uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])

    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("images/example.jpg")


with col2:
    mode = st.radio("Edit mode", ["Replace by description", "Remove with mask", "Remove by description"])
    if mode == "Replace by description":
        mask_prompt = st.text_input("Object to replace", value="peach curtains", help="The mask text")
        prompt_text = st.text_area("Object to add", value="polka dot curtains", height=100, help="The prompt text")
    elif mode == "Remove with mask":
        mask_file = st.file_uploader("Select a removal mask", type=["png", "jpg"])
    else:
        mask_prompt = st.text_input("Object to remove", value="chandelier")
        image_bytes = uploaded_file.getvalue() if uploaded_file else glib.get_bytes_from_file("images/example.jpg")
        text_mask = preview_text_mask(image_bytes, mask_prompt)

    generate_button = st.button("Generate", type="primary")


with col3:
    st.subheader("Result")

    if generate_button:
        if mode == "Remove with mask" and not mask_file:
            st.error("Upload a removal mask matching the image dimensions.")
        elif mode == "Remove by description" and text_mask is None:
            st.error("Preview and confirm the text-selected mask before generating.")
        else:
            with st.spinner("Drawing..."):
                if uploaded_file:
                    image_bytes = uploaded_file.getvalue()
                else:
                    image_bytes = glib.get_bytes_from_file("images/example.jpg")

                if mode == "Remove with mask":
                    generated_image = glib.erase(image_bytes, mask_file.getvalue())
                elif mode == "Remove by description":
                    generated_image = glib.erase(image_bytes, text_mask)
                else:
                    generated_image = glib.replacement(
                        prompt_content=prompt_text,
                        image_bytes=image_bytes,
                        mask_prompt=mask_prompt,
                    )

            st.image(generated_image)

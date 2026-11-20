# M5 and M6 text-selected editing

These are adaptations of the workshop apps, not the original Nova Canvas workflows. CLIPSeg produces an approximate mask locally; the user must preview and confirm the selection. Stability Inpaint edits the surroundings in M5, and Stability Erase removes the selection in M6. Thin details may be missed or unrelated objects selected. Change the description or threshold, or use an image mask, when the preview is wrong.

## Setup

The files overlay an existing copy of the workshop's `labs/multimodal` directory and expect its example images. Back up any changes to those apps first: copying these files replaces the corresponding app and shared backend files.

In the workshop environment, from the root of this repository:

```bash
cp adaptations/m5-m6/stability_lab.py adaptations/m5-m6/mask_preview.py \
   adaptations/m5-m6/text_mask.py /environment/workshop/labs/multimodal/
cp -R adaptations/m5-m6/image_masking/. /environment/workshop/labs/multimodal/image_masking/
cp -R adaptations/m5-m6/image_replacement/. /environment/workshop/labs/multimodal/image_replacement/
python3.11 -m venv --system-site-packages /environment/clipseg-venv
/environment/clipseg-venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The workshop environment also needs `transformers`, Pillow, Streamlit, NumPy, and boto3. My test used the existing workshop packages, including Transformers 5.23.0. On first use, the pinned [CLIPSeg model](https://huggingface.co/CIDAS/clipseg-rd64-refined) downloads approximately 603 MB of weights; inference runs locally. AWS credentials and access to the Stability US inference profiles are needed for editing, which may incur charges.

Run either app from its directory so it can find the workshop images:

```bash
cd /environment/workshop/labs/multimodal/image_masking
/environment/clipseg-venv/bin/python -m streamlit run image_masking_app.py
```

For M6, use `image_replacement` and `image_replacement_app.py` instead. For M5 choose **Prompt**, **OUTPAINTING**, describe the object to keep, and enter a background prompt. For M6 choose **Remove by description** and describe the object to remove. In both, preview and confirm the mask before generating.

The included source builds on the workshop's MIT No Attribution-licensed code. Model weights are not included.

import sys
import types
import unittest
from contextlib import nullcontext
from io import BytesIO
from unittest.mock import Mock, patch

from PIL import Image

sys.modules.setdefault("boto3", types.ModuleType("boto3"))
streamlit = types.ModuleType("streamlit")
sys.modules["streamlit"] = streamlit

import mask_preview
import text_mask


def image_bytes(color):
    buffer = BytesIO()
    Image.new("L", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


class TextMaskTests(unittest.TestCase):
    def test_description_and_threshold_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "object description"):
            text_mask.segment_image(b"", " ", 0.3)
        with self.assertRaisesRegex(ValueError, "threshold"):
            text_mask.segment_image(b"", "desk", 1)

    def test_mask_requires_review_and_is_invalidated_by_new_description(self):
        streamlit.session_state = {}
        streamlit.caption = Mock()
        streamlit.slider = Mock(return_value=0.3)
        streamlit.button = Mock(side_effect=[True, False, False])
        streamlit.spinner = Mock(return_value=nullcontext())
        streamlit.image = Mock()
        streamlit.checkbox = Mock(side_effect=[False, True])
        image, mask = image_bytes(0), image_bytes(255)

        with patch.object(mask_preview, "segment_image", return_value=mask) as segment:
            self.assertIsNone(mask_preview.preview_text_mask(image, "desk"))
            self.assertEqual(mask_preview.preview_text_mask(image, "desk"), mask)
            self.assertIsNone(mask_preview.preview_text_mask(image, "lamp"))
        segment.assert_called_once_with(image, "desk", 0.3)


if __name__ == "__main__":
    unittest.main()

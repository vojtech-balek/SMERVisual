from pathlib import Path
from tkinter import Image
from PIL import Image
import cv2
from transformers import AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
from typing import Union, Optional, List
from openai import OpenAI
import base64
import matplotlib.pyplot as plt
import re
import os
import torch


class BoundingBoxGenerator:
    def __init__(self,
                 data: Union[str, Path],
                 top_words: List[str],
                 openai_key: Optional[str] = None,
                 openai_model: Optional[str] = None,
                 local_model_path: Optional[str] = None):
        self.data_folder = Path(data)
        self.top_words = top_words

        self.model_host = "openai" if openai_model else "local"
        if self.model_host == "openai":
            if not openai_key:
                raise ValueError("OpenAI key must be provided when using OpenAI API.")
            if not openai_model:
                raise ValueError("OpenAI model must be provided when using OpenAI API.")

            self.openai_key = openai_key
            self.openai_model = openai_model
            self.client = OpenAI(api_key=self.openai_key)
        else:
            if not local_model_path:
                raise ValueError("Local model must be provided when not using OpenAI API.")

            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                local_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(local_model_path)

    def __call__(self):
        for file_path, label in self._get_image_files_with_class(self.data_folder):
            if self.model_host == "openai":
                bounding_box = self._get_bounding_boxes_openai(file_path)
            else:
                bounding_box = self._get_bounding_boxes_local(file_path)

            self._save_image_with_bounding_box(file_path, label, bounding_box)

    def _get_bounding_boxes_local(self, file_path: str):
        """
        Retrieve bounding box coordinates from an image using local model.
        :param file_path: Path to the image file.
        :type file_path: str
        :return: Bounding box coordinates in the format (x_min, y_min, x_max, y_max) or None if an error occurs.
        :rtype: tuple or None
        """
        decoded_output = None

        try:
            with open(file_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

            image = Image.open(file_path).convert("RGB")
            prompt = (
                f"Provide bounding box coords for the object from {self.top_words}.\n"
                "Format: x_min, y_min, x_max, y_max with integers only."
            )

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": encoded_image},
                    ]
                }
            ]
            input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=80)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            print(f"Raw bounding box response: '{decoded_output}'")  # Debug statement

            match = re.search(r'(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)', decoded_output)
            if match:
                bounding_box_text = match.group(1)
                print(f"Extracted bounding box text: '{bounding_box_text}'")  # Debug statement

                parts = bounding_box_text.split(',')
                if len(parts) != 4:
                    raise ValueError("Expected four comma-separated values for bounding box.")

                bounding_box = tuple(int(part.strip()) for part in parts)
                print(f"Parsed bounding box: {bounding_box}")  # Debug statement

                return bounding_box
            else:
                raise ValueError("No valid bounding box pattern found in the response.")

        except ValueError as ve:
            print(f"ValueError while parsing bounding box: {ve}")
            if decoded_output:
                print(f"Bounding box text received: '{decoded_output}'")
            else:
                print(f"Bounding box text not received.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _get_bounding_boxes_openai(self, file: str):
        """
        Retrieve bounding box coordinates from an image using OpenAI API.
        :param file: Path to the image file.
        :type file: str
        :return: Bounding box coordinates in the format (x_min, y_min, x_max, y_max) or None if an error occurs.
        :rtype: tuple or None
        """

        try:
            with open(file, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Give me just one bounding box coordinates for the object most likely to be in the image from this list '{self.top_words}' ""in this image in the format: x_min, y_min, x_max, y_max, where all values are integers without any words or letters."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=80
            )
            bounding_box_text = response.choices[0].message.content.strip()
            print(bounding_box_text)
            match = re.search(r'(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)', bounding_box_text)
            if match:
                bounding_box = match.group(1)
                print(f"Extracted bounding box text: '{bounding_box}'")  # Debug statement

                parts = bounding_box_text.split(',')
                if len(parts) != 4:
                    raise ValueError("Expected four comma-separated values for bounding box.")

                bounding_box = tuple(int(part.strip()) for part in parts)
                print(f"Parsed bounding box: {bounding_box}")  # Debug statement

                return bounding_box
            else:
                raise ValueError("No valid bounding box pattern found in the response.")

        except ValueError as ve:
            print(f"ValueError while parsing bounding box: {ve}")
            print(f"Bounding box text received: '{bounding_box_text}'")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    @staticmethod
    def visualize_bounding_box(image_path: str, bounding_box):
        """
        Visualize a bounding box on an image.
        :param image_path: The file path to the image to be visualized.
        :type image_path: str
        :param bounding_box: A tuple containing bounding box coordinates in the format (x_min, y_min, x_max, y_max).
        :type bounding_box: tuple
        """

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if bounding_box:
            x_min, y_min, x_max, y_max = bounding_box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        else:
            print("No bounding box to visualize.")

    @staticmethod
    def _save_image_with_bounding_box(image_path: str, label: str, bounding_box):
        """
        Saves an image with a bounding box in the folder structure `smer_bounding_boxes/{label}`.

        :param image_path: Path to the original image.
        :param bounding_box: Tuple (x_min, y_min, x_max, y_max) defining the bounding box.
        :param label:
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if bounding_box:
            x_min, y_min, x_max, y_max = bounding_box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            output_folder = os.path.join("smer_bounding_boxes", label)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            print("No bounding box to visualize.")

    @staticmethod
    def _get_image_files_with_class(folder_path):
        """Load images with labels based on their parent directory name.
        :param folder_path: path to directory subfolders with images
        :type folder_path: str or PathLike
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions) and '.ipynb_checkpoints' not in root:
                    image_path = os.path.join(root, file)
                    class_name = os.path.basename(root)

                    yield image_path, class_name

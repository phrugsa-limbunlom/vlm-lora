from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


class MathVistaDataset(Dataset):
    """
    MathVista dataset for mathematical reasoning in visual contexts.

    Loads data from HuggingFace and processes it for VLM training.
    """

    def __init__(self, split: str = "testmini", max_samples: Optional[int] = None, processor=None):
        """
        Initialize MathVista dataset.

        Args:
            split: Dataset split to use ("testmini" or "test")
            max_samples: Maximum number of samples to load (None for all)
            processor: LLaVA processor for text and image processing
        """
        self.split = split
        self.processor = processor
        self.data: List[Dict[str, Any]] = []

        # Load dataset from HuggingFace
        print(f"Loading MathVista dataset (split: {split})...")
        try:
            dataset = load_dataset("AI4Math/MathVista")
            raw_data = dataset[split]

            # Convert to list for consistent handling
            try:
                data_list = []
                count = 0
                # Handle datasets that might not have len()
                try:
                    max_items = max_samples or len(raw_data)  # type: ignore
                except:
                    max_items = max_samples or 1000  # Default fallback

                for item in raw_data:
                    if count >= max_items:
                        break
                    # Convert item to dict safely
                    if hasattr(item, 'items'):
                        data_list.append(dict(item))
                    else:
                        data_list.append(item)
                    count += 1

                self.data = data_list

            except Exception as conversion_error:
                print(f"Error converting dataset: {conversion_error}")
                # Fallback to dummy data
                self.data = self._create_dummy_data(max_samples or 50)

            print(f"Loaded {len(self.data)} samples from MathVista {split} split")

        except Exception as e:
            print(f"Error loading MathVista dataset: {e}")
            print("Using dummy data for demonstration...")
            self.data = self._create_dummy_data(max_samples or 50)

        # Build vocabulary for simple tokenization
        self.vocab = self._build_vocab()
        self.max_length = 256  # Maximum sequence length

    def _create_dummy_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create dummy data for demonstration."""
        dummy_data = []
        for i in range(num_samples):
            dummy_data.append({
                'question': f"Sample math question {i}?",
                'answer': str(i),
                'decoded_image': None,
                'metadata': {'split': self.split}
            })
        return dummy_data

    def _build_vocab(self):
        """Build a simple vocabulary from the dataset."""
        vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
        word_count = {}

        # Count words in questions and answers
        for item in self.data:
            # Process question
            question = self._get_item_value(item, 'question')
            if question and isinstance(question, str):
                words = question.lower().split()
                for word in words:
                    word_count[word] = word_count.get(word, 0) + 1

            # Process answer
            answer = self._get_item_value(item, 'answer')
            if answer:
                answer_str = str(answer)
                words = answer_str.lower().split()
                for word in words:
                    word_count[word] = word_count.get(word, 0) + 1

        # Add most common words to vocab
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:10000]:  # Top 10K words
            if word not in vocab:
                vocab[word] = len(vocab)

        return vocab

    def _get_item_value(self, item: Dict[str, Any], key: str) -> Any:
        """Safely get value from dataset item."""
        if isinstance(item, dict):
            return item.get(key)
        elif hasattr(item, 'get'):
            return item.get(key)
        elif hasattr(item, '__getitem__'):
            try:
                return item[key]
            except (KeyError, IndexError):
                return None
        elif hasattr(item, key):
            return getattr(item, key)
        else:
            return None

    def _tokenize_text(self, text: str) -> List[int]:
        """Simple tokenization."""
        if not text:
            return [self.vocab["<pad>"]]

        words = text.lower().split()
        tokens = [self.vocab["<start>"]]

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab["<unk>"])

        tokens.append(self.vocab["<end>"])

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.vocab["<pad>"]] * (self.max_length - len(tokens)))

        return tokens

    def _process_image(self, image) -> torch.Tensor:
        """Process image for the model."""
        if image is None:
            # Return dummy image if no image available
            return torch.randn(3, 224, 224)

        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            # Resize and convert to tensor
            image = image.resize((224, 224))
            image = image.convert('RGB')

            # Convert to tensor and normalize
            image_tensor = torch.tensor(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            return image_tensor
        else:
            # Return dummy image if format is unexpected
            return torch.randn(3, 224, 224)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict[str, Any]]]:
        item = self.data[idx]

        # Get question and answer safely
        question = self._get_item_value(item, 'question') or "Sample question"
        answer = self._get_item_value(item, 'answer') or "Sample answer"

        # Ensure question and answer are strings
        question = str(question)
        answer = str(answer)

        # Process image
        image = self._get_item_value(item, 'decoded_image')
        if image is None:
            # Create a dummy image if no image is available
            image = Image.new('RGB', (336, 336), color=(255, 255, 255))

        # Create conversation format for LLaVA
        conversation = [
            {
                "role": "user",
                "content": f"<image>\nQuestion: {question}"
            },
            {
                "role": "assistant",
                "content": f"Answer: {answer}"
            }
        ]

        # Process with LLaVA processor
        if self.processor is not None:
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

            # Process image and text
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Extract processed data
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs["pixel_values"].squeeze(0)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'labels': input_ids.clone(),  # For language modeling
                'question': question,
                'answer': answer,
                'metadata': self._get_item_value(item, 'metadata') or {}
            }
        else:
            # Fallback to simple processing
            return {
                'input_ids': torch.tensor([1, 2, 3], dtype=torch.long),  # Dummy
                'attention_mask': torch.tensor([1, 1, 1], dtype=torch.long),
                'pixel_values': torch.randn(3, 336, 336),  # Dummy
                'labels': torch.tensor([1, 2, 3], dtype=torch.long),
                'question': question,
                'answer': answer,
                'metadata': self._get_item_value(item, 'metadata') or {}
            }
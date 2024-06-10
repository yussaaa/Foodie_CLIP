from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    AutoProcessor,
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import bisect
import torch
from dataclasses import dataclass


class IndividualClassDataset(Dataset):
    def __init__(self, index, path, description, class_max_datapoints=200):
        self.index = index
        self.path = path + self.index + "/"
        self.description = description
        self.img_path = os.listdir(self.path)
        # truncate the dataset over the max_datapoints
        self.img_path = self.img_path[: min(class_max_datapoints, len(self.img_path))]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        return {
            "image": Image.open(self.path + self.img_path[idx]),
            "text": self.description,
            "text_id": self.index,
        }


class CLIPDataset(Dataset):
    def __init__(self, path="", train=True, class_max_datapoints=200):
        self.path = path
        self.df = pd.read_csv(self.path + "/class_names.csv")
        if train:
            subpath = self.path + "/train600x600/"
        else:
            subpath = self.path + "/val600x600/"
        # removing the chinese name of dish
        self.df.drop("Chinese Name of Dish", axis=1, inplace=True)
        self.datasets = []
        self.presum = 0
        for index in self.df.index:
            description = self.df.loc[index]["English Name of Dish"]
            index = "0" * (3 - len(str(index))) + str(index)
            path = subpath
            class_dataset = IndividualClassDataset(
                index=index,
                path=path,
                description=description,
                class_max_datapoints=class_max_datapoints,
            )
            self.presum += len(class_dataset)
            # store (the dataset for the class ith, its end datapoint index)
            self.datasets.append((class_dataset, self.presum))
        self.n_class = len(self.datasets)

    def __len__(self):
        return self.presum

    def __getitem__(self, idx):
        # using binary search to located the target dataset, which contains the idx th datapoint
        dataset_index = bisect.bisect_right(self.datasets, idx, key=lambda x: x[1])
        dataset = self.datasets[dataset_index][0]
        if dataset_index == 0:
            return dataset[idx]
        else:
            prev_end_idx = self.datasets[dataset_index - 1][1]
            return dataset[idx - prev_end_idx]


# @dataclass
class CLIPDataCollator:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors

    def set_max_length(self, max_length):
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        texts = []
        texts_row_id = {}
        for index, feature in enumerate(features):
            images.append(feature["image"])
            # assign the index th image to the corresponding text
            if feature["text_id"] in texts_row_id:
                texts_row_id[feature["text_id"]]["image_index"].append(index)
            else:
                texts.append(feature["text"])
                texts_row_id[feature["text_id"]] = {
                    "row_id": len(texts) - 1,
                    "image_index": [index],
                }
        # There are m text, n images in total, where m <= n(a text label may associates with multiple images)
        m = len(texts)
        n = len(images)
        # # preprocessing for texts and images
        # tokenize text
        batch_images = self.processor(
            images=images, return_tensors=self.return_tensors, padding=self.padding
        )
        # image preprocess for vit
        batch_texts = self.processor(
            text=texts,
            return_tensors=self.return_tensors,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        # construct the mask matrix(m x n)
        # if mask matrix[i][j] == 1, texts[i] is the label of images[j]
        mask = torch.ones(m, n) * -torch.inf
        for text in texts_row_id.values():
            row_index = text["row_id"]
            col_indices = text["image_index"]
            mask[row_index, col_indices] = 1
        return {
            "texts_input": batch_texts,
            "images_input": batch_images,
            "mask": mask.t(),
        }


if __name__ == "__main__":
    import os

    class_max_datapoints = 2
    clipdataset = CLIPDataset(
        path=os.getcwd() + "/",
        class_max_datapoints=class_max_datapoints,
    )
    dataloader = DataLoader(
        clipdataset,
        batch_size=clipdataset.n_class * class_max_datapoints,
        collate_fn=CLIPDataCollator(
            processor=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ),
    )
    for minibatch in dataloader:
        texts_input = minibatch["texts_input"]
        images_input = minibatch["images_input"]
        mask = minibatch["mask"]
        # print(texts_input)
        # print(images_input)
        print(mask)

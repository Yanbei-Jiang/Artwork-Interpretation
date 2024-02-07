import json
import numpy as np
import time
import logging
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0"*(9-len(ann[:-1]))+ann[:-1]
    assert len(id_) == 9
    ann = server+"/"+id_
    return ann

class ap_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            metadata = each["title"]
            if not sentences:
                continue
            file_root = ""
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            if is_train:
                for caption in sentences:
                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption,
                                         "metadata": metadata})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": gold_caption,
                     "metadata": metadata})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        metadata = ann['metadata']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, metadata, image_id, ann["gold_caption"]


class sa_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False, type="visual"):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            if type=="visual":
                sentences = each["visual_sentences"]
            elif type=="contextual":
                sentences = each["contextual_sentences"]
            elif type=="all":
                sentences = each["visual_sentences"]+each["contextual_sentences"]
                
            metadata = "<AUTHOR> " + each["author"] + ", <TITLE> " + each["title"] + ", <TECHNIQUE> " + each["technique"].split(",")[0] + ", <TYPE> " + each["type"] + ", <SCHOOL> " + each["school"] + ", <TIMEFRAME> " + each["timeframe"]
            if not sentences:
                continue

            file_root = ""

            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            if is_train:

                for caption in sentences:
                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption,
                                         "metadata": metadata})
            else:

                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": gold_caption,
                     "metadata": metadata})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        metadata = ann['metadata']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, metadata, image_id, ann["gold_caption"]


class artcap_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False, type="visual"):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each[0]
            sentences = each[1]
            if not sentences:
                continue

            file_root = ""

            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            if is_train:
                for caption in sentences:
                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": gold_caption})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]

        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, image_id, ann["gold_caption"]


class sa_dataset_heterogeous(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False, type="visual"):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            if type=="visual":
                sentences = each["visual_sentences"]
            elif type=="contextual":
                sentences = each["contextual_sentences"]
            elif type=="all":
                sentences = each["visual_sentences"]+each["contextual_sentences"]
                
            metadata = "<AUTHOR> " + each["author"] + ", <TITLE> " + each["title"] + ", <TECHNIQUE> " + each["technique"].split(",")[0] + ", <TYPE> " + each["type"] + ", <SCHOOL> " + each["school"] + ", <TIMEFRAME> " + each["timeframe"]
            
            metadata_words = {"artwork": each["filename"], "author": each["author"], "title": each["title"], "technique": each["technique"], "artwork_type": each["type"], "school": each["school"], "timeframe": each["timeframe"]}

            if not sentences:
                continue
            file_root = ""
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            if is_train:
                for caption in sentences:
                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption,
                                         "metadata": metadata, "metadata_words_list": metadata_words})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": gold_caption,
                     "metadata": metadata, "metadata_words_list": metadata_words})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        metadata = ann['metadata']
        metadata_words = ann["metadata_words_list"]
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, metadata, image_id, ann["gold_caption"], metadata_words


class new_semart_allcaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["img"]

            new_content = ["<content> "+i for i in each["content"]]
            new_form = ["<form> "+i for i in each["form"]]
            new_context = ["<context> "+i for i in each["context"]]
            sentences = new_content + new_form + new_context
            if not sentences:
                continue
            metadata = "<AUTHOR> " + each["author"] + ", <TITLE> " + each["title"] + ", <TECHNIQUE> " + each["technique"].split(",")[0] + ", <TYPE> " + each["type"] + ", <SCHOOL> " + each["school"] + ", <TIMEFRAME> " + each["timeframe"]

            file_root = ""

            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            for caption in new_content:
                caption = caption.lower()
            for caption in new_form:
                caption = caption.lower()
            for caption in new_context:
                caption = caption.lower()
            if is_train:
                for caption in sentences:
                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption, "metadata": metadata})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": {"content": new_content, "form": new_form, "context": new_context},
                     "metadata": metadata})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        metadata = ann['metadata']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, metadata, image_id, ann["gold_caption"]

class new_semart_allcaps_dataset_heterogeous(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["img"]

            new_content = ["<content> "+i for i in each["content"]]
            new_form = ["<form> "+i for i in each["form"]]
            new_context = ["<context> "+i for i in each["context"]]
            sentences = new_content + new_form + new_context
            if not sentences:
                continue
            metadata = "<AUTHOR> " + each["author"] + ", <TITLE> " + each["title"] + ", <TECHNIQUE> " + each["technique"].split(",")[0] + ", <TYPE> " + each["type"] + ", <SCHOOL> " + each["school"] + ", <TIMEFRAME> " + each["timeframe"]
            metadata_words = {"artwork": each["img"], "author": each["author"], "title": each["title"], "technique": each["technique"], "artwork_type": each["type"], "school": each["school"], "timeframe": each["timeframe"]}
            file_root = ""
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for caption in sentences:
                gold_caption.append(caption.lower())
            for caption in new_content:
                caption = caption.lower()
            for caption in new_form:
                caption = caption.lower()
            for caption in new_context:
                caption = caption.lower()
            if is_train:
                for caption in sentences:

                    self.ann_new.append({"image": image_path, "caption": caption.lower(), "gold_caption": gold_caption, "metadata": metadata, "metadata_words_list": metadata_words})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0].lower(), "gold_caption": {"content": new_content, "form": new_form, "context": new_context},
                     "metadata": metadata, "metadata_words_list": metadata_words})
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        metadata = ann['metadata']
        metadata_words = ann["metadata_words_list"]
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, metadata, image_id, ann["gold_caption"], metadata_words

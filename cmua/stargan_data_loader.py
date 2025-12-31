from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random
import pandas as pd
import face_alignment
import numpy as np



class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.inference_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        self.preprocess()

        if mode == 'train':  
            self.num_images = len(self.train_dataset)
        elif mode == 'inference':
            self.num_images = len(self.inference_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if 2000 <= (i + 1) < 4000:
                # Use the range from 2000 to 3999 for inference dataset.
                self.inference_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # The original male photo is the first photo in the test data
        dataset = self.train_dataset if self.mode == 'train' else self.inference_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))

        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class MAADFace(data.Dataset):
    """Dataset class for the MAAD-Face dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, start_index=0):
        self.image_dir = image_dir # e.g., /scratch/.../train
        self.attr_path = attr_path # e.g., /scratch/.../MAAD_Face_filtered.csv
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.inference_dataset = []
        self.start_index = start_index # Starting index for training dataset
        self.attr2idx = {} 
        self.idx2attr = {}

        self.preprocess()
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == 'inference':
            self.num_images = len(self.inference_dataset)

    def preprocess(self):
        df = pd.read_csv(self.attr_path, encoding='utf-8-sig') # csv file
        # print("column names (47) at MAAD-FACE dataset :", df.columns.tolist())
        attr_names = list(df.columns[1:])  # Exclude 'Filename'

        # attr_name to index mapping
        self.attr2idx = {attr: idx for idx, attr in enumerate(attr_names)}
        self.idx2attr = {idx: attr for attr, idx in self.attr2idx.items()}


        def process_row(row):
            filename = row['Filename'].strip()
            label = [(row[attr] == 1) for attr in self.selected_attrs]
            return (filename, label)
        full_dataset = [process_row(row) for _, row in df.iterrows()]   # Sequential learning on the entire dataset

        random.seed(1234)
        random.shuffle(full_dataset)

        for i, data_item in enumerate(full_dataset):
            if i < 2000:
                self.inference_dataset.append(data_item)
            elif i >= (2000 + self.start_index):
                self.train_dataset.append(data_item) # Start from the index

        print('Finished preprocessing the MAADFace dataset...')


    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.inference_dataset
        filename, label = dataset[index]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = align_face(image, crop_size=178)
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        return self.num_images




fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')
def align_face(image: Image.Image, crop_size=178) -> Image.Image:

    img_np = np.array(image)
    preds = fa.get_landmarks(img_np)

    # Return original image if face detection fails
    if preds is None or len(preds) == 0:
        print(f"[align_face] Face detection failed: {getattr(image, 'filename', 'unknown')}")
        return image.resize((crop_size, crop_size))

    # Select the largest face
    landmarks = max(preds, key=lambda x: x[:, 1].ptp())

    # Calculate tight bounding box for the face area
    x_min = max(int(np.min(landmarks[:, 0])) - 10, 0)
    x_max = min(int(np.max(landmarks[:, 0])) + 10, image.width)
    y_min = max(int(np.min(landmarks[:, 1])) - 10, 0)
    y_max = min(int(np.max(landmarks[:, 1])) + 10, image.height)

    # Crop & resize the face box
    face_box = image.crop((x_min, y_min, x_max, y_max))
    face_resized = face_box.resize((crop_size, crop_size), Image.BILINEAR)

    return face_resized




def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset=None, mode='train', num_workers=1, start_index=0):
    """Build and return a data loader."""
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'MAADFace':
        dataset = MAADFace(image_dir, attr_path, selected_attrs, transform, mode,start_index=start_index)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle = False, 
                                  num_workers=num_workers)
    return data_loader

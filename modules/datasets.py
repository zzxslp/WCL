import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        label = example['label']
        sample = (image_id, image, report_ids, report_masks, seq_length, label) # report_ids: encoded ids
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        label = example['label']
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample

class mimic_abnormal_dataset(Dataset):
    def __init__(self, args, tok, split_type):
        self.max_seq_length = args.max_seq_length

        splits = create_splits(args, tok)
        self.imgpairs = splits[split_type][0]
        self.captions = splits[split_type][1]
        self.imgnames = splits[split_type][2]
        self.labels = splits[split_type][3]

        self.tok = tok

        self.split_type = split_type

    def __len__(self):
        return len(self.imgpairs)

    def __getitem__(self, idx):
        img1, img2 = self.imgpairs[idx]
        image = torch.stack((img1, img2), 0)
        image_id = self.imgnames[idx]
        raw_encodings = self.captions[idx]
        # encode the sentence
        report_ids = raw_encodings[:self.max_seq_length]
        report_masks = [1] * len(report_ids)
        seq_length =  len(report_ids)
        label = self.labels[idx]
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample

def create_splits(args, tok):
    all_imgs = torch.load(args.image_dir)
    with open(args.ann_path) as file:
        data = json.load(file)
    split_imgs, split_caps, split_names, split_labels = dict(), dict(), dict(), dict()
    for split in ['train', 'val', 'test']:
        img_names = [d['id'] for d in data[split]]
        split_imgs[split] = [(all_imgs["{}-0".format(i)], all_imgs["{}-1".format(i)]) for i in img_names]
        split_caps[split] = [tok(d['report']) for d in data[split]]
        split_names[split] = img_names
        split_labels[split] = [d['label'] for d in data[split]]

    print ('Num of train/val/test: {}/{}/{}'.format(len(split_caps['train']), len(split_caps['val']), len(split_caps['test'])))
    splits = {'train':(split_imgs['train'], split_caps['train'], split_names['train'], split_labels['train']), 
                'val':(split_imgs['val'], split_caps['val'], split_names['val'], split_labels['val']), 
                'test':(split_imgs['test'], split_caps['test'], split_names['test'], split_labels['test'])}

    return splits




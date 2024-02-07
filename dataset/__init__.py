import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import ap_dataset, sa_dataset, artcap_dataset, sa_dataset_heterogeous, new_semart_allcaps_dataset, new_semart_allcaps_dataset_heterogeous

from dataset.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode

def create_dataset(dataset, config, epoch=None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   

    if dataset=='artpedia':
        train_dataset = ap_dataset(['./data/Artpedia/artpedia_train_visual.json'], train_transform, './data/Artpedia/images/data',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=conifig['add_object'])
        val_dataset = ap_dataset(['./ata/Artpedia/artpedia_val_visual.json'], test_transform, './data/Artpedia/images/data',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = ap_dataset(['./data/Artpedia/artpedia_test_visual.json'], test_transform, './data/Artpedia/images/data',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset


    elif dataset=='semart1.0_visual':
        train_dataset = sa_dataset(['./data/SemArt/semart1.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="visual")
        val_dataset = sa_dataset(['./data/SemArt/semart1.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="visual")
        test_dataset = sa_dataset(['./data/SemArt/semart1.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="visual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart1.0_contextual':
        train_dataset = sa_dataset(['./data/SemArt/semart1.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="contextual")
        val_dataset = sa_dataset(['./data/SemArt/semart1.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="contextual")
        test_dataset = sa_dataset(['./data/SemArt/semart1.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="contextual")
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'artcap_dataset':
        train_dataset = artcap_dataset(['./data/ArtCap/artcap_train_val.json'], train_transform,
                                   './data/ArtCap/images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=True, add_object=config['add_object'], type="visual")
        val_dataset = artcap_dataset(['./data/ArtCap/artcap_train_val.json'], test_transform,
                                 './data/ArtCap/images',
                                 max_words=config['max_length'], read_local_data=config['read_local_data'],
                                 is_train=False, add_object=config['add_object'], type="visual")
        test_dataset = artcap_dataset(['./data/ArtCap/artcap_test.json'], test_transform,
                                  './data/ArtCap/images',
                                  max_words=config['max_length'], read_local_data=config['read_local_data'],
                                  is_train=False, add_object=config['add_object'], type="visual")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='semart1.0_visual_heterogeous':
        train_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="visual")
        val_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="visual")
        test_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="visual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart1.0_contextual_heterogeous':
        train_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="contextual")
        val_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="contextual")
        test_dataset = sa_dataset_heterogeous(['./data/SemArt/semart1.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="contextual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart2.0':
        train_dataset = new_semart_allcaps_dataset(['./data/SemArt/semart2.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = new_semart_allcaps_dataset(['./data/SemArt/semart2.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = new_semart_allcaps_dataset(['./data/SemArt/semart2.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='semart2.0_heterogeous':
        train_dataset = new_semart_allcaps_dataset_heterogeous(['./data/SemArt/semart2.0_train_val.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = new_semart_allcaps_dataset_heterogeous(['./data/SemArt/semart2.0_train_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = new_semart_allcaps_dataset_heterogeous(['./data/SemArt/semart2.0_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset


def coco_collate_fn(batch):
    image_list, caption_list, metadata_list, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, metadata, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        metadata_list.append(metadata)
    return torch.stack(image_list,dim=0), caption_list, metadata_list, image_id_list, gold_caption_list

def artcap_collate_fn(batch):
    image_list, caption_list, image_id_list, gold_caption_list = [], [], [], []
    for image, caption, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
    return torch.stack(image_list,dim=0), caption_list, image_id_list, gold_caption_list

def semart_heterogeous_collate_fn(batch):
    image_list, caption_list, metadata_list, image_id_list, gold_caption_list, metadata_words_list = [], [], [], [], [], []
    for image, caption, metadata, image_id, gold_caption, metadata_words in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        metadata_list.append(metadata)
        metadata_words_list.append(metadata_words)
    return torch.stack(image_list,dim=0), caption_list, metadata_list, image_id_list, gold_caption_list, metadata_words_list

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = False
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

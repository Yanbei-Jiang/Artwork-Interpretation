import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import ap_dataset, new_semart_dataset, sa_dataset, new_semart_allcaps_dataset, sa_dataset_kg, new_semart_allcaps_kg_dataset, new_semart_dataset_kg, semart_visual_contextual


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
    

    if dataset=='artpedia_visual':
        train_dataset = ap_dataset(['./data/Artpedia/artpedia_train_visual.json'], train_transform, './data/Artpedia/images/data',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = ap_dataset(['./data/Artpedia/artpedia_val_visual.json'], test_transform, './data/Artpedia/images/data',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = ap_dataset(['./data/Artpedia/artpedia_test_visual.json'], test_transform, './data/Artpedia/images/data',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='artpedia_contextual':
        train_dataset = ap_dataset(['./data/Artpedia/artpedia_train_context.json'], train_transform, './data/Artpedia/images/data',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = ap_dataset(['./data/Artpedia/artpedia_val_context.json'], test_transform, './data/Artpedia/images/data',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = ap_dataset(['./data/Artpedia/artpedia_test_context.json'], test_transform, './data/Artpedia/images/data',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='semart_visual':
        train_dataset = sa_dataset(['./data/SemArt/processed_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="visual")
        val_dataset = sa_dataset(['./data/SemArt/semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="visual")
        test_dataset = sa_dataset(['./data/SemArt/semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="visual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart_contextual':
        train_dataset = sa_dataset(['./data/SemArt/processed_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="contextual")
        val_dataset = sa_dataset(['./data/SemArt/semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="contextual")
        test_dataset = sa_dataset(['./data/SemArt/semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="contextual")
        return train_dataset, val_dataset, test_dataset
    elif dataset=='semart_visual_kg':
        train_dataset = sa_dataset_kg(['./data/SemArt/processed_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="visual")
        val_dataset = sa_dataset_kg(['./data/SemArt/semart_val.json'], './data/SemArt/embeddings_val.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="visual")
        test_dataset = sa_dataset_kg(['./data/SemArt/semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="visual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart_contextual_kg':
        train_dataset = sa_dataset_kg(['./data/SemArt/processed_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="contextual")
        val_dataset = sa_dataset_kg(['./data/SemArt/semart_val.json'], './data/SemArt/embeddings_val.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="contextual")
        test_dataset = sa_dataset_kg(['./data/SemArt/semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="contextual")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='semart_all':
        train_dataset = sa_dataset(['./data/SemArt/semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="all")
        val_dataset = sa_dataset(['./data/SemArt/semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="all")
        test_dataset = sa_dataset(['./data/SemArt/semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="all")
        return train_dataset, val_dataset, test_dataset
    elif dataset=='semart_all_kg':
        train_dataset = sa_dataset_kg(['./data/SemArt/semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="all")
        val_dataset = sa_dataset_kg(['./data/SemArt/semart_val.json'], './data/SemArt/embeddings_val.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="all")
        test_dataset = sa_dataset_kg(['./data/SemArt/semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="all")
        return train_dataset, val_dataset, test_dataset
    elif dataset=='new_semart_content':
        train_dataset = new_semart_dataset(['./data/SemArt/processed_new_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="content")
        val_dataset = new_semart_dataset(['./data/SemArt/new_semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="content")
        test_dataset = new_semart_dataset(['./data/SemArt/new_semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="content")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='new_semart_form':
        train_dataset = new_semart_dataset(['./data/SemArt/processed_new_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="form")
        val_dataset = new_semart_dataset(['./data/SemArt/new_semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="form")
        test_dataset = new_semart_dataset(['./data/SemArt/new_semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="form")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='new_semart_context':
        train_dataset = new_semart_dataset(['./data/SemArt/processed_new_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="context")
        val_dataset = new_semart_dataset(['./data/SemArt/new_semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="context")
        test_dataset = new_semart_dataset(['./data/SemArt/new_semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="context")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='new_semart_content_kg':
        train_dataset = new_semart_dataset_kg(['./data/SemArt/processed_new_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="content")
        val_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_val.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="content")
        test_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="content")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='new_semart_form_kg':
        train_dataset = new_semart_dataset_kg(['./data/SemArt/processed_new_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="form")
        val_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_val.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="form")
        test_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="form")
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='new_semart_context_kg':
        train_dataset = new_semart_dataset_kg(['./data/SemArt/processed_new_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'], type="context")
        val_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_val.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'], type="context")
        test_dataset = new_semart_dataset_kg(['./data/SemArt/new_semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'], type="context")
        return train_dataset, val_dataset, test_dataset
    elif dataset=='new_semart_allcaps':
        train_dataset = new_semart_allcaps_dataset(['./data/SemArt/SemArt_three_split/processed_new_semart_train.json'], train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = new_semart_allcaps_dataset(['./data/SemArt/new_semart_val.json'], test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = new_semart_allcaps_dataset(['./data/SemArt/new_semart_test.json'], test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    elif dataset=='new_semart_allcaps_kg':
        train_dataset = new_semart_allcaps_kg_dataset(['./data/SemArt/processed_new_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = new_semart_allcaps_kg_dataset(['./data/SemArt/new_semart_val.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = new_semart_allcaps_kg_dataset(['./data/SemArt/new_semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    elif dataset=='semart_visual_contextual':
        train_dataset = semart_visual_contextual(['./data/SemArt/processed_semart_train.json'], './data/SemArt/embeddings_train.json', train_transform, './data/SemArt/Images',
                                     max_words=config['max_length'], read_local_data=config['read_local_data'],
                                     is_train=True, add_object=config['add_object'])
        val_dataset = semart_visual_contextual(['./data/SemArt/semart_val.json'], './data/SemArt/embeddings_val.json', test_transform, './data/SemArt/Images',
                                   max_words=config['max_length'], read_local_data=config['read_local_data'],
                                   is_train=False, add_object=config['add_object'])
        test_dataset = semart_visual_contextual(['./data/SemArt/semart_test.json'], './data/SemArt/embeddings_test.json', test_transform, './data/SemArt/Images',
                                    max_words=config['max_length'], read_local_data=config['read_local_data'],
                                    is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    
    # elif dataset=='artcap':
    #     train_dataset = artcap_dataset(['./data/ArtCap/artpedia_train_visual.json'], train_transform, './data/Artpedia/images/data',
    #                                  max_words=config['max_length'], read_local_data=config['read_local_data'],
    #                                  is_train=True, add_object=config['add_object'])
    #     val_dataset = artcap_dataset(['./data/Artpedia/artpedia_val_visual.json'], test_transform, './data/Artpedia/images/data',
    #                                max_words=config['max_length'], read_local_data=config['read_local_data'],
    #                                is_train=False, add_object=config['add_object'])
    #     test_dataset = artcap_dataset(['./data/Artpedia/artpedia_test_visual.json'], test_transform, './data/Artpedia/images/data',
    #                                 max_words=config['max_length'], read_local_data=config['read_local_data'],
    #                                 is_train=False, add_object=config['add_object'])
    #     return train_dataset, val_dataset, test_dataset
def videoqa_collate_fn(batch):
    image_list, question_list, answer_list, n = [], [], [], []
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(answer)
        n.append(1)
    return torch.stack(image_list,dim=0), question_list, answer_list, n

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def nocaps_collate_fn(batch):
    image_list, image_id_list = [], []
    for image, image_id in batch:
        image_list.append(image)
        image_id_list.append(image_id)
    return torch.stack(image_list,dim=0), image_id_list


def coco_collate_fn(batch):
    image_list, caption_list, metadata_list, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, metadata, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        metadata_list.append(metadata)
    return torch.stack(image_list,dim=0), caption_list, metadata_list, image_id_list, gold_caption_list

def semart_kg_collate_fn(batch):
    image_list, caption_list, metadata_list, image_id_list, gold_caption_list, embedding_list = [], [], [], [], [], []
    for image, caption, metadata, image_id, gold_caption, embedding in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        metadata_list.append(metadata)
        embedding_list.append(embedding)
    return torch.stack(image_list,dim=0), caption_list, metadata_list, image_id_list, gold_caption_list, embedding_list

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

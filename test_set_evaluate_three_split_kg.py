import argparse
import os
import yaml
import language_evaluation
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_caption_kg_fusion import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, coco_collate_fn, semart_kg_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer, create_two_optimizer_without_metadata
from tqdm import tqdm
import wandb


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 5
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, (image, caption, metadata, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        if config['prompt'] != "":
            caption = [config['prompt'] + each+config['eos'] for each in caption]
        else:
            caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']+" "+each for each in metadata]

        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        # question_input = caption.input_ids[0,0].repeat(caption.input_ids.size(0), 1)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, caption, train=True)

        if accum_steps > 1:
            loss = loss / accum_steps
        wandb.log({'loss': loss})
        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        del image, question_input,caption,loss 

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate test result:'
    print_freq = 50

    result = []


    answer_input = None
    for n, (image, caption, metadata, image_ids, gold_caption, embedding) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        # question_input = [config['bos']+" "+each for each in metadata_list]
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt", add_special_tokens=False).to(device)
        metadata = tokenizer(metadata, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        topk_ids_content, topk_probs_content, topk_ids_form, topk_probs_form, topk_ids_context, topk_probs_context = model(image, metadata, caption, embedding, train=False)

        for image_id, topk_id_content, topk_prob_content, topk_id_form, topk_prob_form, topk_id_context, topk_prob_context, gold_caption_dict in zip(image_ids, topk_ids_content, topk_probs_content, topk_ids_form, topk_probs_form, topk_ids_context, topk_probs_context, gold_caption):
            for type in gold_caption_dict:
                gold_caption_list = gold_caption_dict[type]
                if gold_caption_list:
                    if type == "content":
                        ans = tokenizer.decode(topk_id_content[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace("<content>", "").replace("<form>", "").replace("<context>", "").strip() 
                        gold_caption_list = [i.replace("<content>", "").strip() for i in gold_caption_list]
                    elif type == "form":
                        ans = tokenizer.decode(topk_id_form[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace("<content>", "").replace("<form>", "").replace("<context>", "").strip()
                        gold_caption_list = [i.replace("<form>", "").strip() for i in gold_caption_list]
                    elif type == "context":
                        ans = tokenizer.decode(topk_id_context[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace("<content>", "").replace("<form>", "").replace("<context>", "").strip()
                        gold_caption_list = [i.replace("<context>", "").strip() for i in gold_caption_list]

                    result.append({"question_id":image_id, "type": type, "pred_caption":ans, "gold_caption":gold_caption_list})
    return result


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    predicts = []
    answers = []
    answer_input = None
    for n, (image, caption, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])
        #{'Bleu_1': 0.9999999999863945, 'Bleu_2': 0.9999999999859791, 'Bleu_3': 0.9999999999854866, 'Bleu_4': 0.999999999984889, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 2.7246232035629268, 'SPICE': 0.40389416048620613}
        result = cal_metric(predicts, answers)
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_2'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_3'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_4'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    wandb.log(results)
    print(len(result_list), results)
    return results


def main(args, config):
    # utils.init_distributed_mode(args)
    wandb.init(project="mcs-research-project")
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating datasets")
    datasets = create_dataset('new_semart_allcaps_kg', config)

    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    # else:
    samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[0,0,0],is_trains=[True, False, False],
                                              collate_fns=[semart_kg_collate_fn, semart_kg_collate_fn, semart_kg_collate_fn]) 


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)

    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        arg_opt['lr1'] = float(arg_opt['lr1'])
        arg_opt['lr2'] = float(arg_opt['lr2'])
        arg_opt['weight_decay'] = float(arg_opt['weight_decay'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        arg_opt['lr1'] = float(arg_opt['lr1'])
        arg_opt['lr2'] = float(arg_opt['lr2'])
        arg_opt['weight_decay'] = float(arg_opt['weight_decay'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['lr'] = float(arg_sche['lr'])
    arg_sche['min_lr'] = float(arg_sche['min_lr'])
    arg_sche['warmup_lr'] = float(arg_sche['warmup_lr'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    checkpoint = torch.load(args.checkpoint_dir, map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint['module']

    # reshape positional embedding to accomodate for image resolution change
    if config["clip_name"] == "ViT-B-16":
        num_patches = int(config["image_res"] * config["image_res"]/(16*16))
    elif config["clip_name"] == "ViT-L-14":
        num_patches = int(config["image_res"] * config["image_res"]/(14*14))
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
    pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                            pos_embed.unsqueeze(0))
    state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

    # if not args.evaluate:
    #     for key in list(state_dict.keys()):
    #         if ('fusion' in key or 'bert' in key) and 'decode' not in key:
    #             encoder_key = key.replace('fusion.', '').replace('bert.', '')
    #             state_dict[encoder_key] = state_dict[key]
    #             del state_dict[key]


    msg = model.load_state_dict(state_dict, strict=False)
    # print('load checkpoint from %s' % args.checkpoint)
    # print(msg)

    # model_without_ddp = model
    # if args.distributed:
    #     #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     import apex
    #     model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    #     model_without_ddp = model.module

    # print("Start training")
    # start_time = time.time()
    vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % i)
    #if utils.is_main_process():
    result = cal_metric(os.path.join(args.result_dir, result_file))
    # dist.barrier()
    # wandb.watch(model)
    
    # for epoch in range(start_epoch, max_epoch):
    #     if epoch > 0:
    #         lr_scheduler.step(epoch + warmup_steps)

            
    #     if not args.evaluate:
    #         if args.distributed:
    #             train_loader.sampler.set_epoch(epoch)

    #         train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
    #                             config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

    #     if args.evaluate:
    #         break

    #     vqa_result = evaluation(model, val_loader, tokenizer, device, config)
    #     result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)
    #     if utils.is_main_process():
    #         result = cal_metric(result_file)
    #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                      'epoch': epoch,
    #                      }
    #         with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #         torch.save({
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'lr_scheduler': lr_scheduler.state_dict(),
    #             'config': config,
    #             'epoch': epoch,
    #         }, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

    #     # dist.barrier()

    #     #vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    #     #result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config.yaml')
    parser.add_argument('--checkpoint_dir', default='./model_checkpoint/mplug_large_v2.pth')
    parser.add_argument('--output_dir', default='./evaluation/semart_contextual_processed')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='./bert_base_uncased')
    parser.add_argument('--text_decoder', default='./bert_base_uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=8, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--max_length', default=40, type=int)
    parser.add_argument('--max_input_length', default=40, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--do_two_optim', default=True)
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=2, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    #config['optimizer']['lr'] = args.lr
    #config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
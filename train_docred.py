import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch

import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model_balanceloss import DocREModel
from utils_sample import set_seed, collate_fn
from evaluation import to_official, official_evaluate
from prepro import ReadDataset



def train(args, model, train_features, dev_features, test_features):
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(args.log_dir, 'a+') as f_log:
                f_log.write(s + '\n')
    def finetune(features, optimizer, num_epoch, num_steps, model):
        cur_model = model.module if hasattr(model, 'module') else model
        if args.train_from_saved_model != '':
            best_score = torch.load(args.train_from_saved_model)["best_f1"]
            epoch_delta = torch.load(args.train_from_saved_model)["epoch"] + 1
        else:
            epoch_delta = 0
            best_score = -1
        test_best_score = -1
        dev_best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = [epoch + epoch_delta for epoch in range(num_epoch)]
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)#3053/4*30=22890
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        global_step = 0
        log_step = 100
        total_loss = 0
        


        #scaler = GradScaler()
        for epoch in train_iterator:
            start_time = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                model.train()

                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                #with autocast():
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                total_loss += loss.item()
                #    scaler.scale(loss).backward()
               

                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    #scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(cur_model.parameters(), args.max_grad_norm)
                    #scaler.step(optimizer)
                    #scaler.update()
                    #scheduler.step()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    num_steps += 1
                    if global_step % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        logging(
                            '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f}'.format(
                                epoch, global_step, elapsed / 60, scheduler.get_last_lr(), cur_loss * 1000))
                        total_loss = 0
                        start_time = time.time()

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                # if step ==0:
                    logging('-' * 89)
                    eval_start_time = time.time()
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    logging(
                        '| epoch {:3d} | time: {:5.2f}s | dev_result:{}'.format(epoch, time.time() - eval_start_time,
                                                                                dev_output))
                    logging('-' * 89)
                    if dev_score > dev_best_score:
                        dev_best_score = dev_score
                        logging(
                            '| epoch {:3d} | dev_best_f1:{}'.format(epoch, dev_best_score))
                        pred = report(args, model, test_features)
                        test_save_path= "./submit_result/best_result"+str(step)+".json"
                        with open(test_save_path, "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            save_path=args.save_path+str(step)+".pt"
                            torch.save({
                                'epoch': epoch,
                                'checkpoint': cur_model.state_dict(),
                                'best_f1': best_score,
                                'optimizer': optimizer.state_dict()
                            }, save_path
                            , _use_new_zipfile_serialization=False)
        print(f"dev最佳结果是{dev_best_score}")
        return num_steps

    cur_model = model.module if hasattr(model, 'module') else model
    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ] # 分别是BERT的参数、头尾抽取和双线性层的参数、U型网络的参数

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.train_from_saved_model != '':
        optimizer.load_state_dict(torch.load(args.train_from_saved_model)["optimizer"])
        print("load saved optimizer from {}.".format(args.train_from_saved_model))
    

    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, model)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            total_loss += loss.item()

    average_loss = total_loss / (i + 1)
    preds = np.concatenate(preds, axis=0).astype(np.float32)#(396790, 97)
    ans = to_official(preds, features)
    best_f1, best_f1_ign, re_p, re_r=0,0,0,0
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate(ans, args.data_dir)

    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_re_p": re_p * 100,
        tag + "_re_r": re_r * 100,
        tag + "_average_loss": average_loss
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="./checkpoint/docred/", type=str)
    parser.add_argument("--load_path", default="", type=str)
    # parser.add_argument("--load_path", default=r"C:\Users\dell\Desktop\all\code\docrel\Docunet\DocuNet-main\DocuNet_psp\checkpoint\docred\1.pt", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=3, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_lr", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                         help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    parser.add_argument("--unet_in_dim", type=int, default=3,
                        help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=64,
                        help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256,
                        help="down_dim.")
    parser.add_argument("--channel_type", type=str, default='context-based',
                        help="unet_out_dim.")
    parser.add_argument("--log_dir", type=str, default='./logs/docred/1.log',
                        help="log.")
    parser.add_argument("--max_height", type=int, default=48,
                        help="log.")
    parser.add_argument("--train_from_saved_model", type=str, default='',
                        help="train from a saved model.")
    parser.add_argument("--dataset", type=str, default='docred',
                        help="dataset type")

    args = parser.parse_args()
    print('args:',args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    Dataset = ReadDataset(args.dataset, tokenizer, args.max_seq_length)

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = Dataset.read(train_file)
    dev_features = Dataset.read(dev_file)
    test_features = Dataset.read(test_file)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, args,  model, num_labels=args.num_labels)
    if args.train_from_saved_model != '':
        model.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(args.train_from_saved_model))
    
    if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model.to(device)


    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path)['checkpoint'])
        T_features = test_features  # Testing on the test set
        #T_score, T_output = evaluate(args, model, T_features, tag="test")
        pred = report(args, model, T_features)
        with open("./submit_result/result.json", "w") as fh:
            json.dump(pred, fh)
        print(":over")

if __name__ == "__main__":
    main()

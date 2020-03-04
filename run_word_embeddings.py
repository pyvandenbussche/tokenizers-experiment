# coding=utf-8
""" Fine-tuning the library models for named entity recognition """
from __future__ import absolute_import, division, print_function

import argparse
import beautifultable as bt
import pandas as pd
import spacy
import logging
import os
import random
import torch
import time

from scipy.spatial.distance import cosine
import numpy as np
# import torch
import transformers
from transformers import *

# to get our ideas clear: http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer)
}

nlp = spacy.load("en_core_web_sm")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def load_data(data_fp):
    df = pd.read_csv(data_fp, sep=',', header=0, encoding="ISO-8859-1")
    df.drop('Body', axis=1, inplace=True)
    return df


def get_word_token(sentence):
    doc = nlp(sentence)
    # i=1
    # all_tokens_ids = []
    # data = []
    # for word in doc:
    #     tokenized_text = tokenizer.tokenize(word.text)
    #     tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    #     all_tokens_ids.extend(tokens_ids)
    #     data.append({"word":word.text,"w_start":word.idx,"w_end":word.idx+len(word.text),
    #                  "tokens":tokenized_text,"tokens_id":tokens_ids, "t_start":i, "t_end":i+len(tokens_ids)})
    #     i+= len(tokens_ids)
    return [word.text for word in doc]


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data", default=None, type=str, required=False,
                        help="File path to the input data.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")


    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="Set this flag to test the legacy tokenizer (Transformers<2.5.2).")


    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          output_hidden_states=True, output_attentions=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, use_fast=not args.use_slow_tokenizer)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config) #, output_hidden_states=True,)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)



    df = load_data(args.data)

    # df['q_tokens'] = df['Title'].apply(lambda x: nlp.tokenizer(x))
    titles = df['Title'].tolist()
    print(type(titles))




    time1 = time.time()
    output = tokenizer.batch_encode_plus(titles, return_offsets_mapping= not args.use_slow_tokenizer)
    print(output.keys())
    time2 = time.time()
    df['q_token_ids'] = output["input_ids"]
    df['q_token_offset'] = output["offset_mapping"]
    print('tokenizer function took {:.3f} ms'.format((time2 - time1) * 1000.0))

    print(df.head())


    # Encode text
    text = "COS-7 cells were cultured in DMEM (Mediatech, Inc., Herndon, Virginia) supplemented with 10% fetal bovine serum (Atlanta Biologicals, Lawrenceville, Georgia) and maintained in the 37°C incubator with 5% CO2. H1299 and H358 non-small cell lung cancer cell lines were purchased from American Type Culture Collection (ATCC) and were maintained in RPMI medium (ATCC) containing 10% fetal bovine serum."
    markers = [(29,33), (344,348)]
    # text = "H1299 and H358 non-small cell lung cancer cell lines were purchased from American Type Culture Collection (ATCC) and were maintained in RPMI medium (ATCC) containing 10% fetal bovine serum. All cell lines except GM00847 were grown in RPMI medium containing 10% fetal calf serum."

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    i=1
    all_tokens_ids = []
    data = []
    for word in doc:
        tokenized_text = tokenizer.tokenize(word.text)
        tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        all_tokens_ids.extend(tokens_ids)
        data.append({"word":word.text,"w_start":word.idx,"w_end":word.idx+len(word.text),
                     "tokens":tokenized_text,"tokens_id":tokens_ids, "t_start":i, "t_end":i+len(tokens_ids)})
        i+= len(tokens_ids)

    df = pd.DataFrame(data=data, columns=["word", "w_start", "w_end", "tokens", "tokens_id", "t_start", "t_end"])
    print(df.head(10))

    data_markers = []
    for (start, end) in markers:
        df_temp = df[((df["w_start"]<= start) & (df["w_end"]>= start)) |
                 ((df["w_start"]>= start) & (df["w_end"]<= end)) |
                 ((df["w_start"]<= end) & (df["w_end"]>= end))]
        # print(df_temp)
        # print(df_temp["t_start"].min())
        # print(df_temp["t_end"].max())
        data_markers.append({"annotation":text[start:end], "a_start":start, "a_end":end,
                             "t_start":df_temp["t_start"].min(), "t_end":df_temp["t_end"].max()})
    df_markers = pd.DataFrame(data=data_markers, columns=["annotation", "a_start", "a_end", "t_start", "t_end"])
    print(df_markers.head())

    last_hidden_state, pooler_output, hidden_states, attentions = model(torch.tensor([tokenizer.encode(all_tokens_ids, add_special_tokens=True)]))

    # stack the tensors from tuples to a single tensor
    # of shape: [# layers, # batches, # tokens, # features]
    hidden_states = torch.stack(hidden_states)
    # print(hidden_states.size())

    # squeeze the batch
    # shape: [# layers, # tokens, # features]
    hidden_states = hidden_states.squeeze()
    # print(hidden_states.size())

    # switch dimensions 0 and 1
    # shape: [# tokens, # layers, # features]
    hidden_states = hidden_states.permute(1, 0, 2)
    # print(hidden_states.size())


    # Stores the token vectors, with shape [24 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [24 x 13 x 768] tensor.

    # For each token in the sentence...
    for token in hidden_states:
        # `token` is a [13 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-5:-1], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec.detach().numpy())
    # print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    annotation1 = df_markers.iloc[0]
    annotation2 = df_markers.iloc[1]
    annotation1_mean = np.mean(token_vecs_sum[annotation1["t_start"]:annotation1["t_end"]], axis=0)
    token_rpmi_mean = np.mean(token_vecs_sum[annotation2["t_start"]:annotation2["t_end"]], axis=0)

    cos = 1 - cosine(annotation1_mean, token_rpmi_mean)
    print("Vector similarity between {} and {} meanings: {:.4f}".format(annotation1["annotation"],
                                                                         annotation2["annotation"], cos))



if __name__ == "__main__":
    main()

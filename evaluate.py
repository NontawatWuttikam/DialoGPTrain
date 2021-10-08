import train
import pandas as pd
from typing import Dict, List, Tuple
import os 
import logging
import torch
from tqdm import tqdm
diR = "microsoft/DialoGPT-medium"
args = train.Args()
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
logger = logging.getLogger(__name__)

def construct_conv(row, tokenizer, eos = True):
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

class ConversationDataset(train.Dataset):
    def __init__(self, tokenizer: train.PreTrainedTokenizer, args, df, block_size=1024):
        # block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                if len(conv) > block_size: continue
                self.examples.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            # logger.info("Saving features into cached file %s", cached_features_file)
            # with open(cached_features_file, "wb") as handle:
            #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
    return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)

def evaluate(args, model: train.PreTrainedModel, tokenizer: train.PreTrainedTokenizer, df_trn, df_val, prefix="",write_file = True) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return train.pad_sequence(examples, batch_first=True)
        return train.pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = train.SequentialSampler(eval_dataset)
    eval_dataloader = train.DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last = True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}
    print(result)
    if write_file:
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            if logger is not None:
                logger.info("***** Eval results {} *****".format(prefix))
            else: print("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                if logger is not None:
                    logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result

tokenizer = train.AutoTokenizer.from_pretrained(diR)
model = train.AutoModelWithLMHead.from_pretrained(diR)

df = pd.read_excel('train_sessions.xlsx')

trn_df, val_df = train.train_test_split(df, test_size = 0.1)

tokenizer = train.AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
lens, tok_cnt, word_cnt = train.get_counter_and_lens(trn_df[df.columns].apply(lambda x: ' '.join(x.astype(str)), axis = 1), tokenizer)

from torch.utils.tensorboard import SummaryWriter

trn_df = trn_df.drop('Unnamed: 0',axis=1)
val_df = val_df.drop('Unnamed: 0',axis=1)

idx_to_rm_trn = []
idx_to_rm_val = []

for k,row in trn_df.iterrows():
    for i, item in row.iteritems():
        if item is None: idx_to_rm_trn.append(k); break
        elif type(item) is not str: idx_to_rm_trn.append(k); break
        elif item == "": idx_to_rm_trn.append(k); break

for k,row in val_df.iterrows():
    for i, item in row.iteritems():
        if item is None: idx_to_rm_val.append(k); break
        elif type(item) is not str: idx_to_rm_val.append(k); break
        elif item == "": idx_to_rm_val.append(k); break

trn_df = trn_df.drop(idx_to_rm_trn, axis = 0)
val_df = val_df.drop(idx_to_rm_val, axis = 0)

print("val DF", val_df.shape)
device = torch.device("cpu")
args.n_gpu = 1
args.device = device
evaluate(args, model, tokenizer, trn_df,val_df,write_file=False)
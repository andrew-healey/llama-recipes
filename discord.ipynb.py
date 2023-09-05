%cd ~
!git clone https://github.com/facebookresearch/llama-recipes.git
%cd llama-recipes/

%%bash
# pip install -qqq transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets
# TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
# python ${TRANSFORM} --input_dir models --model_size 7B --output_dir models_hf/7B

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_id="./models_hf/7B"

tokenizer = LlamaTokenizer.from_pretrained(model_id)

model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

from pathlib import Path
import os
import sys
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import discord_dataset

train_dataset = get_preprocessed_dataset(tokenizer, discord_dataset, 'train')



from transformers.generation.logits_process import LogitsProcessorList
import torch
from typing import Tuple
def completion_to_logits(prompt:str,completion:str)->torch.Tensor:
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    target_output = tokenizer(completion, return_tensors="pt")["input_ids"].to("cuda")
    print("target",target_output)

    vocab_size = tokenizer.vocab_size
    num_tokens_to_produce = target_output.shape[1]

    out_logits=torch.zeros((num_tokens_to_produce,1,vocab_size)).to("cuda")
    arange = torch.arange(vocab_size).to("cuda")

    curr_idx = 0
    def logit_collect(input_ids, scores)->torch.Tensor:
        nonlocal curr_idx
        # save logits to out_logits
        out_logits[curr_idx] = scores
        curr_idx += 1

        # set all scores to -inf except the target token
        target_token = input_ids[0,curr_idx]
        scores[:,arange != target_token] = -float("inf")
        return scores
    
    model.generate(**model_input, max_new_tokens=num_tokens_to_produce, logits_processor=LogitsProcessorList([logit_collect]))

    return out_logits
out_logits = completion_to_logits("I like to eat", " pizza")

def logits_to_real(completion: str,logits: torch.Tensor):
    target_output = tokenizer(completion, return_tensors="pt")["input_ids"].to("cuda")
    num_tokens_to_produce = target_output.shape[1]
    out_probs = torch.softmax(logits, dim=-1)

    real = 0

    cumsums = torch.cumsum(out_probs, dim=-1)

    # add 0 to the start of cumsums
    print("cumsums",cumsums.shape)
    cumsums = torch.cat([torch.zeros((num_tokens_to_produce,1,1)).to("cuda"), cumsums], dim=-1)

    interval_starts = cumsums[torch.arange(num_tokens_to_produce),0,target_output[0]]
    interval_ends = cumsums[torch.arange(num_tokens_to_produce),0,target_output[0]+1]
    interval_lengths = interval_ends - interval_starts

    print("Interval starts:",interval_starts)
    print("Interval ends:",interval_ends)
    print("Real num:",(interval_lengths.cumprod(0) * interval_starts).sum().item())
    print("Probs",interval_lengths)

    return interval_starts, interval_ends

def reals_to_completion(prompt:str,interval_starts: torch.Tensor, interval_ends: torch.Tensor)->str:
    # an explicit (but numerically-unstable) algorithm is this:
    
    curr_interval_starts = interval_starts.clone()
    curr_interval_ends = interval_ends.clone()

    curr_interval_start = 0
    curr_interval_end = 1

    arange = torch.arange(tokenizer.vocab_size).to("cuda")

    did_finish_decoding = False
    has_printed = False

    def logit_decode(input_ids,scores)->Tuple[str,bool]:
        nonlocal curr_interval_starts, curr_interval_ends, did_finish_decoding, curr_interval_start, curr_interval_end, has_printed

        # get cumsum of probs
        out_probs = torch.softmax(scores, dim=-1)
        cumsums = torch.cumsum(out_probs, dim=-1)
        # add 1 to end of cumsums
        cumsums = torch.cat([cumsums, torch.ones((1,1)).to("cuda")], dim=-1)

        token_interval_starts = cumsums[0,:-1]
        token_interval_ends = cumsums[0,1:]

        num_intervals_consumed = 0

        # while the curr_interval doesn't fit inside any single token interval:
        def fits_in_token_interval():
            return ((curr_interval_start >= token_interval_starts) & (curr_interval_end <= token_interval_ends)).any()
        while num_intervals_consumed < len(curr_interval_starts) and not fits_in_token_interval():
            # find the token interval that contains the curr_interval
            # (this is the first token interval that starts after the curr_interval)
            curr_interval_length = curr_interval_end - curr_interval_start
            curr_interval_end = curr_interval_start + curr_interval_ends[num_intervals_consumed] * curr_interval_length
            curr_interval_start = curr_interval_start + curr_interval_starts[num_intervals_consumed] * curr_interval_length

            num_intervals_consumed += 1
        
        if fits_in_token_interval():
            # find the token
            # convert these two token masks to a str
            ge_mask = curr_interval_start >= token_interval_starts
            le_mask = curr_interval_end <= token_interval_ends
            # format as 0001111...
            ge_str = "".join(["1" if x else "0" for x in ge_mask])
            le_str = "".join(["1" if x else "0" for x in le_mask])
            if(not has_printed): print(ge_str+"\n"+le_str+"\n"+str(num_intervals_consumed))

            token_mask = (curr_interval_start >= token_interval_starts) & (curr_interval_end <= token_interval_ends)
            token_idx = token_mask.nonzero()[0,0]

            # normalize the curr_interval to the token interval
            curr_interval_length = curr_interval_end - curr_interval_start
            curr_interval_end = (curr_interval_end - token_interval_starts[token_idx]) / (token_interval_ends[token_idx] - token_interval_starts[token_idx])
            curr_interval_start = (curr_interval_start - token_interval_starts[token_idx]) / (token_interval_ends[token_idx] - token_interval_starts[token_idx])
            print("new interval",curr_interval_start,curr_interval_end)

            curr_interval_starts = curr_interval_starts[num_intervals_consumed:]
            curr_interval_ends = curr_interval_ends[num_intervals_consumed:]

            # print decoded token
            # if(not has_printed): print(tokenizer.decode([token_idx,token_idx-1,token_idx+1]))

            # set all scores to -inf except the target token
            print("token_idx",token_idx)
            print(token_interval_starts[token_idx],token_interval_ends[token_idx])
            scores[:,arange != token_idx] = -float("inf")
            did_finish_decoding = True
            has_printed = True

            return scores
        
        else:
            # return EOS
            scores[:,tokenizer.eos_token_id] = 0
            scores[:,arange != tokenizer.eos_token_id] = -float("inf")
            return scores

    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**model_input, logits_processor=LogitsProcessorList([logit_decode]))

    output_str = tokenizer.decode(output[0])
    return output_str, did_finish_decoding

prompt = "I like to eat "
completion = "pizza"
reals_to_completion(prompt, *logits_to_real(completion, completion_to_logits(prompt, completion)))


eval_prompt = """
not fox (15:43 PM): anything i should know
semantic_zone (15:44 PM): Idts
semantic_zone (14:52 PM): I'm gonna hope that you're online here
semantic_zone (14:52 PM): can you please send a screenshot of the newspaper editor/something else that is done?
semantic_zone (16:51 PM): 😮
not fox (16:52 PM): yeah, michael and a bunch of my friends started preassuring me
not fox (16:52 PM): so i finally gave in
semantic_zone (16:57 PM): michael who?
semantic_zone (16:57 PM): no judgement
it's a good game
not fox (16:57 PM): illie
not fox (16:58 PM): i died on the first night
semantic_zone (16:58 PM): Ilie convinced you to play video games?
not fox (16:58 PM): and a bunch of ppl from cty
semantic_zone (16:58 PM): 
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


model.train()

input_dir= "tmp/llama-output-5"
use_ckpt = True

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
        PeftModel,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    if use_ckpt:
        model = PeftModel.from_pretrained(model,input_dir,trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)


from transformers import TrainerCallback
from contextlib import nullcontext
enable_profiler = False
output_dir = "tmp/llama-output-6"

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


from transformers import default_data_collator, Trainer, TrainingArguments


# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()


model.save_pretrained(output_dir)


model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

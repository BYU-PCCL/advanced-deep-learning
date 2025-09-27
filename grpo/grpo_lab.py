from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.inputs import TokensPrompt
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import wandb
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "Qwen/Qwen2.5-3B"
gen_device = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 500
Q_batch_size = 1
num_pre_Q = 3
train_batch_size = 1
gen_update_steps = 6
save_steps = 50
compute_gen_logps = True
clip_param = 0.2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ref-server-port', type=int, default=59875)
args, unknown = parser.parse_known_args()
ref_server = f"http://localhost:{args.ref_server_port}"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "bf16": {"enabled": True},
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch, tokenizer_for_padding):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer_for_padding.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.3, max_model_len=4096, dtype="float16")
    vllm_tokenizer = vllm_gen.get_tokenizer()
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            tip_text.append(vllm_tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [];  ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    from math_verify import parse, verify, ExprExtractionConfig
    def reward_correct(item, answer):
        pattern = r'\d+\.\d+|\d+/\d+|\d+'
        nums = re.findall(pattern, answer) 
        if len(nums) == 0: return -1.0
        lastnum = nums[-1]
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        return 1 if verify(ans, ground_truth) else -1
    def reward_format(item, answer):
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1


    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                rewards.append(reward_correct(inp, a) + reward_format(inp, a))
        prompts_text = [vllm_tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        if it % 5 == 0: print('answers:', answers[0])

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = vllm_tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if ref_server_ver == 'tensor':
                # Preserve unnormalized rewards for logging/analytics
                curr_rewards_unnorm = curr_rewards.clone()
                # Compute format-only accuracy for each generated answer in this group
                # 1 if matches required <think>...</think><answer>...</answer> format, else 0
                fmt_flags = []
                for a in curr_answers:
                    fmt_flags.append(1 if reward_format(None, a) > 0 else 0)
                # Normalize rewards for training stability
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_rewards_unnorm = curr_rewards_unnorm[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    sub_fmt_flags = fmt_flags[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=vllm_tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    # Include unnormalized rewards and format accuracy in JSON header for training-side logging
                    data = [json.dumps({
                                "plen": plen,
                                "unnormalized_rewards": sub_rewards_unnorm.tolist(),
                                "format_accuracy": float(np.mean(sub_fmt_flags)) if len(sub_fmt_flags) > 0 else None
                            }).encode(),
                            tensor_to_bytes(merged_ids),
                            tensor_to_bytes(sub_rewards)]       

                    if compute_gen_logps:
                        zz = vllm_gen.generate(
                            TokensPrompt(prompt_token_ids=merged_ids[0].tolist()),
                            sampling_params=gen_logps_sp,
                            use_tqdm=False
                        )
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'


# Use vLLM's tokenizer for consistency - will be passed from gen_worker
# tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer
    )

    # Initialize Weights & Biases (rank 0 only)
    if dist.get_rank() == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "grpo-math"),
            name=os.environ.get("WANDB_RUN_NAME", f"{os.path.basename(__file__)}"),
            config={
                "model_path": model_path,
                "beta": beta,
                "all_steps": all_steps,
                "Q_batch_size": Q_batch_size,
                "num_pre_Q": num_pre_Q,
                "train_batch_size": train_batch_size,
                "gen_update_steps": gen_update_steps,
                "clip_param": clip_param,
                "compute_gen_logps": compute_gen_logps,
            },
        )

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        # For now, use a fallback tokenizer for pad_token_id - ideally pass vLLM tokenizer
        fallback_tokenizer = AutoTokenizer.from_pretrained(model_path)
        loss = GRPO_step(batch, fallback_tokenizer)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            current_loss = loss.item()
            current_reward = np.mean(batch.get('unnormalized_rewards', [0.0]))
            current_format_acc = batch.get('format_accuracy', None)
            progress.set_description(f"Loss: {current_loss:.6f}, Avg Reward: {current_reward:.3f}")
            log_payload = {
                "train/loss": current_loss,
                "train/avg_reward_raw": current_reward,
                "train/step": step,
            }
            if current_format_acc is not None:
                log_payload["train/format_accuracy"] = current_format_acc
            wandb.log(log_payload)

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        # if step % save_steps == 0:
        #     dist.barrier()
        #     if dist.get_rank() == 0:
        #         print('saving model')
        #         save_name = f"./step_{step}"
        #         state_dict = engine.module.state_dict()
        #         state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
        #         engine.module.save_pretrained(save_name, state_dict=state_dict)
        #         fallback_tokenizer.save_pretrained(save_name)
        #     dist.barrier()

    # Finalize Weights & Biases
    if dist.get_rank() == 0:
        wandb.finish()

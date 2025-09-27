import torch
import re
from math_verify import parse, verify, ExprExtractionConfig

def get_per_token_logps(logits, input_ids):
    """CPU version of get_per_token_logps for testing."""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def reward_correct(item, answer):
    """Same as in grpo_lab.py."""
    # TODO: Implement reward_correct function
    pass

def reward_format(item, answer):
    """Same as in grpo_lab.py."""
    # TODO: Implement reward_format function
    pass

def recompute_grpo_loss(vars_dict):
    """Recompute GRPO loss on CPU using saved variables."""
    # Global Variables
    pad_token_id = vars_dict['pad_token_id']
    clip_param = vars_dict['clip_param']
    beta = vars_dict['beta']
    compute_gen_logps = vars_dict['compute_gen_logps']
    
    # Local Variables
    prompt_length = vars_dict['prompt_length']
    inputs = vars_dict['inputs']
    advantages = vars_dict['advantages']
    logits = vars_dict['logits']
    input_ids = inputs[:, 1:]
    refs_per_token_logps = vars_dict['ref_per_token_logps']
    gen_logps = vars_dict.get('gen_logps', None) if compute_gen_logps else None
    assert gen_logps is not None
    
    # TODO: Implement recomputation of GRPO loss (NOTE: This may change slightly when not CPU).
    pass

if __name__ == '__main__':
    print("Running GRPO lab tests...")
    print("=" * 40)
    print("Recomputing GRPO loss from saved variables...")
    # Load saved variables
    vars_dict = torch.load('grpo_step_vars.pth', weights_only=True)
    expected_loss = vars_dict['loss']
    
    # Recompute loss on CPU
    computed_loss = recompute_grpo_loss(vars_dict)
    
    print(f"Expected Loss: {expected_loss.item():.6f}")
    print(f"Computed Loss: {computed_loss.item():.6f}")
    print(f"Match: {torch.allclose(expected_loss, computed_loss, atol=1e-3)}")
    print("=" * 40)
    
    # Tests for reward functions
    test_item = {'Q': 'What is 2+2?', 'A': '4'}
    
    # Test reward_correct
    print(f"Testing reward_correct...")
    assert reward_correct(test_item, '<think>2+2=4</think><answer>4</answer>') == 1
    assert reward_correct(test_item, '<think>2+2=5</think><answer>5</answer>') == -1
    assert reward_correct(test_item, 'No number') == -1
    print("reward_correct tests passed")
    print("=" * 40)
    
    # Test reward_format
    print(f"Testing reward_format...")
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer>') == 1.25
    assert reward_format(test_item, '<think>reasoning</think> <answer>answer</answer>') == 1.25  # with space
    assert reward_format(test_item, '<think>reasoning<answer>answer</answer>') == -1  # missing </think>
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer><think>extra</think>') == -1  # extra tags
    print("reward_format tests passed")
    print("=" * 40)

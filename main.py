from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_next_token_contrastive_decoding(text, expert_model, amateur_model, tokenizer, alpha=0.1):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    
    # Get model outputs
    with torch.no_grad():
        outputs_expert = expert_model(**inputs)
        outputs_amateur = amateur_model(**inputs)

    # Get logits for the last token
    logits_expert = outputs_expert.logits[0, -1, :]
    logits_amateur = outputs_amateur.logits[0, -1, :]

    # Compute the probablities
    probabilities_expert = torch.nn.functional.softmax(logits_expert, dim=-1)
    probabilities_amateur = torch.nn.functional.softmax(logits_amateur, dim=-1)

    # Compute V_head
    p_max = torch.max(probabilities_expert)
    print(f"p_max: {p_max}")

    # Select all indices where the probability is greater than alpha * p_max
    indices = torch.where(probabilities_expert > alpha * p_max)[0]

    # Get the index where the ratio expert / amateur is maximal
    ratios = probabilities_expert.log() - probabilities_amateur.log()
    mask = torch.ones_like(ratios, dtype=torch.bool)
    mask[indices] = False
    ratios[mask] = -float('inf') # ignore indices not in V_head

    # Get the index with the highest ratio
    next_token_index = torch.argmax(ratios)

    # Get the token
    next_token = tokenizer.decode([next_token_index.item()])

    return next_token, probabilities_expert[next_token_index].item(), probabilities_amateur[next_token_index].item()
    

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # I had some issues with MPS on my Mac, so I used the cpu for now
        return "cpu"
    else:
        return "cpu"

def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    amateur_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    expert_model = "Qwen/Qwen2.5-3B-Instruct"
    
    # Both models should have the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(amateur_model, trust_remote_code=True)
    amateur_model = AutoModelForCausalLM.from_pretrained(
        amateur_model,
        device_map=None,  # Disable automatic device mapping
        trust_remote_code=True
    ).to(device)

    expert_model = AutoModelForCausalLM.from_pretrained(
        expert_model,
        device_map=None,  # Disable automatic device mapping
        trust_remote_code=True
    ).to(device)
    
    # Example text
    text = "Barack Obama was born in Honolulu, Hawaii. He was born in"
    print("\nInput text:")
    print(text)
    
    # Get next token predictions
    #predictions = get_next_token_logits(text, amateur_model, tokenizer)

    for i in range(10):
        prediction_cl = get_next_token_contrastive_decoding(text, expert_model, amateur_model, tokenizer)
        print(f"Next token: {prediction_cl[0]}")
        print(f"Probability expert: {prediction_cl[1]}")
        print(f"Probability amateur: {prediction_cl[2]}")
        text += prediction_cl[0]

if __name__ == "__main__":
    main()

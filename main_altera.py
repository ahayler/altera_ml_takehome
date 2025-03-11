import transformers as tr
from main import get_next_token_contrastive_decoding, get_device


def main():
	amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
	expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

	tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

	user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
		scores,
		results,
		kFactor = 4,
	) {
		for (const result of results) {
			const { first, second, outcome } = result;
			const firstScore = scores[first] ?? 1000;
			const secondScore = scores[second] ?? 1000;

			const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
			const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
			let sa = 0.5;
			if (outcome === 1) {
				sa = 1;
			} else if (outcome === -1) {
				sa = 0;
			}
			scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
			scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
		}
		return scores;
	}\n```"""

	prompt = tokenizer.apply_chat_template(
		[
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": user_message},
		],
		add_generation_prompt=True,
		tokenize=False,
	)
	device = get_device()

	amateur_model = tr.AutoModelForCausalLM.from_pretrained(
		amateur_path,
		device_map=None,
		trust_remote_code=True
	).to(device)

	expert_model = tr.AutoModelForCausalLM.from_pretrained(
		expert_path,
		device_map=None,
		trust_remote_code=True
	).to(device)
     
	generation = contrastive_generation(amateur_model, expert_model, prompt, tokenizer, 10)
	print(generation)


def contrastive_generation(amateur, expert, prompt, tokenizer, max_tokens) -> str:
    
	# Get the next token using contrastive decoding
	for _ in range(max_tokens):
		next_token, _, _ = get_next_token_contrastive_decoding(prompt, expert, amateur, tokenizer)
		prompt += next_token
		
		if next_token == tokenizer.eos_token:
			break

	return prompt

if __name__ == "__main__":
    main()
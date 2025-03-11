1. **What should you do if the two models have different tokenizers?**

In the end, to compute the CD-scores we will need $p_{\text{EXE}}$ and $p_{\text{AMA}}$ to live on the same probability space to compute the fractions over the domain. Changing the tokenizer of one of the LLMs would require significant retraining. The following would be an idea to unify the probability spaces without retraining (This idea definitely has some problems, see end; I have kept it in both for completeness and because I think that with further considerations it still might be useful):

Take the set of possible tokens of the expert $T_E$ and amateur $T_A$ and form their union $T=T_A \cup T_E$. For each $t \in T$, we can compute its probability under $p_{\text{EXE}}$ and $p_{\text{AMA}}$ by “multiplying” through. This is because each token $t$, will have a deterministic tokenization in each tokenizer. Still this method would be computationally very expensive as we will possibly need to query the amateur and expert models $O(|T|)$ of times (if we assume tokens $t$ are of bounded length and we have at most one token per character). 

If the tokenizer are very similar, the computational effort would be a lot lower. Additionally, we could do some more intelligent clipping the $\mathcal{V}_{\text {head }}$ formulation:

- We can compute $\max _w p_{\operatorname{EXP}}\left(w \mid x_{<i}\right)$ without every multiply through (as that can only decrease the probabilities)
- With this we can clip/discard most of the tokens before ever multiplying through

**Problem with this idea:** $p_{\text{EXE}}$ and $p_{\text{AMA}}$ don't define a probability distribution on $T$. The total probability mass over $T$ will generally be larger than $1$, as we could both include a the probability of a token $t$ itself and product involing this probability. Still, I think we the CD-scores we would get might still be enough to do useful sampling. I could imagine a amateur model with a "strictly weaker" tokenizer, i.e. $T_A \subset T_E$. In this case, we would only need to multiply through for $p_{\text{AMA}}$ on the rather small $\mathcal{V}_{\text {head }}$.

2. **Do you think contrastive decoding is used in practice?**

In 6.1 they hypothesize that having scaling the expert model further, would also lead to increased improvements generation quality (coherence, diversity, MAUVE etc.). If this is true, then it would be helpful to employ to use Contrastive Decoding for models with hundreds of billion parameters.

In the limitation section, they mention that contrastive decoding degrades performance in the task-specific generation settings they tested.

Personally, I think that contrastive decoding might be employed by practitioners, who are specifically interested in open-ended text generation. Due to the trade-off described above, I wouldn’t expect it to be used as the “standard” decoding algorithm.
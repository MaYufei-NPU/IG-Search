**We will open-source all code after the company review process is completed.**

# IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning

IG-Search is a reinforcement learning framework designed to improve search-augmented reasoning in large language models. IG-Search provides a step-level reward by measuring the explicit informational value of each individual search query.

## Information Gain (IG)

The central mechanism of IG-Search is quantifying how much a specific set of retrieved documents improves the model's confidence in the gold answer, compared to a counterfactual baseline of random documents. 

For any given search step $t$, the Information Gain is defined as:

$$\text{IG}_t = \log \pi_{\theta_{\text{old}}}(a^* \mid \mathcal{C}_t^{\text{real}}) - \frac{1}{N}\sum_{j=1}^{N} \log \pi_{\theta_{\text{old}}}(a^* \mid \mathcal{C}_t^{\text{rand},j})$$

**Variables Explained:**
* **$\text{IG}_t$**: The step-level Information Gain reward for search step $t$.
* **$a^*$**: The gold answer.
* **$\pi_{\theta_{\text{old}}}$**: The policy model used to generate the trajectories and evaluate the log-probabilities.
* **$\mathcal{C}_t^{\text{real}}$**: The true context up to step $t$, keeping the actual search query, the retrieved documents, and the model's refinement of those documents.
* **$\mathcal{C}_t^{\text{rand},j}$**: A counterfactual context where the documents and refinements at step $t$ are swapped out for randomly sampled ones from other questions. This isolates the value of the *actual* retrieval by comparing it to an uninformative baseline of the same structure and length.
* **$N$**: The total number of random counterfactual contexts sampled to compute the empirical average.

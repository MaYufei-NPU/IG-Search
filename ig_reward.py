"""
Information-Gain reward computation for search-augmented reasoning agents.

Given a rollout with the trajectory structure:

    <search>query</search>
    <documents>...</documents>
    <refine>...</refine>
    <search>query2</search>
    <documents>...</documents>
    <refine>...</refine>
    <answer>...</answer>

we measure the per-step information gain as:

    IG_t = log P(gold | REAL context up to step t)
         - mean_k log P(gold | RANDOM context up to step t)

where REAL keeps the real (documents, refine) pair at step t, and RANDOM
replaces that pair with a pair sampled from other rollouts in the same batch.
The (documents, refine) pair is replaced jointly so that REAL and RANDOM
contexts have comparable lengths, avoiding length-induced bias in log P.

This file contains only the core IG computation pipeline:
    1. Trajectory parsing                (parse_trajectory_steps)
    2. Context construction              (build_context_skeleton)
    3. IG sequence batch construction    (prepare_ig_sequences)
    4. Tokenization                      (tokenize_ig_batch)
    5. IG extraction from token logprobs (extract_ig_from_logprobs)
"""

import re
import random
import string
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict


# 0. QA normalization helpers (for gold-answer extraction)
def _normalize_answer(s: str) -> str:
    """Standard QA answer normalization: lowercase, strip punctuation and articles."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# 1. Trajectory parsing
@dataclass
class SearchStep:
    """One search step in a rollout, consisting of a <documents> block and its paired <refine> block."""
    step_index: int
    info_start: int             # start offset of <documents>
    info_end: int               # end offset of </documents>
    info_text: str              # full <documents>...</documents> text
    context_end: int            # end offset of the whole step (refine_end if present, else info_end)
    refine_text: str = ""       # paired <refine>...</refine> text (empty if missing)
    refine_start: int = -1
    refine_end: int = -1


def _get_assistant_start(full_text: str) -> int:
    """Return the character offset where the assistant turn begins."""
    match = re.search(r'<\|im_start\|>assistant\n', full_text)
    return match.end() if match else 0


def parse_trajectory_steps(full_text: str) -> List[SearchStep]:
    """
    Parse every <documents>...</documents> block in the assistant turn, and
    pair each with the first <refine> block that follows it (before the next
    <documents>).

    The parser is robust to unclosed <refine> tags: if </refine> is missing,
    the refine span ends at the nearest following <search> or <answer>.
    """
    assistant_start = _get_assistant_start(full_text)

    # 1. All <documents>...</documents> blocks inside the assistant turn.
    doc_matches = [
        m for m in re.finditer(r'<documents>.*?</documents>', full_text, re.DOTALL)
        if m.start() >= assistant_start
    ]

    # 2. All <refine> blocks, tolerating missing </refine>.
    refine_blocks = []  # list of (abs_start, abs_end, full_text_of_block)
    for m in re.finditer(r'<refine>', full_text):
        if m.start() < assistant_start:
            continue
        content_start = m.end()
        remaining = full_text[content_start:]

        # End at </refine>, or at the next <search>/<answer> — whichever comes first.
        end_candidates = []
        if (m_close := re.search(r'</refine>', remaining)):
            end_candidates.append(m_close.end())
        if (m_search := re.search(r'\n\s*<search>', remaining)):
            end_candidates.append(m_search.start())
        if (m_answer := re.search(r'\n\s*<answer>', remaining)):
            end_candidates.append(m_answer.start())

        end_offset = min(end_candidates) if end_candidates else len(remaining)
        abs_end = content_start + end_offset

        refine_full = full_text[m.start():abs_end]
        if not refine_full.rstrip().endswith('</refine>'):
            inner = full_text[content_start:abs_end].strip()
            refine_full = '<refine>' + inner + '</refine>'

        refine_blocks.append((m.start(), abs_end, refine_full))

    # 3. Pair each <documents> with the first <refine> between it and the next <documents>.
    steps = []
    for i, doc_match in enumerate(doc_matches):
        refine_text, refine_start, refine_end = "", -1, -1
        next_doc_start = doc_matches[i + 1].start() if i + 1 < len(doc_matches) else len(full_text)

        for r_start, r_end, r_text in refine_blocks:
            if doc_match.end() <= r_start < next_doc_start:
                refine_text, refine_start, refine_end = r_text, r_start, r_end
                break

        steps.append(SearchStep(
            step_index=i,
            info_start=doc_match.start(),
            info_end=doc_match.end(),
            info_text=doc_match.group(),
            context_end=refine_end if refine_end > 0 else doc_match.end(),
            refine_text=refine_text,
            refine_start=refine_start,
            refine_end=refine_end,
        ))

    return steps


def _parse_search_blocks(full_text: str) -> List[str]:
    """Return the list of full <search>...</search> block texts from the assistant turn."""
    assistant_start = _get_assistant_start(full_text)
    return [
        m.group() for m in
        re.finditer(r'<search>.*?</search>', full_text[assistant_start:], re.DOTALL)
    ]


# 2. Context construction — paired (documents, refine) replacement
def build_context_skeleton(
    full_text: str,
    target_step: SearchStep,
    all_steps: List[SearchStep],
    target_replacement: Optional[Tuple[str, str]] = None,
) -> str:
    """
    Build the prefix context used to score log P(gold | context).

    Rules:
        - Steps before `target_step`: the real <search>, <documents>, and <refine>
          blocks are kept as-is.
        - The target step: the <search> block is kept as-is; the <documents> and
          <refine> blocks are either kept (REAL, target_replacement=None) or
          replaced jointly with the given pair (RANDOM).
        - Steps after `target_step`: omitted.

    Replacing <documents> and <refine> jointly is critical: <refine> summarizes
    <documents>, so replacing only one of them creates a context-summary mismatch
    and shortens RANDOM relative to REAL, biasing IG.
    """
    assistant_start = _get_assistant_start(full_text)

    # Prefix = everything up to the first <search> in the assistant turn.
    first_search = re.search(r'<search>', full_text[assistant_start:])
    if not first_search:
        return full_text[:512]
    prefix = full_text[:assistant_start + first_search.start()].rstrip()

    search_blocks = [
        m.group() for m in
        re.finditer(r'<search>.*?</search>', full_text[assistant_start:], re.DOTALL)
    ]

    parts = [prefix]
    for step in all_steps:
        if step.step_index > target_step.step_index:
            break

        search_text = (
            search_blocks[step.step_index]
            if step.step_index < len(search_blocks)
            else "<search>query</search>"
        )

        if step.step_index == target_step.step_index and target_replacement is not None:
            # RANDOM: keep real query, replace (documents, refine) pair.
            repl_docs, repl_refine = target_replacement
            parts.append(search_text)
            parts.append(repl_docs)
            if repl_refine:
                parts.append(repl_refine)
        else:
            # REAL target step, or any earlier step: keep everything.
            parts.append(search_text)
            parts.append(step.info_text)
            if step.refine_text:
                parts.append(step.refine_text)

    return "\n".join(parts)


# 3. IG sequence batch construction
@dataclass
class IGSequence:
    """One scoring sequence: a (context, answer) pair feeding into the policy to compute log P."""
    sample_idx: int
    step_idx: int
    variant_idx: int          # index over gold-answer variants
    seq_type: str             # "real" or "random"
    context_text: str
    answer_text: str


@dataclass
class IGMapping:
    sequences: List[IGSequence] = field(default_factory=list)
    answer_lengths: List[int] = field(default_factory=list)


def _extract_golds(ga: Union[str, dict, list, np.ndarray], max_variants: int) -> List[str]:
    """Extract a list of gold-answer strings from heterogeneous input formats."""
    target = ga['target'] if isinstance(ga, dict) and 'target' in ga else ga
    if isinstance(target, str):
        return [target]
    if isinstance(target, (list, np.ndarray)):
        return [str(t) for t in target[:max_variants]]
    return [str(target)]


def _strip_tags(text: str, tags: List[str]) -> str:
    """Strip the given XML-like tags and return the plain inner text."""
    for tag in tags:
        text = text.replace(tag, '')
    return text.strip()


def prepare_ig_sequences(
    solution_strs: List[str],
    gold_answers_list: List,
    n_random_per_step: int = 3,
    max_gold_variants: int = 1,
) -> IGMapping:
    """
    Build the batch of scoring sequences required to compute per-step IG for every rollout.

    For each valid search step of each rollout, we generate:
        - 1  REAL   sequence:  original (documents, refine) at that step
        - N  RANDOM sequences: (documents, refine) pairs sampled from other rollouts in the batch

    RANDOM baseline pairs are drawn from a single batch-wide pool, excluding the
    current step's own (documents, refine) pair.
    """
    mapping = IGMapping()
    all_steps_per_sample: List[List[SearchStep]] = []

    # Build a batch-wide pool of (documents, refine) pairs usable as RANDOM baselines.
    pair_pool: List[Tuple[str, str]] = []
    for sol_str in solution_strs:
        steps = parse_trajectory_steps(sol_str)
        all_steps_per_sample.append(steps)
        for step in steps:
            doc_content = _strip_tags(step.info_text, ['<documents>', '</documents>'])
            if len(doc_content) >= 10:  # skip empty/malformed <documents> blocks
                pair_pool.append((step.info_text, step.refine_text))

    if not pair_pool or all(len(s) == 0 for s in all_steps_per_sample):
        return mapping

    answer_prefix = "\n<answer> "

    for i, sol_str in enumerate(solution_strs):
        steps = all_steps_per_sample[i]
        if not steps:
            continue
        golds = _extract_golds(gold_answers_list[i], max_gold_variants)

        for step in steps:
            # Skip steps with empty/malformed <documents>.
            doc_content = _strip_tags(step.info_text, ['<documents>', '</documents>'])
            if len(doc_content) < 10:
                continue

            for v_idx, gold in enumerate(golds):
                # REAL sequence.
                ctx_real = build_context_skeleton(sol_str, step, steps, target_replacement=None)
                mapping.sequences.append(IGSequence(
                    sample_idx=i, step_idx=step.step_index, variant_idx=v_idx,
                    seq_type="real",
                    context_text=ctx_real + answer_prefix,
                    answer_text=gold,
                ))

                # RANDOM sequences: sample baseline pairs, excluding the current step's own pair.
                current_pair = (step.info_text, step.refine_text)
                other_pairs = [p for p in pair_pool if p != current_pair] or pair_pool

                n_rand = max(n_random_per_step, 1)
                sampled = (
                    random.sample(other_pairs, n_rand)
                    if len(other_pairs) >= n_rand
                    else random.choices(other_pairs, k=n_rand)
                )

                for rand_docs, rand_refine in sampled:
                    ctx_rand = build_context_skeleton(
                        sol_str, step, steps,
                        target_replacement=(rand_docs, rand_refine),
                    )
                    mapping.sequences.append(IGSequence(
                        sample_idx=i, step_idx=step.step_index, variant_idx=v_idx,
                        seq_type="random",
                        context_text=ctx_rand + answer_prefix,
                        answer_text=gold,
                    ))

    return mapping


# 4. Tokenization
def tokenize_ig_batch(
    mapping: IGMapping,
    tokenizer,
    max_context_length: int = 4096,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Tokenize all (context, answer) pairs into left-padded context + right-padded answer tensors.

    Each context is truncated from the left (keeping its tail) so the answer is
    always adjacent to the context it is scored against. `mapping.answer_lengths`
    is populated in place so downstream logprob slicing knows how many answer
    tokens to read per sequence.
    """
    if not mapping.sequences:
        return None

    pad_id = tokenizer.pad_token_id
    all_ctx, all_ans = [], []
    for seq in mapping.sequences:
        ctx_ids = tokenizer.encode(seq.context_text, add_special_tokens=False)
        if len(ctx_ids) > max_context_length:
            ctx_ids = ctx_ids[-max_context_length:]  # keep tail; answer attends to recent context
        ans_ids = tokenizer.encode(seq.answer_text, add_special_tokens=False)
        all_ctx.append(ctx_ids)
        all_ans.append(ans_ids)
        mapping.answer_lengths.append(len(ans_ids))

    B = len(all_ctx)
    max_ctx = max(len(c) for c in all_ctx)
    max_ans = max(len(a) for a in all_ans)
    total_len = max_ctx + max_ans

    input_ids = torch.full((B, total_len), pad_id, dtype=torch.long)
    responses = torch.full((B, max_ans), pad_id, dtype=torch.long)
    for i in range(B):
        c, a = all_ctx[i], all_ans[i]
        cs = max_ctx - len(c)  # left-pad context
        input_ids[i, cs:max_ctx] = torch.tensor(c, dtype=torch.long)
        input_ids[i, max_ctx:max_ctx + len(a)] = torch.tensor(a, dtype=torch.long)
        responses[i, :len(a)] = torch.tensor(a, dtype=torch.long)

    attention_mask = (input_ids != pad_id).long()
    position_ids = torch.zeros_like(input_ids)
    for i in range(B):
        position_ids[i] = (torch.cumsum(attention_mask[i], 0) - 1).clamp(min=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'responses': responses,
    }


# 5. IG extraction from per-token log probabilities
def extract_ig_from_logprobs(
    log_probs: torch.Tensor,
    mapping: IGMapping,
    n_head_tokens: int = 3,
    head_weight: float = 3.0,
    aggregation: str = "mean",
) -> Tuple[Dict[int, float], Dict[int, List[float]], Dict[int, List[int]], Dict[str, float]]:
    """
    Convert per-token log probabilities into per-step IG and a per-rollout aggregated score.

    For each scoring sequence we compute a weighted mean log P of the gold answer,
    up-weighting the first `n_head_tokens` tokens by `head_weight`. This emphasizes
    the answer's leading tokens, which carry most of the entity-identifying signal.

    Per search step:
        IG_t = best_real_logp - mean(random_baseline_logps)

    where `best_real_logp` is the REAL logp of the gold-answer variant with the
    highest REAL score (trivially the only variant when max_gold_variants=1),
    and the baselines are the RANDOM sequences for that same variant.

    Returns:
        sample_r_ig          : {sample_idx -> aggregated IG across steps}
        sample_step_igs      : {sample_idx -> [IG per step]}
        sample_step_indices  : {sample_idx -> [step_idx per step]}
        ig_stats             : diagnostic summary statistics
    """
    # 1. Compute a weighted mean log P per sequence.
    n_seqs = len(mapping.sequences)
    weighted_logps: List[float] = []
    for i in range(n_seqs):
        ans_len = mapping.answer_lengths[i]
        if ans_len == 0:
            weighted_logps.append(0.0)
            continue
        lp = log_probs[i, :ans_len].float().cpu().numpy()
        w = np.ones(ans_len, dtype=np.float64)
        w[:min(n_head_tokens, ans_len)] = head_weight
        w /= w.mean()
        weighted_logps.append(float(np.sum(lp * w) / ans_len))

    # 2. Group sequences by (sample_idx, step_idx).
    grouped = defaultdict(lambda: {'real': {}, 'random': defaultdict(list)})
    for i, seq in enumerate(mapping.sequences):
        key = (seq.sample_idx, seq.step_idx)
        if seq.seq_type == "real":
            grouped[key]['real'][seq.variant_idx] = weighted_logps[i]
        elif seq.seq_type == "random":
            grouped[key]['random'][seq.variant_idx].append(weighted_logps[i])

    # 3. Compute IG per step.
    sample_step_igs: Dict[int, List[float]] = defaultdict(list)
    sample_step_indices: Dict[int, List[int]] = defaultdict(list)
    all_step_igs: List[float] = []
    all_logp_real: List[float] = []
    all_logp_baseline: List[float] = []

    for (sample_idx, step_idx), data in sorted(grouped.items()):
        if not data['real']:
            continue
        best_vid = max(data['real'], key=data['real'].get)
        best_real = data['real'][best_vid]
        baseline_logps = data['random'].get(best_vid, [])
        if not baseline_logps:
            continue
        mean_baseline = float(np.mean(baseline_logps))

        ig = best_real - mean_baseline

        sample_step_igs[sample_idx].append(ig)
        sample_step_indices[sample_idx].append(step_idx)
        all_step_igs.append(ig)
        all_logp_real.append(best_real)
        all_logp_baseline.append(mean_baseline)

    # 4. Aggregate per rollout.
    sample_r_ig = {
        idx: _aggregate(igs, aggregation) for idx, igs in sample_step_igs.items()
    }

    # 5. Diagnostic statistics.
    ig_stats: Dict[str, float] = {}
    if all_step_igs:
        arr = np.array(all_step_igs)
        ig_stats = {
            "ig/step_ig_mean": float(arr.mean()),
            "ig/step_ig_std": float(arr.std()),
            "ig/step_ig_max": float(arr.max()),
            "ig/step_ig_min": float(arr.min()),
            "ig/logp_real_mean": float(np.mean(all_logp_real)),
            "ig/logp_baseline_mean": float(np.mean(all_logp_baseline)),
            "ig/n_search_steps": len(arr),
            "ig/positive_ig_ratio": float((arr > 0).mean()),
            "ig/r_ig_mean": float(np.mean(list(sample_r_ig.values()))) if sample_r_ig else 0.0,
        }

    return dict(sample_r_ig), dict(sample_step_igs), dict(sample_step_indices), ig_stats


def _aggregate(step_igs: List[float], method: str = "mean") -> float:
    """Aggregate per-step IG values into a single rollout-level score."""
    if not step_igs:
        return 0.0
    if method == "mean":
        return float(np.mean(step_igs))
    if method == "sum":
        return float(np.sum(step_igs))
    if method == "max":
        return float(np.max(step_igs))
    raise ValueError(f"Unknown aggregation: {method}")

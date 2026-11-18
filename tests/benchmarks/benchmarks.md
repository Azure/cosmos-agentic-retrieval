# Benchmarks

Results at commit `7ae43ad` (2026-11-18). The agent and the judge both use
`gpt-5.6-luna` on Azure AI Foundry.

Costs are for the agent's model calls, at the Azure OpenAI Global Standard
price for GPT-5.6-luna (short context) on the [Azure OpenAI pricing page][1]
on 2026-11-18: $0.20 input, $0.02 cached input and $1.20 output per 1M tokens.
Input, cached input and output token counts are those the model reports for
each call; we apply these prices to them.

[1]: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

## Summary

| Metric                      | HippoRAG 2 |
|-----------------------------|------------|
| Questions                   | 5,000      |
| Judge accuracy              | 0.801      |
| Gold-document recall        | 0.798      |
| p50 latency (s)             | 15.6       |
| p99 latency (s)             | 61.4       |
| Avg tokens / question       | 20,329     |
| Cached share of input       | 79%        |
| Avg cost / question (USD)   | 0.00200    |

## HippoRAG 2

Five question sets of 1,000 questions each from the
[HippoRAG 2 collection](https://huggingface.co/datasets/osunlp/HippoRAG_2),
searched over one shared corpus of 45,881 passages.

### Quality

| Subset   | Questions | Judge accuracy | Recall (returned) | Recall (cited) |
|----------|-----------|----------------|-------------------|----------------|
| NQ       | 1,000     | 0.820          | 0.793             | 0.541          |
| PopQA    | 1,000     | 0.769          | 0.475             | 0.350          |
| MuSiQue  | 1,000     | 0.648          | 0.816             | 0.650          |
| 2Wiki    | 1,000     | 0.867          | 0.983             | 0.898          |
| HotpotQA | 1,000     | 0.905          | 0.922             | 0.843          |
| All      | 5,000     | 0.801          | 0.798             | 0.657          |

### Cost

| Subset   | p50 (s) | p99 (s) | Avg tokens | Avg cost (USD) |
|----------|---------|---------|------------|----------------|
| NQ       | 13.8    | 51.8    | 16,367     | 0.00161        |
| PopQA    | 10.3    | 40.0    | 13,629     | 0.00121        |
| MuSiQue  | 28.6    | 78.1    | 39,656     | 0.00389        |
| 2Wiki    | 16.0    | 50.0    | 17,352     | 0.00175        |
| HotpotQA | 14.3    | 42.3    | 14,619     | 0.00153        |
| All      | 15.6    | 61.4    | 20,329     | 0.00200        |

## Metrics

- **Judge accuracy:** share of answers a model judge marks as matching the
  gold answer.
- **Recall (returned):** share of a question's gold documents among the
  documents returned, up to 10, cited ones first.
- **Recall (cited):** the same, counting only the documents the answer cites.
- **Latency:** time per question, from request to response.
- **Avg tokens:** input plus output tokens per question, over all model calls.
- **Avg cost:** the tokens above at the prices listed at the top, with cached
  input at its lower price.

Means are over the questions that got a response.

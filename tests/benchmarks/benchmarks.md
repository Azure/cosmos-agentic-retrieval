# Benchmarks

## Summary

| Metric                      | HippoRAG 2 | EnterpriseRAG-Bench |
|-----------------------------|------------|---------------------|
| Questions                   | 5,000      | 500                 |
| Judge accuracy              | 0.801      | 0.582               |
| Gold-document recall        | 0.798      | 0.605               |
| p50 latency (s)             | 15.6       | 37.3                |
| p99 latency (s)             | 61.4       | 94.4                |
| Avg tokens / question       | 20,329     | 99,432              |
| Cached share of input       | 79%        | 72%                 |
| Avg cost / question (USD)   | 0.00200    | 0.00826             |

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

## EnterpriseRAG-Bench

The 500 questions of
[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench),
in ten categories, searched over its full corpus of 511,962 company
documents, stored as 1,727,869 chunks of 512 tokens with 128 overlapping. An
earlier run on a 20% subset of the corpus is in
[#9](https://github.com/Azure/cosmos-agentic-retrieval/pull/9).

### Quality

| Category         | Questions | Overall | Judge accuracy | Completeness | Recall (returned) | Recall (cited) | Invalid extra docs |
|------------------|-----------|---------|----------------|--------------|-------------------|----------------|--------------------|
| Basic            | 175       | 0.642   | 0.709          | 0.655        | 0.726             | 0.709          | 6.68               |
| Semantic         | 125       | 0.243   | 0.352          | 0.315        | 0.280             | 0.256          | 7.89               |
| Intra-document   | 40        | 0.738   | 0.750          | 0.766        | 0.825             | 0.775          | 6.38               |
| Project related  | 40        | 0.185   | 0.350          | 0.481        | 0.520             | 0.499          | 3.63               |
| Constrained      | 30        | 0.738   | 0.767          | 0.841        | 0.850             | 0.783          | 5.40               |
| Conflicting info | 20        | 0.499   | 0.650          | 0.583        | 0.725             | 0.550          | 4.55               |
| Completeness     | 20        | 0.350   | 0.550          | 0.411        | 0.536             | 0.519          | 4.45               |
| Miscellaneous    | 20        | 0.698   | 0.800          | 0.723        | 0.900             | 0.850          | 7.40               |
| High level       | 10        | 0.433   | 0.500          | 0.542        | —                 | —              | —                  |
| Info not found   | 20        | 0.550   | 0.550          | 0.550        | —                 | —              | —                  |
| All              | 500       | 0.496   | 0.582          | 0.560        | 0.605             | 0.572          | 6.48               |

High level and Info not found questions have no gold documents, so they have
no recall or invalid extra documents.

- **Overall:** an answer's completeness if the judge marks it correct,
  otherwise 0, averaged over questions.
- **Completeness:** share of the question's expected answer facts that the
  judge finds in the answer.
- **Invalid extra docs:** returned documents that are not gold documents, per
  question. EnterpriseRAG-Bench's own scorer also excuses documents its judges
  find relevant, so its counts are lower than ours.

### Cost

| Category         | p50 (s) | p99 (s) | Avg tokens | Avg cost (USD) |
|------------------|---------|---------|------------|----------------|
| Basic            | 30.1    | 75.8    | 72,975     | 0.00626        |
| Semantic         | 50.3    | 95.7    | 151,279    | 0.01147        |
| Intra-document   | 37.6    | 92.2    | 99,161     | 0.00825        |
| Project related  | 37.7    | 90.6    | 57,713     | 0.00616        |
| Constrained      | 31.1    | 82.4    | 65,782     | 0.00631        |
| Conflicting info | 26.5    | 49.2    | 45,687     | 0.00447        |
| Completeness     | 42.6    | 101.2   | 132,530    | 0.01202        |
| Miscellaneous    | 24.0    | 48.5    | 50,509     | 0.00451        |
| High level       | 32.1    | 54.3    | 54,202     | 0.00477        |
| Info not found   | 68.1    | 107.4   | 233,537    | 0.01835        |
| All              | 37.3    | 94.4    | 99,432     | 0.00826        |

## Setup and Metrics

- **Judge accuracy:** share of answers a model judge marks as matching the
  gold answer.
- **Recall (returned):** share of a question's gold documents among the
  documents returned, up to 10, cited ones first.
- **Recall (cited):** the same, counting only the documents the answer cites.
- **Latency:** time per question, from request to response.
- **Avg tokens:** input plus output tokens per question, over all model calls.
- **Avg cost:** the tokens above at the prices below, with cached input at its
  lower price.

Means are over the questions that got a response.

The agent and the judge both use `gpt-5.6-luna` on Azure AI Foundry.
HippoRAG 2 ran at commit `7ae43ad` (2026-11-18), EnterpriseRAG-Bench at
`2b0b96c` (2026-11-19).

Costs are for the agent's model calls, at the Azure OpenAI Global Standard
price for GPT-5.6-luna (short context) on the [Azure OpenAI pricing page][1]
on 2026-11-18: $0.20 input, $0.02 cached input and $1.20 output per 1M tokens.
Input, cached input and output token counts are those the model reports for
each call; we apply these prices to them.

[1]: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

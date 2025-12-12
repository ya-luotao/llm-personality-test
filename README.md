# LLM MBTI Test

Run MBTI personality tests on LLMs via the [OpenMBTI](https://openmbti.org) MCP server using [OpenRouter](https://openrouter.ai) as the LLM gateway.

## Features

- Test any model available on OpenRouter
- Run multiple test iterations for statistical analysis
- Parallel execution for faster batch testing
- Incremental saving (safe to interrupt and resume)
- Batch testing across multiple models
- Detailed results with all Q&A pairs, MBTI type, and dimension scores

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set your OpenRouter API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

## Usage

### Single Model Test

```bash
# Basic run
uv run python main.py -m openai/gpt-4o

# Multiple runs with parallel execution
uv run python main.py -m openai/gpt-4o --runs 10 --parallel 5

# Continue interrupted run
uv run python main.py -m openai/gpt-4o --runs 10 -c results/openai_gpt-4o_20241212.json
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | OpenRouter model ID | required |
| `-r, --runs` | Number of test runs | 1 |
| `-p, --parallel` | Concurrent runs | 3 |
| `-o, --output` | Output JSON path | auto-generated |
| `-c, --continue-from` | Resume from file | - |

### Batch Testing (Multiple Models)

```bash
# Test multiple models
uv run python run_all.py -m openai/gpt-4o anthropic/claude-3.5-sonnet google/gemini-pro

# With custom settings
uv run python run_all.py \
  -m openai/gpt-4o anthropic/claude-3.5-sonnet \
  --run-id experiment001 \
  --runs 10 \
  --parallel 5

# Continue interrupted batch
uv run python run_all.py \
  -m openai/gpt-4o anthropic/claude-3.5-sonnet \
  --run-id experiment001 \
  --runs 10 \
  -c
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --models` | List of model IDs | required |
| `-id, --run-id` | Batch identifier | timestamp |
| `-r, --runs` | Runs per model | 5 |
| `-p, --parallel` | Concurrent runs | 3 |
| `-c, --continue` | Resume existing | false |

## Output

### Single Model Output

Results are saved to `results/<model>_<timestamp>.json`:

```json
{
  "model": "openai/gpt-4o",
  "total_runs": 10,
  "timestamp": "2024-12-12T10:30:00Z",
  "runs": [
    {
      "run_id": 1,
      "mbti_type": "INTJ",
      "dimension_scores": {
        "E_I": {"E": 30, "I": 70},
        "S_N": {"S": 25, "N": 75},
        "T_F": {"T": 80, "F": 20},
        "J_P": {"J": 65, "P": 35}
      },
      "questions": [
        {
          "id": 1,
          "text": "Extraversion vs Introversion",
          "options": ["1 - Strongly: Extraversion", "..."],
          "llm_choice": 4,
          "llm_raw_response": "4"
        }
      ]
    }
  ],
  "summary": {
    "completed_runs": 10,
    "mbti_distribution": {"INTJ": 7, "INFJ": 2, "INTP": 1}
  }
}
```

### Batch Output

Results are saved to `results/<run-id>/`:

```
results/
  experiment001/
    openai_gpt-4o.json
    anthropic_claude-3.5-sonnet.json
    _summary.json
```

## Example Models

```bash
# OpenAI
openai/gpt-4o
openai/gpt-4-turbo

# Anthropic
anthropic/claude-3.5-sonnet
anthropic/claude-3-opus

# Google
google/gemini-pro
google/gemini-pro-1.5

# Meta
meta-llama/llama-3.1-70b-instruct

# Mistral
mistralai/mistral-large
```

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## License

MIT

# LLM Personality Test

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

# Low temperature with 4-point forced choice scale
uv run python main.py -m openai/gpt-4o --runs 10 -t 0.3 -s 4

# Continue interrupted run
uv run python main.py -m openai/gpt-4o --runs 10 -c results/openai_gpt-4o_20241212.json
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | OpenRouter model ID | required |
| `-r, --runs` | Number of test runs | 1 |
| `-p, --parallel` | Concurrent runs | 3 |
| `-t, --temperature` | LLM temperature | 0.7 |
| `-s, --scale` | Answer scale: 4 (forced) or 5 (neutral) | 5 |
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
| `-t, --temperature` | LLM temperature | 0.7 |
| `-s, --scale` | Answer scale: 4 (forced) or 5 (neutral) | 5 |
| `-c, --continue` | Resume existing | false |

### Analyze Results

```bash
# Analyze a batch run
uv run python report.py experiment001

# Analyze specific model in a batch
uv run python report.py experiment001 -m gpt-4o

# Analyze a single result file
uv run python report.py results/openai_gpt-4o_20241212.json
```

**Report includes:**
- MBTI type distribution with histogram
- Choice table showing each question's answers across runs
- Consistency analysis (most/least consistent questions)
- Dimension score analysis with averages and std dev
- Multi-model comparison (when analyzing batch runs)

**Example output:**
```
--- Choices by Question ---
Question                       | R1 R2 R3 R4 R5 | Mode | Agr%
--------------------------------------------------------------
Extraversion vs Introversion   |  4  4  4  3  4 |  4   |  80%
Sensing vs Intuition           |  5  5  4  5  5 |  5   |  80%
Thinking vs Feeling            |  2  2  2  2  3 |  2   |  80%

--- Consistency Analysis ---
Overall consistency: 76.5%

Most inconsistent questions:
  40.0% | Judging vs Perceiving              | 3 4 2 5 3
  60.0% | Sensing vs Intuition               | 4 5 4 5 4
```

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

## Experiment Results (Dec 12, 2024)

Tested 3 frontier models across different configurations (5 runs each):

### Models Tested
- **Claude Opus 4.5** (`anthropic/claude-opus-4.5`)
- **GPT-5.2 Pro** (`openai/gpt-5.2-pro`)
- **Gemini 3 Pro Preview** (`google/gemini-3-pro-preview`)

### 4-Point Scale (Forced Choice, No Neutral)

| Model | Temp 0.0 | Temp 0.5 | Temp 1.0 |
|-------|----------|----------|----------|
| Claude Opus 4.5 | INFJ (5/5) | INFJ (5/5) | INFJ (5/5) |
| GPT-5.2 Pro | INTJ (5/5) | INTJ (5/5) | INTJ (5/5) |
| Gemini 3 Pro | INTJ (5/5) | INTJ (4/5), ISTJ (1/5) | INTJ (4/5), ISTJ (1/5) |

### 5-Point Scale (With Neutral Option)

| Model | Temp 0.0 | Temp 0.5 | Temp 1.0 |
|-------|----------|----------|----------|
| Claude Opus 4.5 | INFJ (5/5) | INFJ (5/5) | INFJ (5/5) |
| GPT-5.2 Pro | ESFJ (4/5), INTJ (1/5) | ESFJ (4/5), INTJ (1/5) | ESFJ (4/5), INTJ (1/5) |
| Gemini 3 Pro | INTJ (5/5) | INTJ (4/5), ISTJ (1/5) | INTJ (5/5) |

### Key Findings

1. **Claude Opus 4.5**: Most consistent - INFJ across all configurations
2. **GPT-5.2 Pro**: Scale matters - INTJ with 4-point forced choice, ESFJ with 5-point scale
3. **Gemini 3 Pro**: Highly consistent INTJ, occasional ISTJ variation at higher temps
4. **Temperature impact**: Minimal effect on personality type, slight increase in variation

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

# Pricing Agent

Estimates what a product should cost, then hunts RSS deal feeds for items priced well below that estimate.

A fine-tuned Llama 3.2 3B model runs on [Modal](https://modal.com) and prices a product description. A local agent framework scans deals, combines that specialist model with a frontier model and a neural network, and sends a Pushover alert when the discount is large enough.

## How it works

The planner runs three steps:

1. **Scanner** reads deal RSS feeds and asks a frontier model to keep the five listings with a clear price and a detailed description.
2. **Ensemble** rewrites the description, then blends three estimates: the Modal specialist (weight 0.1), a RAG frontier model over a Chroma product store (0.8), and a local deep neural network (0.1).
3. **Messenger** writes a short alert with Claude and pushes it through Pushover when the discount is greater than $50.

`price_is_right.py` wraps the same loop in a Gradio UI that logs each run, lists opportunities, and plots the product vector store.

## Layout

| File | What it does |
|---|---|
| `hello.py` | Small Modal function that reports where the container is running |
| `llama.py` | Loads base Llama 3.2 3B on a T4 and generates a short completion |
| `pricer_ephemeral.py` | Prices one description remotely; the model loads on every call |
| `pricer_service.py` | Same pricer as a Modal function |
| `pricer_service2.py` | Deployed `Pricer` class that keeps the model warm and caches Hugging Face weights |
| `pricer_test.py` | Calls the deployed `pricer-service` app |
| `deal_agent_framework.py` | Runs one planning cycle and stores opportunities in `memory.json` |
| `price_is_right.py` | Gradio app, "The Price is Right" |
| `agents/` | Scanner, ensemble, specialist, frontier, neural network, messenger, and planner |

The specialist calls the deployed class by name: Modal app `pricer-service`, class `Pricer`. The weights are the LoRA adapter `ed-donner/price-2025-11-28_18.47.07` on `meta-llama/Llama-3.2-3B`, loaded in 4-bit on a T4.

## Setup

Install the Python packages the scripts import (`modal`, `python-dotenv`, `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `openai`, `litellm`, `chromadb`, `gradio`, `plotly`, `scikit-learn`, `sentence-transformers`, `feedparser`, `beautifulsoup4`, `pydantic`, `requests`). Authenticate the Modal CLI (`modal setup`).

Create a Modal secret named `huggingface-secret` with a Hugging Face token that can read Llama 3.2.

Put local keys in a `.env` file:

```
OPENAI_API_KEY=
PUSHOVER_USER=
PUSHOVER_TOKEN=
PRICER_PREPROCESSOR_MODEL=ollama/llama3.2
```

`PRICER_PREPROCESSOR_MODEL` is optional. The preprocessor defaults to Ollama at `http://localhost:11434`. The neural network agent loads weights from `deep_neural_network.pth`, and the frontier agent reads a Chroma database at `products_vectorstore`.

## Run

```bash
modal run hello.py
modal run llama.py
modal run pricer_ephemeral.py
```

Deploy the warm pricer, then call it:

```bash
modal deploy pricer_service2.py
modal run pricer_test.py
```

Run the deal agent once, or open the UI (it also reruns every 5 minutes):

```bash
python deal_agent_framework.py
python price_is_right.py
```

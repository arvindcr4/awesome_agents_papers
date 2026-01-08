# LLM Fine-Tuning Provider Comparison

Choose the right provider for your fine-tuning needs.

## Quick Comparison

| Feature | Fireworks AI | Together AI | Replicate |
|---------|-------------|-------------|-----------|
| **Free Tier** | ✅ Yes | ❌ No | ❌ No (pay-per-use) |
| **Pricing Model** | Credits + usage | Standard account | Pay-per-use |
| **LoRA/QLoRA** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Min. Examples** | ~10 | ~10 | ~10 |
| **API Style** | OpenAI-compatible | Native + OpenAI | Native |
| **Model Selection** | Good | Excellent | Good |
| **Ease of Use** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## When to Choose Each

### 🔥 Fireworks AI
**Best for:** Experimentation on a budget

Choose Fireworks if you:
- Want to experiment with fine-tuning for free
- Need OpenAI-compatible API (easy migration)
- Value fast inference after training
- Are building prototypes before production

**Notebook:** [fireworks_finetune_colab.ipynb](fireworks_finetune_colab.ipynb)

### 🚀 Together AI
**Best for:** Production workloads with diverse models

Choose Together if you:
- Need access to the widest model selection
- Want detailed training metrics and monitoring
- Are building production systems
- Need competitive pricing at scale

**Notebook:** [together_finetune_colab.ipynb](together_finetune_colab.ipynb)

### 🔄 Replicate
**Best for:** Simple, pay-as-you-go deployment

Choose Replicate if you:
- Want the simplest possible workflow
- Prefer pay-per-use (no monthly commitments)
- Need instant model deployment after training
- Value a clean web UI option

**Notebook:** [replicate_finetune_colab.ipynb](replicate_finetune_colab.ipynb)

## Pricing Estimates (as of 2024)

| Provider | Training Cost | Inference Cost |
|----------|--------------|----------------|
| Fireworks | Free tier + ~$0.50/M tokens | ~$0.20/M tokens |
| Together | ~$2-5/M tokens trained | ~$0.20/M tokens |
| Replicate | ~$0.001/sec training | ~$0.0005/sec inference |

*Prices vary by model size and are subject to change. Check provider websites for current pricing.*

## Dataset Format Summary

### Fireworks AI
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Together AI
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Replicate (Llama format)
```json
{"text": "<s>[INST] <<SYS>>\nSystem prompt\n<</SYS>>\n\nUser message [/INST] Assistant response </s>"}
```

## Getting Started

1. **Sign up** for your chosen provider
2. **Get API key** from the provider's dashboard
3. **Open the notebook** in Google Colab via VSCode
4. **Add API key** to Colab secrets or enter manually
5. **Prepare your data** following the format above
6. **Run the notebook** cells sequentially

## Tips for All Providers

### Data Quality
- **50-1000+ examples** for best results
- **Consistent formatting** across examples
- **Cover edge cases** in your domain
- **Quality > quantity** - well-crafted examples matter more

### Hyperparameters
- **Epochs:** Start with 3, increase if underfitting
- **Learning rate:** 1e-5 is a safe default
- **Batch size:** Larger = faster but uses more memory

### Cost Optimization
- Start with smaller models (7B-8B) for testing
- Use validation sets to detect overfitting early
- Fine-tune on focused tasks (not general knowledge)

## Sample Use Cases

| Use Case | Recommended Provider | Why |
|----------|---------------------|-----|
| Prototype chatbot | Fireworks | Free tier |
| Production assistant | Together | Best model selection |
| Code generation | Replicate | Good Code Llama support |
| Multilingual | Together | Qwen models |
| Low-latency inference | Fireworks | Optimized serving |

## Connecting to Google Colab from VSCode

1. Install "Jupyter" extension in VSCode
2. Open any `.ipynb` file
3. Click "Select Kernel" → "Connect to Google Colab"
4. Authenticate with your Google account
5. Run cells with Shift+Enter

This gives you Colab's free GPU while using VSCode's superior editing experience.

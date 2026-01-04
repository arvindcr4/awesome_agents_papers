# RL Finetuning Papers - Eraser Diagrams Generation Report

**Generated:** 2026-01-04
**Total Papers:** 16
**Successfully Generated:** 2
**Failed:** 14 (API spending limit reached)

## Status Summary

The diagram generation process was **halted after 2 successful generations** due to reaching the Eraser API monthly spending limit ($120/month). The API returned:
- Error: `PaywallError` - "You have reached your team's monthly usage-based spending limit"
- Current spend: $122.40 / $120.00 limit
- API calls made: 102 calls

## Successfully Generated Diagrams (2/16)

### 1. DeepSeek-R1: Reasoning via RL
- **Paper ID:** `deepseek-r1-reasoning`
- **Prompt:** "RL pipeline for reasoning with rejection sampling"
- **Status:** ✓ Success
- **Request ID:** `Y5DjzVFBndGWlBR2EkDg`
- **Image URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A8070edb91cd32219f3e7cff05f32e00f35ea703857a3b4c51732094bd29eb4e0.png
- **Edit URL:** https://app.eraser.io/new?requestId=Y5DjzVFBndGWlBR2EkDg
- **Diagram Code:** Flowchart showing RL pipeline with rejection sampling mechanism including:
  - Environment and policy initialization
  - Reward model setup
  - Candidate trajectory generation
  - Trajectory evaluation with accept/reject decision
  - Policy update cycle
  - Performance evaluation loop

### 2. DeepSeek R1: Implications for AI
- **Paper ID:** `deepseek-r1-implications`
- **Prompt:** "Architectural innovations"
- **Status:** ✓ Success
- **Request ID:** `x5JC4Ep4NVTREPRf429C`
- **Image URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A6b6bb87a66930463219027a5adcf2260ad6852c65e76a3aff58931d34cde44df.png
- **Edit URL:** https://app.eraser.io/new?requestId=x5JC4Ep4NVTREPRf429C
- **Diagram Code:** Flowchart showing architectural innovation process including:
  - Market insights and technology advances
  - Ideation phase with requirements gathering
  - Design phase with digital modeling
  - Prototyping and testing phase
  - Implementation and deployment

## Failed Diagrams (14/16)

All remaining diagrams failed due to **API spending limit exceeded**:

3. **DeepSeek R1: Are Reasoning Models Faithful** - "Reasoning faithfulness analysis"
4. **DeepSeekMath: GRPO** - "Group Relative Policy Optimization"
5. **Guided GRPO** - "Guided policy optimization"
6. **Direct Preference Optimization (DPO)** - "DPO vs PPO algorithms"
7. **Unpacking DPO and PPO** - "Detailed algorithm comparison"
8. **OpenAI O1 Replication** - "Reasoning model training pipeline"
9. **Qwen QwQ** - "Long-chain reasoning architecture"
10. **Sky-T1** - "Distillation for compact reasoning"
11. **s1: Test-Time Scaling** - "Compute scaling for reasoning"
12. **R-Search** - "Search-based multi-step reasoning"
13. **RL Fine-tuning: Instruction Following** - "RLHF pipeline"
14. **RFT Powers Multimodal** - "Reinforcement fine-tuning"
15. **STILL-2: Distilling Reasoning** - "Knowledge distillation"
16. **DeepSeek Janus Pro** - "Multimodal architecture"

## API Details Used

- **Endpoint:** `https://app.eraser.io/api/render/prompt`
- **Authentication:** Bearer token
- **Request Format:**
  ```json
  {
    "text": "<prompt>",
    "diagramType": "flowchart-diagram",
    "mode": "premium"
  }
  ```
- **Response Format:**
  ```json
  {
    "requestId": "<id>",
    "imageUrl": "<url>",
    "createEraserFileUrl": "<url>",
    "diagrams": [...]
  }
  ```

## Files Generated

All results saved to: `/Users/arvind/rlproject/awesome_agents_papers/diagrams/rl-finetuning/`

- `01-deepseek-r1-reasoning.json` - Complete result with diagram code
- `02-deepseek-r1-implications.json` - Complete result with diagram code
- `03-deepseek-r1-faithfulness.json` through `16-deepseek-janus-pro.json` - Error responses
- `_summary.json` - Overall summary
- `generate_diagrams.py` - Generation script

## Next Steps

To complete the remaining 14 diagrams:

1. **Wait for monthly reset** - The API limit will reset at the beginning of next month
2. **Increase spending limit** - Contact Eraser to raise the monthly cap
3. **Use different account** - Use a different API key/account
4. **Use standard mode** - Try with `mode: "standard"` (may have different pricing)

Once the limit is reset, run the script again:
```bash
python3 /Users/arvind/rlproject/awesome_agents_papers/diagrams/rl-finetuning/generate_diagrams.py
```

## Diagram Code Examples

The successfully generated diagrams include editable diagram code that can be modified in the Eraser UI:

**Example 1 - RL Pipeline:**
```python
title RL Pipeline for Reasoning with Rejection Sampling
direction right
// Nodes
Start environment [shape: oval, icon: globe, color: lightblue]
Initialize reward model [shape: rectangle, icon: shield, color: lightblue]
Generate candidate trajectories [shape: rectangle, icon: cpu, color: orange]
Evaluate trajectories [shape: rectangle, icon: bar-chart-2, color: orange]
...
```

## Cost Analysis

- **Cost per diagram:** Approximately $1.20 (based on $122.40 for 102 calls)
- **Total cost for 16 diagrams:** ~$19.20
- **Remaining budget needed:** ~$16.80 (for 14 more diagrams)

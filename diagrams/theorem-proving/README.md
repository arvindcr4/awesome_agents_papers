# Theorem Proving Papers - Eraser Diagrams Summary

Generated flowchart diagrams for 9 theorem proving research papers using the Eraser API.

## Statistics

- **Total Papers:** 9
- **Successful:** 5
- **Failed:** 4
- **Success Rate:** 55.6%

---

## Successfully Generated Diagrams

### 1. LeanDojo: Retrieval-Augmented

**Status:** Success
**Request ID:** `dT9OZgb4DVnnbG0lkOFl`

**Diagram URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3Ae9f16f73f0c80e01c7a2a5f7722caacaf620df21fe20d82b4d10fc747faf7ac3.png

**Local File:** `/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/leandojo-_retrieval-augmented.json`

**Description:** Tactic retrieval architecture for theorem proving showing repository as database, query mechanism, retriever model, selector/generator, and feedback loop.

---

### 2. Autoformalizing Euclidean Geometry

**Status:** Success
**Request ID:** `P9uElLkgnexMBk3keeqB`

**Diagram URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3Ac9d87d89b817a0ff0c74789b354892ff3551df9c8f2bde0d4f2730fc3fb2558f.png

**Local File:** `/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/autoformalizing_euclidean_geometry.json`

**Description:** Geometry formalization pipeline showing input processing, diagram understanding, translation to geometric primitives, formalization, proof generation, and verification.

---

### 3. Draft Sketch Prove

**Status:** Success
**Request ID:** `EKckXoeuYzX14o1mNmTl`

**Diagram URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A4df490b12f7ed8453569736c8589c50a748a0fa576fff9b1d76aee7edacebda0.png

**Local File:** `/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/draft_sketch_prove.json`

**Description:** DSP pipeline for proofs showing draft, sketch, and prove phases with iterative refinement and verification.

---

### 4. miniCTX: Long-Context

**Status:** Success
**Request ID:** `9LRCmGyml5kZoDaj2TAP`

**Diagram URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A409e1d2d3ce1363eda9ffe3e654abf038eedde42b592fa8ab99446e936b91074.png

**Local File:** `/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/minictx-_long-context.json`

**Description:** Long-context theorem proving architecture showing large context window, retrieval mechanism, context encoding, tactic prediction, and multi-step proof generation.

---

### 5. Lean-STaR

**Status:** Success
**Request ID:** `Olz0sh1XYoGuITihOYxE`

**Diagram URL:** https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A7dab7e8f1b917026675fd4a8820572919eea6a1e3b5e74e631cb819cd765443b.png

**Local File:** `/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/lean-star.json`

**Description:** Interleaved reasoning and proving architecture showing reasoning module, translation, proving module, reward signal, and self-improvement loop.

---

## Failed Diagrams

### 6. Autoformalization with LLMs

**Status:** Failed

**Error:** HTTPSConnectionPool(host='app.eraser.io', port=443): Read timed out. (read timeout=120)

**Description:** Natural Language to formal proof pipeline.

---

### 7. ImProver: Proof Optimization

**Status:** Failed

**Error:** 402 Client Error: Payment Required - Monthly usage limit exceeded ($120)

**Description:** Iterative proof improvement system with critic and improver modules.

---

### 8. ICL Agent Theorem Proving

**Status:** Failed

**Error:** 402 Client Error: Payment Required - Monthly usage limit exceeded ($120)

**Description:** In-Context Learning based theorem proving with example retrieval and few-shot learning.

---

### 9. AlphaGeometry

**Status:** Failed

**Error:** 402 Client Error: Payment Required - Monthly usage limit exceeded ($120)

**Description:** Symbolic and neural components for geometry theorem proving.

---

## API Usage Details

- **API Endpoint:** https://app.eraser.io/api/render/prompt
- **Mode:** Premium
- **Diagram Type:** flowchart-diagram
- **Monthly Limit:** $120 (exceeded after 5 successful generations)
- **Total Spend:** $124.80

---

## File Structure

```
/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/
├── _summary.json                          # Summary of all diagram generation attempts
├── _summary.html                          # Interactive HTML report with diagram previews
├── README.md                              # This file
├── leandojo-_retrieval-augmented.json     # LeanDojo diagram metadata
├── autoformalizing_euclidean_geometry.json # Autoformalizing Geometry diagram metadata
├── draft_sketch_prove.json                # Draft Sketch Prove diagram metadata
├── minictx-_long-context.json             # miniCTX diagram metadata
└── lean-star.json                         # Lean-STaR diagram metadata
```

---

## Regenerating Failed Diagrams

To regenerate the failed diagrams once the API limit resets:

```bash
# Use the existing script
python3 /Users/arvind/rlproject/generate_eraser_diagrams_tp.py
```

Or manually retry individual papers using the curl command:

```bash
curl -X POST "https://app.eraser.io/api/render/prompt" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer NUPBy0SOPgDWIAIl3PDk" \
  -d '{
    "text": "YOUR_PROMPT_HERE",
    "mode": "premium",
    "diagramType": "flowchart-diagram",
    "theme": "dark"
  }'
```

---

## Generated On

January 4, 2026

## Output Directory

`/Users/arvind/rlproject/awesome_agents_papers/diagrams/theorem-proving/`

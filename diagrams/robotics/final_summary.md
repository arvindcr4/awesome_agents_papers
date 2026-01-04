# Robotics Papers - Eraser Diagrams - Final Summary

**Generated:** 2026-01-04
**Status:** 4/6 diagrams generated successfully (hit API spending limit)

---

## Successfully Generated Diagrams

### 1. Voyager: Open-Ended Embodied Agent ✓

**Description:** Voyager Minecraft agent with skill library and iterative prompting mechanism

**Prompt:** "Skill library and self-evolving prompts"

**Diagram URL:**
- ![Voyager Diagram](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A06c37e794577cbe6b9d53ea2d68cc0e27048b52203ad7ab392a1e43e60f54600.png)
- Direct link: https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A06c37e794577cbe6b9d53ea2d68cc0e27048b52203ad7ab392a1e43e60f54600.png

**Request ID:** `5U1z7G7EaBCY04X34OWb`

**Generated:** 2026-01-04 17:01:09

---

### 2. Eureka: Human-Level Reward Design ✓

**Description:** Eureka reward design system with LLM generating reward functions for RL

**Prompt:** "LLM-generated reward functions with RL"

**Diagram URL:**
- ![Eureka Diagram](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A4a733bca81c298469d312cf7c9e49ee1fd88af7a06c1f9bea4ab1221b56f88ee.png)
- Direct link: https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A4a733bca81c298469d312cf7c9e49ee1fd88af7a06c1f9bea4ab1221b56f88ee.png

**Request ID:** `ueWiS80UvcoEmzdY9meN`

**Generated:** 2026-01-04 16:55:37

---

### 3. DrEureka: Sim-to-Real ✓

**Description:** DrEureka sim-to-real transfer with domain randomization and adaptation

**Prompt:** "Domain adaptation pipeline"

**Diagram URL:**
- ![DrEureka Diagram](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A63763d4c79589902bd7d0abe8a39833a8245b072595b9b2da45ed697ce6df5b9.png)
- Direct link: https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A63763d4c79589902bd7d0abe8a39833a8245b072595b9b2da45ed697ce6df5b9.png

**Request ID:** `LMvdsa6KeSDKBCAJR9Ye`

**Generated:** 2026-01-04 17:02:54

---

### 4. GR00T N1: Humanoid Foundation Model ✓

**Description:** GR00T N1 humanoid robot with vision, planning, and control systems

**Prompt:** "Vision, planning, control"

**Diagram URL:**
- ![GR00T Diagram](https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A9f40ba0bfb119ca06e2a20d1559cc90ca171f23de57ff9aacbe7f7ab17d8eb86.png)
- Direct link: https://storage.googleapis.com/second-petal-295822.appspot.com/elements/autoDiagram%3A9f40ba0bfb119ca06e2a20d1559cc90ca171f23de57ff9aacbe7f7ab17d8eb86.png

**Request ID:** `TZ17KnfImCmnJLoTn2ag`

**Generated:** 2026-01-04 16:58:14

---

## Failed to Generate (API Limit Reached)

### 5. Gran Turismo: Deep RL Racing ❌

**Reason:** Hit API spending limit after 106 premium API calls ($127.20 spent out of $120 limit)

**Prompt:** "Racing agent with reward design"

**Description:** Gran Turismo Sophy racing agent with reinforcement learning and reward shaping

---

### 6. SLAC: Latent Action ❌

**Reason:** Hit API spending limit

**Prompt:** "Latent action space pretraining"

**Description:** SLAC latent action model with pretraining and reinforcement learning

---

## API Statistics

- **Total Papers:** 6
- **Successfully Generated:** 4 (67%)
- **Failed:** 2 (33%)
- **API Endpoint:** https://app.eraser.io/api/render/prompt
- **Diagram Type:** cloud-architecture-diagram
- **Mode:** premium
- **API Spending:** $127.20 / $120.00 limit (106 calls)
- **Status:** Monthly limit reached

---

## Files Generated

All results stored in: `/Users/arvind/rlproject/awesome_agents_papers/diagrams/robotics/`

- `voyager.json` - Voyager diagram metadata
- `eureka.json` - Eureka diagram metadata
- `dreureka.json` - DrEureka diagram metadata
- `gr00t.json` - GR00T diagram metadata
- `all_robotics_diagrams.json` - Combined results
- `summary.txt` - Plain text summary
- `README.md` - Original markdown summary
- `final_summary.md` - This comprehensive summary

---

## Notes

1. The Eraser API required the field name "text" instead of "prompt" in the request payload
2. Premium mode diagrams take 90-100 seconds to generate on average
3. API spending limit was reached after 4 successful diagrams
4. To generate the remaining 2 diagrams (Gran Turismo and SLAC), you'll need to:
   - Wait for the monthly spending limit to reset, OR
   - Increase the spending limit in your Eraser account
   - Prompts are ready and can be retried with the same scripts

---

## Commands to Regenerate Failed Diagrams

Once the API limit is reset or increased:

```bash
# Retry all 6 papers
python3 /Users/arvind/rlproject/generate_robotics_diagrams.py

# OR retry only the failed ones
python3 /Users/arvind/rlproject/retry_failed_diagrams.py
```

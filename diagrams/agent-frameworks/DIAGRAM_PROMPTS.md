# Eraser Diagram Prompts - Agent Frameworks Papers

This document contains the detailed prompts for generating flowchart diagrams for 10 Agent Frameworks papers using the Eraser API.

## API Configuration

```
Endpoint: https://app.eraser.io/api/render/prompt
Method: POST
Headers:
  Content-Type: application/json
  Authorization: Bearer NUPBy0SOPgDWIAIl3PDk

Body:
{
  "text": "<prompt from below>",
  "mode": "premium",
  "diagramType": "flowchart-diagram"
}
```

---

## Paper 1: ReAct - Synergizing Reasoning and Acting

**Prompt:**
```
Create a flowchart diagram showing the ReAct pattern: Start with User Query -> Thought (reasoning step) -> Action (tool/function call) -> Observation (result) -> loop back to Thought until done, then Final Answer. Show the iterative nature of reasoning and acting.
```

**Key Elements:**
- User Query input
- Thought (reasoning)
- Action (tool execution)
- Observation (result)
- Loop until completion
- Final Answer output

---

## Paper 2: AutoGen - Multi-Agent Conversation

**Prompt:**
```
Create a flowchart showing AutoGen multi-agent architecture: User -> Assistant Agent (generates code/response) -> User Proxy Agent (executes code, provides feedback) -> Assistant Agent, loop until completion. Show multiple agents with different roles communicating.
```

**Key Elements:**
- User input
- Assistant Agent role
- User Proxy Agent role
- Conversation loop
- Multi-agent communication
- Task completion

---

## Paper 3: StateFlow - State-Driven LLM Task-Solving

**Prompt:**
```
Create a flowchart showing StateFlow finite state machine: Initial State -> Task Decomposition -> Subtask States (parallel execution) -> State Transitions based on conditions -> Final State. Show state nodes and transition edges with decision points.
```

**Key Elements:**
- Initial State
- Task Decomposition
- Subtask States
- Parallel Execution
- State Transitions
- Decision Points
- Final State
- Finite State Machine structure

---

## Paper 4: DSPy - Compiling Declarative Language Model

**Prompt:**
```
Create a flowchart showing DSPy pipeline: Input -> Signature (declare input/output) -> Module (process with LLM) -> Optimizer (compile and optimize prompts) -> Output. Show the compilation process from declarative to optimized.
```

**Key Elements:**
- Input data
- Signature (input/output declaration)
- Module (LLM processing)
- Optimizer (prompt compilation)
- Output result
- Declarative to optimized transformation

---

## Paper 5: FireAct - Language Agent Fine-tuning

**Prompt:**
```
Create a flowchart showing FireAct: Task Execution -> Agent Trajectories (collect experience) -> Training Data (successful paths) -> Fine-tune LLM -> Improved Agent. Show the learning loop from execution to fine-tuning.
```

**Key Elements:**
- Task Execution
- Trajectory Collection
- Experience gathering
- Training Data preparation
- LLM Fine-tuning
- Agent Improvement
- Learning feedback loop

---

## Paper 6: ARTIST - Agentic Reasoning with Tool Integration

**Prompt:**
```
Create a flowchart showing ARTIST: Query -> Reasoning (plan approach) -> Tool Selection -> Tool Execution -> Verification (check result) -> iterate if needed -> Answer. Show tool use with verification loop.
```

**Key Elements:**
- Query input
- Reasoning/Planning
- Tool Selection
- Tool Execution
- Verification step
- Iteration if needed
- Final Answer
- Tool integration

---

## Paper 7: OpenHands - AI Software Developers

**Prompt:**
```
Create a flowchart showing OpenHands agent architecture: User Request -> Task Analysis -> Code Generation -> Code Execution -> Test Results -> Error Analysis -> Code Revision -> loop until tests pass -> Final Solution. Show software development cycle.
```

**Key Elements:**
- User Request
- Task Analysis
- Code Generation
- Code Execution
- Test Results
- Error Analysis
- Code Revision
- Iteration loop
- Final Solution
- SDLC representation

---

## Paper 8: TapeAgents - Holistic Framework

**Prompt:**
```
Create a flowchart showing TapeAgents: Input -> Tape (record all steps) -> Reasoning (plan next step) -> Action (execute) -> Tape Update (append step) -> loop with learning from tape history. Show tape-based recording and learning.
```

**Key Elements:**
- Input task
- Tape initialization
- Step recording
- Reasoning/Planning
- Action execution
- Tape update
- History learning
- Continuous loop
- Tape-based paradigm

---

## Paper 9: Paper2Agent - Research Papers to AI Agents

**Prompt:**
```
Create a flowchart showing Paper2Agent pipeline: Research Paper (PDF) -> Paper Analysis -> Extract Methods/Algorithms -> Agent Specification -> Generate Agent Code -> Test Agent -> Deploy Agent. Show transformation pipeline.
```

**Key Elements:**
- Research Paper input (PDF)
- Paper Analysis
- Method/Algorithm Extraction
- Agent Specification
- Code Generation
- Agent Testing
- Agent Deployment
- Transformation pipeline

---

## Paper 10: Teaching LLMs to Self-Debug

**Prompt:**
```
Create a flowchart showing Self-Debugging: Generate Initial Solution -> Execute/Test -> Error Detection -> Bug Analysis -> Generate Fix (self-reflection) -> Apply Fix -> Re-test -> loop until no errors. Show the debugging feedback loop.
```

**Key Elements:**
- Initial Solution generation
- Code execution/testing
- Error detection
- Bug analysis
- Self-reflection
- Fix generation
- Fix application
- Re-testing
- Debugging loop
- Iteration until success

---

## Usage Instructions

1. Make sure your Eraser API account has sufficient credits
2. Use the Python script `generate_diagrams.py` to automate generation
3. Each diagram will be saved as a JSON file with:
   - `imageUrl`: Direct link to the diagram image
   - `fileUrl`: Download link for the diagram file
   - `requestId`: Unique identifier for the generation request

4. To generate a single diagram manually:
```bash
curl -X POST https://app.eraser.io/api/render/prompt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer NUPBy0SOPgDWIAIl3PDk" \
  -d '{
    "text": "<paste prompt from above>",
    "mode": "premium",
    "diagramType": "flowchart-diagram"
  }'
```

---

## Notes

- All prompts are optimized for flowchart diagram type
- The "premium" mode provides higher quality diagrams
- Diagram generation typically takes 10-30 seconds per diagram
- The prompts include specific visual instructions (loops, parallel execution, etc.)
- Each prompt describes the key architectural elements of the framework

---

## Troubleshooting

If you encounter errors:

1. **402 Payment Required**: Monthly spending limit reached
   - Solution: Increase limit in Eraser dashboard or wait for reset

2. **Timeout (60s)**: Request processing time exceeded
   - Solution: Increase timeout value or retry the request

3. **400 Bad Request**: Invalid prompt format
   - Solution: Ensure the "text" field is properly escaped and quoted

4. **401 Unauthorized**: Invalid API key
   - Solution: Verify API key is correct and active

---

For updates or to regenerate diagrams, refer to the main generation script:
`generate_diagrams.py`

Here's the raw markdown — copy everything between the lines:

```markdown
# Agentic Supply Chain Layer (logi-cc)

An autonomous supply-chain disruption-mitigation agent. A discrete-event
logistics simulation feeds live network telemetry into a Spatio-Temporal
Graph Attention Network (ST-GAT) that predicts cascading "blast radius"
risk across the hub network. A ReAct agent (powered by Gemini via
LangChain/LangGraph) observes that risk, reasons about root cause, and
acts — rerouting shipments, reordering inventory, or escalating to a
human — with every action passing through a human-in-the-loop guardrail
layer before it executes.

Available as both a live Rich CLI dashboard (`main.py`) and an interactive
Streamlit command center (`streamlit_app.py`).

## How it works

```
SimPy tick → env.tick()
    ↓
env.get_pyg_tensors() → build Data object
    ↓
model.forward(data) → raw_risk_tensor            (ST-GAT, ~150k params)
    ↓
propagate_blast_radius() → final risk_scores dict
    ↓
agent.run_cycle(risk_scores, env_state)
  ├─ observe()   → observation string
  ├─ reason()    → chain-of-thought + decision (Gemini LLM)
  └─ act()       → guardrail.intercept() → tool call
    ↓
dashboard.render() → CLI / Streamlit refresh
```

### Components

| Module | Role |
|---|---|
| `supply_chain/environment.py` | Discrete-event logistics simulation (nodes = hubs, edges = carrier routes). Models throughput, backlog, health, and failure injection/propagation. |
| `supply_chain/gnn_model.py` | ST-GAT model — 3-layer Graph Attention Network + temporal attention head predicting per-node risk (1-, 2-, and 3-hop cascading failure). |
| `supply_chain/agent.py` | ReAct agent: observe → reason (LLM chain-of-thought) → act (tool call) → reflect (memory update). Tools: `reroute_shipment`, `reorder_inventory`, `request_human_intervention`. Uses ChromaDB for long-term memory of past incidents. |
| `supply_chain/guardrails.py` | Intercepts every tool call. Enforces cost (> $500) and volume (> 10%) approval gates, flags escalation requests, and keeps an audit log. |
| `supply_chain/dashboard.py` | Live Rich-powered CLI dashboard renderer. |
| `supply_chain/hub_coords.py` | Hub geo-coordinates and route definitions used by the Streamlit map view. |
| `train_gnn.py` | Training script for the ST-GAT model; produces `trained_gnn_weights.pt`. |

## Getting started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and add your Gemini API key (create one in
[Google AI Studio](https://aistudio.google.com/)):

```bash
cp .env.example .env
```

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Never commit your `.env` file** — it's already excluded via `.gitignore`.

### 3. Run it

**CLI dashboard:**

```bash
python main.py                                   # defaults: 20 ticks, HUB-NYC fails at tick 3
python main.py --fail-node HUB-CHI                # inject failure at a different hub
python main.py --ticks 30 --delay 1.5             # longer run, slower refresh
python main.py --auto-approve                     # demo mode: auto-approve guardrail escalations
python main.py --fail-tick 5 --severity 0.9 --resolve-tick 12
```

**Streamlit command center:**

```bash
streamlit run streamlit_app.py
```

The Streamlit UI lets you step through ticks manually, choose which hub
fails and when, switch between trained and heuristic GNN weights, and
approve or decline high-cost actions and human-intervention requests from
the agent in real time.

### 4. (Optional) Train the GNN

```bash
python train_gnn.py
```

This produces `trained_gnn_weights.pt`, which both `main.py` and
`streamlit_app.py` will load automatically if present (falling back to a
heuristic warm-start otherwise).

## Human-in-the-loop guardrails

Every proposed agent action is intercepted before execution:

- **Cost gate** — any action with estimated cost > $500 requires approval.
- **Volume gate** — any action rerouting > 10% of total network volume requires approval.
- **Escalation** — any `request_human_intervention` call is always flagged.
- **Audit log** — every intercepted command is recorded, approved or not.

In the CLI, `--auto-approve` simulates an operator approving reasonable
escalations under a secondary cap. In Streamlit, approval is a real
button click — nothing executes until you approve or decline it.

## Requirements

See `requirements.txt`. Key dependencies: `torch`, `torch-geometric`,
`simpy`, `langchain` / `langgraph` / `langchain-google-genai`, `chromadb`,
`streamlit`, `plotly`, `rich`.
```


# Instacart Multi-Agent AI Customer Service Assistant

**Agentic AI | San José State University**

A multi-agent AI customer service assistant that uses historical Instacart purchasing data to answer customer questions, retrieve order history, and recommend grocery products.

## Project Overview

This project combines data engineering with agentic AI. We use Python and DuckDB to process a large e-commerce dataset, and integrates specialized agents using LangGraph and optional OpenAI calls.

### Dataset

**Source:** [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)

| Dataset | Records |
|---|---:|
| Orders | 3,421,083 |
| Prior order products | 32,434,489 |
| Training order products | 1,384,617 |
| Products | 49,688 |
| Aisles | 134 |
| Departments | 21 |

The dataset contains anonymized historical transactions. It does not include live delivery tracking, prices, or refund records.

## AI Agent Architecture

The system includes three specialized agents:

1. **Order History Agent** — Retrieves historical customer orders and purchased items.
2. **Product Recommendation Agent** — Recommends grocery products using purchasing patterns.
3. **Customer Support Agent** — Answers customer questions using a clearly identified simulated support policy.

A LangGraph router directs requests to the appropriate agent. MemorySaver keeps in-process conversation state; new chats, customer changes, and server restarts start fresh conversations.

## Technology Stack

- **Python:** Data processing and application logic
- **DuckDB:** SQL analytics across millions of records
- **Pandas:** Data exploration and validation
- **OpenAI API:** Language model integration
- **LangGraph:** Agent routing and orchestration
- **MemorySaver:** Conversation memory
- **Git and GitHub:** Version control and team collaboration

## Project Structure

```text
Agentic-AI-customer-service/
├── agents/
│   ├── order_agent.py
│   ├── product_agent.py
│   ├── refund_agent.py
│   └── router.py
├── data/
│   ├── raw/
│   └── policies/
│       └── refund_policy.txt
├── scripts/
│   ├── download_data.py
│   ├── data_check.py
│   └── data_quality.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Mohamed254-pixel/Agentic-AI-customer-service.git
cd Agentic-AI-customer-service
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download the six CSV files from Kaggle and place them in:

`data/raw/instacart/`

Large dataset files are excluded from GitHub.

### 5. Run data validation

```bash
python scripts/data_check.py
python scripts/data_quality.py
```

## Development Progress

### Run Cartside locally

With your virtual environment activated, run:

```bash
python -m pip install -r requirements-preview.txt
python scripts/build_db.py --dataset sample
python -m streamlit run main.py
```

Open the localhost URL printed in the terminal. The preview includes a demo
customer selector, historical order summaries, and a wide chat connected to
the order, recommendation, and simulated refund agents. Order history is
available in the sidebar.

Product photo assets are retained for future use but are not displayed in the
chat layout. Photos are representative; the dataset does not verify historical
brands or package sizes. Sources are recorded in `assets/products/sources.json`.

- [x] Create GitHub repository
- [x] Set up Python environment
- [x] Download Instacart dataset
- [x] Verify all six dataset row counts
- [x] Validate order and product ID uniqueness
- [x] Check initial missing values
- [ ] Complete full data quality validation
- [x] Create persistent DuckDB database
- [x] Implement three agents
- [x] Connect agents using LangGraph
- [x] Implement MemorySaver
- [x] Test offline conversations and agent responses
- [ ] Verify live OpenAI responses with a valid API key
- [ ] Complete project presentation

## Team Collaboration

Developed as a four-member group project for BUS4 118S at San José State University.

Team members contribute through GitHub branches and pull requests.

## Project Status

**In development:** Integrated agents and frontend verified in offline demo mode. Live OpenAI validation and the final presentation remain to be completed.

### OpenAI mode and testing

Copy `.env.example` to `.env`. For offline mode, keep `DEMO_MODE=1`.
For OpenAI mode, set `DEMO_MODE=0` and supply your own `OPENAI_API_KEY`.
Restart Streamlit after changing settings. Never commit `.env` or an API key.
The app labels offline replies and falls back to database/policy responses if
OpenAI is unavailable. The OpenAI API is required for the assignment's live AI
requirement; offline mode is for development and does not satisfy it by itself.

The refund policy is simulated, not Instacart's policy. No refunds, live tracking,
prices, payments, or order changes are available.

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_db.py --dataset sample
DEMO_MODE=1 python -m pytest tests -q
```

Example three-turn order demo (customer `5117`):
1. `What was in my latest order?`
2. `Show that again`
3. `What was in it?`

Also try `Recommend bananas` and `What is the refund policy?`.
Memory tests use both real local graph execution and a fake model that checks
history delivery without making paid API calls. Arbitrary follow-up understanding
is more limited in offline mode than OpenAI mode.

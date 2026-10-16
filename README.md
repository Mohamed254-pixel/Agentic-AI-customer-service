# Instacart Multi-Agent AI Customer Service Assistant

**Agentic AI | San José State University**

A multi-agent AI customer service assistant that uses historical Instacart purchasing data to answer customer questions, retrieve order history, and recommend grocery products.

## Project Overview

This project combines data engineering with agentic AI. We use Python and DuckDB to process a large e-commerce dataset, then plan to integrate specialized AI agents using LangGraph and OpenAI.

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

The planned system includes three specialized agents:

1. **Order History Agent** — Retrieves historical customer orders and purchased items.
2. **Product Recommendation Agent** — Recommends grocery products using purchasing patterns.
3. **Customer Support Agent** — Answers customer questions using a clearly identified simulated support policy.

A LangGraph router will direct requests to the appropriate agent. MemorySaver will maintain conversation context across multiple turns.

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

### Local frontend preview

With your virtual environment activated, run:

```bash
python -m pip install -r requirements-preview.txt
python scripts/build_db.py --dataset sample
python -m streamlit run main.py
```

Open the localhost URL printed in the terminal. The preview includes a demo
customer selector, historical order summaries, and a wide chat placeholder
with a bottom input bar. Order history is available in the sidebar.
AI agents are not connected.

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
- [ ] Create persistent DuckDB database
- [ ] Implement three AI agents
- [ ] Connect agents using LangGraph
- [ ] Implement MemorySaver
- [ ] Test conversations and agent responses
- [ ] Complete project presentation

## Team Collaboration

Developed as a four-member group project for BUS4 118S at San José State University.

Team members contribute through GitHub branches and pull requests.

## Project Status

**In development:** Dataset preparation and initial validation completed. AI agent implementation is upcoming.

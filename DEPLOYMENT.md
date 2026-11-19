# Deploy Cartside to Streamlit Community Cloud

## Publish the free offline demo

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io) with a GitHub account that can access `Mohamed254-pixel/Agentic-AI-customer-service`.
2. Select **Create app** and deploy from the existing GitHub repository. Choose the branch containing this release and enter `main.py` for the main file path.
3. In **Advanced settings**, select Python **3.13**. Leave **Secrets** empty, or use `DEMO_MODE = "1"`. The public app uses no OpenAI key.
4. Deploy and wait for the app to finish starting. Set its sharing visibility to public and copy the assigned `.streamlit.app` URL.
5. Complete the smoke test below in a signed-out browser before adding that URL to the README.

The first session builds DuckDB from the six bundled CSVs and creates a separate simulated pricing database. Both generated files are disposable. Startup requires no Kaggle downloads, external services, or full historical data. The sample contains 10,000 orders and 97,855 basket rows. Community Cloud can put inactive apps to sleep. An anonymous visitor can wake a public app from its sleep screen.

[Community Cloud deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [Resource limits and hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Release smoke test

| Check | Expected result |
| --- | --- |
| Open the URL while signed out | Cartside loads and shows **Demo mode** and the portfolio notice. |
| Customer 5117 → latest order | Order `1405149` lists Lemon Yogurt, Blueberry Yoghurt and Original No Pulp 100% Florida Orange Juice. |
| Ask for the first order, then ask “Show that again” | Both replies refer to order `1662937`. The first basket contains four catalog items. |
| Click the purchase-history starter in a new chat | Recommendations use this customer's basket history and explain the ranking. |
| Click the order-help starter in a new chat | The reply explains historical order lookup, real-order support, and the simulated policy. |
| Switch customer or click **New chat** | The prior transcript and recommendation context disappear. |
| Open **Orders** and **Shopping list** | Historical baskets load. Selecting a catalog item shows an explicitly simulated USD price and total. |

For a local HTTP check, start the server and query its health endpoint. Run the existing AppTest integrations to execute the actual app script and exercise interactive behavior.

```bash
DEMO_MODE=1 OPENAI_API_KEY= python -m streamlit run main.py --server.headless=true --server.address=127.0.0.1
curl --fail http://127.0.0.1:8501/_stcore/health
```

## Private OpenAI testing

Copy `.env.example` to `.env`. Put your key in that ignored local file and set `DEMO_MODE=0`. Start Streamlit on loopback with `python -m streamlit run main.py --server.address=127.0.0.1`. Keep this instance private. The Streamlit integration requires loopback binding and opt-in, with a request timeout, zero retries and an output-token cap. The public Community Cloud deployment uses offline mode. The helper CLI agents also require explicit `DEMO_MODE=0` before enabling live requests.

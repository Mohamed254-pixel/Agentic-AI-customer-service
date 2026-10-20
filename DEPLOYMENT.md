# Publish Cartside

The app builds its sample DuckDB database automatically on first launch from
`data/sample/`. No Kaggle download or committed database is required. Conversation
memory belongs to each browser session and is cleared by New chat, customer
changes, or server restarts.

1. Sign in to https://share.streamlit.io with the GitHub account that can access
   `Mohamed254-pixel/Agentic-AI-customer-service`.
2. Create an app from that repository. Choose the branch containing this update
   and `main.py` as the entry point. Select Python 3.13 in advanced settings.
3. Copy the format from `.streamlit/secrets.toml.example` into the Cloud Secrets
   field. Enter the real API key there and use `DEMO_MODE = "0"`. For an offline
   public demo, omit the key and set `DEMO_MODE = "1"`.
4. Deploy, then verify customer 5117: latest order 1405149, first order 1662937,
   recommendations for bananas, and the clearly simulated refund policy.
5. Test a three-turn conversation, New chat, and switching customers before
   sharing the generated `.streamlit.app` URL.

Keep `.env` and `.streamlit/secrets.toml` out of Git. Hosted API calls use the
key configured in Cloud Secrets. Adjust the app's sharing settings to match
the intended class audience before sharing the link.

Official instructions:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

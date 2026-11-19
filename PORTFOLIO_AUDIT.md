# Cartside completion checklist

The starting checkout had three LangGraph specialists, deterministic simulated prices, a session shopping list, sample CSVs, and automatic database creation. The original offline suite reported `74 passed in 16.43s`.

| Priority | Audit finding | Completion |
| --- | --- | --- |
| P0 | The purchase-history prompt did not use the selected customer's history. | Rank products from customer baskets and explain frequency, recency, and sample-wide fallback. |
| P0 | A reused checkpoint could retain the previous customer's messages and recommendation state. | Clear the active graph context on customer change and test the same thread with two customers. |
| P0 | Broad routing sent some order-item and support prompts to the wrong specialist. Model prose could add unsupported facts. | Give explicit order, recommendation and support intents precedence. Render order records and policy replies from verified sources. |
| P0 | The deployment guide encouraged an API key on a potentially public app. | Make the hosted runtime offline even when a key is present. Require an explicit loopback launch and opt-in for private OpenAI testing. |
| P0 | The existing database builder inferred CSV types and published without integrity checks. | Check CSV headers, schema types, key uniqueness, references, order fields, basket positions and split membership before atomic publication. |
| P1 | Orders opened a DuckDB connection for each basket. | Fetch the selected customer's baskets in one query. Keep the historical tables read-only. |
| P1 | The repository lacked GitHub Actions and current verification evidence. | Run offline tests and static checks in CI. Verify startup, package resolution, and dependency advisories locally. |
| P1 | The onboarding prompts did not demonstrate personalization and order help. | Add prompts for those two routes and a visible historical-data and simulated-policy notice. |
| P1 | A clean Community Cloud deployment and anonymous visitor check are required. | Follow the release smoke test in [Deployment](DEPLOYMENT.md). |
| P2 | Full-data checks, browser screenshots, and a paid API smoke test are independent of the offline sample. | Record verification separately when the raw dataset, browser or paid test authorization is available. |

from src.data_store import sync_data
from src.enrichment.asset_metadata import AssetEnricher
import pandas as pd

print("Starting full sync...")
# 1. Scrape
df = sync_data()
print(f"Scraped {len(df)} records.")

# 2. Enrich
print("Starting enrichment...")
enricher = AssetEnricher()
df_enriched = enricher.enrich_dataframe(df)
print("Enrichment complete.")

# Save back just in case sync_data didn't save the enriched version (it doesn't, it saves raw)
# But wait, the app pipeline enriches on the fly or saves?
# app.py: get_data_pipeline calls sync_data(), then enricher.enrich_dataframe(df). 
# It does NOT save the enriched data back to CSV. It caches it in memory with @st.cache_data.
# So the CSV remains raw. That's fine.
# However, to "test" if errors are gone, I should run the enrichment here.
print("Done.")


# AIReady Scanner MVP

A simple Streamlit web app that scans a business website and creates an AI visibility readiness report.

## What it checks

- robots.txt
- sitemap.xml
- page crawlability/status codes
- title, meta description, H1
- Schema.org JSON-LD
- LocalBusiness / Organization schema
- phone, email, address, hours
- service/booking language
- social/profile links
- FAQ/review schema signals
- Open Graph/Twitter metadata
- image alt text
- simple page speed/page size indicators

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Simplest hosting option

Use Streamlit Community Cloud:

1. Create a GitHub repo.
2. Upload these files:
   - `app.py`
   - `ai_readiness_scanner.py`
   - `report_pdf.py`
   - `ai_simulation.py`
   - `requirements.txt`
3. Go to Streamlit Community Cloud.
4. Choose your repo, branch, and `app.py`.
5. Click Deploy.

## Optional: AI-answer simulation

The "What would an AI assistant say about you?" feature calls the Anthropic
API to show, in real words, what an assistant would tell a customer based only
on what it can read on the page. It only appears when an Anthropic API key is
available to the app (set `ANTHROPIC_API_KEY` — on Streamlit Community Cloud,
add it under the app's Secrets). Without a key the app runs exactly as before,
just without that one section.

## MVP note

The built-in lead capture creates a downloadable CSV for now. For a real MVP, connect the form to one of these:

- Tally form embedded/link
- Airtable form
- Google Form
- Make/Zapier webhook
- Supabase table
- HubSpot form

Fastest path: use Tally or Airtable for lead capture and put the link under the report.

## Suggested product naming

Brand: AIReady  
Tool: AIReady Scanner  
Report metric: AI Readiness Score

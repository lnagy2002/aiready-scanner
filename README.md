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
   - `requirements.txt`
3. Go to Streamlit Community Cloud.
4. Choose your repo, branch, and `app.py`.
5. Click Deploy.

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

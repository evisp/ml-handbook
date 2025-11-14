# Data Collection: Working with APIs

This tutorial covers how to retrieve data from web APIs using Python. You'll learn to make HTTP requests, handle common API patterns like pagination and rate limits, and transform JSON responses into usable datasets.

**Estimated time:** 35 minutes

## Why This Matters

**Problem statement:** 

> Most valuable data lives behind APIs, not in downloadable files.

**Real-world data sources** include weather services, social media platforms, financial markets, government databases, and business tools. These services expose data through APIs (Application Programming Interfaces) that require specific request patterns, authentication, and response handling.

**Models need training data**, and APIs provide access to fresh, structured information that CSV files can't match.

**Practical benefits:** API skills let you collect current data, automate data pipelines, and access datasets that don't exist as files. You'll build systems that fetch updated information on demand rather than relying on static downloads.

**Professional context:** Data scientists regularly integrate multiple API sources. Companies that master API collection build dynamic systems; those that rely only on manual downloads face outdated information and scaling problems.

> API access is standard practice. Every data role expects you to fetch and integrate external data.

![API Workflow](https://i.imgur.com/4jdT6SU.png)

## Core Concepts

### What Is an API?

An **API (Application Programming Interface) is a service that accepts structured requests and returns structured data**. Instead of clicking through a website, you send HTTP requests directly to endpoints and receive responses in JSON or XML format.

**Key terminology:**

**Endpoint:** A specific URL that provides access to a resource (e.g., `https://api.service.com/users`)

**HTTP Methods:** Actions you can perform:

- **GET:** Retrieve data (most common)
- **POST:** Send new data
- **PUT/PATCH:** Update existing data
- **DELETE:** Remove data

**Request:** What you send to the API (URL, parameters, headers)

**Response:** What the API returns (status code, data, headers)

**JSON:** JavaScript Object Notation; the standard format for API data exchange

**Rate Limit:** Maximum number of requests allowed per time period (e.g., 100 requests/hour)

**Authentication:** Credentials proving you're authorized to access the API (API keys, tokens)

### The Request-Response Cycle

Every API interaction follows the same pattern:

![API Cycle](https://i.imgur.com/3CmF8fM.png)

**1. You send request → 2. API processes → 3. API returns response → 4. You parse data**

Understanding this cycle helps you debug issues and handle errors systematically.

## Step-by-Step Guide

### Phase 1: Making Your First API Request

> Start simple: fetch one resource and examine the response.

**Install the requests library:**

```bash
pip install requests
```

**Basic GET request:**

```python
import requests

# Define endpoint
url = "https://swapi-api.hbtn.io/api/people/1/"

# Make request
response = requests.get(url)

# Check if successful
print(f"Status Code: {response.status_code}")

# View raw response
print(f"Response Text: {response.text}")
```

**Status codes tell you what happened:**

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Parse the data |
| 400 | Bad Request | Check your parameters |
| 401 | Unauthorized | Verify authentication |
| 404 | Not Found | Endpoint or resource doesn't exist |
| 429 | Too Many Requests | You hit rate limit |
| 500 | Server Error | API issue; try again later |

**Tip:** Always check `response.status_code` before parsing data. Attempting to parse a failed request causes errors.

### Phase 2: Working with JSON Data

Most APIs return JSON, a structured format that Python handles natively.

**Converting JSON to Python:**

```python
# Fetch data
response = requests.get("https://swapi-api.hbtn.io/api/people/1/")

# Parse JSON into dictionary
if response.status_code == 200:
    data = response.json()          # Converts JSON string to Python dict
    print(f"Name: {data['name']}")
    print(f"Height: {data['height']}")
    print(f"Films: {data['films']}")
else:
    print(f"Error: {response.status_code}")
```

**JSON structure maps to Python types:**

- JSON object `{}` → Python dictionary
- JSON array `[]` → Python list
- JSON string → Python string
- JSON number → Python int/float
- JSON boolean → Python True/False
- JSON null → Python None

**Navigating nested JSON:**

```python
# Access nested fields
data = response.json()
homeworld_url = data['homeworld']
films_list = data['films']

# Extract specific values{films_list}")
```

### Phase 3: Adding Parameters to Requests

APIs accept parameters to filter, search, or customize responses.

**Query parameters:**

```python
# Search for characters named "Luke"
url = "https://swapi-api.hbtn.io/api/people/"
params = {"search": "Luke"}

response = requests.get(url, params=params)
data = response.json()

print(f"Results found: {data['count']}")
print(f"First result: {data['results']['name']}")
```

**The `params` argument automatically formats the URL:**
```
https://swapi-api.hbtn.io/api/people/?search=Luke
```

**Multiple parameters:**

```python
params = {
    "search": "Skywalker",
    "format": "json"
}
response = requests.get(url, params=params)
```

**Tip:** Use dictionaries for parameters. The requests library handles URL encoding automatically.

### Phase 4: Handling Pagination

APIs split large datasets across multiple pages. You must request each page to get complete data.

**Detecting pagination:**

```python
response = requests.get("https://swapi-api.hbtn.io/api/people/")
data = response.json()

print(f"Total results: {data['count']}")
print(f"Next page: {data['next']}")
print(f"Results on this page: {len(data['results'])}")
```

**Collecting all pages:**

```python
def fetch_all_pages(base_url):
    all_results = []
    url = base_url
    
    while url:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
            
        data = response.json()
        all_results.extend(data['results'])  # Add this page's results
        url = data['next']  # Get next page URL (None if last page)
        
    return all_results

# Fetch all Star Wars characters
characters = fetch_all_pages("https://swapi-api.hbtn.io/api/people/")
print(f"Total characters collected: {len(characters)}")
```

**Pagination patterns vary by API:**

- **Link-based:** API provides `next` URL (like SWAPI)
- **Offset-based:** You increment `?offset=20` or `?page=2`
- **Cursor-based:** API provides token for next page

Always read API documentation to understand the pagination method.

### Phase 5: Respecting Rate Limits

APIs restrict request frequency to prevent abuse and maintain performance.

**Common rate limit formats:**

- 100 requests per hour
- 10 requests per second
- 1000 requests per day

**Handling rate limits:**

```python
import time

def fetch_with_rate_limit(urls, delay=1):
    """Fetch multiple URLs with delay between requests."""
    results = []
    
    for url in urls:
        response = requests.get(url)
        
        if response.status_code == 429:  # Rate limit hit
            print("Rate limit reached. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url)  # Retry
            
        if response.status_code == 200:
            results.append(response.json())
            
        time.sleep(delay)  # Delay between requests
        
    return results

# Fetch multiple characters with 1-second delay
character_urls = [
    "https://swapi-api.hbtn.io/api/people/1/",
    "https://swapi-api.hbtn.io/api/people/2/",
    "https://swapi-api.hbtn.io/api/people/3/"
]
characters = fetch_with_rate_limit(character_urls, delay=1)
```

**Best practices:**

- **Read headers:** Many APIs include rate limit info in response headers (`X-RateLimit-Remaining`)
- **Add delays:** Insert `time.sleep()` between requests
- **Handle 429 responses:** Wait and retry when rate limited
- **Cache results:** Store responses to avoid redundant requests

### Phase 6: Transforming API Data into DataFrames

Once collected, convert JSON to pandas DataFrames for analysis.

**Basic conversion:**

```python
import pandas as pd

# Fetch data
response = requests.get("https://swapi-api.hbtn.io/api/people/")
data = response.json()

# Convert results to DataFrame
df = pd.DataFrame(data['results'])
print(df[['name', 'height', 'mass', 'birth_year']].head())
```

**Handling nested fields:**

```python 
# Some fields contain lists or nested objects
# Flatten or extract specific information

# Extract first film URL for each character
dff len(x) > 0 else None)

# Convert height to numeric (some values are "unknown")
df['height'] = pd.to_numeric(df['height'], errors='coerce')

# Clean and prepare for analysis
df_clean = df[['name', 'height', 'mass', 'gender']].dropna()
```

**Enriching data with related resources:**

```python
def get_homeworld_name(homeworld_url):
    """Fetch homeworld name from URL."""
    response = requests.get(homeworld_url)
    if response.status_code == 200:
        return response.json()['name']
    return None

# Add homeworld names
df['homeworld_name'] = df['homeworld'].apply(get_homeworld_name)
```

**Warning:** Fetching related resources for every row can trigger rate limits. Add delays or batch requests.

## Common API Challenges

### Authentication Required

Many APIs require credentials.

**API key in headers:**

```python
headers = {"Authorization": "Bearer YOUR_API_KEY"}
response = requests.get(url, headers=headers)
```

**API key in parameters:**

```python
params = {"api_key": "YOUR_API_KEY"}
response = requests.get(url, params=params)
```

> Always store API keys in environment variables, never in code:

```python
import os
api_key = os.environ.get("API_KEY")
```

### Network Errors and Timeouts

Requests can fail due to network issues.

**Handling errors:**

```python
try:
    response = requests.get(url, timeout=10)  # Wait max 10 seconds
    response.raise_for_status()  # Raises exception for 4xx/5xx codes
    data = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Connection failed")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
```

### Inconsistent Data Formats

APIs sometimes return inconsistent structures.

**Defensive parsing:**

```
# Handle missing or inconsistent fields
name = data.get('name', 'Unknown')
height = data.get('height', None)
films = data.get('films', [])

# Validate expected structure
if 'results' in data and isinstance(data['results'], list):
    process_results(data['results'])
else:
    print("Unexpected response format")
```

## Best Practices

**Read API documentation first:** Spend 15 minutes understanding endpoints, parameters, authentication, and rate limits before writing code. Saves hours of debugging.

**Test with small requests:** Fetch one resource successfully before building loops that make hundreds of requests.

**Handle errors gracefully:** APIs fail. Always check status codes and include try-except blocks for network errors.

**Respect rate limits:** Add delays between requests. Getting blocked wastes more time than being patient.

**Cache responses:** Save API responses to files during development. Avoids re-requesting the same data during testing.

```python
import json

# Save response
with open('cache.json', 'w') as f:
    json.dump(response.json(), f)

# Load cached data
with open('cache.json', 'r') as f:
    data = json.load(f)
```

**Log your requests:** Track what you've fetched to avoid duplicates and debug issues.

**Version API endpoints:** APIs change. Note the API version you're using (e.g., `/v1/`, `/v2/`) to ensure reproducibility.

## Quick Reference

### Essential requests Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `requests.get(url)` | Fetch data | `response = requests.get(url)` |
| `response.status_code` | Check success | `if response.status_code == 200:` |
| `response.json()` | Parse JSON | `data = response.json()` |
| `requests.get(url, params=dict)` | Add parameters | `requests.get(url, params={'search': 'Luke'})` |
| `requests.get(url, headers=dict)` | Add headers | `requests.get(url, headers={'Authorization': 'Bearer KEY'})` |
| `requests.get(url, timeout=10)` | Set timeout | `requests.get(url, timeout=10)` |

### Pagination Pattern Template

```
def fetch_all_pages(base_url):
    results = []
    url = base_url
    
    while url:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results.extend(data['results'])
            url = data.get('next')  # Or handle your API's pagination
        else:
            break
            
    return results
```

### Error Handling Template

```python
def safe_api_request(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None
```

## Summary & Next Steps

**Key accomplishments:** You've learned to make HTTP GET requests and parse JSON responses, handle pagination to collect complete datasets, respect rate limits and implement delays, transform API data into pandas DataFrames, and handle common errors and network issues.

**Critical insights:**

- **APIs are structured:** Every API follows request-response patterns with consistent status codes
- **Documentation matters:** Reading docs prevents hours of trial-and-error debugging
- **Be respectful:** Rate limits exist for good reasons; always add delays
- **Errors happen:** Network issues are normal; build retry logic and error handling

**What's next:**

With API data collection skills, you can now build automated data pipelines that fetch fresh information daily. The next step is combining API data with other sources and preprocessing it for analysis.

**Practice resources:**

- [SWAPI - Star Wars API](https://swapi-api.hbtn.io/) - free, no authentication required
- [JSONPlaceholder](https://jsonplaceholder.typicode.com/) - fake API for testing
- [Public APIs List](https://github.com/public-apis/public-apis) - hundreds of free APIs to practice with

**External resources:**

- [requests Documentation](https://requests.readthedocs.io/) - comprehensive guide to the requests library
- [HTTP Status Codes](https://httpstatuses.com/) - understand what each code means
- [REST API Tutorial](https://restfulapi.net/) - deeper dive into API design patterns

> **Remember:** APIs unlock real-time data. Master these patterns, and you'll build systems that stay current instead of relying on stale files.
```
"""
Test RapidAPI LinkedIn Job Search
"""
import requests
import json

RAPIDAPI_KEY = "2a86f3cdbfmsh5d8dfaabd0dd421p190709jsnf3fe78ba9559"

def test_simple_search():
    """Test with a very simple search"""
    
    url = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-7d"
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"
    }
    
    # Very simple query
    querystring = {
        "limit": "10",
        "offset": "0",
        "title_filter": "\"Software Engineer\"",
        "location_filter": "\"United States\"",
        "description_type": "text"
    }
    
    print("="*60)
    print("🧪 TESTING RAPIDAPI CONNECTION")
    print("="*60)
    print(f"URL: {url}")
    print(f"Query: {querystring}")
    print(f"API Key: {RAPIDAPI_KEY[:20]}...")
    print("="*60)
    
    try:
        print("\n⏳ Sending request to RapidAPI...")
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        
        print(f"\n📊 Response Status Code: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}\n")
        
        if response.status_code == 200:
            print("✅ API Request Successful!\n")
            
            data = response.json()
            print(f"📦 Response Type: {type(data)}")
            print(f"📦 Response Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
            
            # Pretty print the response
            print("\n📄 Full Response:")
            print(json.dumps(data, indent=2)[:2000])  # First 2000 chars
            
            # Try to extract jobs
            if isinstance(data, list):
                print(f"\n✅ Found {len(data)} jobs (list format)")
                if data:
                    print(f"\n📋 First job sample:")
                    print(json.dumps(data[0], indent=2))
            elif isinstance(data, dict):
                jobs = data.get('data', data.get('jobs', data.get('results', [])))
                print(f"\n✅ Found {len(jobs)} jobs (dict format)")
                if jobs:
                    print(f"\n📋 First job sample:")
                    print(json.dumps(jobs[0], indent=2)[:500])
            else:
                print("\n❌ Unexpected response format")
        
        elif response.status_code == 403:
            print("❌ 403 FORBIDDEN - API Key might be invalid or expired")
            print(f"Response: {response.text}")
        
        elif response.status_code == 429:
            print("❌ 429 TOO MANY REQUESTS - Rate limit exceeded")
            print(f"Response: {response.text}")
        
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_search()
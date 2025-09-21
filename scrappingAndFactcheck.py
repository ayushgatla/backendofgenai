# scrappingAndFactcheck.py - Production Ready Version

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime
import logging
import random
import threading
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("key.env")

# Load proxies from environment variable
PROXIES = os.getenv('PROXIES').split(',') if os.getenv('PROXIES') else []

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Google Custom Search API configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# Trusted website lists for fact-checking by domain
GENERAL_TRUSTED_WEBSITES = [
    "wikipedia.org", "britannica.com", "nationalgeographic.com",
    "apnews.com", "reuters.com", "bbc.com/news", "nytimes.com",
    "wsj.com", "factcheck.org", "snopes.com", "politifact.com",
]

HEALTH_TRUSTED_WEBSITES = [
    # Indian Government/Health Organizations
    "mohfw.gov.in", "icmr.gov.in", "aiims.edu", "nhp.gov.in",
    "phfi.org", "nihfw.org", "indianpediatrics.net", "fssai.gov.in",
    "mciindia.org", "ncdc.gov.in", "tmc.gov.in", "pgimer.edu.in",
    "sctimst.ac.in",
    # International Health Organizations
    "cdc.gov", "mayoclinic.org", "medlineplus.gov", "fda.gov",
    "health.gov", "webmd.com", "healthline.com", "nhs.uk",
    "health.harvard.edu", "heart.org", "hopkinsmedicine.org",
    "medicalnewtoday.com", "nia.nih.gov", "thelancet.com",
    "wikipedia.org", "everydayhealth.com", "clevelandclinic.org",
    "onlymyhealth.com", "health.economictimes.indiatimes.com",
    "maxhealthcare.in", "netmeds.com", "1mg.com", "cabidigitallibrary.org",
]

FINANCE_TRUSTED_WEBSITES = [
    "rbi.org.in", "sebi.gov.in", "bseindia.com", "nseindia.com",
    "moneycontrol.com", "economictimes.indiatimes.com",
    "business-standard.com", "financialexpress.com", "livemint.com",
    "businesstoday.in", "crisil.com", "icra.in", "tradingeconomics.com",
    "investindia.gov.in", "ibef.org", "pib.gov.in", "taxmann.com",
    "caindia.org", "policybazaar.com", "india.gov.in",
    # International finance sources
    "investopedia.com", "bloomberg.com", "reuters.com", "wsj.com",
    "ft.com", "cnbc.com", "fidelity.com", "zacks.com", "fool.com",
    "wikipedia.org",
]

# Map domains to trusted websites
DOMAIN_TRUSTED_WEBSITES = {
    "Health": HEALTH_TRUSTED_WEBSITES,
    "Finance": FINANCE_TRUSTED_WEBSITES,
    "General": GENERAL_TRUSTED_WEBSITES,
    "Other": GENERAL_TRUSTED_WEBSITES,
}

# Real-time sources by domain
REALTIME_SOURCES_GENERAL = [
    "reddit.com", "reuters.com", "apnews.com", "bbc.com", "cnn.com",
    "indianexpress.com", "thehindu.com", "timesofindia.indiatimes.com",
    "hindustantimes.com", "thewire.in", "republicworld.com",
    "indiatoday.in", "news18.com", "zeenews.india.com"
]

REALTIME_SOURCES_FINANCE = [
    "moneycontrol.com", "bloomberg.com", "cnbc.com", 
    "economictimes.indiatimes.com", "livemint.com", "reuters.com", "apnews.com"
]

REALTIME_SOURCES_HEALTH = [
    "reuters.com", "apnews.com", "bbc.com", "cnn.com", "reddit.com"
]

REALTIME_DOMAIN_SOURCES = {
    "Health": REALTIME_SOURCES_HEALTH,
    "Finance": REALTIME_SOURCES_FINANCE,
    "General": REALTIME_SOURCES_GENERAL,
    "Other": REALTIME_SOURCES_GENERAL,
}

# Thread-local storage for Gemini models
_local = threading.local()

def get_gemini_model(config: Optional[Dict] = None):
    """Get thread-local Gemini model instance"""
    cache_key = str(config) if config else "default"
    
    if not hasattr(_local, 'models'):
        _local.models = {}
    
    if cache_key not in _local.models:
        default_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 500,
        }
        if config:
            default_config.update(config)
            
        _local.models[cache_key] = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=default_config,
        )
    
    return _local.models[cache_key]

class FactCheckResult:
    """Structured result class for fact-checking operations"""
    def __init__(self, news_id=None):
        self.news_id = news_id
        self.trusted_urls = []
        self.scraped_contents = []
        self.summarized_answer = ""
        self.fact_check_assessment = ""
        self.further_education_suggestions = ""
        self.trust_score = 0.0
        self.processing_errors = []
        self.sources_used = []
        self.scraped_content_count = 0
        self.success = False
        self.debug_data = {}

    def to_dict(self):
        """Convert result to dictionary for JSON serialization"""
        return {
            'news_id': self.news_id,
            'trusted_urls': self.trusted_urls,
            'scraped_content_count': self.scraped_content_count,
            'summarized_answer': self.summarized_answer,
            'fact_check_assessment': self.fact_check_assessment,
            'further_education_suggestions': self.further_education_suggestions,
            'trust_score': self.trust_score,
            'processing_errors': self.processing_errors,
            'sources_used': self.sources_used,
            'success': self.success,
            'debug_data': self.debug_data
        }

# Connection pool management for better performance
class ConnectionManager:
    """Manages aiohttp session with connection pooling"""
    
    def __init__(self):
        self._session = None
        self._lock = asyncio.Lock()
    
    async def get_session(self):
        """Get or create aiohttp session with connection pooling"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=15, connect=10)
                    connector = aiohttp.TCPConnector(
                        limit=20,
                        limit_per_host=10,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                    )
                    self._session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
                        }
                    )
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

# Global connection manager
_connection_manager = ConnectionManager()

async def perform_google_search(query: str, start_index: int, num_results: int = 10) -> Dict:
    """Perform Google Custom Search API call with error handling"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Google API Key or CSE ID not configured")
        return {}
        
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
        "start": start_index
    }

    try:
        response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during Google Custom Search API call (page {start_index}): {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error during Google Custom Search (page {start_index}): {e}")
        return {}

async def google_search_and_filter(query: str, misinformation_domain: str, max_total_results: int = 50, num_per_request: int = 10) -> List[str]:
    """Search Google and filter for trusted domains (Evergreen)"""
    logger.info(f"Searching Google for evergreen sources: \"{query}\" in {misinformation_domain} domain...")
    
    trusted_domains_list = DOMAIN_TRUSTED_WEBSITES.get(misinformation_domain, GENERAL_TRUSTED_WEBSITES)
    all_search_items = []
    
    for i in range(0, max_total_results, num_per_request):
        start_index = i + 1
        search_results = await perform_google_search(query, start_index, num_per_request)
        if search_results and search_results.get("items"):
            all_search_items.extend(search_results.get("items"))
        else:
            break
        await asyncio.sleep(1)  # Rate limiting

    logger.info(f"Found {len(all_search_items)} total search results for '{query}'")

    filtered_urls = []
    for item in all_search_items:
        url = item.get("link")
        if url:
            for trusted_domain in trusted_domains_list:
                if trusted_domain in url and url not in filtered_urls:
                    logger.info(f"Adding trusted URL (Evergreen): {url} (matched {trusted_domain})")
                    filtered_urls.append(url)
                    if len(filtered_urls) >= 5:
                        return filtered_urls
    
    logger.info(f"Filtered (Evergreen) {len(filtered_urls)} URLs for query '{query}'")
    return filtered_urls

async def google_search_realtime(query: str, misinformation_domain: str, max_total_results: int = 30, num_per_request: int = 10) -> List[str]:
    """Search Google for real-time sources"""
    logger.info(f"Searching Google for real-time sources: \"{query}\"...")
    
    preferred_domains = REALTIME_DOMAIN_SOURCES.get(misinformation_domain, REALTIME_SOURCES_GENERAL)
    all_search_items = []
    
    for i in range(0, max_total_results, num_per_request):
        start_index = i + 1
        search_results = await perform_google_search(query, start_index, num_per_request)
        if search_results and search_results.get("items"):
            all_search_items.extend(search_results.get("items"))
        else:
            break
        await asyncio.sleep(1)  # Rate limiting

    filtered_urls = []
    keywords = query.lower().split()[:5]  # Use first 5 words as keywords
    
    for item in all_search_items:
        url = item.get("link")
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        
        # Check relevance
        if not any(keyword in title or keyword in snippet for keyword in keywords):
            continue

        if not url:
            continue
            
        for domain in preferred_domains:
            if domain in url and url not in filtered_urls:
                logger.info(f"Adding preferred real-time URL: {url} (matched {domain})")
                filtered_urls.append(url)
                break
        
        if len(filtered_urls) >= 8:
            break

    # Fallback for insufficient results
    if len(filtered_urls) < 3:
        logger.info(f"Not enough preferred real-time URLs found. Using fallback for query '{query}'.")
        for item in all_search_items:
            url = item.get("link")
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()
            
            if not any(keyword in title or keyword in snippet for keyword in keywords):
                continue

            if not url or url in filtered_urls:
                continue
                
            if any(d in url for d in ["news", "live", "breaking", "latest"]):
                logger.info(f"Adding fallback real-time URL: {url}")
                filtered_urls.append(url)
                
            if len(filtered_urls) >= 8:
                break

    logger.info(f"Filtered (Real-time) {len(filtered_urls)} URLs for query '{query}'")
    return filtered_urls

async def scrape_url_with_retry(session: aiohttp.ClientSession, url: str, proxy: Optional[str] = None, max_retries: int = 2) -> Optional[str]:
    """Scrape content from a single URL with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            kwargs = {}
            if proxy:
                kwargs['proxy'] = proxy
                
            async with session.get(url, **kwargs) as response:
                if response.status == 429:  # Rate limited
                    if attempt < max_retries:
                        wait_time = random.uniform(5, 15) * (attempt + 1)
                        logger.warning(f"Rate limited on {url}, waiting {wait_time:.1f}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Final attempt failed for {url} due to rate limiting")
                        return None
                
                response.raise_for_status()
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract text from paragraphs
                paragraphs = soup.find_all('p')
                text_content = ' '.join([p.get_text() for p in paragraphs])
                if not text_content:
                    text_content = soup.get_text()
                
                return text_content.strip()
                
        except aiohttp.ClientResponseError as e:
            if e.status == 403:
                logger.warning(f"Access forbidden for {url}. Skipping.")
                return None
            elif attempt < max_retries:
                logger.warning(f"HTTP error {e.status} for {url}, retry {attempt + 1}")
                await asyncio.sleep(random.uniform(2, 5))
                continue
            else:
                logger.warning(f"Final HTTP error {e.status} for {url}: {e}")
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries:
                logger.warning(f"Error fetching {url}, retry {attempt + 1}: {e}")
                await asyncio.sleep(random.uniform(2, 5))
                continue
            else:
                logger.warning(f"Final error fetching {url}: {e}")
                
        except Exception as e:
            logger.warning(f"Unexpected error scraping {url}: {e}")
            break
    
    return None

async def async_scrape(urls: List[str]) -> List[str]:
    """Asynchronously scrape multiple URLs with connection pooling and proxy rotation"""
    logger.info(f"Scraping {len(urls)} URLs...")
    scraped_contents = []
    
    # Create proxy cycle if available
    proxy_cycle = None
    if PROXIES:
        random.shuffle(PROXIES)
        proxy_cycle = iter(PROXIES * ((len(urls) // len(PROXIES)) + 1))
    
    session = await _connection_manager.get_session()
    
    # Process URLs with concurrency control
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    async def scrape_with_semaphore(url):
        async with semaphore:
            proxy = next(proxy_cycle, None) if proxy_cycle else None
            content = await scrape_url_with_retry(session, url, proxy)
            if content:
                logger.info(f"Successfully scraped content from {url}. Length: {len(content)}")
                return content
            else:
                logger.warning(f"No content scraped from {url}")
                return None
    
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, str):
            scraped_contents.append(result)
        elif isinstance(result, Exception):
            logger.error(f"Scraping task failed: {result}")
    
    logger.info(f"Successfully scraped {len(scraped_contents)} out of {len(urls)} URLs")
    return scraped_contents

async def fact_check_evergreen_misinformation(input_news_text: str, scraped_data: List[str]) -> str:
    """Compare input news with trusted sources using Gemini"""
    logger.info("Performing evergreen fact-check analysis...")
    
    combined_trusted_content = " ".join(scraped_data)
    model = get_gemini_model()

    prompt = f"""Given the following original news text and content from trusted sources, analyze if the original news text contains misinformation related to evergreen topics.
    Focus on factual accuracy and consistency with the trusted sources. Provide a direct assessment of 'True', 'Potentially Misleading', or 'False' with a concise explanation of 2-3 lines, referencing sources for key confirmations or contradictions.

    Original News Text: {input_news_text}

    Trusted Sources Content: {combined_trusted_content[:8000]}

    Assessment:"""

    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during Gemini evergreen fact-check: {e}")
        return f"Fact-checking failed: {str(e)}"

async def fact_check_realtime_misinformation(input_news_text: str, scraped_data: List[str]) -> str:
    """Apply real-time fact-checking logic"""
    logger.info("Applying real-time fact-checking logic...")
    
    combined_trusted_content = " ".join(scraped_data)
    model = get_gemini_model({
        "temperature": 0.15,
        "max_output_tokens": 500,
    })

    prompt = f"""Given the following current claim and content scraped from real-time sources (news wires, social feeds, latest articles), assess if the claim is likely true, needs verification, or false.
    Specifically, compare the claim with the real-time source content. Provide a direct judgment: 'True', 'Needs Verification', or 'False', with a concise reasoning of 2-3 lines. Prioritize wire services and reputable outlets.

Claim: {input_news_text}

Real-time Source Content: {combined_trusted_content[:8000]}

Judgment:"""

    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during Gemini real-time fact-check: {e}")
        return f"Real-time fact-checking failed: {str(e)}"

async def summarize_scraped_data_with_gemini(scraped_data: List[str]) -> str:
    """Summarize scraped data using Gemini"""
    logger.info("Summarizing scraped data...")
    
    combined_content = "\n\n".join(scraped_data)
    model = get_gemini_model({
        "temperature": 0.2,
        "max_output_tokens": 500,
    })

    prompt = f"""Based on the following content from trusted sources, provide a concise and direct summary of the key information related to the topic, keeping the length to a minimum while retaining essential facts.

    Trusted Sources Content:
    {combined_content[:4000]}

    Summary:"""

    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during Gemini summarization: {e}")
        return f"Summarization failed: {str(e)}"

async def generate_further_education(news_text: str, misinformation_domain: str) -> str:
    """Generate educational suggestions using Gemini"""
    logger.info("Generating education suggestions...")

    model = get_gemini_model({
        "temperature": 0.3,
        "max_output_tokens": 300,
    })

    prompt = f"""Given the original news topic: "{news_text}" (categorized as {misinformation_domain} misinformation), suggest 3-5 concise key areas or reputable resources for an individual to further educate themselves to avoid similar misinformation in the future. Focus on critical thinking, media literacy, and understanding the {misinformation_domain} domain. Provide each suggestion on a new line, keeping each point to a single sentence.

    Suggestions:"""

    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating further education: {e}")
        return f"Further education suggestions could not be generated: {str(e)}"

def calculate_trust_score(fact_check_assessment: str) -> float:
    """Calculate trust score based on fact-check assessment"""
    assessment_lower = fact_check_assessment.lower()
    
    if "true" in assessment_lower and "false" not in assessment_lower:
        return 9.0
    elif "potentially misleading" in assessment_lower or "needs verification" in assessment_lower:
        return 5.0
    elif "false" in assessment_lower:
        return 1.0
    else:
        return 0.0

def save_debug_data(result: FactCheckResult, news_text: str, news_type: str, misinformation_domain: str) -> Optional[str]:
    """Save debug data to JSON file with error handling"""
    debug_data = {
        "input_news_text": news_text,
        "news_type": news_type,
        "misinformation_domain": misinformation_domain,
        "trusted_urls_found": result.trusted_urls,
        "scraped_content_count": result.scraped_content_count,
        "trust_score": result.trust_score,
        "fact_check_assessment": result.fact_check_assessment,
        "processing_errors": result.processing_errors,
        "timestamp": datetime.now().isoformat()
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"scraped_data_{misinformation_domain}_{timestamp}.json"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Debug data saved to {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"Error saving debug data to JSON: {e}")
        return None

async def initialize_fact_checker(news_type: str, news_text: str, misinformation_domain: str, news_id: Optional[str] = None) -> FactCheckResult:
    """Main fact-checking function - production ready"""
    result = FactCheckResult(news_id=news_id)
    
    try:
        if news_type == "Evergreen News":
            logger.info(f"Starting evergreen fact-check for: {news_text[:100]}...")
            
            # Search for trusted URLs
            result.trusted_urls = await google_search_and_filter(news_text, misinformation_domain)
            
            if not result.trusted_urls:
                result.processing_errors.append("No trusted sources found")
                result.fact_check_assessment = "N/A - No trusted sources found"
                result.trust_score = 0.0
                return result

            logger.info(f"Found {len(result.trusted_urls)} trusted URLs for scraping")
            result.sources_used = result.trusted_urls.copy()
            
            # Scrape content
            result.scraped_contents = await async_scrape(result.trusted_urls)
            result.scraped_content_count = len(result.scraped_contents)
            
            if not result.scraped_contents:
                result.processing_errors.append("Could not scrape content from any trusted URLs")
                result.fact_check_assessment = "N/A - No content scraped from trusted URLs"
                result.trust_score = 0.0
                return result

            logger.info(f"Successfully scraped content from {result.scraped_content_count} URLs")

            # Generate all analyses concurrently for better performance
            summary_task = summarize_scraped_data_with_gemini(result.scraped_contents)
            education_task = generate_further_education(news_text, misinformation_domain)
            fact_check_task = fact_check_evergreen_misinformation(news_text, result.scraped_contents)
            
            # Wait for all tasks to complete
            result.summarized_answer, result.further_education_suggestions, result.fact_check_assessment = await asyncio.gather(
                summary_task, education_task, fact_check_task
            )

            # Calculate trust score
            result.trust_score = calculate_trust_score(result.fact_check_assessment)

            # Save debug data
            debug_filename = save_debug_data(result, news_text, news_type, misinformation_domain)
            if debug_filename:
                result.debug_data['saved_file'] = debug_filename

            result.success = True
            logger.info("Evergreen fact-checking completed successfully")

        elif news_type == "Real-time News":
            logger.info(f"Starting real-time fact-check for: {news_text[:100]}...")
            
            # Search for real-time sources
            result.trusted_urls = await google_search_realtime(news_text, misinformation_domain)

            if not result.trusted_urls:
                result.processing_errors.append("No real-time sources found")
                result.fact_check_assessment = "N/A - No real-time sources found"
                result.trust_score = 0.0
                return result
            
            logger.info(f"Found {len(result.trusted_urls)} real-time URLs for scraping")
            result.sources_used = result.trusted_urls.copy()

            # Scrape content
            result.scraped_contents = await async_scrape(result.trusted_urls)
            result.scraped_content_count = len(result.scraped_contents)

            if not result.scraped_contents:
                result.processing_errors.append("Could not scrape content from any real-time URLs")
                result.fact_check_assessment = "N/A - No content scraped from real-time URLs"
                result.trust_score = 0.0
                return result
            
            logger.info(f"Successfully scraped content from {result.scraped_content_count} URLs")

            # Generate analyses concurrently
            summary_task = summarize_scraped_data_with_gemini(result.scraped_contents)
            education_task = generate_further_education(news_text, misinformation_domain)
            fact_check_task = fact_check_realtime_misinformation(news_text, result.scraped_contents)
            
            result.summarized_answer, result.further_education_suggestions, result.fact_check_assessment = await asyncio.gather(
                summary_task, education_task, fact_check_task
            )
            
            # Calculate trust score for real-time
            result.trust_score = calculate_trust_score(result.fact_check_assessment)
            
            # Save debug data
            debug_filename = save_debug_data(result, news_text, news_type, misinformation_domain)
            if debug_filename:
                result.debug_data['saved_file'] = debug_filename

            result.success = True
            logger.info("Real-time fact-checking completed successfully")

        else:
            result.fact_check_assessment = f"News type '{news_type}' not recognized for fact-checking."
            result.processing_errors.append(f"Invalid news type: {news_type}")
            result.success = False

    except Exception as e:
        logger.error(f"Error during fact-checking: {e}")
        result.processing_errors.append(f"Fact-checking failed: {str(e)}")
        result.fact_check_assessment = f"Fact-checking failed due to error: {str(e)}"
        result.success = False

    return result

# Cleanup function
async def cleanup_connections():
    """Cleanup aiohttp connections"""
    await _connection_manager.close()

# Register cleanup on exit
import atexit

def _sync_cleanup():
    try:
        # Attempt to get the current loop, if it exists and is running.
        # Otherwise, create a new loop for cleanup.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(cleanup_connections(), loop)
            else:
                # If loop is not running, run cleanup in a new temporary loop
                asyncio.run(cleanup_connections())
        except RuntimeError: # No current event loop
            asyncio.run(cleanup_connections())
    except Exception as e:
        logger.error(f"Error during synchronous connection cleanup: {e}")

atexit.register(_sync_cleanup)

if __name__ == "__main__":
    # Test the updated system
    async def test_fact_checker():
        test_news_text = "Eating rice makes you fat and should be avoided for weight loss."
        test_news_type = "Evergreen News"
        test_misinformation_domain = "Health"
        
        result = await initialize_fact_checker(test_news_type, test_news_text, test_misinformation_domain)
        print("=== Fact-Check Result ===")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        
        # Cleanup
        await cleanup_connections()
    
    asyncio.run(test_fact_checker())
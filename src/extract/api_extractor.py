# src/extract/api_extractor.py

import requests
import logging
import time
from typing import Optional, List, Dict, Any, Union, Tuple

# Get a logger for this module
logger = logging.getLogger(__name__)

class ApiExtractor:
    """
    A class to extract data from REST APIs.
    """

    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 api_key_header: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 auth_token_type: str = "Bearer",
                 default_headers: Optional[Dict[str, str]] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: int = 5): # seconds
        """
        Initializes the ApiExtractor.

        Args:
            base_url (str): The base URL for the API (e.g., "https://api.example.com/v1").
            api_key (Optional[str]): API key for authentication.
            api_key_header (Optional[str]): The header name for the API key (e.g., "X-API-Key").
            auth_token (Optional[str]): Authentication token (e.g., OAuth Bearer token).
            auth_token_type (str): Type of auth token, defaults to "Bearer".
            default_headers (Optional[Dict[str, str]]): Default headers to include in all requests.
            timeout (int): Request timeout in seconds.
            max_retries (int): Maximum number of retries for failed requests.
            retry_delay (int): Delay in seconds between retries.
        """
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.auth_token = auth_token
        self.auth_token_type = auth_token_type
        self.default_headers = default_headers if default_headers is not None else {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session() # Use a session object for connection pooling

        logger.info(f"ApiExtractor initialized for base URL: {self.base_url}")

    def _prepare_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepares request headers, including authentication."""
        headers = self.default_headers.copy()
        headers['Accept'] = 'application/json' # Common default

        if self.api_key and self.api_key_header:
            headers[self.api_key_header] = self.api_key
            logger.debug(f"API Key header '{self.api_key_header}' added.")
        elif self.api_key: # If header not specified, try a common one or warn
            logger.warning("API key provided but api_key_header not specified. Not adding API key to headers by default.")


        if self.auth_token:
            headers['Authorization'] = f"{self.auth_token_type} {self.auth_token}"
            logger.debug("Authorization header added.")

        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _make_request(self,
                      method: str,
                      endpoint: str,
                      params: Optional[Dict[str, Any]] = None,
                      json_payload: Optional[Dict[str, Any]] = None,
                      custom_headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
        """Makes an HTTP request with retries."""
        full_url = f"{self.base_url}{endpoint.lstrip('/')}"
        headers = self._prepare_headers(custom_headers)

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making {method} request to {full_url} with params: {params}, attempt: {attempt + 1}")
                response = self.session.request(method,
                                                full_url,
                                                params=params,
                                                json=json_payload,
                                                headers=headers,
                                                timeout=self.timeout)
                response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
                logger.info(f"Successfully received {method} response from {full_url}, status: {response.status_code}")
                return response
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error during {method} request to {full_url}: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 429: # Too Many Requests - rate limiting
                    logger.warning(f"Rate limit hit (429). Retrying after {self.retry_delay * (attempt + 1)} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff could be better
                elif 500 <= e.response.status_code < 600 and attempt < self.max_retries: # Server-side errors
                    logger.warning(f"Server error ({e.response.status_code}). Retrying after {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else: # Other client errors or max retries reached for server errors
                    return response # Return the failed response for further inspection if needed
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error during {method} request to {full_url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout error during {method} request to {full_url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"An unexpected request exception occurred for {full_url}: {e}")
                # Do not retry for generic request exceptions unless sure it's safe
                break # Exit retry loop
        logger.error(f"Failed to get a successful response from {full_url} after {self.max_retries + 1} attempts.")
        return None

    def fetch_all_data(self,
                       endpoint: str,
                       params: Optional[Dict[str, Any]] = None,
                       results_key: Optional[str] = None,
                       next_page_url_key: Optional[str] = None, # e.g., 'next' in response JSON
                       page_param_name: Optional[str] = None, # e.g., 'page' or 'offset'
                       limit_param_name: Optional[str] = None, # e.g., 'limit' or 'per_page'
                       limit_per_page: int = 100,
                       max_pages: Optional[int] = None) -> List[Dict[Any, Any]]:
        """
        Fetches all data from a paginated API endpoint.

        Args:
            endpoint (str): The API endpoint path.
            params (Optional[Dict[str, Any]]): Initial query parameters.
            results_key (Optional[str]): Key in the JSON response where the list of results is found.
                                         If None, the entire JSON response (if a list) is taken as results.
            next_page_url_key (Optional[str]): Key in the JSON response that contains the URL for the next page.
            page_param_name (Optional[str]): Name of the query parameter for page number/offset.
            limit_param_name (Optional[str]): Name of the query parameter for results per page.
            limit_per_page (int): Number of results to request per page if using page_param_name.
            max_pages (Optional[int]): Maximum number of pages to fetch.

        Returns:
            List[Dict[Any, Any]]: A list of all records fetched from the API.
        """
        all_records: List[Dict[Any, Any]] = []
        current_params = params.copy() if params else {}
        current_page = 1 # For page_param_name strategy
        next_url_to_fetch = f"{self.base_url}{endpoint.lstrip('/')}" # Initial URL

        if page_param_name and limit_param_name:
            current_params[limit_param_name] = limit_per_page
            logger.info(f"Using page parameter pagination: page_param='{page_param_name}', limit_param='{limit_param_name}' with {limit_per_page} items/page.")
        elif next_page_url_key:
            logger.info(f"Using next page URL key pagination: next_page_url_key='{next_page_url_key}'.")
        else:
            logger.info("No pagination strategy specified, fetching single page.")

        pages_fetched = 0
        while (max_pages is None or pages_fetched < max_pages):
            if page_param_name:
                current_params[page_param_name] = current_page
                response = self._make_request("GET", endpoint, params=current_params)
            elif next_page_url_key and pages_fetched > 0 and not next_url_to_fetch: # No more next URL
                 break
            elif next_page_url_key: # Use the full URL provided by the API
                # We need to make the request without prepending base_url if next_url_to_fetch is absolute
                # This part needs refinement based on how `_make_request` handles absolute URLs.
                # For simplicity, assuming `_make_request` always uses base_url + endpoint.
                # A more robust solution would parse next_url_to_fetch.
                if next_url_to_fetch.startswith(self.base_url):
                    relative_endpoint = next_url_to_fetch[len(self.base_url):]
                    response = self._make_request("GET", relative_endpoint, params=current_params if pages_fetched == 0 else None)
                else: # It's an absolute URL different from base_url, or a relative path not from base_url
                    # This scenario requires a more flexible _make_request or direct requests.get call here
                    logger.warning(f"Next URL '{next_url_to_fetch}' is absolute or not relative to base_url. This pagination case might need specific handling.")
                    # Fallback to direct request for this specific case:
                    try:
                        direct_response = self.session.get(next_url_to_fetch, headers=self._prepare_headers(), params=current_params if pages_fetched == 0 else None, timeout=self.timeout)
                        direct_response.raise_for_status()
                        response = direct_response
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Direct request to {next_url_to_fetch} failed: {e}")
                        response = None
            else: # No pagination
                response = self._make_request("GET", endpoint, params=current_params)


            if not response or response.status_code != 200:
                logger.error(f"Failed to fetch data from {endpoint} (page {current_page if page_param_name else pages_fetched + 1}). Status: {response.status_code if response else 'None'}")
                break

            try:
                data = response.json()
            except ValueError: # Includes json.JSONDecodeError
                logger.error(f"Failed to decode JSON response from {endpoint}")
                break

            records_on_page: List[Dict[Any, Any]] = []
            if results_key:
                records_on_page = data.get(results_key, [])
                if not isinstance(records_on_page, list):
                    logger.error(f"Expected a list for results_key '{results_key}', but got {type(records_on_page)}.")
                    break
            elif isinstance(data, list):
                records_on_page = data
            else:
                logger.warning(f"Response from {endpoint} is not a list and no results_key provided. Assuming single object response.")
                if isinstance(data, dict):
                    all_records.append(data) # Add the single dict and break if no pagination
                if not (page_param_name or next_page_url_key): # if no pagination, we are done
                    break


            if not records_on_page and (page_param_name or next_page_url_key): # Empty page usually means end of data for paginated APIs
                logger.info(f"No more records found on page {current_page if page_param_name else pages_fetched + 1}. Ending pagination.")
                break

            all_records.extend(records_on_page)
            logger.info(f"Fetched {len(records_on_page)} records from page {current_page if page_param_name else pages_fetched + 1}. Total records: {len(all_records)}")
            pages_fetched += 1

            if next_page_url_key:
                next_url_to_fetch = data.get(next_page_url_key)
                if not next_url_to_fetch:
                    logger.info("No 'next_page_url_key' found in response. Ending pagination.")
                    break
                current_params = {} # Params are usually in the next_url itself
            elif page_param_name:
                current_page += 1
            else: # No pagination strategy, so break after first fetch
                break

        logger.info(f"Finished fetching data. Total records extracted: {len(all_records)} from {pages_fetched} pages.")
        return all_records

    def get_single_record(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[Any, Any]]:
        """Fetches a single record (expected to be a JSON object)."""
        response = self._make_request("GET", endpoint, params=params)
        if response and response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                logger.error(f"Failed to decode JSON for single record from {endpoint}")
                return None
        return None

    def close_session(self):
        """Closes the requests session."""
        if self.session:
            self.session.close()
            logger.info("Requests session closed.")


if __name__ == '__main__':
    # This is for basic testing of the ApiExtractor.
    # In a real application, logging would be configured by the main script.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example using JSONPlaceholder (a public test API)
    logger.info("-" * 30)
    logger.info("Testing ApiExtractor with JSONPlaceholder...")

    # 1. Test fetching all posts (uses page_param_name for pagination)
    # JSONPlaceholder uses _page and _limit for pagination
    jsonplaceholder_extractor = ApiExtractor(base_url="https://jsonplaceholder.typicode.com")
    
    logger.info("\nFetching all posts (paginated)...")
    # JSONPlaceholder returns a list directly, so no results_key is needed.
    # It doesn't provide a 'next_page_url_key'.
    # We will use page_param_name and limit_param_name.
    all_posts = jsonplaceholder_extractor.fetch_all_data(
        endpoint="posts",
        page_param_name="_page",
        limit_param_name="_limit",
        limit_per_page=10, # Fetch 10 posts per page
        max_pages=3 # Fetch a maximum of 3 pages for this test
    )
    if all_posts:
        logger.info(f"Successfully fetched {len(all_posts)} posts in total.")
        # logger.info("First 2 posts:")
        # for post in all_posts[:2]:
        #     logger.info(f"  ID: {post.get('id')}, Title: {post.get('title')[:30]}...")
    else:
        logger.error("Failed to fetch posts.")

    # 2. Test fetching a single post
    logger.info("\nFetching a single post (ID: 1)...")
    single_post = jsonplaceholder_extractor.get_single_record(endpoint="posts/1")
    if single_post:
        logger.info(f"Successfully fetched post ID 1: {single_post.get('title')}")
    else:
        logger.error("Failed to fetch single post.")

    # 3. Test with a non-existent endpoint (should log errors)
    logger.info("\nTesting non-existent endpoint (posts/99999)...")
    non_existent_post = jsonplaceholder_extractor.get_single_record(endpoint="posts/99999")
    if non_existent_post is None: # Expecting None due to 404
        logger.info("Correctly handled non-existent endpoint (returned None, error logged).")

    jsonplaceholder_extractor.close_session()

    # Example with an API requiring an API key (conceptual)
    # logger.info("-" * 30)
    # logger.info("Conceptual test for API with API Key (will not run)...")
    # conceptual_api_key = "YOUR_ACTUAL_API_KEY"
    # if conceptual_api_key == "YOUR_ACTUAL_API_KEY":
    #     logger.warning("Skipping API key test as placeholder key is used.")
    # else:
    #     key_extractor = ApiExtractor(
    #         base_url="https://api.some-service.com/v2", # Replace with a real API base URL
    #         api_key=conceptual_api_key,
    #         api_key_header="X-SomeService-Key"
    #     )
    #     # some_data = key_extractor.fetch_all_data(endpoint="items")
    #     # if some_data:
    #     #     logger.info(f"Fetched {len(some_data)} items using API key.")
    #     key_extractor.close_session()

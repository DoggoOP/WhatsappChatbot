import os
import json
import time
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

from serpapi import GoogleSearch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException)
from bs4 import BeautifulSoup
import json, re, html
from apscheduler.schedulers.blocking import BlockingScheduler

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RX_NEXT = re.compile(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

def next_blob(url, headers):
    """Return the parsed JSON sitting in <script id="__NEXT_DATA__"> on *url*."""
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    m = RX_NEXT.search(r.text)
    if not m:
        return {}
    return json.loads(html.unescape(m.group(1)))

def _extract_next_data(html_text: str) -> dict | None:
    """
    Grab the JSON sitting in <script id="__NEXT_DATA__"> … </script>
    and return it as a Python dict.
    """
    m = re.search(
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html_text,
        flags=re.S,
    )
    if not m:
        return None
    try:
        return json.loads(html.unescape(m.group(1).strip()))
    except json.JSONDecodeError:
        return None


def gql(query: str, variables: dict | None = None, headers: dict | None = None):
    """Helper to POST a GraphQL query and return JSON."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    r = requests.post("https://www.d2place.com/graphql", headers=h, json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data.get("data", {})


class D2PlaceScraper:
    def __init__(self):
        self.base_url = "https://www.d2place.com"
        self.data = {
            "home_happenings": [],
            "dining": [],
            "shopping": [],
            "events": [],
            "play": [],
            "mall_info": {
                "about": "",
                "location": "",
                "parking": "",
                "leasing": "",
            }
        }
        
        self.headers = {                      #  <<<  ADD THIS
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        }
        
        self.serpapi_key = os.environ.get("SERP_API_KEY")
        self.setup_driver()
        
        

    def _harvest_detail(self, url: str) -> dict:
        """
        Pull phone, hours, website, FB, IG from a store / restaurant page.
        Tries Next.js JSON first; falls back to BeautifulSoup if needed.
        """
        blank = {
            "phone": "",
            "opening_hours": "",
            "facebook": "",
            "instagram": "",
            "website": "",
        }
        
        def _find(blob, wanted):
            """recursive search for first string value whose key is in *wanted*."""
            if isinstance(blob, dict):
                for k, v in blob.items():
                    if k in wanted and isinstance(v, str) and v.strip():
                        return v.strip()
                    found = _find(v, wanted)
                    if found:
                        return found
            elif isinstance(blob, list):
                for item in blob:
                    found = _find(item, wanted)
                    if found:
                        return found
            return ""

        # 1) pull the page
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
        except Exception:
            return blank

        # 2) try the Next.js blob
        blob = _extract_next_data(resp.text)
        if blob:
            try:
                meta = blob["props"]["pageProps"].get("meta", {})
                extracted = {
                    "phone":         meta.get("tel",          "").strip(),
                    "opening_hours": meta.get("openingHours", "").strip(),
                    "facebook":      meta.get("facebook",     "").strip(),
                    "instagram":     meta.get("instagram",    "").strip(),
                    "website":       meta.get("website",      "").strip(),
                }
                # only return if *any* real data was found
                if any(extracted.values()):
                    return extracted
            except (KeyError, TypeError):
                pass
            
         # ... after resp.text has been loaded into *blob* ...
        if blob:
            blank.update({
                "phone":         _find(blob, {"tel", "telephone", "phone"}),
                "opening_hours": _find(blob, {"openingHours", "opening_hours"}),
                "facebook":      _find(blob, {"facebook", "facebookUrl", "fb"}),
                "instagram":     _find(blob, {"instagram", "instagramUrl", "ig"}),
                "website":       _find(blob, {"website", "site"}),
            })
            # if we found at least one piece of data we can already return
            if any(blank.values()):
                return blank


        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
        except Exception:
            return blank

        # 2) fallback – parse the HTML
        soup = BeautifulSoup(resp.text, "html.parser")

        txt = soup.get_text(" ", strip=True)

        m_phone = re.search(r"\+?852[-\s]?\d{4}[-\s]?\d{4}", txt)
        if m_phone:
            blank["phone"] = m_phone.group(0)

        m_hours = re.search(
            #  ↓ new, accepts “Mon – Sun 11:00-22:00”, “星期一至日 11:00-22:00” …
            r"(Mon|Monday|星期[一二三四五六日]).{0,30}?(\d{1,2}:\d{2}).{0,5}?[-–~至to]{1,3}.{0,5}?(\d{1,2}:\d{2})",
            txt, re.I)
        if m_hours:
            blank["opening_hours"] = m_hours.group(0)

        # sometimes the site stores the socials in data-attributes
        for attr in ("data-facebook", "data-fb", "data-instagram", "data-ig"):
            if attr in soup.attrs:
                if "facebook" in attr:
                    blank["facebook"] = soup[attr]
                else:
                    blank["instagram"] = soup[attr]

        for a in soup.select("a[href]"):
            href = a["href"]
            if "facebook.com" in href and not blank["facebook"]:
                blank["facebook"] = href
            elif "instagram.com" in href and not blank["instagram"]:
                blank["instagram"] = href
            elif href.startswith("http") and not blank["website"]:
                blank["website"] = href

        return blank


    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                    """
                },
            )
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def wait_for_element(self, selector, timeout=20, by=By.CSS_SELECTOR):
        """Wait for an element to be present and visible."""
        try:
            # First wait for presence
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            # Then wait for visibility
            WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            logger.warning(f"Timeout waiting for element: {selector}")
            return None
        
    def extract_next_data(html_text: str) -> dict | None:
        """
        Pull the JSON sitting inside <script id="__NEXT_DATA__">…</script>
        and turn it into a Python dict.
        """
        m = re.search(
            r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            html_text,
            flags=re.S,
        )
        if not m:
            return None
        raw = html.unescape(m.group(1).strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
        
    

    def load_page(self, url, wait_selector=None, wait_time=20):
        """Load a page and wait for content to be ready."""
        try:
            logger.info(f"Loading page: {url}")
            self.driver.get(url)
            time.sleep(2)  # Initial wait for page load

            # Scroll in increments to trigger lazy loading
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 5  # Limit scroll attempts to prevent infinite loops
            
            while scroll_attempts < max_scroll_attempts:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1

            if wait_selector:
                # Try multiple times with increasing timeouts
                for timeout in [10, 20, 30]:
                    element = self.wait_for_element(wait_selector, timeout)
                    if element:
                        return self.driver.page_source
                    logger.warning(f"Retrying with longer timeout ({timeout}s) for selector: {wait_selector}")
            
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error loading {url}: {e}")
            return None

    def debug_save(self, filename, html):
        """Helper to save debug HTML if needed."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Saved debug HTML to {filename}")
        except Exception as e:
            logger.error(f"Error saving debug file {filename}: {e}")

    # ================== MALL INFO (Direct Page Loads) ==================
    def scrape_about_us(self):
        url = f"{self.base_url}/about-us"
        html = self.load_page(url)
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one(".section-content") or soup.select_one("#__next")
        if container:
            text = container.get_text(separator="\n", strip=True)
            self.data["mall_info"]["about"] = text
            logger.info("Scraped About Us info")

    def scrape_location(self):
        url = f"{self.base_url}/location"
        html = self.load_page(url)
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one("#__next main") or soup.select_one("#__next")
        if container:
            text = container.get_text(separator="\n", strip=True)
            self.data["mall_info"]["location"] = text
            logger.info("Scraped location info")

    def scrape_parking(self):
        url = f"{self.base_url}/parking"
        html = self.load_page(url)
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)
        self.data["mall_info"]["parking"] = f"{page_title}\n{text}"
        logger.info("Scraped parking info")

    def scrape_leasing(self):
        url = f"{self.base_url}/leasing-and-renting"
        html = self.load_page(url)
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        self.data["mall_info"]["leasing"] = text
        logger.info("Scraped leasing/renting info")

    # ================== HOME HAPPENINGS (Homepage) ==================
    def scrape_home_happenings(self):
        url = self.base_url
        html = self.load_page(url, wait_selector="div.Home_eventListContainer__e1Gdt")
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one("div.Home_eventListContainer__e1Gdt")
        if not container:
            logger.warning("No 'Home_eventListContainer__e1Gdt' found on homepage")
            return
        items = container.select("div.flex.flex-col.h-full.cursor-pointer")
        for it in items:
            try:
                title_elem = it.select_one("p.common_eventTitle__DFL3G")
                detail_spans = it.select("div.common_eventDetail__407n6 span")
                title = title_elem.get_text(strip=True) if title_elem else "Unknown"
                date = detail_spans[0].get_text(strip=True) if len(detail_spans) > 0 else ""
                venue = detail_spans[1].get_text(strip=True) if len(detail_spans) > 1 else ""
                event_data = {
                    "title": title,
                    "date": date,
                    "venue": venue,
                }
                self.data["home_happenings"].append(event_data)
            except Exception as e:
                logger.error(f"Error scraping a home happening item: {e}")
        logger.info("Found %d home happenings", len(self.data["home_happenings"]))
        
    def _card_to_item(self, card) -> dict:
        """
        Return {"name", "location", "detail_url"} for one shop card.
        Works even if the anchor or the onclick handler sits on a
        descendant element instead of `card` itself.
        """
        name = card.find_element(
            By.CSS_SELECTOR, "p[class*='shopName']"
        ).text.strip()
        location = card.find_element(
            By.CSS_SELECTOR, "p[class*='shopLocation']"
        ).text.strip()

        # ── 1) Any <a href> underneath the card that points to /shops/… ──
        detail_path = None
        try:
            a = card.find_element(
                By.XPATH,
                ".//a[contains(@href, '/shops/')]"
            )
            detail_path = a.get_attribute("href")
        except NoSuchElementException:
            pass

        # ── 2) Fallback: scan all onclick/onClick on card & children ──
        if not detail_path:
            for el in ([card] + card.find_elements(By.CSS_SELECTOR, "*[onclick], *[onClick]")):
                onclick = el.get_attribute("onclick") or el.get_attribute("onClick") or ""
                m = re.search(r"/shops/[^'\"\)]+", onclick)
                if m:
                    detail_path = m.group(0)
                    break

        # ── 3) Final fallback: take the very first <a> in the card ──
        if not detail_path:
            try:
                first_a = card.find_element(By.TAG_NAME, "a")
                detail_path = first_a.get_attribute("href")
            except NoSuchElementException:
                detail_path = None

        #── 4) Normalize to full URL ──
        detail_url = ""
        if detail_path:
            detail_url = detail_path
            if detail_url.startswith("/"):
                detail_url = f"{self.base_url}{detail_url}"

        print(f"[CARD] {name!r} → detail_url={detail_url}")

        return {"name": name, "location": location, "detail_url": detail_url}

    # ================== DINING (Example) ==================
    def extract_card_details(self, card_element, mode="auto"):
        """Extract detailed information from a card element. Mode: 'auto', 'modal', or 'page'."""
        details = {}
        try:
            # Get basic info
            name = card_element.find_element(By.CSS_SELECTOR, "p.shop_shopName___oote").text.strip()
            location = card_element.find_element(By.CSS_SELECTOR, "p.shop_shopLocationContainer__JwLEp span").text.strip()
            detail_url = None
            try:
                link = card_element.find_element(By.TAG_NAME, "a")
                detail_url = link.get_attribute("href")
            except NoSuchElementException:
                pass

            # Decide extraction mode
            if mode == "auto":
                if detail_url:
                    mode = "page"
                else:
                    mode = "modal"

            if mode == "page" and detail_url:
                # Save current URL to return after extraction
                main_url = self.driver.current_url
                try:
                    self.driver.get(detail_url)
                    time.sleep(2)
                    # Extract details from detail page
                    page_source = self.driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")
                    # Description
                    desc_elem = soup.select_one(".shopDescription__container, .shop_shopDescription__container")
                    if desc_elem:
                        details["description"] = desc_elem.get_text(strip=True)
                    # Opening hours
                    hours_elem = (
                        soup.select_one("div.shop_shopHours__text")
                        or soup.select_one("div[class*='shopHours']")
                        or soup.find(string=re.compile(r"(Mon|星期).*:\d{2}"))
                    )
                    if hours_elem:
                        details["opening_hours"] = hours_elem.get_text(strip=True)
                    # Phone number
                    phone_elem = soup.find("a", href=lambda h: h and h.startswith("tel:"))
                    if phone_elem:
                        details["phone"] = phone_elem.get_text(strip=True)
                    # Social links
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                    if "facebook.com" in href and not details["facebook"]:
                        details["facebook"] = href
                    elif "instagram.com" in href and not details["instagram"]:
                        details["instagram"] = href
                except Exception as e:
                    logger.error(f"Error extracting details from detail page for {name}: {e}")
                finally:
                    self.driver.get(main_url)
                    time.sleep(1)
            elif mode == "modal":
                try:
                    card_element.click()
                    time.sleep(2)
                    modal = self.wait_for_element("div.modal-content", timeout=10)
                    if modal:
                        # Description
                        try:
                            desc_elem = modal.find_element(By.CSS_SELECTOR, "div.shop_shopDescription__container")
                            if desc_elem:
                                details["description"] = desc_elem.text.strip()
                        except NoSuchElementException:
                            pass
                        # Social links
                        try:
                            social_links = modal.find_elements(By.CSS_SELECTOR, "a[href*='instagram.com'], a[href*='facebook.com']")
                            for link in social_links:
                                href = link.get_attribute("href")
                                if "instagram.com" in href:
                                    details["instagram"] = href
                                elif "facebook.com" in href:
                                    details["facebook"] = href
                        except NoSuchElementException:
                            pass
                        # Opening hours
                        try:
                            hours_text = None
                            hours_selectors = [
                                "div.shop_shopHours__container",
                                "div.shop_shopHours__text",
                                "div.shop_shopHours",
                                "div[class*='shopHours']",
                                "div[class*='opening-hours']"
                            ]
                            for selector in hours_selectors:
                                try:
                                    hours_elem = modal.find_element(By.CSS_SELECTOR, selector)
                                    if hours_elem and hours_elem.text.strip():
                                        hours_text = hours_elem.text.strip()
                                        break
                                except NoSuchElementException:
                                    continue
                            if not hours_text:
                                try:
                                    all_text = modal.text
                                    import re
                                    match = re.search(r'(Monday[^\n]+\d{2}:\d{2}[^\n]*)', all_text)
                                    if match:
                                        hours_text = match.group(1)
                                except Exception as e:
                                    logger.error(f"Regex fallback for opening hours failed: {e}")
                            if hours_text:
                                details["opening_hours"] = hours_text
                                logger.info(f"Found opening hours for {name}: {hours_text}")
                            else:
                                logger.warning(f"No opening hours found for {name} using any selector or pattern")
                        except Exception as e:
                            logger.error(f"Error extracting opening hours for {name}: {e}")
                        # Phone number
                        try:
                            phone = None
                            try:
                                tel_link = modal.find_element(By.CSS_SELECTOR, "a[href^='tel']")
                                phone = tel_link.text.strip()
                            except NoSuchElementException:
                                import re
                                all_text = modal.text
                                match = re.search(r'(\d{4} \d{4})', all_text)
                                if match:
                                    phone = match.group(1)
                            if phone:
                                details["phone"] = phone
                                logger.info(f"Found phone for {name}: {phone}")
                            else:
                                logger.warning(f"No phone found for {name}")
                        except Exception as e:
                            logger.error(f"Error extracting phone for {name}: {e}")
                        # Close modal
                        try:
                            close_btn = modal.find_element(By.CSS_SELECTOR, "button.close")
                            if close_btn:
                                close_btn.click()
                                time.sleep(1)
                        except NoSuchElementException:
                            pass
                except Exception as e:
                    logger.error(f"Error extracting details from modal for {name}: {e}")
            # Always return the basic info plus any additional details we found
            return {
                "name": name,
                "location": location,
                "detail_url": detail_url,
                **details
            }
        except Exception as e:
            logger.error(f"Error extracting card details: {e}")
            return None

    def scrape_dining(self):
        cats = gql("{findManyShopCategory(where:{categoryType:{equals:DINING}}){id}}")
        for c in cats.get("findManyShopCategory", []):
            shops = gql(
                """
                query($cid:Int!){
                    findManyShop(where:{shopCategoryId:{equals:$cid}}, take:1000){
                        nameEn nameTc addressEn addressTc phoneNumber
                        displayOpeningHoursEn displayOpeningHoursTc
                        facebookUrl instagramUrl websiteUrl passcode
                    }
                }
                """,
                {"cid": c["id"]},
                headers=self.headers,
            ).get("findManyShop", [])

            for shop in shops:
                item = {
                    "name": shop.get("nameEn") or shop.get("nameTc", ""),
                    "location": shop.get("addressEn") or shop.get("addressTc", ""),
                    "detail_url": f"{self.base_url}/shops/{shop['passcode']}",
                    "phone": shop.get("phoneNumber", ""),
                    "opening_hours": shop.get("displayOpeningHoursEn") or shop.get("displayOpeningHoursTc", ""),
                    "facebook": shop.get("facebookUrl", ""),
                    "instagram": shop.get("instagramUrl", ""),
                    "website": shop.get("websiteUrl", ""),
                }
                self.data["dining"].append(item)

        logger.info("Dining scraped via GraphQL → %d entries", len(self.data["dining"]))


    def scrape_shopping(self):
        cats = gql("{findManyShopCategory(where:{categoryType:{equals:SHOP}}){id}}")
        for c in cats.get("findManyShopCategory", []):
            shops = gql(
                """
                query($cid:Int!){
                    findManyShop(where:{shopCategoryId:{equals:$cid}}, take:1000){
                        nameEn nameTc addressEn addressTc phoneNumber
                        displayOpeningHoursEn displayOpeningHoursTc
                        facebookUrl instagramUrl websiteUrl passcode
                    }
                }
                """,
                {"cid": c["id"]},
                headers=self.headers,
            ).get("findManyShop", [])

            for shop in shops:
                item = {
                    "name": shop.get("nameEn") or shop.get("nameTc", ""),
                    "location": shop.get("addressEn") or shop.get("addressTc", ""),
                    "detail_url": f"{self.base_url}/shops/{shop['passcode']}",
                    "phone": shop.get("phoneNumber", ""),
                    "opening_hours": shop.get("displayOpeningHoursEn") or shop.get("displayOpeningHoursTc", ""),
                    "facebook": shop.get("facebookUrl", ""),
                    "instagram": shop.get("instagramUrl", ""),
                    "website": shop.get("websiteUrl", ""),
                }
                self.data["shopping"].append(item)

        logger.info("Shopping scraped via GraphQL → %d entries", len(self.data["shopping"]))

    def scrape_events(self):
        events = gql(
            """
            query($take:Int!){
                findManyEventPublic(take:$take){
                    nameEn nameTc venueEn venueTc alias
                    displayOpeningHoursEn displayOpeningHoursTc
                    eventStartDate eventEndDate
                }
            }
            """,
            {"take": 1000},
            headers=self.headers,
        ).get("findManyEventPublic", [])

        for ev in events:
            try:
                start = datetime.fromisoformat(ev["eventStartDate"].replace("Z", "+00:00")).date()
                end = datetime.fromisoformat(ev["eventEndDate"].replace("Z", "+00:00")).date()
                date_str = f"{start} - {end}"
            except Exception:
                date_str = ""

            item = {
                "title": ev.get("nameEn") or ev.get("nameTc", ""),
                "date": date_str,
                "opening_hours": ev.get("displayOpeningHoursEn") or ev.get("displayOpeningHoursTc", ""),
                "venue": ev.get("venueEn") or ev.get("venueTc", ""),
                "detail_url": f"{self.base_url}/events/{ev['alias']}",
            }
            self.data["events"].append(item)

        logger.info("Events scraped via GraphQL → %d entries", len(self.data["events"]))

    def scrape_play(self):
        cats = gql("{findManyShopCategory(where:{categoryType:{equals:PLAY}}){id}}")
        for c in cats.get("findManyShopCategory", []):
            shops = gql(
                """
                query($cid:Int!){
                    findManyShop(where:{shopCategoryId:{equals:$cid}}, take:1000){
                        nameEn nameTc addressEn addressTc phoneNumber
                        displayOpeningHoursEn displayOpeningHoursTc
                        facebookUrl instagramUrl websiteUrl passcode
                    }
                }
                """,
                {"cid": c["id"]},
                headers=self.headers,
            ).get("findManyShop", [])

            for shop in shops:
                item = {
                    "name": shop.get("nameEn") or shop.get("nameTc", ""),
                    "location": shop.get("addressEn") or shop.get("addressTc", ""),
                    "detail_url": f"{self.base_url}/shops/{shop['passcode']}",
                    "phone": shop.get("phoneNumber", ""),
                    "opening_hours": shop.get("displayOpeningHoursEn") or shop.get("displayOpeningHoursTc", ""),
                    "facebook": shop.get("facebookUrl", ""),
                    "instagram": shop.get("instagramUrl", ""),
                    "website": shop.get("websiteUrl", ""),
                }
                self.data["play"].append(item)

        logger.info("Play scraped via GraphQL → %d entries", len(self.data["play"]))

    # ================== Facebook Page ==================
    def scrape_facebook_page(self, shop, fb_url):
        logger.info(f"Attempting to load Facebook page for {shop['name']}: {fb_url}")
        try:
            r = requests.get(fb_url, timeout=10)
            if r.status_code == 200:
                fb_soup = BeautifulSoup(r.text, "html.parser")
                fb_title = fb_soup.title.get_text(strip=True) if fb_soup.title else ""
                shop["facebook_page_title"] = fb_title
                logger.info("Got FB page title for %s: %s", shop["name"], fb_title)
            else:
                logger.warning(f"FB request returned status {r.status_code}")
        except Exception as e:
            logger.warning(f"Failed to load FB page for {shop['name']}: {e}")

    # ================== SERP API: Detailed Google Maps ==================
    def serp_google_reviews(self, query):
        """
        Use the 'google_maps' engine from SerpAPI to retrieve more structured data
        like rating, total reviews, website, etc.
        """
        if not self.serpapi_key:
            return {"error": "No SERP API key in environment; skipping real search."}

        logger.info(f"Searching Google Maps for '{query}' with SerpAPI ...")
        params = {
            "engine": "google_maps",
            "q": query,
            "type": "search",
            "hl": "en",
            "google_domain": "google.com.hk",
            "gl": "hk",
            "api_key": self.serpapi_key,
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            place_results = results.get("place_results") or results.get("local_results", [])
            if not place_results:
                return {"message": "(No place_results found)"}

            place = place_results[0]
            rating = place.get("rating")
            reviews_count = place.get("reviews")
            address = place.get("address", "")
            phone = place.get("phone", "")
            website = place.get("website", "")

            review_data = {
                "rating": rating,
                "reviews_count": reviews_count,
                "address": address,
                "phone": phone,
                "website": website,
                "serpapi_link": results.get("search_metadata", {}).get("google_maps_url", ""),
            }
            return review_data

        except Exception as e:
            logger.error(f"SERP API error: {e}")
            return {"error": str(e)}

    # ================== SERP API: Extra Normal Google Search ==================
    def serpapi_extra_google_info(self, query):
        """
        Use the 'google' engine to do a normal Google search for the restaurant,
        potentially finding star dishes, cuisine type, menu links, or
        other knowledge panel info.
        """
        if not self.serpapi_key:
            return {"message": "No SERP API key in environment; skipping extra google info"}

        logger.info(f"Doing normal Google search for '{query}' ...")
        params = {
            "engine": "google",
            "q": query,
            "gl": "hk",
            "google_domain": "google.com.hk",
            "api_key": self.serpapi_key,
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            logger.error(f"SERP API error (google engine): {e}")
            return {"error": str(e)}

        info_data = {}
        # Possibly parse knowledge panel if present
        knowledge_graph = results.get("knowledge_graph", {})
        if knowledge_graph:
            info_data["kg_title"] = knowledge_graph.get("title")
            info_data["kg_type"] = knowledge_graph.get("type")
            info_data["kg_description"] = knowledge_graph.get("description")
            info_data["kg_menu_link"] = knowledge_graph.get("menu")
            info_data["kg_website"] = knowledge_graph.get("website")
            # Some knowledge panels show "reviews" or "serves_dishes"
            info_data["kg_reviews"] = knowledge_graph.get("reviews")
            info_data["kg_cuisines"] = knowledge_graph.get("serves_cuisine")

        # Also check top organic results
        organic_results = results.get("organic_results", [])
        if organic_results:
            # e.g. store snippet from the first result
            info_data["top_snippet"] = organic_results[0].get("snippet", "")

        return info_data

    # ================== MAIN ==================
    def scrape_all(self):
        try:
            logger.info("Starting full scrape of D2 Place")
            
            # Scrape mall info
            self.scrape_about_us()
            self.scrape_location()
            self.scrape_parking()
            self.scrape_leasing()
            
            # Scrape dynamic content
            self.scrape_home_happenings()
            self.scrape_dining()
            self.scrape_shopping()
            self.scrape_events()
            self.scrape_play()
            
            # 7) Google Maps + Normal Google search for each dining shop
            for shop in self.data["dining"]:
                # A) Google Maps details
                logger.info(f"Fetching detailed Google reviews for '{shop['name']}' ...")
                shop["google_review_data"] = self.serp_google_reviews(shop["name"])

                # B) Extra normal Google info
                # e.g. "Thai-ger D2 Place HK menu" or just "Thai-ger D2 Place HK"
                normal_query = f"{shop['name']} D2 Place HK menu"
                logger.info(f"Fetching extra Google info for '{normal_query}' ...")
                shop["extra_google_info"] = self.serpapi_extra_google_info(normal_query)

            self.save_data()
            logger.info("Full scrape completed successfully")
        except Exception as e:
            logger.error(f"Error during full scrape: {e}")
            self.save_data()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user → saving partial data…")
            self.save_data()
            raise
        finally:
            self.driver.quit()
            logger.info("WebDriver closed")

    def save_data(self):
        """Save scraped data to JSON file."""
        try:
            with open("d2place_data.json", "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.info("Data saved to d2place_data.json")
            stats = {
                "happenings": len(self.data["home_happenings"]),
                "dining": len(self.data["dining"]),
                "shopping": len(self.data["shopping"]),
                "events": len(self.data["events"]),
                "play": len(self.data["play"]),
            }
            logger.info(f"Scraped data statistics: {stats}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        filename='scraper_schedule.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    scraper = D2PlaceScraper()

    scheduler = BlockingScheduler(timezone="Asia/Hong_Kong")
    
    scraper.scrape_all()
    
    # run once a week: every Monday at 02:00
    scheduler.add_job(
        scraper.scrape_all,
        trigger='cron',
        day_of_week='mon',
        hour=2,
        minute=0,
        id='weekly_d2_scrape'
    )

    logger.info("Starting weekly scraper job (every Monday at 02:00 HK time)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown()
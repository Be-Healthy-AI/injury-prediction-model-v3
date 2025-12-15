"""
Fill missing team countries by scraping Transfermarkt
Searches for each team and extracts country from the club page
"""

import pandas as pd
import sys
import os
import time
import re
from typing import Optional, Dict, Any
import logging

# Add scripts directory to path
# From data_exports/transfermarkt/england/20251205/ to scripts/
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'scripts'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_collection.transfermarkt_scraper import TransfermarktScraper, ScraperConfig
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_team_name_for_search(team_name: str) -> str:
    """Normalize team name for better search results"""
    # Remove common suffixes that might not be in Transfermarkt
    name = team_name.strip()
    
    # Remove youth team indicators
    name = re.sub(r'\s*(U\d+|U-\d+|YL|Youth|Junior|Reserve|B Team|B-Team)\s*$', '', name, flags=re.IGNORECASE)
    
    # Remove leading numbers and dots (e.g., "1. FC Brno" -> "FC Brno")
    name = re.sub(r'^\d+\.\s*', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def search_team_on_transfermarkt(scraper: TransfermarktScraper, team_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a team on Transfermarkt and return club_id and slug.
    Uses the search functionality on Transfermarkt.
    """
    try:
        # Try original name first
        search_queries = [team_name]
        
        # Add normalized version
        normalized = normalize_team_name_for_search(team_name)
        if normalized != team_name:
            search_queries.append(normalized)
        
        # Try without common prefixes
        if team_name.startswith(('1. ', '1.', 'FC ', 'SC ', 'AC ', 'AS ')):
            without_prefix = re.sub(r'^(\d+\.?\s*)?(FC|SC|AC|AS)\s+', '', team_name, flags=re.IGNORECASE)
            if without_prefix != team_name:
                search_queries.append(without_prefix)
        
        for query in search_queries:
            try:
                # Use Transfermarkt search
                search_url = f"{scraper.config.base_url}/schnellsuche/ergebnis/schnellsuche"
                params = {
                    'query': query,
                    'Spieler': '',
                    'Verein': '1',  # Search for clubs/teams
                    'Wettbewerb': '',
                    'Trainer': '',
                    'Schiedsrichter': '',
                }
                
                # Use _request directly since _fetch_soup doesn't support params
                response = scraper._request(search_url, params=params)
                soup = BeautifulSoup(response.text, "lxml")
                
                # Look for club links in search results
                club_links = soup.select('a[href*="/verein/"]')
                
                for link in club_links[:10]:  # Check first 10 results
                    href = link.get('href', '')
                    # Extract club ID
                    match = re.search(r'/verein/(\d+)', href)
                    if not match:
                        continue
                    
                    club_id = int(match.group(1))
                    
                    # Extract slug from href
                    parts = [p for p in href.split('/') if p]
                    if len(parts) >= 2:
                        # Find the part before /verein/
                        verein_idx = -1
                        for i, part in enumerate(parts):
                            if part == 'verein':
                                verein_idx = i
                                break
                        if verein_idx > 0:
                            club_slug = parts[verein_idx - 1]
                        else:
                            continue
                    else:
                        continue
                    
                    # Get club name from link text
                    club_name = link.get_text(strip=True)
                    
                    # Check if this looks like a match
                    if club_name and len(club_name) > 2:
                        # Simple similarity check - if normalized names share significant words
                        query_words = set(normalize_team_name_for_search(query).lower().split())
                        club_words = set(normalize_team_name_for_search(club_name).lower().split())
                        
                        # Remove common words
                        common_words = {'fc', 'sc', 'ac', 'as', 'club', 'team', 'cf', 'fk'}
                        query_words -= common_words
                        club_words -= common_words
                        
                        # If there's overlap, it's likely a match
                        if query_words and club_words and (query_words & club_words):
                            return {
                                'club_id': club_id,
                                'club_slug': club_slug,
                                'club_name': club_name
                            }
                        # Or if the query is contained in club name (for partial matches)
                        elif query_words:
                            query_clean = ' '.join(sorted(query_words))
                            club_clean = ' '.join(sorted(club_words))
                            if query_clean in club_clean or any(qw in club_clean for qw in query_words if len(qw) > 3):
                                return {
                                    'club_id': club_id,
                                    'club_slug': club_slug,
                                    'club_name': club_name
                                }
                
                # Rate limit between search attempts
                time.sleep(0.5)
            
            except Exception as e:
                logger.debug(f"Search attempt failed for '{query}': {e}")
                continue
        
        return None
    
    except Exception as e:
        logger.warning(f"Error searching for team '{team_name}': {e}")
        return None

def extract_country_from_club_page(scraper: TransfermarktScraper, club_slug: str, club_id: int) -> Optional[str]:
    """
    Extract country from a club's Transfermarkt page.
    Looks for the country flag/label in the club info section.
    """
    try:
        # Method 1: Try API first (most reliable)
        try:
            club_profile = scraper.fetch_club_profile(str(club_id))
            if club_profile:
                # Check various possible fields for country
                country = (
                    club_profile.get('country') or
                    club_profile.get('countryName') or
                    club_profile.get('baseDetails', {}).get('country') or
                    club_profile.get('baseDetails', {}).get('countryName') or
                    club_profile.get('baseDetails', {}).get('country', {}).get('name') if isinstance(club_profile.get('baseDetails', {}).get('country'), dict) else None
                )
                if country:
                    return str(country).strip()
        except Exception as api_error:
            logger.debug(f"API method failed for {club_id}: {api_error}")
        
        # Method 2: Fetch club page HTML
        url = f"{scraper.config.base_url}/{club_slug}/startseite/verein/{club_id}"
        soup = scraper._fetch_soup(url)
        
        # Look for country in info table (similar to player profiles)
        info_table = soup.select_one("div.info-table")
        if info_table:
            spans = info_table.select("span.info-table__content")
            for i in range(len(spans) - 1):
                label_span = spans[i]
                value_span = spans[i + 1] if i + 1 < len(spans) else None
                
                if not value_span:
                    continue
                
                label_text = label_text = label_span.get_text(strip=True).lower()
                
                # Check if this is a country/location field
                country_keywords = ['country', 'land', 'país', 'country of origin', 'location', 'nation', 'staat']
                if any(keyword in label_text for keyword in country_keywords):
                    # Look for flag images
                    flag_images = value_span.select('img.flaggenrahmen, img.flag, img[class*="flag"]')
                    if flag_images:
                        for img in flag_images:
                            country_name = img.get('alt') or img.get('title') or img.get('data-original-title')
                            if country_name:
                                country_clean = str(country_name).strip()
                                if country_clean and len(country_clean) < 50:
                                    return country_clean
                    
                    # Or get text directly
                    country_text = value_span.get_text(strip=True)
                    if country_text and len(country_text) < 50:
                        return country_text
        
        # Method 3: Look for country in breadcrumb navigation
        breadcrumb = soup.select_one('ol.breadcrumb, nav.breadcrumb, div.breadcrumb')
        if breadcrumb:
            # Usually format: Home > Country > League > Team
            links = breadcrumb.select('a')
            if len(links) >= 2:
                # Second link is often the country (skip "Home" which is usually first)
                for link in links[1:3]:  # Check 2nd and 3rd links
                    country_name = link.get_text(strip=True)
                    # Country names are usually short and don't contain "league" or "cup"
                    if (country_name and 
                        len(country_name) < 30 and 
                        'league' not in country_name.lower() and
                        'cup' not in country_name.lower() and
                        'championship' not in country_name.lower()):
                        return country_name
        
        # Method 4: Look for flag in header or main content
        header_flags = soup.select('header img[class*="flag"], .main img[class*="flag"]')
        for flag_img in header_flags[:3]:  # Check first 3 flags
            country_name = flag_img.get('alt') or flag_img.get('title')
            if country_name:
                country_clean = str(country_name).strip()
                if country_clean and len(country_clean) < 50:
                    return country_clean
        
        return None
    
    except Exception as e:
        logger.warning(f"Error extracting country from club page ({club_slug}/{club_id}): {e}")
        return None

def get_team_country(scraper: TransfermarktScraper, team_name: str) -> Optional[str]:
    """
    Get country for a team by searching Transfermarkt and extracting from club page.
    """
    # Step 1: Search for the team
    team_info = search_team_on_transfermarkt(scraper, team_name)
    
    if not team_info:
        logger.warning(f"Could not find team '{team_name}' on Transfermarkt")
        return None
    
    # Step 2: Extract country from club page
    country = extract_country_from_club_page(
        scraper,
        team_info['club_slug'],
        team_info['club_id']
    )
    
    if country:
        logger.info(f"Found country for '{team_name}': {country}")
    else:
        logger.warning(f"Could not extract country for '{team_name}' (found club: {team_info['club_name']})")
    
    return country

def main():
    """Main function"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    teams_file = os.path.join(base_dir, 'teams_data.csv')
    output_file = os.path.join(base_dir, 'teams_data.csv')
    
    logger.info("=" * 60)
    logger.info("Filling missing team countries from Transfermarkt")
    logger.info("=" * 60)
    
    # Load teams data
    logger.info(f"Loading teams from {teams_file}...")
    teams_df = pd.read_csv(teams_file, sep=';', encoding='utf-8-sig')
    
    # Find teams without countries
    missing_country = teams_df[teams_df['country'].isna() | (teams_df['country'] == '')]
    logger.info(f"Found {len(missing_country)} teams without countries (out of {len(teams_df)} total)")
    
    if len(missing_country) == 0:
        logger.info("All teams already have countries!")
        return
    
    # Initialize scraper
    scraper = TransfermarktScraper()
    
    try:
        # Process each team
        updated_count = 0
        failed_count = 0
        
        for idx, row in missing_country.iterrows():
            team_name = row['team']
            logger.info(f"\n[{idx+1}/{len(missing_country)}] Processing: {team_name}")
            
            try:
                country = get_team_country(scraper, team_name)
                
                if country:
                    teams_df.at[idx, 'country'] = country
                    updated_count += 1
                    logger.info(f"✓ Updated: {team_name} -> {country}")
                else:
                    failed_count += 1
                    logger.warning(f"✗ Failed to find country for: {team_name}")
                
                # Rate limiting - be respectful
                time.sleep(scraper.config.rate_limit_seconds)
            
            except Exception as e:
                logger.error(f"Error processing {team_name}: {e}")
                failed_count += 1
                continue
        
        # Save updated file
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary:")
        logger.info(f"  Updated: {updated_count} teams")
        logger.info(f"  Failed: {failed_count} teams")
        logger.info(f"  Remaining without country: {len(teams_df[teams_df['country'].isna() | (teams_df['country'] == '')])}")
        logger.info(f"{'='*60}")
        
        # Save with UTF-8 BOM for Excel compatibility
        teams_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
        logger.info(f"\nSaved updated teams_data.csv to: {output_file}")
    
    finally:
        scraper.close()

if __name__ == '__main__':
    main()


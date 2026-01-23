"""
Script to calculate Premier League injuries by season (2020/21 to 2024/25)
and estimate global Top 5 leagues injury costs for 2024/25.

This script:
1. Fetches PL clubs from TransferMarkt for each season
2. Loads injuries and player career data
3. Filters injuries to only count those when players were at PL clubs
4. Calculates injury counts per season
5. Estimates 2024/25 costs based on historical trends
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from collections import defaultdict
import numpy as np

# Configuration
DATA_FOLDER = r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\raw_data"
INJURIES_FILE = f"{DATA_FOLDER}\injuries_data.csv"
CAREER_FILE = f"{DATA_FOLDER}\players_career.csv"

# Season mapping: season format -> TransferMarkt saison_id
SEASONS = {
    "20/21": "2020",
    "21/22": "2021", 
    "22/23": "2022",
    "23/24": "2023",
    "24/25": "2024"
}

# Historical Top 5 leagues spending (in million euros)
HISTORICAL_COSTS = {
    "20/21": 376,
    "21/22": 496,
    "22/23": 696,
    "23/24": 732
}


def fetch_pl_clubs_from_transfermarkt(saison_id):
    """
    Fetch Premier League clubs from TransferMarkt for a given season.
    
    Args:
        saison_id: Season ID (e.g., "2024" for 2024/25)
    
    Returns:
        List of club names as they appear on TransferMarkt
    """
    url = f"https://www.transfermarkt.pt/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={saison_id}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with clubs
        clubs = []
        # Look for club links in the table
        club_links = soup.find_all('a', href=re.compile(r'/startseite/verein/\d+/saison_id'))
        
        for link in club_links:
            club_name = link.get_text(strip=True)
            if club_name and club_name not in clubs:
                clubs.append(club_name)
        
        # Alternative: look for table rows with club data
        if not clubs:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    for cell in cells:
                        links = cell.find_all('a', href=re.compile(r'/startseite/verein'))
                        for link in links:
                            club_name = link.get_text(strip=True)
                            if club_name and len(club_name) > 2 and club_name not in clubs:
                                clubs.append(club_name)
        
        print(f"Found {len(clubs)} clubs for season {saison_id}")
        return clubs
    
    except Exception as e:
        print(f"Error fetching clubs for season {saison_id}: {e}")
        # Fallback: return known PL clubs for recent seasons
        return get_fallback_pl_clubs(saison_id)


def get_fallback_pl_clubs(saison_id):
    """Fallback list of PL clubs if web scraping fails."""
    # Based on the web search results provided
    if saison_id == "2024":
        return [
            "Manchester City FC", "Chelsea FC", "FC Arsenal", "Liverpool FC",
            "Manchester United FC", "Tottenham Hotspur", "Aston Villa FC",
            "Newcastle United", "Brighton & Hove Albion", "Crystal Palace FC",
            "AFC Bournemouth", "Nottingham Forest", "Wolverhampton Wanderers",
            "Brentford FC", "West Ham United", "FC Everton", "FC Fulham",
            "FC Southampton", "Ipswich Town FC", "Leicester City FC"
        ]
    elif saison_id == "2023":
        return [
            "Manchester City FC", "FC Arsenal", "Chelsea FC", "Liverpool FC",
            "Tottenham Hotspur", "Manchester United FC", "Aston Villa FC",
            "Newcastle United", "Brighton & Hove Albion", "Nottingham Forest",
            "West Ham United", "Crystal Palace FC", "Wolverhampton Wanderers",
            "Brentford FC", "AFC Bournemouth", "FC Everton", "FC Fulham",
            "FC Burnley", "Sheffield United", "Luton Town"
        ]
    else:
        # Generic list for other seasons
        return [
            "Manchester City FC", "Chelsea FC", "FC Arsenal", "Liverpool FC",
            "Manchester United FC", "Tottenham Hotspur", "Aston Villa FC",
            "Newcastle United", "Brighton & Hove Albion", "Crystal Palace FC",
            "AFC Bournemouth", "Nottingham Forest", "Wolverhampton Wanderers",
            "Brentford FC", "West Ham United", "FC Everton", "FC Fulham",
            "FC Southampton", "Leicester City FC", "West Bromwich Albion"
        ]


def normalize_club_name(name):
    """
    Normalize club names to handle variations.
    Maps different name formats to a standard format.
    """
    if not name or pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Common name variations - map to standard names
    name_lower = name.lower()
    
    # Direct mappings for common variations
    direct_mappings = {
        "manchester city": "Manchester City",
        "man city": "Manchester City",
        "mancity": "Manchester City",
        "arsenal": "Arsenal",
        "chelsea": "Chelsea FC",
        "liverpool": "Liverpool FC",
        "manchester united": "Manchester United",
        "man united": "Manchester United",
        "manchester utd": "Manchester United",
        "tottenham": "Tottenham Hotspur",
        "tottenham hotspur": "Tottenham Hotspur",
        "spurs": "Tottenham Hotspur",
        "aston villa": "Aston Villa",
        "villa": "Aston Villa",
        "newcastle": "Newcastle United",
        "newcastle united": "Newcastle United",
        "brighton": "Brighton & Hove Albion",
        "brighton & hove albion": "Brighton & Hove Albion",
        "crystal palace": "Crystal Palace",
        "bournemouth": "AFC Bournemouth",
        "afc bournemouth": "AFC Bournemouth",
        "nottingham forest": "Nottingham Forest",
        "wolverhampton": "Wolverhampton Wanderers",
        "wolves": "Wolverhampton Wanderers",
        "wolverhampton wanderers": "Wolverhampton Wanderers",
        "brentford": "Brentford FC",
        "west ham": "West Ham United",
        "west ham united": "West Ham United",
        "everton": "Everton FC",
        "fulham": "FC Fulham",
        "southampton": "Southampton FC",
        "ipswich": "Ipswich Town",
        "ipswich town": "Ipswich Town",
        "leicester": "Leicester City",
        "leicester city": "Leicester City",
        "burnley": "Burnley FC",
        "sheffield united": "Sheffield United",
        "luton": "Luton Town",
        "luton town": "Luton Town",
        "west brom": "West Bromwich Albion",
        "west bromwich albion": "West Bromwich Albion"
    }
    
    # Try direct mapping first
    if name_lower in direct_mappings:
        return direct_mappings[name_lower]
    
    # Try partial matches
    for key, value in direct_mappings.items():
        if key in name_lower or name_lower in key:
            return value
    
    # Remove common suffixes and try again
    suffixes = [" fc", " united", " city", " town", " fc.", " fc,"]
    for suffix in suffixes:
        if name_lower.endswith(suffix):
            base = name_lower[:-len(suffix)].strip()
            if base in direct_mappings:
                return direct_mappings[base]
    
    return name


def parse_date(date_str):
    """Parse date string in format DD/MM/YYYY or DD/MM/YY."""
    if pd.isna(date_str) or not date_str:
        return None
    
    try:
        # Try DD/MM/YYYY format
        if len(date_str.split('/')) == 3:
            parts = date_str.split('/')
            if len(parts[2]) == 2:
                # Convert YY to YYYY
                year = int(parts[2])
                if year < 50:
                    year += 2000
                else:
                    year += 1900
                return datetime(year, int(parts[1]), int(parts[0]))
            else:
                return datetime(int(parts[2]), int(parts[1]), int(parts[0]))
    except:
        pass
    
    return None


def get_season_dates(season):
    """Get start and end dates for a season."""
    # Premier League seasons typically run from August to May
    year = int(season.split('/')[0])
    if year < 50:
        start_year = 2000 + year
    else:
        start_year = 1900 + year
    
    start_date = datetime(start_year, 8, 1)  # August 1
    end_date = datetime(start_year + 1, 6, 30)  # June 30 next year
    
    return start_date, end_date


def load_and_process_data():
    """Load injuries and career data."""
    print("Loading injuries data...")
    injuries = pd.read_csv(INJURIES_FILE, sep=';', encoding='utf-8')
    print(f"Loaded {len(injuries)} injury records")
    
    print("Loading career data...")
    career = pd.read_csv(CAREER_FILE, sep=';', encoding='utf-8')
    print(f"Loaded {len(career)} career records")
    
    return injuries, career


def build_player_club_timeline(career_df):
    """
    Build a timeline for each player showing which club they were at during each period.
    
    Returns:
        dict: {player_id: [(start_date, end_date, club_name), ...]}
    """
    print("Building player club timeline...")
    
    timeline = defaultdict(list)
    
    for _, row in career_df.iterrows():
        player_id = row['id']
        date_str = row['Date']
        to_club = row['To']
        
        if pd.isna(to_club) or to_club == '':
            continue
        
        date = parse_date(date_str)
        if not date:
            continue
        
        # Normalize club name
        club = normalize_club_name(str(to_club))
        
        timeline[player_id].append((date, club))
    
    # Sort by date for each player
    for player_id in timeline:
        timeline[player_id].sort(key=lambda x: x[0])
    
    # Convert to date ranges
    timeline_ranges = {}
    for player_id, events in timeline.items():
        ranges = []
        for i, (date, club) in enumerate(events):
            if i < len(events) - 1:
                end_date = events[i + 1][0]
            else:
                # Last event - assume they stay until end of 2025
                end_date = datetime(2025, 12, 31)
            ranges.append((date, end_date, club))
        timeline_ranges[player_id] = ranges
    
    return timeline_ranges


def get_player_club_at_date(player_id, date, timeline):
    """Get the club a player was at on a specific date."""
    if player_id not in timeline:
        return None
    
    for start_date, end_date, club in timeline[player_id]:
        if start_date <= date <= end_date:
            return club
    
    return None


def calculate_pl_injuries_by_season(injuries_df, career_df, pl_clubs_by_season):
    """
    Calculate injuries that occurred when players were at PL clubs.
    
    Args:
        injuries_df: DataFrame with injuries
        career_df: DataFrame with player careers
        pl_clubs_by_season: dict {season: set of PL club names}
    
    Returns:
        dict: {season: injury_count}
    """
    print("Calculating PL injuries by season...")
    
    # Build player club timeline
    timeline = build_player_club_timeline(career_df)
    
    # Normalize PL club names for each season
    pl_clubs_normalized = {}
    for season, clubs in pl_clubs_by_season.items():
        pl_clubs_normalized[season] = {normalize_club_name(c) for c in clubs}
    
    # Count injuries
    injury_counts = defaultdict(int)
    injury_details = defaultdict(list)
    
    for _, injury in injuries_df.iterrows():
        season = injury['season']
        player_id = injury['player_id']
        from_date_str = injury['fromDate']
        club_in_injury = injury['clubs']
        
        # Only process relevant seasons
        if season not in SEASONS:
            continue
        
        # Parse injury date
        injury_date = parse_date(from_date_str)
        if not injury_date:
            continue
        
        # Get player's club at injury date
        player_club = get_player_club_at_date(player_id, injury_date, timeline)
        
        # Check if player was at a PL club
        pl_clubs = pl_clubs_normalized[season]
        player_club_normalized = normalize_club_name(player_club) if player_club else ""
        
        # Check if player was at PL club (either from timeline or injury record)
        is_pl_injury = False
        matched_club = None
        
        # Method 1: Check player's club from timeline
        if player_club_normalized:
            player_club_lower = player_club_normalized.lower()
            for pl_club in pl_clubs:
                pl_club_normalized = normalize_club_name(pl_club)
                pl_club_lower = pl_club_normalized.lower()
                # Check if names match (either contains or exact match)
                if (player_club_lower == pl_club_lower or
                    player_club_lower in pl_club_lower or 
                    pl_club_lower in player_club_lower):
                    is_pl_injury = True
                    matched_club = pl_club_normalized
                    break
        
        # Method 2: Check the club mentioned in injury record
        if not is_pl_injury and pd.notna(club_in_injury):
            club_normalized = normalize_club_name(str(club_in_injury))
            club_lower = club_normalized.lower()
            for pl_club in pl_clubs:
                pl_club_normalized = normalize_club_name(pl_club)
                pl_club_lower = pl_club_normalized.lower()
                if (club_lower == pl_club_lower or
                    club_lower in pl_club_lower or 
                    pl_club_lower in club_lower):
                    is_pl_injury = True
                    matched_club = pl_club_normalized
                    break
        
        if is_pl_injury:
            injury_counts[season] += 1
            injury_details[season].append({
                'player_id': player_id,
                'injury_type': injury['injury_type'],
                'date': injury_date,
                'club': player_club or club_in_injury
            })
    
    return dict(injury_counts), injury_details


def estimate_2024_25_costs(injury_counts, historical_costs):
    """
    Estimate 2024/25 costs based on injury trends and historical spending.
    
    Args:
        injury_counts: dict {season: count}
        historical_costs: dict {season: cost_in_millions}
    
    Returns:
        Estimated cost for 2024/25 in million euros
    """
    print("\nEstimating 2024/25 costs...")
    
    # Calculate injury growth rates
    seasons = ["20/21", "21/22", "22/23", "23/24", "24/25"]
    injury_rates = []
    cost_rates = []
    
    for i in range(len(seasons) - 1):
        if seasons[i] in injury_counts and seasons[i+1] in injury_counts:
            if injury_counts[seasons[i]] > 0:
                injury_rate = injury_counts[seasons[i+1]] / injury_counts[seasons[i]]
                injury_rates.append(injury_rate)
        
        if seasons[i] in historical_costs and seasons[i+1] in historical_costs:
            if historical_costs[seasons[i]] > 0:
                cost_rate = historical_costs[seasons[i+1]] / historical_costs[seasons[i]]
                cost_rates.append(cost_rate)
    
    # Average growth rates
    avg_injury_rate = np.mean(injury_rates) if injury_rates else 1.0
    avg_cost_rate = np.mean(cost_rates) if cost_rates else 1.0
    
    print(f"Average injury growth rate: {avg_injury_rate:.3f}")
    print(f"Average cost growth rate: {avg_cost_rate:.3f}")
    
    # Method 1: Extrapolate based on cost growth trend
    if "23/24" in historical_costs:
        estimated_cost_trend = historical_costs["23/24"] * avg_cost_rate
    else:
        estimated_cost_trend = 732 * avg_cost_rate
    
    # Method 2: Use injury-to-cost ratio
    if "23/24" in injury_counts and injury_counts["23/24"] > 0:
        cost_per_injury_23_24 = historical_costs["23/24"] / injury_counts["23/24"]
        if "24/25" in injury_counts:
            estimated_cost_ratio = injury_counts["24/25"] * cost_per_injury_23_24
        else:
            estimated_cost_ratio = injury_counts["23/24"] * avg_injury_rate * cost_per_injury_23_24
    else:
        estimated_cost_ratio = estimated_cost_trend
    
    # Average of both methods
    estimated_cost = (estimated_cost_trend + estimated_cost_ratio) / 2
    
    return estimated_cost, {
        'injury_rates': injury_rates,
        'cost_rates': cost_rates,
        'avg_injury_rate': avg_injury_rate,
        'avg_cost_rate': avg_cost_rate,
        'estimated_cost_trend': estimated_cost_trend,
        'estimated_cost_ratio': estimated_cost_ratio
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("Premier League Injury Analysis (2020/21 to 2024/25)")
    print("=" * 80)
    
    # Step 1: Fetch PL clubs for each season
    print("\nStep 1: Fetching PL clubs from TransferMarkt...")
    pl_clubs_by_season = {}
    for season, saison_id in SEASONS.items():
        print(f"\nFetching clubs for season {season} (saison_id={saison_id})...")
        clubs = fetch_pl_clubs_from_transfermarkt(saison_id)
        pl_clubs_by_season[season] = clubs
        print(f"PL clubs for {season}: {len(clubs)} clubs")
        print(f"  {', '.join(clubs[:5])}...")  # Show first 5
    
    # Step 2: Load data
    print("\n" + "=" * 80)
    print("Step 2: Loading data files...")
    injuries_df, career_df = load_and_process_data()
    
    # Step 3: Calculate PL injuries
    print("\n" + "=" * 80)
    print("Step 3: Calculating PL injuries by season...")
    injury_counts, injury_details = calculate_pl_injuries_by_season(
        injuries_df, career_df, pl_clubs_by_season
    )
    
    # Step 4: Display results
    print("\n" + "=" * 80)
    print("RESULTS: Premier League Injuries by Season")
    print("=" * 80)
    print(f"{'Season':<12} {'Injury Count':<15} {'Historical Cost (M€)':<20}")
    print("-" * 50)
    
    for season in ["20/21", "21/22", "22/23", "23/24", "24/25"]:
        count = injury_counts.get(season, 0)
        cost = HISTORICAL_COSTS.get(season, "N/A")
        print(f"{season:<12} {count:<15} {cost}")
    
    # Step 5: Estimate 2024/25 costs
    print("\n" + "=" * 80)
    print("ESTIMATION: Top 5 Leagues Injury Costs for 2024/25")
    print("=" * 80)
    
    estimated_cost, estimation_details = estimate_2024_25_costs(
        injury_counts, HISTORICAL_COSTS
    )
    
    print(f"\nEstimated cost for Top 5 leagues in 2024/25: {estimated_cost:.2f} million euros")
    print(f"\nEstimation details:")
    print(f"  - Based on cost trend: {estimation_details['estimated_cost_trend']:.2f} M€")
    print(f"  - Based on injury ratio: {estimation_details['estimated_cost_ratio']:.2f} M€")
    print(f"  - Average injury growth: {estimation_details['avg_injury_rate']:.3f}")
    print(f"  - Average cost growth: {estimation_details['avg_cost_rate']:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total PL injuries (2020/21 to 2024/25): {sum(injury_counts.values())}")
    print(f"Estimated Top 5 leagues cost for 2024/25: {estimated_cost:.2f} million euros")
    
    # Calculate average injuries per season
    avg_injuries = sum(injury_counts.values()) / len([s for s in injury_counts.values() if s > 0])
    print(f"Average injuries per season: {avg_injuries:.1f}")
    
    # Show year-over-year changes
    print("\nYear-over-year changes:")
    prev_count = None
    for season in ["20/21", "21/22", "22/23", "23/24", "24/25"]:
        count = injury_counts.get(season, 0)
        if prev_count is not None and prev_count > 0:
            change_pct = ((count - prev_count) / prev_count) * 100
            print(f"  {season}: {count} injuries ({change_pct:+.1f}% vs previous season)")
        else:
            print(f"  {season}: {count} injuries")
        prev_count = count
    
    print("=" * 80)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'season': season,
            'injury_count': injury_counts.get(season, 0),
            'historical_cost_millions_euros': HISTORICAL_COSTS.get(season, None),
            'estimated_cost_millions_euros': estimated_cost if season == "24/25" else None
        }
        for season in ["20/21", "21/22", "22/23", "23/24", "24/25"]
    ])
    
    output_file = "pl_injury_analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return injury_counts, estimated_cost, injury_details


if __name__ == "__main__":
    injury_counts, estimated_cost, injury_details = main()

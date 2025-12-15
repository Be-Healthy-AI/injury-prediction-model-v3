"""
Generate teams_data.csv from match data
Extracts unique teams, normalizes names, and infers countries from competitions
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Set, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Competition to country mappings
COMPETITION_COUNTRY_MAP = {
    # England
    'premier league': 'England',
    'championship': 'England',
    'league one': 'England',
    'league two': 'England',
    'fa cup': 'England',
    'efl cup': 'England',
    'community shield': 'England',
    
    # Spain
    'la liga': 'Spain',
    'segunda división': 'Spain',
    'copa del rey': 'Spain',
    'supercopa': 'Spain',
    
    # Italy
    'serie a': 'Italy',
    'serie b': 'Italy',
    'coppa italia': 'Italy',
    'supercoppa': 'Italy',
    
    # Germany
    'bundesliga': 'Germany',
    '2. bundesliga': 'Germany',
    'dfb-pokal': 'Germany',
    'dfb supercup': 'Germany',
    
    # France
    'ligue 1': 'France',
    'ligue 2': 'France',
    'coupe de france': 'France',
    'trophée des champions': 'France',
    
    # Portugal
    'primeira liga': 'Portugal',
    'liga portugal': 'Portugal',
    'segunda liga': 'Portugal',
    'taça de portugal': 'Portugal',
    'taça da liga': 'Portugal',
    'supertaça': 'Portugal',
    
    # Netherlands
    'eredivisie': 'Netherlands',
    'eerste divisie': 'Netherlands',
    'knvb beker': 'Netherlands',
    
    # Brazil
    'brasileirão': 'Brazil',
    'série a': 'Brazil',
    'série b': 'Brazil',
    'copa do brasil': 'Brazil',
    
    # Argentina
    'primera división': 'Argentina',
    'copa argentina': 'Argentina',
    
    # International
    'champions league': 'International',
    'europa league': 'International',
    'conference league': 'International',
    'world cup': 'International',
    'euro': 'International',
    'copa america': 'International',
    'africa cup': 'International',
    'international friendlies': 'International',
}

# Team name patterns for country inference
TEAM_PATTERNS = {
    'England': [
        'manchester', 'liverpool', 'chelsea', 'arsenal', 'tottenham', 'everton',
        'newcastle', 'leeds', 'west ham', 'aston villa', 'brighton', 'crystal palace',
        'fulham', 'wolves', 'southampton', 'burnley', 'watford', 'norwich',
        'brentford', 'bournemouth', 'sheffield', 'hull', 'cardiff', 'swansea',
        'stoke', 'sunderland', 'middlesbrough', 'derby', 'reading', 'birmingham',
        'blackburn', 'wigan', 'bolton', 'portsmouth', 'qpr', 'millwall',
    ],
    'Spain': [
        'real madrid', 'barcelona', 'atlético', 'atletico', 'sevilla', 'valencia',
        'villarreal', 'real sociedad', 'athletic', 'betis', 'celta', 'espanyol',
        'getafe', 'osasuna', 'mallorca', 'granada', 'levante', 'alavés', 'alaves',
        'eibar', 'valladolid', 'cádiz', 'cadiz', 'elche', 'huesca',
    ],
    'Italy': [
        'juventus', 'milan', 'inter', 'napoli', 'roma', 'lazio', 'atalanta',
        'fiorentina', 'torino', 'sampdoria', 'genoa', 'bologna', 'udinese',
        'sassuolo', 'cagliari', 'verona', 'parma', 'benevento', 'spezia',
        'venezia', 'salernitana', 'empoli', 'lecce', 'monza',
    ],
    'Germany': [
        'bayern', 'dortmund', 'leipzig', 'leverkusen', 'mönchengladbach', 'monchengladbach',
        'wolfsburg', 'frankfurt', 'hoffenheim', 'stuttgart', 'hertha', 'union',
        'augsburg', 'mainz', 'freiburg', 'köln', 'koln', 'bremen', 'schalke',
        'hamburg', 'nürnberg', 'nurnberg', 'düsseldorf', 'dusseldorf',
    ],
    'France': [
        'psg', 'paris saint', 'lyon', 'marseille', 'monaco', 'lille', 'nice',
        'rennes', 'bordeaux', 'saint-étienne', 'saint-etienne', 'toulouse',
        'nantes', 'montpellier', 'strasbourg', 'reims', 'lens', 'angers',
        'metz', 'lorient', 'brest', 'dijon', 'amiens',
    ],
    'Portugal': [
        'benfica', 'porto', 'sporting', 'braga', 'vitória', 'vitoria', 'guimarães', 'guimaraes',
        'marítimo', 'maritimo', 'moreirense', 'paços', 'pacos', 'boavista', 'rio ave',
        'estoril', 'tondela', 'portimonense', 'santa clara', 'famalicão', 'famalicao',
        'arouca', 'chaves', 'gil vicente', 'vizela', 'feirense', 'alverca',
    ],
    'Brazil': [
        'flamengo', 'palmeiras', 'corinthians', 'são paulo', 'sao paulo', 'santos',
        'grêmio', 'gremio', 'internacional', 'atlético mineiro', 'atletico mineiro',
        'cruzeiro', 'fluminense', 'botafogo', 'vasco', 'bahia', 'vitória', 'vitoria',
        'sport', 'nautico', 'santa cruz', 'figueirense', 'criciúma', 'criciuma',
        'atlético paranaense', 'atletico paranaense', 'coritiba', 'paraná', 'parana',
        'goiás', 'goias', 'atlético goianiense', 'atletico goianiense',
    ],
    'Netherlands': [
        'ajax', 'psv', 'feyenoord', 'az alkmaar', 'utrecht', 'vitesse', 'heerenveen',
        'groningen', 'twente', 'sparta', 'willem ii', 'heracles', 'fortuna',
    ],
}

def normalize_team_name(team_name: str) -> str:
    """Normalize team name for comparison"""
    if pd.isna(team_name) or not team_name:
        return ''
    
    # Convert to string and lowercase
    name = str(team_name).strip().lower()
    
    # Remove common suffixes/prefixes that don't affect identity
    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
    name = re.sub(r'\b(fc|cf|sc|ac|bc|bk|cf)\b', '', name)  # Remove common prefixes
    name = re.sub(r'\b(u\d+|u-\d+|youth|junior|reserve|b team)\b', '', name)  # Remove age groups
    name = re.sub(r'\s+', ' ', name).strip()  # Clean up whitespace again
    
    return name

def get_competition_country(competition: str) -> Optional[str]:
    """Infer country from competition name"""
    if pd.isna(competition) or not competition:
        return None
    
    comp_lower = str(competition).lower().strip()
    
    # Check exact matches first
    for key, country in COMPETITION_COUNTRY_MAP.items():
        if key in comp_lower:
            return country
    
    # Check for league patterns
    if 'premier' in comp_lower and 'league' in comp_lower:
        return 'England'
    if 'la liga' in comp_lower or 'primera división' in comp_lower:
        return 'Spain'
    if 'serie a' in comp_lower or 'serie b' in comp_lower:
        return 'Italy'
    if 'bundesliga' in comp_lower:
        return 'Germany'
    if 'ligue 1' in comp_lower or 'ligue 2' in comp_lower:
        return 'France'
    if 'primeira liga' in comp_lower or 'liga portugal' in comp_lower:
        return 'Portugal'
    if 'eredivisie' in comp_lower:
        return 'Netherlands'
    if 'brasileirão' in comp_lower or 'série a' in comp_lower or 'série b' in comp_lower:
        return 'Brazil'
    
    return None

def infer_team_country_from_patterns(team_name: str) -> Optional[str]:
    """Infer country from team name patterns"""
    if pd.isna(team_name) or not team_name:
        return None
    
    team_lower = str(team_name).lower()
    
    for country, patterns in TEAM_PATTERNS.items():
        for pattern in patterns:
            if pattern in team_lower:
                return country
    
    return None

def extract_teams_from_matches(match_data_dir: str) -> Tuple[pd.DataFrame, Dict[str, Counter]]:
    """Extract all unique teams from match files - OPTIMIZED"""
    logger.info("Scanning match files...")
    
    all_teams = set()
    team_competitions = defaultdict(Counter)  # team -> {competition: count}
    match_files = glob.glob(os.path.join(match_data_dir, 'match_*.csv'))
    
    logger.info(f"Found {len(match_files)} match files")
    
    # Read only necessary columns for speed
    usecols = ['home_team', 'away_team', 'competition']
    
    for i, match_file in enumerate(match_files):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{len(match_files)} files... ({len(all_teams)} teams so far)")
        
        try:
            # Read only necessary columns
            df = pd.read_csv(match_file, encoding='utf-8', sep=',', usecols=usecols, low_memory=False)
            
            # Extract teams from home_team and away_team - vectorized
            if 'home_team' in df.columns:
                home_teams = df['home_team'].dropna().unique()
                all_teams.update(home_teams)
                
                # Track competitions for each team - vectorized
                if 'competition' in df.columns:
                    for team in home_teams:
                        team_comps = df[df['home_team'] == team]['competition'].dropna()
                        if len(team_comps) > 0:
                            team_competitions[team].update(team_comps.tolist())
            
            if 'away_team' in df.columns:
                away_teams = df['away_team'].dropna().unique()
                all_teams.update(away_teams)
                
                # Track competitions for each team - vectorized
                if 'competition' in df.columns:
                    for team in away_teams:
                        team_comps = df[df['away_team'] == team]['competition'].dropna()
                        if len(team_comps) > 0:
                            team_competitions[team].update(team_comps.tolist())
        
        except Exception as e:
            logger.warning(f"Error reading {match_file}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_teams)} unique teams")
    
    # Create DataFrame
    teams_df = pd.DataFrame({'team': sorted(all_teams)})
    
    return teams_df, team_competitions

def infer_countries(teams_df: pd.DataFrame, team_competitions: Dict[str, Counter]) -> pd.DataFrame:
    """Infer countries for teams"""
    logger.info("Inferring countries for teams...")
    
    countries = []
    confidence = []
    methods = []
    
    for team in teams_df['team']:
        country = None
        conf = 'low'
        method = 'none'
        
        # Method 1: Infer from most common competition
        if team in team_competitions:
            comp_counts = team_competitions[team]
            if comp_counts:
                most_common_comp = comp_counts.most_common(1)[0][0]
                country = get_competition_country(most_common_comp)
                if country:
                    conf = 'medium'
                    method = 'competition'
        
        # Method 2: Infer from team name patterns
        if not country:
            country = infer_team_country_from_patterns(team)
            if country:
                conf = 'medium'
                method = 'pattern'
        
        # Method 3: Check all competitions for this team
        if not country and team in team_competitions:
            for comp, _ in team_competitions[team].most_common():
                country = get_competition_country(comp)
                if country and country != 'International':
                    conf = 'high'
                    method = 'competition_analysis'
                    break
        
        countries.append(country)
        confidence.append(conf)
        methods.append(method)
    
    teams_df['country'] = countries
    teams_df['confidence'] = confidence
    teams_df['method'] = methods
    
    # Count statistics
    inferred = teams_df['country'].notna().sum()
    logger.info(f"Inferred countries for {inferred}/{len(teams_df)} teams ({inferred*100/len(teams_df):.1f}%)")
    logger.info(f"  High confidence: {(teams_df['confidence'] == 'high').sum()}")
    logger.info(f"  Medium confidence: {(teams_df['confidence'] == 'medium').sum()}")
    logger.info(f"  Low confidence: {(teams_df['confidence'] == 'low').sum()}")
    
    return teams_df

def group_similar_teams(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Group similar team names (variations of the same team)"""
    logger.info("Grouping similar team names...")
    
    # Create normalized version for grouping
    teams_df['normalized'] = teams_df['team'].apply(normalize_team_name)
    
    # Group by normalized name
    groups = teams_df.groupby('normalized')
    
    # For each group, pick the most common original name and country
    grouped_teams = []
    for norm_name, group in groups:
        if len(group) > 1:
            # Multiple variations - pick the most common one
            # Use the one with highest confidence country, or first one
            best_row = group.sort_values('confidence', ascending=False).iloc[0]
            grouped_teams.append({
                'team': best_row['team'],
                'country': best_row['country'],
                'confidence': best_row['confidence'],
                'method': best_row['method'],
                'variations': ', '.join(group['team'].tolist())
            })
        else:
            # Single team
            row = group.iloc[0]
            grouped_teams.append({
                'team': row['team'],
                'country': row['country'],
                'confidence': row['confidence'],
                'method': row['method'],
                'variations': row['team']
            })
    
    result_df = pd.DataFrame(grouped_teams)
    logger.info(f"Grouped {len(teams_df)} teams into {len(result_df)} unique teams")
    
    return result_df

def main():
    """Main function"""
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    match_data_dir = os.path.join(base_dir, 'match_data')
    output_file = os.path.join(base_dir, 'teams_data.csv')
    
    logger.info("=" * 60)
    logger.info("Generating teams_data.csv")
    logger.info("=" * 60)
    
    # Step 1: Extract teams from match files
    teams_df, team_competitions = extract_teams_from_matches(match_data_dir)
    
    # Step 2: Infer countries
    teams_df = infer_countries(teams_df, team_competitions)
    
    # Step 3: Group similar teams
    teams_df = group_similar_teams(teams_df)
    
    # Step 4: Sort and prepare final output
    teams_df = teams_df.sort_values('team').reset_index(drop=True)
    
    # Create final output (only team and country columns for compatibility)
    output_df = teams_df[['team', 'country']].copy()
    
    # Save main file (use utf-8-sig for Excel compatibility on Windows)
    output_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    logger.info(f"\nSaved teams_data.csv to: {output_file}")
    logger.info(f"Total teams: {len(output_df)}")
    logger.info(f"Teams with country: {output_df['country'].notna().sum()}")
    logger.info(f"Teams without country: {output_df['country'].isna().sum()}")
    
    # Save detailed version with confidence and variations (use utf-8-sig for Excel compatibility)
    detailed_file = os.path.join(base_dir, 'teams_data_detailed.csv')
    teams_df.to_csv(detailed_file, index=False, sep=';', encoding='utf-8-sig')
    logger.info(f"\nSaved detailed version to: {detailed_file}")
    
    # Show sample of teams without country (for manual review)
    missing_country = teams_df[teams_df['country'].isna()]
    if len(missing_country) > 0:
        logger.info(f"\nSample teams without country (first 20):")
        for team in missing_country['team'].head(20):
            logger.info(f"  - {team}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()


# scripts/generate_parameter_files.py

from pathlib import Path
import pandas as pd
import json
import csv
from typing import Dict, Any, List

def generate_test_data(output_dir: Path) -> None:
    """Generate test content data in both English and Finnish."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # English content
    df_en = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            "Hello! I'm interested in learning programming. Which courses would you recommend for a beginner?",
            "Hi! I'd like to know more about your marketing courses. Do you offer part-time options?",
            "Hello! I'm considering a career change. How could I best utilize my sales experience when transitioning to IT?",
            "Hi there! I'm interested in your online courses. Are they flexible in terms of scheduling?",
            "Greetings! I'm an entrepreneur looking for business development training. What would you recommend?"
        ]
    })
    
    # Finnish content
    df_fi = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'keskustelu': [
            "Hei! Olen kiinnostunut opiskelemaan ohjelmointia. Mitä kursseja suosittelisitte aloittelijalle?",
            "Terve! Haluaisin tietää lisää teidän markkinoinnin koulutuksista. Onko teillä tarjolla osa-aikaisia vaihtoehtoja?",
            "Moi! Olen harkinnut alanvaihtoa. Miten voisin parhaiten hyödyntää aiempaa kokemustani myynnissä IT-alalle siirtyessäni?",
            "Hei siellä! Kiinnostaisi tietää enemmän teidän verkkokursseista. Ovatko ne joustavia aikataulultaan?",
            "Tervehdys! Olen yrittäjä ja etsin koulutusta liiketoiminnan kehittämiseen. Mitä voisitte suositella?"
        ]
    })
    
    # Save content files
    df_en.to_excel(output_dir / 'sample_content_en.xlsx', index=False)
    df_en.to_csv(output_dir / 'sample_content_en.csv', index=False, encoding='utf-8')
    
    df_fi.to_excel(output_dir / 'sample_content_fi.xlsx', index=False)
    df_fi.to_csv(output_dir / 'sample_content_fi.csv', index=False, encoding='utf-8')

def get_parameter_data(language: str = "en") -> Dict[str, pd.DataFrame]:
    """Get parameter data for specified language."""
    if language == "fi":
        # Finnish parameters
        general_params = pd.DataFrame({
            'parametri': [
                'max_kws', 'max_themes', 'focus_on', 'language',
                'additional_context', 'column_name_to_analyze'
            ],
            'arvo': [
                8, 3, 'koulutukseen ja työelämään liittyvät aiheet', 'Finnish',
                'Koulutuspalveluita tarjoavan yrityksen asiakaspalvelukeskustelut', 'keskustelu'
            ]
        })
        
        categories = pd.DataFrame({
            'kategoria': ['koulutusmuoto', 'urasuunnittelu', 'koulutuksen_sisalto', 'aikataulu', 'yrityspalvelut'],
            'kuvaus': [
                'Koulutuksen toteutustapa (esim. verkko, lähi, osa-aika)',
                'Uraan, alanvaihtoon ja työllistymiseen liittyvät aiheet',
                'Koulutuksen sisältöön ja vaatimuksiin liittyvät asiat',
                'Koulutuksen aikatauluun ja kestoon liittyvät asiat',
                'Yrittäjille ja yrityksille suunnatut palvelut'
            ]
        })
        
        predefined_keywords = pd.DataFrame({
            'avainsana': [
                'ohjelmointi', 'markkinointi', 'alanvaihto', 'verkkokurssi',
                'yrittäjyys', 'osa-aikainen', 'IT-ala', 'myynti',
                'liiketoiminnan kehittäminen'
            ]
        })
        
        excluded_keywords = pd.DataFrame({
            'avainsana': ['koulutus', 'opiskelu']
        })
        
        sheets = {
            'general': ('yleiset säännöt', general_params),
            'categories': ('kategoriat', categories),
            'keywords': ('haettavat avainsanat', predefined_keywords),
            'excluded': ('älä käytä', excluded_keywords)
        }
        
    else:
        # English parameters
        general_params = pd.DataFrame({
            'parameter': [
                'max_kws', 'max_themes', 'focus_on', 'language',
                'additional_context', 'column_name_to_analyze'
            ],
            'value': [
                8, 3, 'education and career-related topics', 'English',
                'Customer service conversations for an educational services company', 'text'
            ]
        })
        
        categories = pd.DataFrame({
            'category': ['education_type', 'career_planning', 'course_content', 'scheduling', 'business_services'],
            'description': [
                'Type of education (e.g., online, in-person, part-time)',
                'Career, career change, and employment topics',
                'Course content and requirements',
                'Course scheduling and duration',
                'Services for entrepreneurs and businesses'
            ]
        })
        
        predefined_keywords = pd.DataFrame({
            'keyword': [
                'programming', 'marketing', 'career change', 'online course',
                'entrepreneurship', 'part-time', 'IT sector', 'sales',
                'business development'
            ]
        })
        
        excluded_keywords = pd.DataFrame({
            'keyword': ['education', 'study']
        })
        
        sheets = {
            'general': ('General Parameters', general_params),
            'categories': ('Categories', categories),
            'keywords': ('Predefined Keywords', predefined_keywords),
            'excluded': ('Excluded Keywords', excluded_keywords)
        }
    
    return sheets

def generate_parameter_files(output_dir: Path, language: str = "en") -> None:
    """Generate parameter Excel and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameter data
    sheets = get_parameter_data(language)
    
    # Save Excel file
    excel_path = output_dir / f"parameters_{language}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for sheet_name, (tab_name, df) in sheets.items():
            df.to_excel(writer, sheet_name=tab_name, index=False)
    
    # Extract DataFrames from sheets
    general_df = sheets['general'][1]
    categories_df = sheets['categories'][1]
    keywords_df = sheets['keywords'][1]
    excluded_df = sheets['excluded'][1]
    
    # Set up column names based on language
    param_key = 'parametri' if language == 'fi' else 'parameter'
    value_key = 'arvo' if language == 'fi' else 'value'
    
    # Convert category data based on language
    if language == "fi":
        category_dict = dict(zip(categories_df['kategoria'], categories_df['kuvaus']))
        keyword_list = keywords_df['avainsana'].tolist()
        excluded_list = excluded_df['avainsana'].tolist()
    else:
        category_dict = dict(zip(categories_df['category'], categories_df['description']))
        keyword_list = keywords_df['keyword'].tolist()
        excluded_list = excluded_df['keyword'].tolist()
    
    # Create combined parameters
    combined_params = pd.concat([
        general_df,
        pd.DataFrame({
            param_key: ['categories'],
            value_key: [json.dumps(category_dict, ensure_ascii=False)]
        }),
        pd.DataFrame({
            param_key: ['predefined_keywords'],
            value_key: [json.dumps(keyword_list, ensure_ascii=False)]
        }),
        pd.DataFrame({
            param_key: ['excluded_keywords'],
            value_key: [json.dumps(excluded_list, ensure_ascii=False)]
        })
    ])
    
    # Save CSV
    csv_path = output_dir / f"parameters_{language}.csv"
    combined_params.to_csv(
        csv_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8-sig'  # For better Excel compatibility
    )

def main():
    """Generate all test data and parameter files."""
    # output_dir = Path("example_parameters")
    output_dir = Path("data/raw/test_data")
    
    # Generate test content
    print("Generating test content...")
    generate_test_data(output_dir)
    
    # Generate parameter files
    for lang in ["en", "fi"]:
        print(f"Generating {lang} parameters...")
        generate_parameter_files(output_dir, language=lang)
        
    print("\nFiles generated successfully:")
    print(f"Output directory: {output_dir.absolute()}")
    for file in output_dir.glob("*"):
        print(f"- {file.name}")

if __name__ == "__main__":
    main()
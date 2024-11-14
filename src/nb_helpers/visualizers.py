# src/nb_helpers/visualizers.py
from typing import Any, Dict, List
import pandas as pd

def format_confidence_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def create_analysis_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Create analysis summary DataFrame."""
    keywords = []
    if results.get("keywords") and hasattr(results["keywords"], "keywords"):
        keywords = [k.keyword for k in results["keywords"].keywords]
        
    categories = []
    if results.get("categories") and hasattr(results["categories"], "categories"):
        categories = [c.name for c in results["categories"].categories]
        
    themes = []
    if results.get("themes") and hasattr(results["themes"], "themes"):
        themes = [t.name for t in results["themes"].themes]
        
    return pd.DataFrame({
        "keywords": [", ".join(keywords)],
        "categories": [", ".join(categories)],
        "themes": [", ".join(themes)]
    })

def display_analysis_report(results: Dict[str, Any], show_confidence: bool = True) -> None:
    """Display formatted analysis report."""
    print("\nAnalysis Report")
    print("=" * 50)
    
    for analysis_type, data in results.items():
        if analysis_type not in ["keywords", "themes", "categories"]:
            continue
            
        print(f"\n{analysis_type.title()}:")
        print("-" * 20)
        
        if hasattr(data, "error") and data.error:
            print(f"Error: {data.error}")
            continue
            
        items = getattr(data, analysis_type, [])
        for item in items:
            if show_confidence and hasattr(item, "confidence"):
                bar = format_confidence_bar(item.confidence)
                print(f"• {item.name:<20} [{bar}] ({item.confidence:.2f})")
            else:
                print(f"• {item.name}")
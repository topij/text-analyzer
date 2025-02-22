import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from typing import Optional, Tuple, Set, List, Dict, Any
import tempfile
import asyncio
from langdetect import detect
import atexit
from text_manager import get_ui_text_manager
from FileUtils import FileUtils

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analyzers.lite_analyzer import LiteSemanticAnalyzer
from src.core.managers import EnvironmentManager, EnvironmentConfig
from src.core.llm.factory import create_llm
from src.core.config import AnalyzerConfig
from src.loaders.parameter_handler import ParameterHandler
from ui.text_manager import UITextManager, get_ui_text_manager
from src.config import ConfigManager

# Initialize session state first
if 'ui_language' not in st.session_state:
    st.session_state.ui_language = 'fi'  # Default to Finnish
if 'previous_ui_language' not in st.session_state:
    st.session_state.previous_ui_language = 'fi'
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'show_parameters' not in st.session_state:
    st.session_state.show_parameters = False
if 'texts_df' not in st.session_state:
    st.session_state.texts_df = None
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = None
if 'temp_parameter_file' not in st.session_state:
    st.session_state.temp_parameter_file = None
if 'file_utils' not in st.session_state:
    st.session_state.file_utils = FileUtils()
if 'text_manager' not in st.session_state:
    st.session_state.text_manager = get_ui_text_manager(file_utils=st.session_state.file_utils)

# Get text function that uses cached text manager
def get_text(category: str, key: str, language: Optional[str] = None, **kwargs) -> str:
    return st.session_state.text_manager.get_text(category, key, language, **kwargs)

# Page configuration
st.set_page_config(
    page_title=get_text("titles", "main", language=st.session_state.ui_language),
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Text Analyzer - A powerful tool for semantic text analysis"
    }
)

def get_available_categories(parameter_file: Optional[Path] = None) -> Set[str]:
    """Get available categories from parameter file."""
    if not parameter_file:
        return set()
        
    try:
        parameter_handler = ParameterHandler(parameter_file)
        parameters = parameter_handler.get_parameters()
        return {cat for cat in parameters.categories.keys()} if parameters.categories else set()
    except Exception as e:
        st.warning(f"Failed to load categories from parameter file: {e}")
        return set()

def initialize_analyzer():
    """Initialize the analyzer with proper configuration."""
    # Get components from environment manager
    components = EnvironmentManager.get_instance().get_components()
    config_manager = components["config_manager"]
    
    # Get configurations
    base_config = config_manager.get_analyzer_config("base")
    config = config_manager.get_config()
    
    # Get default provider and model from config
    model_config = config.get("models", {})
    provider = model_config.get("default_provider", "openai")
    model = model_config.get("default_model", "gpt-4o-mini")
    
    # Create analyzer config
    base_config.update({
        "models": {
            "default_provider": provider,
            "default_model": model,
        }
    })
    analyzer_config = AnalyzerConfig(base_config)
    
    # Create LLM instance
    llm = create_llm(
        analyzer_config=analyzer_config,
        provider=provider,
        model=model
    )
    
    # Get available categories from parameter file
    available_categories = set()
    if hasattr(st.session_state, 'temp_parameter_file') and st.session_state.temp_parameter_file:
        available_categories = get_available_categories(st.session_state.temp_parameter_file)
    
    # Create analyzer
    analyzer = LiteSemanticAnalyzer(
        llm=llm,  # Pass LLM directly without modification
        file_utils=st.session_state.file_utils,
        parameter_file=st.session_state.temp_parameter_file if hasattr(st.session_state, 'temp_parameter_file') else None,
        language=st.session_state.get('ui_language', 'en'),
        available_categories=available_categories
    )
    
    return analyzer

# Initialize session state
if 'analyzer' not in st.session_state:
    # Configure environment
    env_config = EnvironmentConfig(
        project_root=Path().resolve(),
        log_level="ERROR"
    )

    # Initialize environment
    environment = EnvironmentManager(env_config)
    components = environment.get_components()
    
    # Get configurations
    config_manager = components["config_manager"]
    base_config = config_manager.get_analyzer_config("base")
    config = config_manager.get_config()
    
    # Get default provider and model from config
    model_config = config.get("models", {})
    provider = model_config.get("default_provider", "openai")
    model = model_config.get("default_model", "gpt-4o-mini")
    
    # Create analyzer config
    base_config.update({
        "models": {
            "default_provider": provider,
            "default_model": model,
        }
    })
    analyzer_config = AnalyzerConfig(base_config)
    
    # Create LLM instance
    llm = create_llm(
        analyzer_config=analyzer_config,
        provider=provider,
        model=model
    )
    
    # Get available categories from parameter file
    available_categories = get_available_categories(st.session_state.temp_parameter_file) if hasattr(st.session_state, 'temp_parameter_file') else set()
    
    # Create analyzer
    st.session_state.analyzer = LiteSemanticAnalyzer(
        llm=llm,
        parameter_file=st.session_state.temp_parameter_file if hasattr(st.session_state, 'temp_parameter_file') else None,
        file_utils=st.session_state.file_utils,  # Use cached FileUtils instance
        language=st.session_state.get('ui_language', 'en'),
        available_categories=available_categories
    )

def detect_content_language(df: pd.DataFrame, text_column: str) -> str:
    """Detect the primary language of the content"""
    try:
        # Sample up to 5 texts to determine the dominant language
        sample_texts = df[text_column].dropna().head(5).tolist()
        languages = [detect(text) for text in sample_texts if text.strip()]
        # Return most common language, defaulting to 'en' if can't determine
        if languages:
            from collections import Counter
            return Counter(languages).most_common(1)[0][0]
    except Exception as e:
        st.warning(f"Error detecting language: {e}")
    return 'en'

def handle_file_upload(uploaded_file, file_type: str) -> Tuple[bool, Optional[str]]:
    """Handle file upload with validation and feedback."""
    if uploaded_file is not None:
        try:
            # Get data directory and create temp subdirectory
            data_dir = st.session_state.file_utils.get_data_path("raw")
            temp_dir = data_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            if file_type == "texts":
                # Save text file to temp directory
                tmp_path = temp_dir / uploaded_file.name
                with open(tmp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    df = pd.read_excel(tmp_path) if tmp_path.suffix == '.xlsx' else pd.read_csv(tmp_path)
                    st.session_state.texts_df = df
                    
                    if 'column_name_to_analyze' in st.session_state.params:
                        col_name = st.session_state.params['column_name_to_analyze']
                        if col_name in df.columns:
                            detected_lang = detect_content_language(df, col_name)
                            st.session_state.detected_language = detected_lang
                            st.info(f"Detected content language: {detected_lang}")
                finally:
                    # Clean up temporary text file
                    tmp_path.unlink()
            
            elif file_type == "parameters":
                # Clean up previous parameter file if it exists
                if st.session_state.temp_parameter_file and st.session_state.temp_parameter_file.exists():
                    st.session_state.temp_parameter_file.unlink()
                
                # Save new parameter file in temp directory
                param_file = temp_dir / uploaded_file.name
                with open(param_file, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Update session state with new parameter file path
                st.session_state.temp_parameter_file = param_file
                
                try:
                    # Load parameters using ParameterHandler
                    parameter_handler = ParameterHandler(param_file)
                    parameters = parameter_handler.get_parameters()
                    
                    # Update session state with parameters
                    st.session_state.params = {
                        'column_name_to_analyze': parameters.general.column_name_to_analyze,
                        'max_keywords': parameters.general.max_keywords,
                        'max_themes': parameters.general.max_themes,
                        'focus': parameters.general.focus_on,
                        'language': parameters.general.language
                    }
                    
                    # Reinitialize analyzer with new parameters
                    st.session_state.analyzer = initialize_analyzer()
                    
                except Exception as e:
                    # If parameter loading fails, clean up the file
                    param_file.unlink()
                    st.session_state.temp_parameter_file = None
                    raise e
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    return False, None

def display_help(title: str, content: str):
    """Display help tooltip with consistent styling."""
    with st.expander(f"ℹ️ {title}"):
        st.markdown(content)

async def analyze_texts(texts: List[str]) -> Dict[str, Any]:
    """Analyze multiple texts."""
    try:
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = initialize_analyzer()
            
        # Update both parameters and config from UI
        if hasattr(st.session_state.analyzer, 'parameters'):
            st.session_state.analyzer.parameters.general.max_keywords = st.session_state.max_keywords
            st.session_state.analyzer.parameters.general.max_themes = st.session_state.max_themes
            st.session_state.analyzer.parameters.general.focus_on = st.session_state.focus
            
        # Update config as well
        if hasattr(st.session_state.analyzer, 'config'):
            st.session_state.analyzer.config.update({
                'max_keywords': st.session_state.max_keywords,
                'max_themes': st.session_state.max_themes,
                'focus_on': st.session_state.focus,
                'include_compounds': True,  # Keep existing compound word handling
                'min_confidence': 0.6  # Keep existing confidence threshold
            })
            
        # Reinitialize the analyzer to ensure changes take effect
        st.session_state.analyzer = initialize_analyzer()
            
        results = {
            "keywords": [],
            "themes": [],
            "categories": []
        }
        
        success_count = 0
        total_texts = len(texts)
        
        with st.spinner(get_text("messages", "analyzing", language=st.session_state.ui_language)):
            for text in texts:
                try:
                    result = await st.session_state.analyzer.analyze(
                        text=text,
                        analysis_types=["keywords", "themes", "categories"]
                    )
                    
                    if result and result.success:
                        if result.keywords and result.keywords.success:
                            results["keywords"].append(result.keywords)
                        if result.themes and result.themes.success:
                            results["themes"].append(result.themes)
                        if result.categories and result.categories.success:
                            results["categories"].append(result.categories)
                        success_count += 1
                    else:
                        # Store empty results for failed analyses to maintain text index alignment
                        results["keywords"].append(None)
                        results["themes"].append(None)
                        results["categories"].append(None)
                        st.warning(f"Analysis failed for text: {text[:100]}...")
                except Exception as e:
                    st.warning(f"Analysis failed for text: {text[:100]}... Error: {str(e)}")
                    results["keywords"].append(None)
                    results["themes"].append(None)
                    results["categories"].append(None)
            
            st.session_state.analysis_results = results
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                get_text("tabs", "keywords", language=st.session_state.ui_language),
                get_text("tabs", "themes", language=st.session_state.ui_language),
                get_text("tabs", "categories", language=st.session_state.ui_language)
            ])
            
            with tab1:
                if 'keywords' in st.session_state.analysis_results:
                    # Create a list to store rows for the keywords table
                    keywords_data = []
                    for idx, result in enumerate(st.session_state.analysis_results['keywords']):
                        text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                        if result and result.success:
                            # Sort keywords by score and take only the top N based on UI setting
                            sorted_keywords = sorted(result.keywords, key=lambda x: x.score, reverse=True)
                            top_keywords = sorted_keywords[:st.session_state.max_keywords]
                            keywords_str = ", ".join([
                                f"{kw.keyword} ({kw.score:.2f})"
                                for kw in top_keywords
                            ])
                        else:
                            keywords_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                        keywords_data.append({
                            get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                            get_text("table_headers", "keywords", language=st.session_state.ui_language): keywords_str
                        })
                    
                    # Display keywords table
                    if keywords_data:
                        st.dataframe(
                            pd.DataFrame(keywords_data),
                            use_container_width=True,
                            hide_index=True
                        )
            
            with tab2:
                if 'themes' in st.session_state.analysis_results:
                    # Create a list to store rows for the themes table
                    themes_data = []
                    for idx, result in enumerate(st.session_state.analysis_results['themes']):
                        text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                        if result and result.success:
                            themes_str = ", ".join([f"{theme.name} ({theme.confidence:.2f})" for theme in result.themes])
                            if result.theme_hierarchy:
                                themes_str += f"\n\n{get_text('result_labels', 'hierarchy', language=st.session_state.ui_language)}:\n" + "\n".join([
                                    f"- {main_theme}: {', '.join(sub_themes)}"
                                    for main_theme, sub_themes in result.theme_hierarchy.items()
                                ])
                        else:
                            themes_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                        themes_data.append({
                            get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                            get_text("table_headers", "themes", language=st.session_state.ui_language): themes_str
                        })
                    
                    # Display themes table
                    if themes_data:
                        st.dataframe(
                            pd.DataFrame(themes_data),
                            use_container_width=True,
                            hide_index=True
                        )
            
            with tab3:
                if 'categories' in st.session_state.analysis_results:
                    # Create a list to store rows for the categories table
                    categories_data = []
                    for idx, result in enumerate(st.session_state.analysis_results['categories']):
                        text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                        if result and result.success:
                            categories_str = ""
                            for cat in result.matches:
                                categories_str += f"{cat.name} ({cat.confidence:.2f})\n"
                                if cat.evidence:
                                    categories_str += f"{get_text('result_labels', 'evidence', language=st.session_state.ui_language)}:\n" + "\n".join([
                                        f"- {evidence.text} (relevance: {evidence.relevance:.2f})"
                                        for evidence in cat.evidence
                                    ]) + "\n"
                                if hasattr(cat, 'themes') and cat.themes:
                                    categories_str += f"{get_text('result_labels', 'related_themes', language=st.session_state.ui_language)}: {', '.join(cat.themes)}\n"
                                categories_str += "\n"
                        else:
                            categories_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                        categories_data.append({
                            get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                            get_text("table_headers", "categories", language=st.session_state.ui_language): categories_str
                        })
                    
                    # Display categories table
                    if categories_data:
                        st.dataframe(
                            pd.DataFrame(categories_data),
                            use_container_width=True,
                            hide_index=True
                        )

            if success_count == 0:
                st.error(get_text("messages", "analysis_failed_all", language=st.session_state.ui_language, total_texts=total_texts))
            elif success_count < total_texts:
                st.warning(get_text("messages", "analysis_partial", language=st.session_state.ui_language, success_count=success_count, total_texts=total_texts))
            else:
                st.success(get_text("messages", "analysis_complete", language=st.session_state.ui_language))
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def main():
    st.title(get_text("titles", "main", language=st.session_state.ui_language))
    
    # Sidebar
    with st.sidebar:
        st.header(get_text("titles", "help", language=st.session_state.ui_language))
        display_help(
            get_text("help_texts", "what_is_title", language=st.session_state.ui_language),
            get_text("help_texts", "what_is", language=st.session_state.ui_language)
        )
        
        display_help(
            get_text("help_texts", "file_requirements_title", language=st.session_state.ui_language),
            get_text("help_texts", "file_requirements", language=st.session_state.ui_language)
        )
        
        # Language selector
        st.header(get_text("titles", "language_settings", language=st.session_state.ui_language))
        ui_language = st.selectbox(
            get_text("labels", "interface_language", language=st.session_state.ui_language),
            options=['fi', 'en'],
            index=0 if st.session_state.ui_language == 'fi' else 1,
            format_func=lambda x: 'Suomi' if x == 'fi' else 'English',
            key='ui_language'
        )
        
        if ui_language != st.session_state.get('previous_ui_language'):
            st.session_state.previous_ui_language = ui_language
            st.rerun()
    
    # Main content area
    st.subheader(get_text("titles", "file_upload", language=st.session_state.ui_language), divider='grey')
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(get_text("labels", "upload_texts", language=st.session_state.ui_language))
        texts_file = st.file_uploader(
            get_text("labels", "choose_texts", language=st.session_state.ui_language),
            type=['xlsx', 'csv'],
            key='texts_uploader'
        )
        if texts_file:
            success, error = handle_file_upload(texts_file, "texts")
            if success:
                st.success(get_text("messages", "texts_loaded", language=st.session_state.ui_language))
            else:
                st.error(get_text("messages", "error_load", language=st.session_state.ui_language, error=error))
    
    with col2:
        st.markdown(get_text("labels", "upload_parameters", language=st.session_state.ui_language))
        params_file = st.file_uploader(
            get_text("labels", "choose_parameters", language=st.session_state.ui_language),
            type=['xlsx'],
            key='params_uploader'
        )
        if params_file:
            success, error = handle_file_upload(params_file, "parameters")
            if success:
                st.success(get_text("messages", "parameters_loaded", language=st.session_state.ui_language))
            else:
                st.error(get_text("messages", "error_load", language=st.session_state.ui_language, error=error))
    
    # Parameter settings
    st.subheader(get_text("titles", "parameters", language=st.session_state.ui_language), divider='grey')
    
    if st.session_state.texts_df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.number_input(
                get_text("labels", "max_keywords", language=st.session_state.ui_language),
                min_value=1,
                max_value=20,
                value=st.session_state.params.get('max_keywords', 8),
                key='max_keywords',
                help=get_text("help_texts", "max_keywords_help", language=st.session_state.ui_language)
            )
        
        with col2:
            st.number_input(
                get_text("labels", "max_themes", language=st.session_state.ui_language),
                min_value=1,
                max_value=10,
                value=st.session_state.params.get('max_themes', 3),
                key='max_themes',
                help=get_text("help_texts", "max_themes_help", language=st.session_state.ui_language)
            )
        
        with col3:
            st.text_input(
                get_text("labels", "focus", language=st.session_state.ui_language),
                value=st.session_state.params.get('focus', "general topics"),
                key='focus',
                help=get_text("help_texts", "focus_on_help", language=st.session_state.ui_language)
            )
        
        # Analysis button
        if st.button(get_text("buttons", "analyze", language=st.session_state.ui_language), type="primary", use_container_width=True):
            asyncio.run(analyze_texts(st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].tolist()))
    
    # Results section
    if st.session_state.analysis_results:
        st.subheader(get_text("titles", "results", language=st.session_state.ui_language), divider='grey')
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs([
            get_text("tabs", "keywords", language=st.session_state.ui_language),
            get_text("tabs", "themes", language=st.session_state.ui_language),
            get_text("tabs", "categories", language=st.session_state.ui_language)
        ])
        
        with tab1:
            if 'keywords' in st.session_state.analysis_results:
                # Create a list to store rows for the keywords table
                keywords_data = []
                for idx, result in enumerate(st.session_state.analysis_results['keywords']):
                    text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                    if result and result.success:
                        # Sort keywords by score and take only the top N based on UI setting
                        sorted_keywords = sorted(result.keywords, key=lambda x: x.score, reverse=True)
                        top_keywords = sorted_keywords[:st.session_state.max_keywords]
                        keywords_str = ", ".join([
                            f"{kw.keyword} ({kw.score:.2f})"
                            for kw in top_keywords
                        ])
                    else:
                        keywords_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                    keywords_data.append({
                        get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                        get_text("table_headers", "keywords", language=st.session_state.ui_language): keywords_str
                    })
                
                # Display keywords table
                if keywords_data:
                    st.dataframe(
                        pd.DataFrame(keywords_data),
                        use_container_width=True,
                        hide_index=True
                    )
        
        with tab2:
            if 'themes' in st.session_state.analysis_results:
                # Create a list to store rows for the themes table
                themes_data = []
                for idx, result in enumerate(st.session_state.analysis_results['themes']):
                    text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                    if result and result.success:
                        themes_str = ", ".join([f"{theme.name} ({theme.confidence:.2f})" for theme in result.themes])
                        if result.theme_hierarchy:
                            themes_str += f"\n\n{get_text('result_labels', 'hierarchy', language=st.session_state.ui_language)}:\n" + "\n".join([
                                f"- {main_theme}: {', '.join(sub_themes)}"
                                for main_theme, sub_themes in result.theme_hierarchy.items()
                            ])
                    else:
                        themes_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                    themes_data.append({
                        get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                        get_text("table_headers", "themes", language=st.session_state.ui_language): themes_str
                    })
                
                # Display themes table
                if themes_data:
                    st.dataframe(
                        pd.DataFrame(themes_data),
                        use_container_width=True,
                        hide_index=True
                    )
        
        with tab3:
            if 'categories' in st.session_state.analysis_results:
                # Create a list to store rows for the categories table
                categories_data = []
                for idx, result in enumerate(st.session_state.analysis_results['categories']):
                    text = st.session_state.texts_df[st.session_state.params['column_name_to_analyze']].iloc[idx]
                    if result and result.success:
                        categories_str = ""
                        for cat in result.matches:
                            categories_str += f"{cat.name} ({cat.confidence:.2f})\n"
                            if cat.evidence:
                                categories_str += f"{get_text('result_labels', 'evidence', language=st.session_state.ui_language)}:\n" + "\n".join([
                                    f"- {evidence.text} (relevance: {evidence.relevance:.2f})"
                                    for evidence in cat.evidence
                                ]) + "\n"
                            if hasattr(cat, 'themes') and cat.themes:
                                categories_str += f"{get_text('result_labels', 'related_themes', language=st.session_state.ui_language)}: {', '.join(cat.themes)}\n"
                            categories_str += "\n"
                    else:
                        categories_str = get_text("result_labels", "analysis_failed", language=st.session_state.ui_language)
                    categories_data.append({
                        get_text("table_headers", "text_content", language=st.session_state.ui_language): text,
                        get_text("table_headers", "categories", language=st.session_state.ui_language): categories_str
                    })
                
                # Display categories table
                if categories_data:
                    st.dataframe(
                        pd.DataFrame(categories_data),
                        use_container_width=True,
                        hide_index=True
                    )

# Update cleanup function to clean up temp directory
def cleanup_temp_files():
    """Clean up temporary files when the app stops."""
    if 'file_utils' in st.session_state:
        temp_dir = st.session_state.file_utils.get_data_path("raw") / "temp"
        if temp_dir.exists():
            # Clean up all files in temp directory
            for file in temp_dir.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    st.warning(f"Failed to clean up temporary file {file}: {e}")
            # Try to remove the temp directory
            try:
                temp_dir.rmdir()
            except Exception:
                pass  # Ignore if directory can't be removed

# Register cleanup function
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    main() 
"""
Keyword Extractor Module

This module provides functionality for extracting keywords, themes, and categories from text
using a combination of TF-IDF (Term Frequency-Inverse Document Frequency) and LLM (Large Language Model)
based approaches. It also includes language detection capabilities.

Key Features:
- Keyword extraction using TF-IDF and LLM
- Theme identification
- Text categorization
- Language detection
- Dynamic keyword limit based on text length
- Customizable stop words

Main Components:
- ExtractorOutput: A Pydantic model for structured output
- LLMFactory: A factory class for creating Language Model instances
- Extractor: The main class for performing text extraction
- get_extractor: A function to create and configure an Extractor instance
- apply_extraction: A helper function for applying extraction to text

Usage Example:
    from keyword_extractor import get_extractor, apply_extraction

    # Initialize the extractor
    extractor = get_extractor(llm_provider="openai", model_version="gpt-4o-mini")

    # Define extraction parameters
    params = {
        "max_kws": 8,
        "max_themes": 3,
        "focus_on": "main topics and their impacts",
        "categories": {
            "technology": "AI and related technologies",
            "environment": "Climate change and environmental issues",
            "impact": "Effects on society and industry"
        }
    }

    # Apply extraction to a text
    text = "Artificial intelligence is revolutionizing various industries..."
    result = apply_extraction(text, extractor, **params)

    print(result['extracted_keywords'])
    print(result['extracted_themes'])
    print(result['extracted_categories'])
    print(result['summary'])
    print(result['detected_language'])

Dependencies:
- langchain_core
- langchain_openai
- pydantic
- sklearn
- nltk
- langdetect
- python-dotenv

Note:
This module requires an OpenAI API key to be set in the environment variables or in a .env file.
Ensure that you have the necessary API credentials before using this module.

Author: Topi Järvinen
Version: 0.2.1
Date: 20.10.2024
"""

'''
TODO: 
- Refine the prompt: keywords should be the base version of the word (esihenkilöksi -> esihenkilö)
- Check how excluded keywords are handled (not used?)
'''

# src/keyword_extractor.py

import os
import sys
from pathlib import Path

# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from file_utils import FileUtils  # Now we can import from src directly
import pandas as pd
import json
import re
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from langdetect import detect

# Initialize FileUtils

nltk.download("punkt", quiet=True)
load_dotenv()

file_utils = FileUtils()
logger = file_utils.get_logger(__name__)

class ExtractorOutput(BaseModel):
    """
    A Pydantic model for structured output from the Extractor.

    Attributes:
        keywords (List[str]): The most relevant keywords and phrases from the text.
        themes (List[str]): The main themes of the text.
        categories (List[Dict[str, float]]): The most appropriate categories for the text with their importance scores.
        summary (str): A one-sentence summary of the key topic.
        language (str): The detected language of the text.
    """

    keywords: Optional[List[str]] = Field(
        default=None, description="The most relevant keywords and phrases from the text"
    )
    themes: Optional[List[str]] = Field(
        default=None, description="The main themes of the text"
    )
    categories: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="The most appropriate categories for the text with their importance scores",
    )
    summary: Optional[str] = Field(
        default=None, description="A one-sentence summary of the key topic"
    )
    language: Optional[str] = Field(
        default=None, description="The detected language of the text"
    )


class LLMFactory:
    """
    A factory class for creating Language Model instances.
    """

    @staticmethod
    def create(llm_provider: str, model_version: str, **kwargs) -> BaseLanguageModel:
        """
        Create a Language Model instance based on the provider and version.

        Args:
            llm_provider (str): The provider of the language model (e.g., "openai").
            model_version (str): The specific model version to use.
            **kwargs: Additional keyword arguments for the model initialization.

        Returns:
            BaseLanguageModel: An instance of the specified language model.

        Raises:
            ValueError: If an unsupported LLM provider is specified.
        """
        if llm_provider == "openai":
            return ChatOpenAI(model=model_version, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")


class Extractor:
    """
    The main class for performing text extraction, combining TF-IDF and LLM-based approaches.

    Attributes:
        llm (BaseLanguageModel): The language model used for extraction.
        tfidf_weight (float): The weight given to TF-IDF results in the final extraction.
        custom_stop_words (List[str]): Custom stop words to be used in addition to the default ones.
        tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for keyword extraction.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tfidf_weight: float = 0.5,
        custom_stop_words: List[str] = None,
    ):
        """
        Initialize the Extractor.

        Args:
            llm (BaseLanguageModel): The language model to use for extraction.
            tfidf_weight (float, optional): The weight to give TF-IDF results. Defaults to 0.5.
            custom_stop_words (List[str], optional): Custom stop words to use. Defaults to None.
        """
        self.llm = llm
        self.tfidf_weight = tfidf_weight
        self.custom_stop_words = custom_stop_words or []
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), stop_words="english"
        )
        if self.custom_stop_words:
            self.tfidf_vectorizer.stop_words_.update(self.custom_stop_words)

    def create_extraction_chain(self, prompt_template: str, output_key: str):
        """
        Create an extraction chain using the provided prompt template.

        Args:
            prompt_template (str): The template for the extraction prompt.
            output_key (str): The key for the output in the chain.

        Returns:
            Callable: A function that can be invoked with kwargs to run the extraction chain.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        return lambda **kwargs: chain.invoke(kwargs)

    def extract_keywords_tfidf(self, content: str, max_keywords: int) -> List[str]:
        """
        Extract keywords from the content using TF-IDF.

        Args:
            content (str): The text content to extract keywords from.
            max_keywords (int): The maximum number of keywords to extract.

        Returns:
            List[str]: The extracted keywords.
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        keyword_scores = [
            (word, score)
            for word, score in zip(feature_names, tfidf_scores)
            if score > 0
        ]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in keyword_scores[:max_keywords]]

    def detect_language(self, content: str) -> str:
        """
        Detect the language of the given content.

        Args:
            content (str): The text content to detect the language for.

        Returns:
            str: The detected language code or "unknown" if detection fails.
        """
        try:
            return detect(content)
        except:
            return "unknown"

    def calculate_dynamic_keyword_limit(
        self, content: str, base_limit: int = 8, max_limit: int = 15
    ) -> int:
        """
        Calculate a dynamic keyword limit based on the content length.

        Args:
            content (str): The text content to calculate the limit for.
            base_limit (int, optional): The base number of keywords. Defaults to 8.
            max_limit (int, optional): The maximum number of keywords. Defaults to 15.

        Returns:
            int: The calculated keyword limit.
        """
        word_count = len(content.split())
        return min(max(base_limit, word_count // 50), max_limit)
    
    def extract_all(self, content: str, **kwargs) -> Dict[str, Any]:
        detected_language = self.detect_language(content)
        dynamic_keyword_limit = self.calculate_dynamic_keyword_limit(
            content, base_limit=kwargs.get("max_kws", 8)
        )

        """
        Perform full extraction on the given content.

        Args:
            content (str): The text content to analyze.
            **kwargs: Additional keyword arguments for extraction parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted information.
        """

        prompt_template = """
        You are a keyword and theme extraction expert. Given the following text and context, extract the main keywords and themes, categorize the text, and provide a summary.
        
        Context: {additional_context}
        
        Text: {content}
        
        Instructions:
        1. Consider the following TF-IDF extracted keywords and phrases: {tfidf_keywords}
        2. Extract up to {max_kws} keywords and phrases. Focus on {focus_on}. Exclude keywords of type {excluded_keyword_types} and the following specific keywords: {excluded_keywords}.
        3. Extract up to {max_themes} main themes.
        4. Categorize the text into up to 3 of the following categories, ordered by importance (most important first): {categories}
           Provide the categories as a list of dictionaries, where each dictionary contains a single key-value pair of the category name and its importance score (0-1).
        5. Provide a one-sentence summary of the key topic of the text.
        6. The detected language of the text is: {detected_language}. Please ensure your response is appropriate for this language.
        
        Use the language of the input text for keywords, themes, and summary.
        If predefined keywords are provided, prioritize them: {predefined_keywords}
        
        Note: The TF-IDF keywords have a weight of {tfidf_weight} in the final selection.
        
        Provide the response as a JSON object with "keywords", "themes", "categories", "summary", and "language" keys.
        """

        tfidf_keywords = self.extract_keywords_tfidf(content, max_keywords=dynamic_keyword_limit)
        chain = self.create_extraction_chain(prompt_template, output_key=None)

        try:
            logger.debug(f"Extracting from content: {content[:100]}...")
            result = chain(
                content=content,
                tfidf_keywords=", ".join(tfidf_keywords),
                tfidf_weight=self.tfidf_weight,
                max_kws=dynamic_keyword_limit,
                max_themes=kwargs.get("max_themes", 3),
                focus_on=kwargs.get("focus_on", "general topics"),
                excluded_keyword_types=kwargs.get("excluded_keyword_types", "none"),
                detected_language=detected_language,
                predefined_keywords=", ".join(kwargs.get("predefined_keywords", [])) or "none provided",
                excluded_keywords=", ".join(kwargs.get("excluded_keywords", [])) or "none",
                categories=", ".join(f"{k}: {v}" for k, v in kwargs.get("categories", {}).items()),
                additional_context=kwargs.get("additional_context", "General conversation"),
            )

            cleaned_output = re.sub(r"^```json\s*|\s*```$", "", result.content.strip())
            parsed_result = json.loads(cleaned_output)
            logger.debug(f"Cleaned JSON output: {cleaned_output}")
            logger.debug(f"Parsed result: {parsed_result}")

            categories = parsed_result.get("categories", [])
            filtered_categories = sorted(
                [cat for cat in categories if list(cat.values())[0] > 0],
                key=lambda x: list(x.values())[0],
                reverse=True,
            )
            top_categories = filtered_categories[:3]
            categories_str = ", ".join(
                [f"{list(cat.keys())[0]}({list(cat.values())[0]:.2f})" for cat in top_categories]
            )

            return {
                "extracted_keywords": ", ".join(parsed_result.get("keywords", [])),
                "extracted_themes": ", ".join(parsed_result.get("themes", [])),
                "extracted_categories": categories_str,
                "summary": parsed_result.get("summary", ""),
                "detected_language": parsed_result.get("language", detected_language),
                "error": "",
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            logger.error(f"Raw output: {result.content}")
            return {
                "extracted_keywords": "",
                "extracted_themes": "",
                "extracted_categories": "",
                "summary": "",
                "detected_language": detected_language,
                "error": f"JSON parsing error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            return {
                "extracted_keywords": "",
                "extracted_themes": "",
                "extracted_categories": "",
                "summary": "",
                "detected_language": detected_language,
                "error": str(e),
            }

def get_extractor(
    llm_provider: str = "openai",
    model_version: str = "gpt-4o-mini",
    tfidf_weight: float = 0.5,
    custom_stop_words: List[str] = None,
    **kwargs,
) -> Extractor:
    """
    Create and configure an Extractor instance.

    Args:
        llm_provider (str, optional): The provider of the language model. Defaults to "openai".
        model_version (str, optional): The specific model version to use. Defaults to "gpt-4o-mini".
        tfidf_weight (float, optional): The weight to give TF-IDF results. Defaults to 0.5.
        custom_stop_words (List[str], optional): Custom stop words to use. Defaults to None.
        **kwargs: Additional keyword arguments for the language model initialization.

    Returns:
        Extractor: An initialized Extractor instance.
    """
    llm = LLMFactory.create(llm_provider, model_version, **kwargs)
    return Extractor(
        llm, tfidf_weight=tfidf_weight, custom_stop_words=custom_stop_words
    )


def apply_extraction(
    text: str, extractor: Extractor, verbose: bool = False, **kwargs
) -> Dict[str, str]:
    """
    Apply extraction to a given text using the provided Extractor.

    This function is designed to be used with pandas DataFrame's apply method.

    Args:
        text (str): The text to extract information from.
        extractor (Extractor): The Extractor instance to use.
        **kwargs: Additional keyword arguments for the extraction process.

    Returns:
        Dict[str, str]: A dictionary containing the extracted information.
    """
    if verbose:
        print(
            f"Applying extraction to text: {text[:100]}..."
        )  # Print first 100 chars of text
    result = extractor.extract_all(content=text, verbose=False, **kwargs)
    if verbose:
        print(f"Extraction result: {result}")  # Print the result
    return result


def main():
    print("Testing Keyword Extractor")

    # Initialize FileUtils
    file_utils = FileUtils()

    # Load parameters from the new location
    params_path = (
        file_utils.get_data_path("interim") / "configurations" / "extraction_params.csv"
    )
    params_df = file_utils.load_single_file(params_path)
    params = dict(zip(params_df["parameter"], params_df["value"]))

    # Convert string representations back to appropriate types
    params["max_kws"] = int(params["max_kws"])
    params["max_themes"] = int(params["max_themes"])
    params["predefined_keywords"] = json.loads(
        params["predefined_keywords"].replace("'", '"')
    )
    params["categories"] = json.loads(params["categories"].replace("'", '"'))

    # Sample text for testing (you might want to load this from a file in data/raw/)
    sample_text_path = file_utils.get_data_path("raw") / "sample_text.txt"

    try:
        with open(sample_text_path, "r", encoding="utf-8") as file:
            sample_text = file.read()
    except FileNotFoundError:
        print(
            f"Sample text file not found at {sample_text_path}. Using default sample text."
        )
        sample_text = """
        Artificial intelligence (AI) is revolutionizing various industries, from healthcare to finance. 
        Machine learning algorithms, a subset of AI, are being used to analyze large datasets and make 
        predictions. Natural language processing, another AI technology, is improving communication 
        between humans and machines. As AI continues to advance, it raises important ethical considerations 
        regarding privacy, bias, and job displacement.
        """

    # Initialize extractor
    extractor = get_extractor(llm_provider="openai", model_version="gpt-4o-mini")

    # Perform extraction
    result = extractor.extract_all(content=sample_text, verbose=False, **params)

    # Print results
    print("\nExtraction Results:")
    print(f"Keywords: {result['extracted_keywords']}")
    print(f"Themes: {result['extracted_themes']}")
    print(f"Categories: {result['extracted_categories']}")
    print(f"Summary: {result['summary']}")

    if result["error"]:
        print(f"Error: {result['error']}")

    # Save results to data/processed/
    output_df = pd.DataFrame([result])
    file_utils.save_data_to_disk(
        {"extraction_results": output_df}, output_type="processed"
    )

    print(f"Results saved to {file_utils.get_data_path('processed')}")


if __name__ == "__main__":
    main()

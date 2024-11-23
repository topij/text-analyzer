# tests/test_components/test_category_analyzer.py

import pytest
from src.analyzers.category_analyzer import CategoryAnalyzer
from src.schemas import CategoryOutput, CategoryMatch

from src.core.language_processing import create_text_processor
from src.core.llm.factory import create_llm


class TestCategoryAnalyzer:
    @pytest.fixture
    def analyzer(self, file_utils):
        # Load test categories from parameter file
        return CategoryAnalyzer(
            categories={
                "technical": {"keywords": ["machine learning", "data"]},
                "business": {"keywords": ["revenue", "growth"]},
            },
            llm=create_llm(model="gpt-4o-mini"),
            config={"min_confidence": 0.3},
            language_processor=create_text_processor(language="en"),
        )

    @pytest.mark.asyncio
    async def test_category_matching(self, analyzer, test_data):
        """Test category matching and evidence gathering."""
        result = await analyzer.analyze(test_data["en_technical"])

        assert isinstance(result, CategoryOutput)
        assert result.success
        assert len(result.categories) > 0

        for cat in result.categories:
            assert isinstance(cat, CategoryMatch)
            assert cat.name
            assert 0 <= cat.confidence <= 1.0
            assert len(cat.evidence) > 0

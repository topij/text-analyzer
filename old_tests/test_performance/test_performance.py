# tests/test_performance/test_performance.py

import pytest
import time
from src.semantic_analyzer import SemanticAnalyzer


class TestPerformance:
    @pytest.fixture
    def analyzer(self, file_utils):
        return SemanticAnalyzer(parameter_file="parameters_en.xlsx")

    @pytest.mark.asyncio
    async def test_processing_time(self, analyzer, test_data):
        """Test processing time benchmarks."""
        start_time = time.time()
        result = await analyzer.analyze(test_data["en_technical"])
        processing_time = time.time() - start_time

        assert processing_time < 10.0  # Maximum allowed processing time
        assert (
            result.processing_time < processing_time
        )  # Internal time should be less

    @pytest.mark.asyncio
    async def test_batch_performance(self, analyzer, test_data):
        """Test batch processing performance."""
        texts = [test_data["en_technical"]] * 3
        start_time = time.time()
        results = await analyzer.analyze_batch(texts, batch_size=2)
        total_time = time.time() - start_time

        assert len(results) == 3
        assert total_time < 30.0  # Maximum allowed time for batch
        assert all(r.success for r in results)

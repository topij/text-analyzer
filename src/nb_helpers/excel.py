# src/nb_helpers/excel.py
from typing import Any, Dict, Optional

import pandas as pd

from src.nb_helpers.testers import CategoryTester, KeywordTester, ThemeTester
from src.nb_helpers.visualizers import create_analysis_summary
from src.utils.FileUtils.file_utils import FileUtils


class ExcelAnalyzer:
    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        parameter_file: Optional[str] = None,
    ):
        self.file_utils = file_utils or FileUtils()
        self.keyword_tester = KeywordTester()
        self.theme_tester = ThemeTester()
        self.category_tester = CategoryTester(parameter_file=parameter_file)

    async def analyze_excel(
        self, input_file: str, output_file: str, content_column: str = "content"
    ) -> None:
        df = self.file_utils.load_single_file(input_file)
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found")

        results = []
        total = len(df)

        for idx, row in df.iterrows():
            text = str(row[content_column])
            print(f"\nProcessing text {idx + 1}/{total}")

            try:
                analysis = await self._analyze_text(text)
                results.append(create_analysis_summary(analysis).iloc[0])
                print("✓ Analysis complete")
            except Exception as e:
                print(f"✗ Analysis failed: {e}")
                results.append(
                    pd.Series({"keywords": "", "categories": "", "themes": ""})
                )

        output_df = pd.DataFrame(results)
        output_df.insert(0, content_column, df[content_column])

        self.file_utils.save_data_to_disk(
            data={"Analysis Results": output_df},
            output_filetype="xlsx",
            file_name=output_file,
        )

        print(f"\nResults saved to: {output_file}")

    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        return {
            "keywords": await self.keyword_tester.analyze(text),
            "themes": await self.theme_tester.analyze(text),
            "categories": await self.category_tester.analyze(text),
        }


async def analyze_excel_content(
    input_file: str,
    output_file: str,
    content_column: str = "content",
    parameter_file: Optional[str] = None,
) -> None:
    analyzer = ExcelAnalyzer(parameter_file=parameter_file)
    await analyzer.analyze_excel(input_file, output_file, content_column)

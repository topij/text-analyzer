# Text Analyzer UI Guide

## Overview
Text Analyzer is a powerful tool for semantic text analysis that extracts keywords, identifies themes, and categorizes content using advanced natural language processing. The UI provides an intuitive interface for analyzing text content in both Finnish and English.

## Visual Guide
![Text Analyzer UI](../assets/images/ui_guide_optimized.png)

The Text Analyzer interface consists of several key sections:

### Left Sidebar
1. **Help & Documentation**
   - Quick access to tool documentation
   - File requirements and usage guidelines
   - Expandable help sections for detailed information

2. **Sample Files**
   - Download sample content files
   - Download sample parameter files
   - Examples to help you get started

3. **Language Settings**
   - Switch between Finnish and English interface
   - Language selector dropdown menu

### Main Content Area
1. **File Upload Section**
   - Text file upload (required)
     - Drag and drop interface
     - 20MB file size limit
     - XLSX format support
   - Parameter file upload (optional)
     - Customize analysis settings
     - XLSX format support

2. **Parameter Settings**
   - Content Column selector
   - Max Keywords slider (1-20)
   - Max Themes slider (1-10)
   - Analysis Focus input
   - Reset Parameters button

3. **Analysis Button**
   - Primary action button
   - Initiates text analysis
   - Shows progress during analysis

4. **Results Section**
   - Three tabs for different result types:
     - Keywords
     - Themes
     - Categories
   - Export Results button at bottom

### Data Preview
- Expandable sections showing:
  - Uploaded content details
  - Parameter settings
  - Column information
  - Data samples

## Features
- Upload and analyze text content from Excel or CSV files
- Configure analysis parameters
- View detailed analysis results including keywords, themes, and categories
- Export results to Excel format
- Multilingual support (Finnish and English)

## Getting Started

### 1. File Upload
- **Text Content**: Upload your text data in XLSX format
  - File must contain a column with the text content to analyze
  - Maximum file size: 20MB
- **Parameters** (optional): Upload parameter file in XLSX format to customize analysis settings

### 2. Parameter Settings
Configure analysis parameters:
- **Content Column**: Select which column contains the text to analyze
- **Max Keywords**: Set maximum number of keywords to extract (1-20)
- **Max Themes**: Set maximum number of themes to identify (1-10)
- **Analysis Focus**: Specify what aspects of the text to focus on

### 3. Analysis
Click the "Analyze" button to start the analysis process. The tool will:
- Process each text entry
- Show progress with a progress bar
- Display success/error messages for each analysis

### 4. Results
Results are displayed in three tabs:

#### Keywords Tab
- Shows extracted keywords for each text
- Includes confidence scores for each keyword
- Keywords are sorted by relevance

#### Themes Tab
- Displays identified themes
- Shows theme hierarchy and relationships
- Includes confidence scores for each theme

#### Categories Tab
- Shows matched categories
- Includes confidence scores and supporting evidence
- Displays related themes for each category

### 5. Exporting Results
- Click "Export Results" to save analysis results
- Results are saved in Excel format
- Exported file includes all analyzed texts and their results
- File is automatically named with timestamp

## Language Settings
- Switch between Finnish and English interface using the language selector
- Interface language can be changed at any time
- Content language is automatically detected

## Sample Files
The UI includes sample files to help you get started:
- Sample content file demonstrates the required format
- Sample parameter file shows available customization options

## Tips
- Preview uploaded data before analysis to verify content
- Use parameter settings to fine-tune analysis results
- Monitor analysis progress in real-time
- Check error messages if analysis fails
- Export results for further processing or sharing 

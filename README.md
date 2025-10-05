
# LayoutLMDatasetPrep

This repository provides a complete pipeline for preparing datasets for LayoutLM-based models. The process involves extracting images and text from PDFs, performing header tagging, identifying content blocks and tables, tokenizing, chunking, and generating output files for downstream tasks.

## Pipeline Overview

1. **Extract Images from PDF**
	- PDF files are processed to extract page images.

2. **Extract Text using Doctr**
	- The [Doctr](https://mindee.github.io/doctr/) library is used for Optical Character Recognition (OCR) to extract text from the page images.
	- Extracted text is used for further processing.

3. **Header Tagging using Rules**
	- Custom rule-based logic is applied to the extracted text to identify and tag headers.

4. **Identifying Tables using imag2table**
	- The [imag2table](https://github.com/naiveHobo/imag2table) library is used to detect and extract tables from the page images.

6. **Tokenization**
	- The text content is tokenized using a LayoutLM-compatible tokenizer.
	- Tokenization includes mapping tokens to their corresponding bounding boxes on the page.

7. **Chunking**
	- Tokenized data is chunked into manageable segments for model training or inference.

8. **Output Generation**
	- Processed data is saved as CSV files in the `output/` directory.
	- Each PDF has a corresponding output CSV file (e.g., `output/test_pdf1_output.csv`).


## Test Notebook

A test notebook is provided at [`notebooks/layoutlm_ds_prep.ipynb`](https://github.com/nimrisha26/LayoutLMDatasetPrep/blob/main/notebooks/layoutlm_ds_prep.ipynb) to demonstrate and validate the pipeline.


## Output Locations

- **Outputs:** `output/<pdf_name>_output.csv`
- **PDFs:** `pdf/<pdf_name>.pdf`
- **highlight_images:** `highlight_images/`

**GitHub Output Locations Link:**
[View Output Folder](https://github.com/nimrisha26/LayoutLMDatasetPrep/tree/main/output)
[View Highlighted Images](https://github.com/nimrisha26/LayoutLMDatasetPrep/tree/main/highlight_images)


## Setup Instructions

1. **Create a Python virtual environment:**
	 ```sh
	 python -m venv venv
	 ```
	 Activate the virtual environment:
	 - On Windows (PowerShell):
		 ```sh
		 .\venv\Scripts\Activate.ps1
		 ```
	 - On Windows (Command Prompt):
		 ```sh
		 .\venv\Scripts\activate.bat
		 ```
	 - On macOS/Linux:
		 ```sh
		 source venv/bin/activate
		 ```

2. **Install dependencies:**
	 ```sh
	 pip install -r requirements.txt
	 ```


## Streamlit App

An interactive Streamlit app is provided for easy PDF upload, processing, and visualization of results.

### How to Run the Streamlit App

1. Make sure your virtual environment is activated and dependencies are installed (see Setup Instructions above).
2. Run the following command from the project root:
	```sh
	streamlit run src/app/main.py
	```
3. The app will open in your browser. You can:
	- Upload a PDF file
	- Run OCR and dataset preparation
	- View extracted data in a table
	- Download the output CSV
	- View highlighted images for each page using a dropdown selector


### Demo Video

A demo video showing the app workflow (uploading a PDF, displaying the output, and viewing highlighted images) is available in the `videos/` folder:

- `videos/Streamlit Layout LM DS Prep Video.mp4`

**GitHub Video Link:**
<!-- Replace the URL below with your actual GitHub video link -->
[Watch Demo Video on GitHub](https://github.com/nimrisha26/LayoutLMDatasetPrep/blob/main/videos/Streamlit%20Layout%20LM%20DS%20Prep%20Video.mp4)

---


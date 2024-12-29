# Invoice Data Extractor 📄👨‍🔧

This project is a Streamlit-based application for extracting data from invoices in various file formats, such as images (PNG, JPG), PDFs, and DOCX files. The extracted data is displayed in a tabular format, saved into a pandas DataFrame with standardized formatting, and exportable in multiple formats.

## Features 🛠️

- **File Format Support**: Supports PNG, JPG, PDF, and DOCX files.
- **Data Extraction**: Extracts key information, such as:
  - Invoice Number
  - Invoice Date/Time
  - Invoice Amount
  - Total Amount
  - GST
  - Subtotal
  - Invoiced From
  - Invoiced To
- **Standardized Output**:
  - Dates in `YYYY-MM-DD` format.
  - Currency fields prefixed with `INR`.
  - Missing or invalid fields replaced with `None`.
- **Export Options**:
  - Export data as CSV, Excel, or JSON.
- **Error Handling**: Handles invalid inputs and unsupported file types gracefully.

## Technologies Used 🤖

- **Streamlit**: For building the user interface.
- **Google Gemini AI**: For document understanding and data extraction.
- **Python Libraries**:
  - `pandas`: Data manipulation and storage.
  - `Pillow`: Image handling.
  - `fitz` (PyMuPDF): PDF processing.
  - `re`: Regular expressions for data parsing.

## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/invoice-extractor.git
   cd invoice-extractor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Google Gemini API:
   - Obtain your API key from Google Gemini AI.
   - Set the API key in the project:
     ```python
     import google.generativeai as genai
     genai.configure(api_key="your_actual_api_key")
     ```

## Usage 🔄

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Upload a file (PNG, JPG, PDF, or DOCX).
3. View extracted data in tabular format.
4. Export the data to your preferred format (CSV, Excel, JSON).

## File Processing Workflow ⏳

1. **Input Handling**:
   - Uploaded files are identified by type.
   - Images are loaded with `Pillow`.
   - PDFs are converted to images using `fitz`.
2. **Data Extraction**:
   - Google Gemini AI processes the content to extract relevant fields.
   - Extracted data is parsed and formatted into a pandas DataFrame.
3. **Standardization**:
   - Dates are converted to a uniform format (`YYYY-MM-DD`).
   - Numeric fields are prefixed with `INR` and rounded to two decimal places.
   - Missing data is replaced with `None`.
4. **Export Options**:
   - Data can be exported as CSV, Excel, or JSON.

## Example Code 🔧

### Data Parsing and Standardization
```python
import pandas as pd
import re

def standardize_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d.%m.%Y", errors="coerce")
    except Exception:
        return None

def standardize_inr(value):
    try:
        return f"INR {float(value):,.2f}" if value else None
    except ValueError:
        return None

# Apply standardization to DataFrame
currency_fields = ["invoiced_amount", "total", "gst", "subtotal"]
for field in currency_fields:
    df[field] = df[field].apply(standardize_inr)

df["invoiced_date_time"] = df["invoiced_date_time"].apply(standardize_date)
```

## Example Output 📊

| Invoice Number | Invoice Date/Time | Invoice Amount | Total      | GST     | Subtotal  |
|----------------|-------------------|----------------|------------|---------|-----------|
| IN-MAA4-3685  | 2020-08-25        | INR 2,450.00   | INR 2,450.00 | INR 373.72 | INR 2,076.28 |
| IN-MAA4-3686  | NaT               | INR 3,737.50   | INR 3,737.50 | INR 567.82 | INR 3,170.18 |
| None           | NaT               | None           | None       | None    | None      |

## License 🔒

This project is licensed under the MIT License. See the LICENSE file for details.


## Contact 📧

For any queries, feel free to contact [ramusiva77@gmail.com](mailto:ramusiva77@gmail.com).


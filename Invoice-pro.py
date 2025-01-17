import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import os, io, json, re
import fitz
from dotenv import load_dotenv
load_dotenv()
# api_key = os.getenv('Google_API_KEY')
api_key = st.secrets('Google_API_KEY')

# Initialize Gemini AI client
genai.configure(api_key=api_key)

#Function to load Gemini Pro
model = genai.GenerativeModel('models/gemini-1.5-flash')

def extract_invoice_data(file, file_extension):
    # Use Gemini AI's document understanding capabilities to extract data
    # response = model.generate_content([
    #     prompt=f"Extract the following information from the document: invoice_number, invoiced_date_time, invoiced_amount, total, gst, subtotal, invoiced_from, invoiced_to",
    #     image=file]
    # )
    file_type = file_extension
    try:
        if file_type in ["png", "jpg", "jpeg"]:
            # Ensure the file is a PIL Image
            image = Image.open(file)
            input_data = {"image": image}
        elif file_type == "pdf":
            # Convert PDF to image using PyMuPDF
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            page = pdf[0]  # Process only the first page for simplicity
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            input_data = {"image": image}
        else:
            return "Unsupported file type!"

        # Generate content using Gemini AI
        response = model.generate_content(
            [ "Extract the following information from the document: invoice_number, invoiced_date_time, invoiced_amount, total_amount, total_amont_in_words, gst, subtotal, invoiced_from, invoiced_to, item_description, item_quantity, item_hsncode, item_price, item_total, item_2_description, item_2_quantity, item_2_hsncode, item_2_price, item_2_total",
            image]
        )

        # Debugging: Print raw response
        print("Raw response:", response)
        # response = json.dumps(response)
        # Extract and return results
        #extracted_data = response.get("result", "No data extracted")
        # return extracted_data

    except Exception as e:
        response = None
        print("Error occurred:", e)

    # Parse the extracted data into a dictionary
    extracted_data = {}
    if response:
        # Extract the text from the response
        # text = response.result["candidates"][0]["content"]["parts"][0]["text"]
        text = response.candidates[0].content.parts[0].text

        # Define a regex pattern to extract key-value pairs
        pattern = r"\*\*([\w_]+):\*\*\s+(.*)"
        matches = re.findall(pattern, text)

        # Convert matches into a dictionary
        extracted_data = {key.strip().lower(): value.strip() for key, value in matches}
        # for line in response.text.split('\n'):
        #     key, value = line.split(':')
        #     data[key.strip()] = value.strip()

    return extracted_data

def main():
    st.title("Multi Language Invoice Extractor using LLM")

    # File uploader
    uploaded_files = st.file_uploader("Upload Invoice Files", accept_multiple_files=True)

    if uploaded_files:
        data_list = []
        for file in uploaded_files:
            file_extension = file.name.split('.')[-1] # Extract file extension
            data = extract_invoice_data(file, file_extension)
            data_list.append(data)

        # Create a Pandas DataFrame
        df = pd.DataFrame(data_list)

        # Generalize Date Format
        # def standardize_date(date_str):
        #     try:
        #         return pd.to_datetime(date_str, format="%d.%m.%Y", errors="coerce")
        #     except Exception:
        #         return None

        # df["invoiced_date_time"] = df["invoiced_date_time"].apply(standardize_date)
        try: 
            # Standardize Numeric Fields (Add INR if needed)
            def standardize_inr(value):
                try:
                    return f"INR {float(value):,.2f}" if value else None
                except ValueError:
                    return None

            currency_fields = ["invoiced_amount", "total_amount", "gst", "subtotal", "item_price", "item_2_price", "item_total", "item_2_total"]
            for field in currency_fields:
                df[field] = df[field].apply(standardize_inr)
        except Exception as e:
            print(e)
        # Replace None and Invalid with a Standard Value
        #df.fillna(value=None, inplace=True)

        # Display the DataFrame
        st.dataframe(df)

        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "Excel"])
        if st.button("Export"):
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "invoice_data.csv", mime='text/csv')
            elif export_format == "Excel":
                excel = df.to_excel("invoice_data.xlsx", index=False, engine='openpyxl')
                st.download_button("Download Excel", excel, "invoice_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    main()
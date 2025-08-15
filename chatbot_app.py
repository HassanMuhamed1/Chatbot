import streamlit as st
from chatbot_core import prepare_doc, highlight_extraction, flexible_qa, build_qa_chain
import pandas as pd
from docx import Document
import os
import signal

st.title("Chatbot Assistant")

upload_file = st.file_uploader("Upload a DOCX file", type=["docx"])
question = st.text_input("How can I help you?")

tables = []
if upload_file:
    text = prepare_doc(upload_file)
    st.expander("Show Extracted Document").write(text)
    qa_chain = None
    if len(text) > 12000:
        qa_chain = build_qa_chain(text)

    # Extract tables as DataFrames and keep in a list
    doc = Document(upload_file)
    for table in doc.tables:
        df = pd.DataFrame([[cell.text.strip() for cell in row.cells] for row in table.rows])
        tables.append(df)

    if question:
        st.subheader("Answer:")
        answer = flexible_qa(text, question, qa_chain)
        st.write(answer)

        # Try to detect if the user asked for a specific table (e.g., "table 2")
        import re
        match = re.search(r'table\s*(\d+)', question, re.IGNORECASE)
        if match:
            table_idx = int(match.group(1)) - 1
            if 0 <= table_idx < len(tables):
                st.write(f"Extracted Table {table_idx+1}:")
                st.table(tables[table_idx])
            else:
                st.info("Requested table number does not exist.")
    else:
        st.subheader(" Highlights")
        highlights = highlight_extraction(text)
        st.markdown(highlights.replace("- ", "• "), unsafe_allow_html=True)


st.button("Exit App", on_click=lambda: os._exit(0))

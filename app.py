import pytesseract
from pdf2image import convert_from_path
import tempfile
import streamlit as st
import logging
from haystack.nodes import FARMReader
from PyPDF2 import PdfReader

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

if 'my_model' not in st.session_state:
    st.session_state.my_model = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
my_model = st.session_state.my_model

st.set_page_config(
    page_title="Question Answering System",
    layout="centered",
    initial_sidebar_state="expanded"
)
hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title('Question Answering System')

uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

texts = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    pdf = PdfReader(tmp_path)
    text = ''
    if pdf.is_encrypted:
        pdf.decrypt('')
    if len(pdf.pages) > 0 and pdf.pages[0].extract_text() != '':
        for page in pdf.pages:
            text += page.extract_text()
    if text.strip() == '':
        images = convert_from_path(tmp_path)
        for image in images:
            text += pytesseract.image_to_string(image)
    texts.append(text)

st.session_state.texts = texts

num_results = st.slider("Number of results to display", value=10, min_value=5, max_value=30, step=5)

question = st.text_input('Enter the Question')
if question:
    with st.spinner('Please wait...'):
        ans = my_model.predict_on_texts(question, st.session_state.texts, num_results)
    st.subheader("Answers")
    for answer in ans["answers"]:
        with st.expander(f"**Answer**: {answer.answer} - **Score**: {answer.score:.2f}"):
            st.markdown(f"**Context:** {answer.context}")

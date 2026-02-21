import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from docx import Document
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import re 

# =====================================
# ABBREVIATION DICTIONARY
# =====================================

ABBREVIATIONS = {
    "UN": "United Nations",
    "WHO": "World Health Organization",
    "GATE": "Graduate Aptitude Test in Engineering",
    "NASA": "National Aeronautics and Space Administration",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning"
}

def expand_abbreviations(text):
    words = text.split()
    expanded_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        
        if clean_word.upper() in ABBREVIATIONS:
            full_form = ABBREVIATIONS[clean_word.upper()]
            expanded_words.append(full_form)
        else:
            expanded_words.append(word)
            
    return " ".join(expanded_words)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Neural Machine Translator",
    page_icon="🌍",
    layout="wide"
)

# =====================================
# GOOGLE STYLE CSS
# =====================================
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}

.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #1a73e8;
    text-align: center;
    margin-bottom: 20px;
}

.language-bar {
    background: white;
    padding: 15px 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.panel {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

textarea {
    border-radius: 8px !important;
    border: 1px solid #dcdfe3 !important;
    font-size: 18px !important;
}

.stButton>button {
    background-color: #1a73e8;
    color: white;
    border-radius: 6px;
    height: 42px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown('<div class="main-title">🌍 Neural Machine Translator</div>', unsafe_allow_html=True)

# =====================================
# LOAD TRANSLATION MODEL
# =====================================
@st.cache_resource
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# =====================================
# LOAD EMOTION MODEL
# =====================================
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_translation_model()
emotion_tokenizer, emotion_model = load_emotion_model()

# =====================================
# LANGUAGE MAP
# =====================================
language_map = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Oriya": "ory_Orya",
    "Assamese": "asm_Beng",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Arabic": "arb_Arab",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans"
}

quick_languages = ["English", "Hindi", "Tamil", "Telugu"]


# =====================================
# LANGUAGE BAR
# =====================================
st.markdown('<div class="language-bar">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([5,1,5])

with col1:
    source_quick = st.radio("From", quick_languages + ["More"], horizontal=True)
    source_language = st.selectbox("Select Source Language", list(language_map.keys())) if source_quick == "More" else source_quick

with col2:
    st.markdown("<div style='text-align:center;font-size:22px;'>⇄</div>", unsafe_allow_html=True)

with col3:
    target_quick = st.radio("To", quick_languages + ["More"], horizontal=True)
    target_language = st.selectbox("Select Target Language", list(language_map.keys()), index=1) if target_quick == "More" else target_quick

st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# MAIN PANELS
# =====================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    input_text = st.text_area("", height=220, placeholder="Enter text or upload document (Max 1000 words)")
    uploaded_file = st.file_uploader("Upload Document (.txt or .docx)", type=["txt", "docx"])

    if uploaded_file:
        if uploaded_file.type == "text/plain":
            input_text = uploaded_file.read().decode("utf-8")
        else:
            doc = Document(uploaded_file)
            input_text = "\n".join([p.text for p in doc.paragraphs])

    word_count = len(input_text.split())
    st.caption(f"Word Count: {word_count} / 1000")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    output_area = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

translate_button = st.button("Translate")

# =====================================
# EMOTION FUNCTION
# =====================================
def analyze_emotions(text):
    inputs = emotion_tokenizer(text, return_tensors="pt")
    outputs = emotion_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    labels = emotion_model.config.id2label

    base_scores = {labels[i]: round(probs[i].item(), 4) for i in range(len(labels))}

    extended_scores = {
        "anger": base_scores.get("anger", 0),
        "disgust": base_scores.get("disgust", 0),
        "fear": base_scores.get("fear", 0),
        "joy": base_scores.get("joy", 0),
        "sadness": base_scores.get("sadness", 0),
        "surprise": base_scores.get("surprise", 0),
        "neutral": base_scores.get("neutral", 0),
        "frustration": round(base_scores.get("anger", 0) * 0.6, 4),
        "excitement": round(base_scores.get("joy", 0) * 0.6, 4),
        "sarcasm": 0
    }

    return extended_scores

# =====================================
# TRANSLATION + EMOTION
# =====================================
if translate_button:

    if not input_text.strip():
        st.warning("Please enter text or upload a document.")
    elif word_count > 1000:
        st.error("Maximum 1000 words allowed.")
    else:
        with st.spinner("Translating..."):

            source_code = language_map[source_language]
            target_code = language_map[target_language]

            tokenizer.src_lang = source_code

            # ✅ STEP 3: Expand abbreviations BEFORE translation
            expanded_text = expand_abbreviations(input_text)
            encoded = tokenizer(expanded_text, return_tensors="pt", truncation=True, max_length=1024)
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                max_length=1200
            )

            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            output_area.success(translated_text)

        # =====================================
        # EMOTION PANEL (Below Translation)
        # =====================================
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🎭 Emotion Analysis")

        emotions = analyze_emotions(input_text)

        for emotion, score in emotions.items():
            st.write(f"**{emotion.capitalize()}** : {score}")

        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================
        # EMOTION BAR GRAPH
        # =====================================
        st.subheader("📊 Emotion Distribution Graph")

        emotion_names = list(emotions.keys())
        emotion_values = list(emotions.values())

        fig = plt.figure()
        plt.bar(emotion_names, emotion_values)
        plt.xticks(rotation=45)
        plt.ylabel("Probability Score")
        plt.xlabel("Emotions")
        plt.title("Emotion Probability Distribution")

        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)
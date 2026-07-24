import streamlit as st
import numpy as np
import pandas as pd
import re
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multimodal AI Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Multimodal Embedding Studio (PyTorch)")
st.markdown("""
A unified AI workspace for generating and analyzing deep learning embeddings. Toggle between extracting **Word Embeddings (Skip-Gram)** from text datasets or **Visual Features (ResNet18)** from images.
""")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.header("⚙️ Workspace Mode")
mode = st.sidebar.radio(
    "Select your AI pipeline:",
    ["📝 NLP: Text Embeddings", "👁️ CV: Image Embeddings"]
)
st.sidebar.markdown("---")

# ==========================================
# CORE NLP FUNCTIONS
# ==========================================
def preprocess_multilingual_text(text):
    text = text.lower()
    sentences = text.split('\n')
    tokenized_sentences = []
    for sent in sentences:
        tokens = re.findall(r'\w+', sent, re.UNICODE)
        if tokens:
            tokenized_sentences.append(tokens)
    return tokenized_sentences

def generate_skipgram_pairs(tokenized_sentences, window_size=2):
    pairs = []
    for sentence in tokenized_sentences:
        for i, center_word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((center_word, sentence[j]))
    return pairs

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        return self.linear(embeds)

# ==========================================
# CORE CV FUNCTIONS
# ==========================================
@st.cache_resource
def load_feature_extractor():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return feature_extractor.to(device), device

cv_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# 📝 NLP WORKSPACE
# ==========================================
if mode == "📝 NLP: Text Embeddings":
    st.header("Text Processing & Skip-Gram Training")
    
    uploaded_file = st.sidebar.file_uploader("Upload Text Dataset (.txt, .csv)", type=["txt", "csv"])
    window_size = st.sidebar.slider("Context Window Size", 1, 5, 2)
    embed_size = st.sidebar.select_slider("Embedding Dimensions", [16, 32, 50, 100, 200], value=50)
    epochs = st.sidebar.slider("Training Epochs", 5, 100, 25, step=5)
    batch_size = st.sidebar.selectbox("Batch Size", [64, 128, 256, 512], index=2)

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            text_data = " ".join(df.astype(str).values.flatten())
        else:
            text_data = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()

        tokenized_data = preprocess_multilingual_text(text_data)
        bigrams = generate_skipgram_pairs(tokenized_data, window_size=window_size)
        all_words = sorted(list(set([word for sent in tokenized_data for word in sent])))
        vocab_size = len(all_words)
        
        word_to_id = {word: idx for idx, word in enumerate(all_words)}
        id_to_word = {idx: word for idx, word in enumerate(all_words)}

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Vocabulary Size", f"{vocab_size:,}")
        kpi2.metric("Training Pairs", f"{len(bigrams):,}")
        kpi3.metric("Vector Size", f"{embed_size} d")
        st.markdown("---")

        if vocab_size > 1:
            # FIXED: Removed the nested duplicate button and added a unique key
            train_clicked = st.button("🚀 Train Word Embeddings", key="train_nlp_btn", use_container_width=True)

            if train_clicked:
                m1, m2 = st.columns(2)
                epoch_metric = m1.metric("Current Epoch", f"0/{epochs}")
                loss_metric = m2.metric("Current Loss", "0.0000")
                progress_bar = st.progress(0.0)
                chart_holder = st.empty()

                X = [word_to_id[pair[0]] for pair in bigrams]
                Y = [word_to_id[pair[1]] for pair in bigrams]
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = SkipGramModel(vocab_size, embed_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                losses = []
                for epoch in range(epochs):
                    total_loss = 0
                    for batch_X, batch_Y in dataloader:
                        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(batch_X), batch_Y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        
                    avg_loss = total_loss / len(dataloader)
                    losses.append(avg_loss)
                    
                    progress_bar.progress((epoch + 1) / epochs)
                    epoch_metric.metric("Current Epoch", f"{epoch + 1}/{epochs}")
                    loss_metric.metric("Current Loss", f"{avg_loss:.4f}")
                    
                    loss_df = pd.DataFrame({"Epoch": range(1, len(losses) + 1), "Loss": losses})
                    chart_holder.plotly_chart(px.line(loss_df, x="Epoch", y="Loss", title="Training Loss"), use_container_width=True)

                with torch.no_grad():
                    st.session_state['nlp_weights'] = model.embeddings.weight.cpu().numpy()
                    st.session_state['nlp_vocab'] = all_words
                    st.session_state['nlp_word_to_id'] = word_to_id
                    st.session_state['nlp_id_to_word'] = id_to_word
                st.session_state['nlp_trained'] = True

            if 'nlp_trained' in st.session_state:
                weights = st.session_state['nlp_weights']
                vocab = st.session_state['nlp_vocab']
                
                st.markdown("### 🔍 NLP Vector Analysis")
                tab1, tab2 = st.tabs(["🌐 PCA Visualizer", "🔎 Semantic Similarity"])
                
                with tab1:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(weights)
                    df_pca = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
                    df_pca["Word"] = vocab
                    st.plotly_chart(px.scatter(df_pca, x="PCA 1", y="PCA 2", text="Word"), use_container_width=True)
                    
                with tab2:
                    target = st.selectbox("Select Word:", vocab)
                    if target:
                        norm_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
                        target_vec = norm_weights[st.session_state['nlp_word_to_id'][target]]
                        sims = np.dot(norm_weights, target_vec)
                        top_idx = np.argsort(sims)[::-1][1:11]
                        
                        results = [(st.session_state['nlp_id_to_word'][i], float(sims[i])) for i in top_idx]
                        st.table(pd.DataFrame(results, columns=["Similar Word", "Cosine Score"]))
        else:
            st.error("Text file too small. Need more vocabulary.")
    else:
        st.info("👆 Upload text data to start.")

# ==========================================
# 👁️ CV WORKSPACE
# ==========================================
elif mode == "👁️ CV: Image Embeddings":
    st.header("Deep Visual Feature Extraction (ResNet18)")
    
    uploaded_files = st.sidebar.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        num_images = len(uploaded_files)
        st.metric("Images Queued", num_images)
        
        if num_images >= 3:
            if st.button("🚀 Extract Visual Embeddings", use_container_width=True):
                model, device = load_feature_extractor()
                image_names, image_vectors, images_cache = [], [], {}
                
                progress_bar = st.progress(0.0)
                status = st.empty()
                
                with torch.no_grad():
                    for i, file in enumerate(uploaded_files):
                        img = Image.open(file).convert('RGB')
                        images_cache[file.name] = img
                        image_names.append(file.name)
                        
                        img_t = cv_transforms(img).unsqueeze(0).to(device)
                        vector = model(img_t).squeeze().cpu().numpy()
                        image_vectors.append(vector)
                        
                        progress_bar.progress((i + 1) / num_images)
                        status.text(f"Processed: {file.name}")
                        
                st.session_state['cv_vectors'] = np.array(image_vectors)
                st.session_state['cv_names'] = image_names
                st.session_state['cv_cache'] = images_cache
                st.session_state['cv_trained'] = True
                
            if 'cv_trained' in st.session_state:
                vectors = st.session_state['cv_vectors']
                names = st.session_state['cv_names']
                cache = st.session_state['cv_cache']
                
                st.markdown("### 🔍 CV Vector Analysis")
                tab1, tab2 = st.tabs(["🌐 Image PCA Visualizer", "🖼️ Reverse Image Search"])
                
                with tab1:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(vectors)
                    df_pca = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
                    df_pca["Image"] = names
                    st.plotly_chart(px.scatter(df_pca, x="PCA 1", y="PCA 2", text="Image"), use_container_width=True)
                    
                with tab2:
                    target_img = st.selectbox("Select Query Image:", names)
                    if target_img:
                        st.image(cache[target_img], caption="Query Image", width=200)
                        
                        target_idx = names.index(target_img)
                        query_vec = vectors[target_idx].reshape(1, -1)
                        sims = cosine_similarity(query_vec, vectors).flatten()
                        
                        top_idx = np.argsort(sims)[::-1]
                        top_idx = [idx for idx in top_idx if idx != target_idx][:4]
                        
                        st.markdown("**Top Visual Matches:**")
                        cols = st.columns(len(top_idx))
                        for col, idx in zip(cols, top_idx):
                            with col:
                                st.image(cache[names[idx]], use_column_width=True)
                                st.caption(f"Sim: {sims[idx]:.3f}")
        else:
            st.warning("Upload at least 3 images for PCA and similarity comparisons.")
    else:
        st.info("👆 Upload a batch of images to start.")
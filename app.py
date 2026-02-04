import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tempfile
import os
import noisereduce as nr
import soundfile as sf

@st.cache_resource
def load_model():
    with open("audio_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

def extract_features(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        spec_contrast_mean,
        zcr_mean
    ])

    return features, audio, sr


st.title("🎧 Audio Noise Detection")
st.write("Upload an audio file and analyze whether it is **Noisy** or **Noiseless**")

uploaded_file = st.file_uploader(
    "Upload WAV file",
    type=["wav"]
)


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav") 

    submit = st.button("🚀 Analyze Audio")

    if submit:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        features, audio, sr = extract_features(temp_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        if prediction == 1:
            st.error(f"🔊 Noisy Audio (Confidence: {confidence:.2f})")

            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=1.0
            )

            # Save denoised audio
            denoised_path = temp_path.replace(".wav", "_denoised.wav")
            sf.write(denoised_path, reduced_noise, sr)

            st.subheader("🔇 Noise Reduced Audio")
            st.audio(denoised_path, format="audio/wav")

        else:
            st.success(f"🔇 Noiseless Audio (Confidence: {confidence:.2f})")

        st.subheader("📊 Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)

        os.remove(temp_path)

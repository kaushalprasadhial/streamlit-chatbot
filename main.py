import streamlit as st
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tempfile
import os

# Page config
st.set_page_config(page_title="Video Q&A Chatbot", page_icon="📹", layout="wide")

st.title("📹 Video Q&A Chatbot with Qwen2.5-VL")
st.markdown("Upload a video and ask questions about it!")

# Sidebar for model settings
with st.sidebar:
    st.header("Settings")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    st.info(f"Using model: `{model_id}`")
    
    if not torch.cuda.is_available():
        st.warning("⚠️ CUDA is not available. Running on CPU will be extremely slow.")

@st.cache_resource
def load_model():
    """Loads the Qwen2.5-VL model and processor."""
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
with st.spinner("Loading Qwen2.5-VL model... (this may take a while first time)"):
    model, processor = load_model()

if model is None:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Display video
    st.video(video_path)
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing video..."):
                try:
                    # Prepare messages for the model
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    "max_pixels": 360 * 420, # Limit resolution for memory efficiency
                                    "fps": 1.0, # Sample 1 frame per second
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    
                    # Prepare inputs
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)

                    # Generate output
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})

                except Exception as e:
                    st.error(f"Error generating response: {e}")

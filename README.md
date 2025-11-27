# streamlit-chatbot

I have implemented a Streamlit chatbot that uses Qwen2.5-VL-3B-Instruct to answer questions about uploaded videos.

### Prerequisites
Ensure you have the required packages installed:

~~~
pip install -r requirements.txt
~~~
### Running the Application  
Run the Streamlit app with the following command:

~~~
streamlit run main.py
~~~
### How to Use
`Upload a Video:` Use the sidebar or main area to upload a video file (MP4, AVI, MOV, MKV).
Wait for Processing: The video will be displayed.  
`Ask Questions:` Type your question in the chat input (e.g., "Describe what is happening in this video", "What color is the car?").  
`View Responses:` The model will analyze the video frames and answer your question.

### Notes
`First Run:` The first time you run the app, it will download the Qwen2.5-VL-3B model (several GBs). This may take some time.  
`GPU Recommended:` Running this model on a CPU will be very slow. A GPU with at least 8GB VRAM is recommended for reasonable performance.  
`Memory Usage:` I have limited the video resolution and frame rate in the code to manage memory usage.
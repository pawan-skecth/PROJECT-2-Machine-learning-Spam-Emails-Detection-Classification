!pip install gradio
import gradio as gr

# Define the prediction function
def predict_spam(message):
    transformed_message = preprocess_text(message)  # Preprocess the input
    vectorized_message = vectorizer.transform([transformed_message])  # Vectorize input
    result = model.predict(vectorized_message)[0]  # Predict
    return "🚫 Spam" if result == 1 else "✅ Not Spam"

# Enhanced interface with custom CSS for full-page design
with gr.Blocks() as interface:
    # Custom CSS for styling
    interface.css = """
    body {
        background: linear-gradient(to bottom right, #6a11cb, #2575fc);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    #title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-top: 30px;
    }
    #description {
        text-align: center;
        font-size: 20px;
        margin-bottom: 30px;
        color: #e0e0e0;
    }
    #input-box {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        color: black;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    #output-box {
        background-color: #ffffff;
        border: 2px solid #f44336;
        color: black;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    #submit-button {
        background-color: #ff5722;
        color: white;
        font-size: 20px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    #submit-button:hover {
        background-color: #e64a19;
    }
    footer {
        text-align: center;
        margin-top: 40px;
        font-size: 16px;
        color: #e0e0e0;
    }
    """

    gr.Markdown("<div id='title'>📱 SMS Spam Detector</div>")
    gr.Markdown(
        "<div id='description'>Enter your message below to find out if it’s spam or not.</div>"
    )

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="Your SMS Message:",
                placeholder="Type your message here...",
                lines=5,
                interactive=True,
                elem_id="input-box",
            )
        with gr.Column():
            output_box = gr.Textbox(
                label="Prediction Result:",
                interactive=False,
                elem_id="output-box",
            )

    with gr.Row():
        submit_button = gr.Button("🔍 Check Spam", elem_id="submit-button")

    submit_button.click(predict_spam, inputs=input_box, outputs=output_box)

    gr.Markdown(
        "<footer><i>Made with ❤️ using Gradio. Your SMS model at work!</i></footer>"
    )

# Launch the app
interface.launch()

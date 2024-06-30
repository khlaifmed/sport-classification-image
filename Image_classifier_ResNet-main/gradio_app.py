import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Define the predict_single_image function
def predict_single_image(image_array):
    # Preprocess the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Display the image and the predicted class
    plt.imshow(image_array[0].astype(np.uint8))
    plt.title(f"Predicted Class: {predicted_class_label}")
    plt.axis('off')
    plt.show()

    return image_array[0].astype(int), f"Predicted Class: {predicted_class_label}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_single_image,
    inputs=gr.inputs.Image(shape=(224, 224), label="Upload Image"),
    outputs=[gr.outputs.Image(type="numpy"), gr.outputs.Textbox()]
)

# Launch the interface
iface.launch()

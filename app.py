import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("classes.pkl","rb") as f:
    classes = pickle.load(f)

model = models.resnet50(weights =None )

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def predict(files):
    results = []

    for file in files:
        image = Image.open(file.name).convert("RGB")

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            prob = torch.softmax(outputs, dim =1)[0]

        confidence, pred = torch.max(prob, 0)

        results.append([file.name.split("/")[-1],
                       classes[pred.item()],
                       f"{confidence.item()*100:.2f}%"])

    return results


# Gradio UI

with gr.Blocks(title="Image Classifier") as demo:
    gr.Markdown(
        """
        Image Classification System

        Upload images to classify them into one of these categories:

        **Airplane · Candle · Christmas Tree · Jacket · Miscellaneous · Snowman**
        """
    )

    with gr.Row():
        image_input = gr.File(
            label="Upload Multiple Object Images",
            file_count="multiple",
            file_types=["image"],
            type="filepath"
        )

    predict_button = gr.Button("🔍 Detect object", variant="primary")

    output = gr.Dataframe(
        headers=["Image", "Predicted Class", "Confidence"],
        label="Prediction Results",
        interactive=False
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=output
    )
    theme = "soft"

demo.launch()
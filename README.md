# TeaLeaf AI — Final Streamlit Application

This version follows the approved reference interface: dark-green sidebar, TeaLeaf AI branding, Faridpur Engineering College logo, three top result cards, model comparison, probability chart, recognizable class gallery, and the decision-support footer.

## Models
- ResNet50 — `models/best_resnet50.pth`
- EfficientNet-B0 — `models/best_efficientnet_b0.pth`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app performs real inference using the supplied trained models. Confidence values come from the model softmax output and are not hard-coded.

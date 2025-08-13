
````markdown
# Activation Steering for Bias Mitigation

This repository contains the official implementation for the paper:  
**Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs**

This work introduces a complete, end-to-end system that uses techniques from **mechanistic interpretability** to both **identify** and **actively mitigate bias** directly within a model's internal workings.

---

## 🚀 Overview

Instead of treating large language models as black boxes, this project *"looks inside"* to understand **how** and **where** they represent abstract concepts like bias.  
We use this understanding to perform **real-time interventions**, steering the model away from generating harmful content **without any retraining**.

**Core Pipeline:**
1. **Detect** → Train simple linear *probes* on the internal activations of `gpt2-large` to find where the model represents bias.  
   - **Key finding:** Bias becomes linearly separable and highly detectable in the **later layers** of the model.
2. **Steer** → Compute *steering vectors* from these biased representations.  
   - Add these vectors to the model’s activations **during inference** to guide text generation toward **safer, more neutral outputs**.

---

## ✨ Key Features

- **Layer-wise Bias Analysis**  
  First systematic evaluation of where bias is represented across all 36 layers of `gpt2-large`.

- **End-to-End Reproducible System**  
  Complete documented pipeline: data generation → real-time steering.

- **TransformerLens Integration**  
  Clean, educational implementation using the standard **TransformerLens** interpretability library.

- **Real-Time Mitigation**  
  Bias mitigation during inference without costly fine-tuning or retraining.



## 🛠️ Setup and Installation

Optimized for **Google Colab** with GPU.

```bash
# Clone the repository
git clone https://github.com/punctualprocrastinator/Activation-Steering-for-Bias-Mitigation.git
cd Activation-Steering-for-Bias-Mitigation

# Install dependencies
pip install -r requirements.txt
````

---

## ⚡ How to Run

Run the **full demo** (training probes + testing steering):

```bash
python main.py
```

The script will:

1. Load `gpt2-large`
2. Generate the bias dataset
3. Train a diagnostic probe for each layer
4. Compute the steering vectors
5. Run tests for **bias detection** & **mitigation**

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@misc{dubey2025activation,
      title={Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs}, 
      author={Shivam Dubey},
      year={2025},
      eprint={2508.09019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.



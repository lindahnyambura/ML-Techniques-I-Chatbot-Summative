# *Flan Kimathi*: A Dedan Kimathi Trial Chatbot

A historically immersive question-answering chatbot that simulates responses grounded in the 1956 trial of Kenyan freedom fighter Dedan Kimathi. 
Built using FLAN-T5-small and fine-tuned on curated archival transcripts and critical commentary.


### Project Summary
This chatbot was built to reconstruct the voice and historical context of Dedan Kimathi during his 1956 colonial trial, allowing users to ask natural language questions and receive responses grounded in primary source material.

It is a finetuned FLAN-T5-small on 342 curated QA pairs from MacArthur’s Dedan Kimathi on Trial and Ngũgĩ wa Thiong’o & Mugo’s play The Trial of Dedan Kimathi. It is grounded in real historical data with fallback safety for uncertain answers.
Prompt engineering and generation filters ensure conversational quality, confidence, and factuality.
It is built with TensorFlow, Hugging Face Transformers, and Gradio.



### Dataset Overview

This project uses a custom-built question-answering dataset drawn from two primary historical sources:

* *Dedan Kimathi on Trial* (edited by Julie MacArthur, 2017)
* *The Trial of Dedan Kimathi* (play by Ngũgĩ wa Thiong’o & Micere Githae Mugo, 1976)

The dataset consists of **342 QA pairs**, manually curated and annotated, covering key themes such as:

* Colonial legal procedures
* Symbolism and dramatic moments
* Gender and resistance
* Witness credibility and courtroom dynamics
* Press/media narratives

#### Directory Breakdown

```
data/
├── cleaned_text/                         # Cleaned and segmented text files
│   ├── dedan_kimathi_on_trial_cleaned.txt
│   ├── the_trial_of_dedan_kimathi_cleaned.txt
│   ├── dedan_kimathi_on_trial_cleaned/         # 472 sections (e.g. section_005.txt)
│   └── the_trial_of_dedan_kimathi_cleaned/     # 62 sections
├── extracted_text/                      # Raw extracted .txt files from PDFs
├── knowledge_base/                      # Manually built metadata files
│   ├── entities/
│   ├── relationships/
│   ├── themes/
│   └── timelines/
├── metadata_preprocessing/              # Code to extract and clean metadata
├── qa_pairs/                            # Final QA data
│   ├── manual/
│   │   └── kimathi_qa.csv               # Final manually authored QA pairs (342 entries)
│   └── automated/                       # (Empty - placeholder for auto QA scripts)
├── raw_text/                            # Original source PDFs
│   ├── dedan_kimathi_on_trial.pdf
│   └── the_trial_of_dedan_kimathi.pdf
└── qa_generation.py                     # QA pair generation script (not used in final build)
```

>  Note: The QA generation script was initially used for automation but did not yield high-quality results. The final dataset was curated manually based on deep reading of both texts.


### 🧠 Model & Training

This project fine-tunes a lightweight generative model to play the role of **Dedan Kimathi**, answering questions grounded in the historical context of his 1956 trial.

#### Model Architecture

We use:

* **Model**: `google/flan-t5-small` (60M parameters)
* **Framework**: TensorFlow + Hugging Face Transformers
* **Interface**: Gradio (local + web-based)

The model was chosen for its **low compute footprint** (suitable for Colab/VSCode) and **instruction tuning** abilities out-of-the-box.

#### Training Configuration

```python
config = {
    "model_name": "google/flan-t5-small",
    "learning_rate": 8e-5,
    "batch_size": 8,
    "epochs": 6,
    "warmup_steps": 100,
    "max_input_length": 64,
    "max_target_length": 128,
    "generation_params": {
        "num_beams": 4,
        "no_repeat_ngram_size": 2,
        "do_sample": False,
        "early_stopping": True,
        "max_new_tokens": 60,
        "repetition_penalty": 2.0
    }
}
```

#### Training & Evaluation

* **Loss convergence** observed across 6 epochs
* **Validation Loss** plateaued around \~0.91
* **Metrics** (on held-out QA pairs):

  * `BLEU-1`: **0.215**
  * `BLEU-4`: **0.026**
  * `ROUGE-L`: **0.202**
  * `Perplexity`: **2.49**

#### Prompt Engineering

We experimented with several prompt styles to maximize model grounding and persona retention. Final format:

```
Question: Why was Kimathi carrying a revolver?
Answer:
```

Fallback mechanisms were added using:

* Confidence estimation
* Fact-checking functions
* Custom rule-based filters

> The model gracefully responds “I don’t know” when unsure — helping preserve historical integrity over hallucinated confidence.

#### Model Files

Fine-tuned models are saved under:

```
model/
└── flan-kimathi-model-v7/
    ├── config.json
    ├── tokenizer_config.json
    ├── tf_model.h5
    └── ...
```

Perfect — here's a crisp, well-organized **App & Interface** section for your `README.md`:

---

### App & Interface

The chatbot is deployed through a **Gradio web interface**, allowing users to interact with Dedan Kimathi in a natural, question-answer format.

#### How It Works

* You type a question about Kimathi’s trial (e.g., *“Who sentenced Kimathi?”*).
* The model responds **in Kimathi’s voice**, grounded in factual trial data.
* If uncertain or unverifiable, it gracefully falls back with a humanlike disclaimer (e.g., *“I’m not sure. Here’s what I got...”*).


####  Behind the Scenes

We wrap the core `chat_with_model()` function in logic that:

* Checks factuality with `verify_answer()`
* Estimates confidence with `is_confident()`
* Falls back if either check fails


#### Running the App Locally

To launch the chatbot on your machine (VSCode or terminal):

```bash
python app.py
```

This opens a Gradio interface at a local URL, e.g.:

```
Running on http://127.0.0.1:7860
```

You can also **share** your app online with:

```python
gr.Interface(...).launch(share=True)
```

This gives you a link like:

```
https://24471c0b4fbdd7be59.gradio.live
```

> The link lasts \~72 hours. For permanent deployment, see [Hugging Face Spaces](https://huggingface.co/spaces/lnyambura/kimathi-bot).



### Usage & Deployment

This section shows you how to install dependencies, run the chatbot locally, and optionally deploy it for others to access.

---

#### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/lindahnyambura/kimathi-chatbot.git
cd kimathi-chatbot
pip install -r requirements.txt
```

>  Python 3.9+ recommended
>  Tested on Google Colab and local environments (VSCode + TensorFlow backend)

---

#### Running the Chatbot Locally

Once installed, launch the Gradio app using:

```bash
python app.py
```

This starts a local web server:

```
Running on http://127.0.0.1:7860
```

Visit the URL in your browser and ask historical questions like:

> “What was the final verdict?”
> “Did Kimathi testify?”
> “Was there a defense witness?”

---

### Limitations

While the chatbot offers an immersive experience into the historical trial of Dedan Kimathi, there are a few important limitations:

* **Small Dataset**: The model was fine-tuned on \~340 manually curated Q\&A pairs, which limits generalization beyond known trial facts.
* **Model Size**: I used `flan-t5-small` due to compute constraints. A larger model (e.g., `flan-t5-base`) could likely improve factual accuracy and coherence.
* **Factual Consistency**: Despite verification checks, the model may sometimes "hallucinate" facts or present confident but incorrect answers.
* **No Full Context Memory**: The chatbot answers each question independently. It doesn’t track past conversations or remember prior questions.
* **Biases & Gaps**: The responses depend on the curated source materials and may reflect their colonial, editorial, or interpretive biases.
* **Offline Access**: Gradio interfaces are ephemeral unless hosted via Hugging Face or other services.

> OOD (Out-of-distribution) questions are filtered using confidence + fact verification logic to fall back safely with an “I don’t know” response.

---

### Acknowledgements & Credits

This project draws inspiration and content from the following sources:

 **Primary Sources**

* *Dedan Kimathi on Trial* – Edited by Julie MacArthur (2017)
* *The Trial of Dedan Kimathi* – By Ngũgĩ wa Thiong’o & Micere Githae Mugo (1976)

 **Frameworks & Libraries**

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [TensorFlow](https://www.tensorflow.org/)
* [Gradio](https://gradio.app/)
* [NLTK](https://www.nltk.org/) for text preprocessing

 **Development Tools**

* Google Colab (TPU and CPU)
* Visual Studio Code
* GitHub for version control

 **Project Lead**
Crafted, engineered, and debugged under pressure by Lindah Nyambura

### Video demo

https://drive.google.com/file/d/1t8O-SUiqpfYHUF_OMMUXGMsETt64e4Xj/view?usp=sharing

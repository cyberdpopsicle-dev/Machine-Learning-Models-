# Day 12 : Localized Incremental Language Engine (LILE)

A highly efficient, open-source framework designed to bootstrap datasets and machine translation for low-resource, endangered, or undocumented languages with minimal digital footprints.

Unlike massive, opaque large language models, LILE uses an incremental Differentiable Symbolic Memory architecture. It enables field researchers and linguists to convert raw document images into structured dictionaries while supporting a dynamic, zero-weight unlearning protocol for instantly correcting human feedback mistakes.

---

# Key Features

- **Visual Ingestion Layer** — Uses Optical Character Recognition (OCR) to extract unfamiliar scripts directly from images or document scans.
- **Familiar-Word In-Place Translation** — Preserves the original document structure while replacing known tokens with their English equivalents.
- **Automated Unknown Token Batching** — Groups unseen words into standalone `.txt` patch files for manual translation.
- **Deterministic Weight Erasure (Unlearning)** — Removes an incorrect translation in a single operation without retraining or affecting the rest of the model.
- **Part-of-Speech (POS) Profiling** — Automatically classifies newly learned vocabulary into grammatical categories such as nouns, verbs, and adjectives.
- **Lightweight Architecture** — Runs efficiently on standard consumer hardware without requiring GPU clusters.

---

# Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/incremental-language-engine.git
cd incremental-language-engine

pip install easyocr spacy numpy torch pandas opencv-python
```

Download the default spaCy language model.

```bash
python -m spacy download en_core_web_sm
```

---

# Project Structure

```
.
├── model_storage/          # Serialized engine state (.pt)
├── input_images/           # Input document images
├── untranslated_dumps/     # Generated translation patch files
├── incremental_engine.py   # Core engine implementation
└── main.py                 # Execution pipeline
```

---

# Usage

## 1. Process an Image

Place an image in `input_images/` and process it to generate a translation patch file for unknown words.

```python
from incremental_engine import (
    IncrementalLanguageEngine,
    ImageLanguageProcessor,
)

# Initialize or load the language model
engine = IncrementalLanguageEngine(
    storage_path="model_storage/my_language.pt"
)

# Create the image processor
processor = ImageLanguageProcessor(engine)

# Process a document page
output_layout, patch_file_path = processor.process_page(
    image_path="input_images/page_01.png",
    page_id="page_01",
)

print("Current layout:", output_layout)
print("Translation file:", patch_file_path)
```

---

## 2. Provide Human Translations

Open the generated file:

```
untranslated_dumps/needed_translations_page_01.txt
```

Example:

```text
# Translation required for page: page_01
# Format: unknown_word = english_translation

koshur =
lukh =
khyon =
```

Fill in the translations:

```text
koshur = kashmiri
lukh = people
khyon = eat
```

Save the file when finished.

---

## 3. Learn the New Vocabulary

Import the edited translation file back into the engine.

```python
from main import feed_batch_translations

feed_batch_translations(
    "untranslated_dumps/needed_translations_page_01.txt"
)
```

The engine will:

- Update its dictionary.
- Save the new vocabulary.
- Automatically assign Part-of-Speech (POS) tags.

Future documents containing these words will be translated automatically.

---

## 4. Correct an Incorrect Translation

If a translation is incorrect, use the unlearning function to remove it and replace it immediately.

```python
from main import correct_mistake

correct_mistake(
    wrong_unknown_word="koshur",
    correct_english_translation="kashmiri",
)
```

The previous mapping is erased without retraining the entire model.

---
# License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

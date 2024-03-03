# JavaCopilot

This is the official codebase for **JavaCopilot**.

## Introduction

JavaCopilot is an RAG framework-based learning assistant for Java programming, utilizing textbooks, syllabi, and common Java questions as document corpus. It is entirely developed on open-source local models, with the embedded model being [m3e-base] (https://huggingface.co/moka-ai/m3e-base) and the large model being [Qwen-7B-Chat] (https://huggingface.co/Qwen/Qwen-7B-Chat), aiming to provide Java learning assistance to teachers and students alike.Here is an example.

<p align="center">
  <img src="https://github.com/hujili007/JavaCopilot/blob/dafdaa70f67b1eebc2e01a3fc042b8c5e25e5298/javacopilot.jpeg" alt="GastroBot architecture diagram" border="0" width=50%>
</p>

## Getting Started

### step 1 Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name javarag python=3.10.13 -y
$ conda activate javarag
# install requirements
$ pip install -r requirements.txt
```

### step 2 Create index

```
Clone the repo.
$ git clone https://github.com/hujili007/JavaCopilot.git
$ cd JavaCopilot
```

### step 3 Run app.py

```
$ streamlit run app.py
```

## Acknowledgement

- llamaindex: [https://www.llamaindex.ai/](https://www.llamaindex.ai/)
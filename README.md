# Chatbot Unicamp 2024

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Running the Application](#running-the-application)
- [License](#license)

## Overview

The "Chatbot Unicamp 2024" project aims to develop a conversational
assistant based on RAG - Retrieval Augmented Generation to answer questions
related to the Unicamp 2024 Entrance Exam. The assistant was built using
LlamaIndex framework, which is a simple, flexible data framework for connecting
custom data sources to large language models (LLMs). 

The project used information from Resolution GR-031/2023, dated 07/13/2023,
which defines the rules and information for the entrance exam.

In addition to meeting the needs of entrance exam candidates, this project also serves as part of NeuralMind's 
selection process, where students will have the opportunity to apply their AI knowledge in practice.

## Getting Started

### Prerequisites

- OpenAI API Key (obtainable from the OpenAI platform)

### Installation

Inside the [env](.env) file, put the OpenAI API key, as shown below:
```bash
OPENAI_API_KEY = "sk-################################################"
```

Then install the necessary libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
chainlit run app.py
```

## License

This project is licensed under the MIT License.

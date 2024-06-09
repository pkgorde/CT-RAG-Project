## RAG Pipeline setup for Clinical T5 and Llama3

This is an applicatory project, made with association with UC Davis Health to create a local LLM RAG-setup for extracting information from Clinical Trials documents written by UC Davis Health professionals to submit into CT.gov's portal.

To use directly, You will have to get access to Clinical T5 model from Physionet, which also requires a CITI certification from MIT Affiliates. You can find it here: https://physionet.org/content/clinical-t5/1.0.0/

For implementing Foundational model like Llama3 and Llama2, Use RAG/app.py directly, which uses Ollama based backbone for using those models.

To use special models from Huggingface, or in hf format, use RAG/huggingface_app.py to do so and change the model and tokenizer objects accordingly. 



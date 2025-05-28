## Setup

1. Install my [search-and-learn](https://github.com/tnguyen9210/search-and-learn) forked from huggingface
   ```
   git clone https://github.com/tnguyen9210/search-and-learn
   cd search-and-learn
   pip install -e '.[dev]'
   ```
2. Install `requirements.txt`
   ```
   pip install -r requirements.txt
   ```
3. Download [PRM800K](https://github.com/openai/prm800k/tree/main) dataset,
   and set `data_dir` to the directory where it was saved.  
4. Download [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model for LLM,
   and set `llm_tokenizer_dir` to its downloaded path. 
5. Download [Llama3.1-8B-PRM-Deepseek-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data) model for the PRM,
   and set `prm_tokenizer_dir` to the corresponding directory.
Notes: You'll need install [Git LFS](https://git-lfs.com/) and turn it on to properly clone these large files. 

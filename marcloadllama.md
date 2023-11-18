```python
!pip install huggingface_hub ipywidgets
```

    Collecting huggingface_hub
      Obtaining dependency information for huggingface_hub from https://files.pythonhosted.org/packages/05/09/1945ca6ba3ad8ad6e2872ba682ce8d68c5e63c8e55458ed8ab4885709f1d/huggingface_hub-0.19.4-py3-none-any.whl.metadata
      Downloading huggingface_hub-0.19.4-py3-none-any.whl.metadata (14 kB)
    Collecting ipywidgets
      Obtaining dependency information for ipywidgets from https://files.pythonhosted.org/packages/4a/0e/57ed498fafbc60419a9332d872e929879ceba2d73cb11d284d7112472b3e/ipywidgets-8.1.1-py3-none-any.whl.metadata
      Downloading ipywidgets-8.1.1-py3-none-any.whl.metadata (2.4 kB)
    Collecting filelock (from huggingface_hub)
      Obtaining dependency information for filelock from https://files.pythonhosted.org/packages/81/54/84d42a0bee35edba99dee7b59a8d4970eccdd44b99fe728ed912106fc781/filelock-3.13.1-py3-none-any.whl.metadata
      Downloading filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
    Collecting fsspec>=2023.5.0 (from huggingface_hub)
      Obtaining dependency information for fsspec>=2023.5.0 from https://files.pythonhosted.org/packages/e8/f6/3eccfb530aac90ad1301c582da228e4763f19e719ac8200752a4841b0b2d/fsspec-2023.10.0-py3-none-any.whl.metadata
      Downloading fsspec-2023.10.0-py3-none-any.whl.metadata (6.8 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/site-packages (from huggingface_hub) (2.31.0)
    Collecting tqdm>=4.42.1 (from huggingface_hub)
      Obtaining dependency information for tqdm>=4.42.1 from https://files.pythonhosted.org/packages/00/e5/f12a80907d0884e6dff9c16d0c0114d81b8cd07dc3ae54c5e962cc83037e/tqdm-4.66.1-py3-none-any.whl.metadata
      Downloading tqdm-4.66.1-py3-none-any.whl.metadata (57 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.6/57.6 kB[0m [31m4.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/site-packages (from huggingface_hub) (6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/site-packages (from huggingface_hub) (4.5.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/site-packages (from huggingface_hub) (23.1)
    Requirement already satisfied: comm>=0.1.3 in /usr/local/lib/python3.10/site-packages (from ipywidgets) (0.1.3)
    Requirement already satisfied: ipython>=6.1.0 in /usr/local/lib/python3.10/site-packages (from ipywidgets) (8.10.0)
    Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.10/site-packages (from ipywidgets) (5.9.0)
    Collecting widgetsnbextension~=4.0.9 (from ipywidgets)
      Obtaining dependency information for widgetsnbextension~=4.0.9 from https://files.pythonhosted.org/packages/29/03/107d96077c4befed191f7ad1a12c7b52a8f9d2778a5836d59f9855c105f6/widgetsnbextension-4.0.9-py3-none-any.whl.metadata
      Downloading widgetsnbextension-4.0.9-py3-none-any.whl.metadata (1.6 kB)
    Collecting jupyterlab-widgets~=3.0.9 (from ipywidgets)
      Obtaining dependency information for jupyterlab-widgets~=3.0.9 from https://files.pythonhosted.org/packages/e8/05/0ebab152288693b5ec7b339aab857362947031143b282853b4c2dd4b5b40/jupyterlab_widgets-3.0.9-py3-none-any.whl.metadata
      Downloading jupyterlab_widgets-3.0.9-py3-none-any.whl.metadata (4.1 kB)
    Requirement already satisfied: backcall in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.2.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (4.4.2)
    Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.18.2)
    Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.3)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.7.5)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.30 in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (3.0.38)
    Requirement already satisfied: pygments>=2.4.0 in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (2.15.1)
    Requirement already satisfied: stack-data in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.6.2)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (4.8.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/site-packages (from requests->huggingface_hub) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/site-packages (from requests->huggingface_hub) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/site-packages (from requests->huggingface_hub) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/site-packages (from requests->huggingface_hub) (2023.7.22)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.10/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.10/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets) (0.6.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/site-packages (from prompt-toolkit<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets) (0.2.5)
    Requirement already satisfied: executing>=1.2.0 in /usr/local/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (1.2.0)
    Requirement already satisfied: asttokens>=2.1.0 in /usr/local/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.2.1)
    Requirement already satisfied: pure-eval in /usr/local/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.2.2)
    Requirement already satisfied: six in /usr/local/lib/python3.10/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets) (1.16.0)
    Downloading huggingface_hub-0.19.4-py3-none-any.whl (311 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m311.7/311.7 kB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading ipywidgets-8.1.1-py3-none-any.whl (139 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m139.4/139.4 kB[0m [31m19.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2023.10.0-py3-none-any.whl (166 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.4/166.4 kB[0m [31m23.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jupyterlab_widgets-3.0.9-py3-none-any.whl (214 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m214.9/214.9 kB[0m [31m32.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tqdm-4.66.1-py3-none-any.whl (78 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.3/78.3 kB[0m [31m11.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading widgetsnbextension-4.0.9-py3-none-any.whl (2.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m71.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading filelock-3.13.1-py3-none-any.whl (11 kB)
    Installing collected packages: widgetsnbextension, tqdm, jupyterlab-widgets, fsspec, filelock, huggingface_hub, ipywidgets
    Successfully installed filelock-3.13.1 fsspec-2023.10.0 huggingface_hub-0.19.4 ipywidgets-8.1.1 jupyterlab-widgets-3.0.9 tqdm-4.66.1 widgetsnbextension-4.0.9



```python
!pip install --upgrade huggingface-hub
!pip install --upgrade transformers

# get your account token from https://huggingface.co/settings/tokens


# import the relavant libraries for loggin in
from huggingface_hub import HfApi, HfFolder

hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="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", # Token is not persisted on the machine.
)
folder = HfFolder()
folder.save_token(token)
```

    Requirement already satisfied: huggingface-hub in ./.local/lib/python3.10/site-packages (0.19.4)
    Requirement already satisfied: filelock in ./.local/lib/python3.10/site-packages (from huggingface-hub) (3.13.1)
    Requirement already satisfied: fsspec>=2023.5.0 in ./.local/lib/python3.10/site-packages (from huggingface-hub) (2023.10.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/site-packages (from huggingface-hub) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in ./.local/lib/python3.10/site-packages (from huggingface-hub) (4.66.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/site-packages (from huggingface-hub) (6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/site-packages (from huggingface-hub) (4.5.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/site-packages (from huggingface-hub) (23.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/site-packages (from requests->huggingface-hub) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/site-packages (from requests->huggingface-hub) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/site-packages (from requests->huggingface-hub) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/site-packages (from requests->huggingface-hub) (2023.7.22)
    Requirement already satisfied: transformers in ./.local/lib/python3.10/site-packages (4.35.2)
    Requirement already satisfied: filelock in ./.local/lib/python3.10/site-packages (from transformers) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in ./.local/lib/python3.10/site-packages (from transformers) (0.19.4)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/site-packages (from transformers) (1.24.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/site-packages (from transformers) (23.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/site-packages (from transformers) (6.0)
    Requirement already satisfied: regex!=2019.12.17 in ./.local/lib/python3.10/site-packages (from transformers) (2023.10.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/site-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in ./.local/lib/python3.10/site-packages (from transformers) (0.15.0)
    Requirement already satisfied: safetensors>=0.3.1 in ./.local/lib/python3.10/site-packages (from transformers) (0.4.0)
    Requirement already satisfied: tqdm>=4.27 in ./.local/lib/python3.10/site-packages (from transformers) (4.66.1)
    Requirement already satisfied: fsspec>=2023.5.0 in ./.local/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.16.4->transformers) (2023.10.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.16.4->transformers) (4.5.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/site-packages (from requests->transformers) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/site-packages (from requests->transformers) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/site-packages (from requests->transformers) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/site-packages (from requests->transformers) (2023.7.22)



```python
!pip install --quiet bitsandbytes
!pip install --quiet transformers
!pip install --quiet accelerate
!pip install scipy numpy
!pip install torch==2.0.1
```

    Requirement already satisfied: scipy in ./.local/lib/python3.10/site-packages (1.11.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/site-packages (1.24.2)
    Requirement already satisfied: torch==2.0.1 in ./.local/lib/python3.10/site-packages (2.0.1)
    Requirement already satisfied: filelock in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/site-packages (from torch==2.0.1) (4.5.0)
    Requirement already satisfied: sympy in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (1.12)
    Requirement already satisfied: networkx in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/site-packages (from torch==2.0.1) (3.1.2)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.7.99)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.7.99)
    Requirement already satisfied: nvidia-cuda-cupti-cu11==11.7.101 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.7.101)
    Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (8.5.0.96)
    Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.10.3.66)
    Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (10.9.0.58)
    Requirement already satisfied: nvidia-curand-cu11==10.2.10.91 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (10.2.10.91)
    Requirement already satisfied: nvidia-cusolver-cu11==11.4.0.1 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.4.0.1)
    Requirement already satisfied: nvidia-cusparse-cu11==11.7.4.91 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.7.4.91)
    Requirement already satisfied: nvidia-nccl-cu11==2.14.3 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (2.14.3)
    Requirement already satisfied: nvidia-nvtx-cu11==11.7.91 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (11.7.91)
    Requirement already satisfied: triton==2.0.0 in ./.local/lib/python3.10/site-packages (from torch==2.0.1) (2.0.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (65.5.0)
    Requirement already satisfied: wheel in /usr/local/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (0.41.2)
    Requirement already satisfied: cmake in ./.local/lib/python3.10/site-packages (from triton==2.0.0->torch==2.0.1) (3.27.7)
    Requirement already satisfied: lit in ./.local/lib/python3.10/site-packages (from triton==2.0.0->torch==2.0.1) (17.0.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/site-packages (from jinja2->torch==2.0.1) (2.1.2)
    Requirement already satisfied: mpmath>=0.19 in ./.local/lib/python3.10/site-packages (from sympy->torch==2.0.1) (1.3.0)



```python
import torch
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

    Number of GPUs available: 0



```python
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path="./huggingface/llama7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16)

#Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True) 
tokenizer.save_pretrained(base_model_path, from_pt=True)
```


    tokenizer_config.json:   0%|          | 0.00/1.62k [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.84M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/414 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/614 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/26.8k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]



    model-00001-of-00002.safetensors:   0%|          | 0.00/9.98G [00:00<?, ?B/s]



    model-00002-of-00002.safetensors:   0%|          | 0.00/3.50G [00:00<?, ?B/s]



```python

```

#### Installing Ollama (Linux, Mac, Windows)

To Install Ollama we can follow this repository by [Ollama](https://github.com/ollama/ollama). Remember it is better if your laptop/pc/macbook has GPU otherwise running local LLMs isn't a good idea unless CPU specs are really good with a decent iGPU.

#### Downloading Model For Ollama (Hugging Face)

We can download any local model from Hugging Face's [website](https://huggingface.co/models). Look for a small model to run locally for example if your GPU has 8 GB Vram these can be some ideal models (make sure you download in GGUF Format). 

|                          |      |     |                                                                       |                                   |
| ------------------------ | ---- | --- | --------------------------------------------------------------------- | --------------------------------- |
| Mistral-7B-Instruct-v0.2 | 7B   | 32K | [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | Great for 8GB VRAM (Q4_K_M)       |
| Phi-3-mini-4k-instruct   | 3.8B | 4K  | [Link](https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF)   | Microsoft’s efficient small model |
| Llama-3-8B-Instruct (Q4) | 8B   | 8K  | [Link](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)      | May fit 8GB VRAM at Q4 quant      |
| Gemma-7B-it (Q4_K_M)     | 7B   | 8K  | [Link](https://huggingface.co/TheBloke/gemma-7b-it-GGUF)              | Google’s lightweight model        |
| Qwen1.5-4B-Chat-GGUF     | 4B   | 32K | [Link](https://huggingface.co/TheBloke/Qwen1.5-4B-Chat-GGUF)          | Good multilingual support         |

##### 8 GB Vram Graphic Cards

Look for Q8, Q6, or other quantized (“quant”) models. Keep testing different models, but try limiting the context length — you can set this in your Modelfile. By default, context size is 4096 tokens (Mbs). If you have lower VRAM, try Q4 or similar lower-bit quantizations. As a rule of thumb, your model size should be less than your available VRAM. For example: a 6GB model runs best on 8GB+ VRAM. But you can load larger models say, 10–11GB on an 8GB GPU. Ollama will load as many layers as possible into VRAM, and spill the rest over to CPU/system RAM. It’ll still work just slower. For a better understanding check this [blog](https://medium.com/%40samanch70/goodbye-vram-limits-how-to-run-massive-llms-across-your-gpus-b2636f6ae6cf) out. 

#### Sample Modelfile and How to create it 

Now I am assuming you have Ollama installed, Open your System's terminal i.e. Powershell for windows or Terminal on both Linux and Mac. 

<pre>
Ollama serve
</pre>
This command will start Ollama server. After this we need to create a Modelfile. Keep this running and open a new tab/windows of your terminal.

##### Mac/Linux

Create a directory called `modelfiles` in your home directory.

##### Windows

Create a directory called `modelfiles` in your users directory like this:
`C:\Users\<YourUsername>\modelfiles\`

Rest of the steps are same for Linux/Mac/Windows

Now using you terminal cd into modelfiles directory and:

<pre>mkdir name-you-want-to-give-to-model</pre>

For example the complete model name is `Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf` you can name your model something like qwen3-4b-q8, then we'll run:

<pre> mkdir qwen3-4b-q8 </pre>

After creating this directory we need to cd into it:

<pre> cd qwen3-4b-q8 </pre>
#### Example Modelfile

First make sure our GGUF file has been downloaded, now we can keep it in root/C drive or some other external drive, Let's say i keep the GGUF files in a drive other than root/C drive in case of windows, for example i have placed the GGUF models in:

`/run/media/cdev/data-storage/local-models/gguf`

Now to create Modelfile we need to first add parameters into it:

we need to now run:

<pre> nano Modelfile </pre>

This will create an empty Modelfile in the current directory, we need to add parameters to it, in Line 1 we need to tell it where to load the model from so line 1 can look like:

<code>FROM /run/media/cdev/data-storage/local-models/gguf/Name-of-Model.gguf
</code>

In case of our Qwen Model we can use this:

<code>FROM /run/media/cdev/data-storage/local-models/gguf/Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf
</code>

See how I have made sure to use exact model name here unlike in the directory of Modelfile.

##### GPU and Context Parameters

Now Let's how many layers we can offload to GPU, typically (I maybe wrong) and 8 GB Vram GPU can support 45 layers, even if set more layers than the GPU can support, Ollama will default to the correct version it can support, so instead of setting up 45 layers let's say we want to 30 layers so can type in Line 2 of the Modelfile:

<pre>
PARAMETER num_gpu 30
</pre>

So this means Ollama will offload 30 layers to GPU, we still have 2 more layers, how do i know that? See: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF

![](https://i.ibb.co/G43H3kML/image.png)

Here it says the model has 36 Layers and can support context length of 262,144, but we can not use all that context on our 8 GB GPU, we should start small i.e. from 4096 context. The 2 remaining layers will automatically be offloaded to CPU by Ollama and some CPU RAM will also be used. 

Here's how we can set context:

<pre> PARAMETER num_ctx 4096 </pre>

We can try increasing this later for better results, Now we can save the Modelfile and close it. And after that we need to create it by running: 

<pre> ollama create model-name-we-want-to-give -f Modelfile </pre>

For example for the qwen3-4b-q8 model i'd like to go with the same name (I used for directory) , so we can use:

<pre> ollama create qwen3-4b-q8 -f Modelfile </pre>

Remember Modelfile has defaults for running our model we can always override the defaults and keep experimenting. 

Now this is how our complete Modelfile looks like:

```bash
FROM /run/media/cdev/data-storage/local-models/gguf/Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf
PARAMETER num_gpu 30
PARAMETER num_ctx 4096
```
#### Running the Model

To run the model we can simply run:

<pre>
ollama run model-name </pre>
In case of qwen model we are using it will be:

<pre>
ollama run qwen3-4b-q8 </pre>

Now to ensure gpu is being used we can ctrl + f on windows and ctrl + shift + f on linux to search `gpu`, it will show something like this: 

![](https://i.ibb.co/5xh35q88/image.png)

###### Mac/Linux

It means gpu is being used. Mac/Linux users can also type:

`nvidia-smi` and they will see an output like:

![](https://i.ibb.co/QvLwWJBk/image.png)

This means 7050 MBs are been used by Model

###### Windows 

Windows users can see task manager and see that their GPU is being used.
#### Overriding defaults set in Modelfile 

Now there can be several reason why we might want to override defaults of a Modelfile, close the current instance of ollama model by entering `/bye` in the terminal tab where we are running ollama model. 

Then we need to make changes to the Modelfile again and the recreate Modelfile, but this time change model name a bit, else previous model will be used again for exampling changing name of `qwen3-4b-q8` to `qwen3-4b-q8-32kcontext`

For detailed info about parameters we can use in Model files see:

https://github.com/ollama/ollama/blob/main/docs/modelfile.md?spm=a2ty_o01.29997173.0.0.737bc921TXntoi&file=modelfile.md

https://github.com/ollama/ollama/issues/9512

#### Deleting an Ollama Model

Let's say you don't need a model any more or want to create a new model based on entirely new setting, you can remove the model by typing:

<pre> ollama rm modelname </pre>
For example in case of Qwen model:

<pre>
ollama rm qwen3-4b-q8 </pre>
This will not remove your Modelfile, so you can edit and reuse that to create new ollama model. 

#### Some parameters to check when ollama serve is running

Sometimes we might need to check if our running local model has to correct context window or other correct parameters, these are some parameters you can look for in terminal:

###### Context size (`n_ctx`)

Sample Output:

`llama_context: n_ctx = 4096`

This means this is the current context window running in the model, if you setup more context window in model file it might still return this, this is because either the quantized version supports a maximum context window of this size or ollama is reverting to this due to some other reason. 

###### Trainable context (`n_ctx_train`)

Sample Output: 

`print_info: n_ctx_train      = 131072`

This the maximum theoretical context window of the model.

###### GPU Offloaded Layers

This shows how many transformer layers are being offloaded to GPU.

Sample Output:

```
load_tensors: offloading 26 repeating layers to GPU
load_tensors: offloaded 26/33 layers to GPU
```

###### KV Cache Size

What is the KV cache?

- In transformer models like LLaMA, during inference the model stores:
    
    - K (Keys) and V (Values) for each token in each attention layer.
        
- This allows the model to reuse past computations instead of recalculating them every time a new token is generated.
    
- The memory used for this is called the KV cache.

Sample Output:

```
llama_kv_cache_unified: kv_size = 4096, type_k = 'f16', type_v = 'f16', n_layer = 32, can_shift = 1, padding = 32
```

This means current model's KV cache is configured for a 4096-token context, uses FP16 precision, allocates space for 32 layers, supports sliding window eviction, and pads allocations to 32 tokens for efficiency.

#### Running this Local Chatbot

The code exists in `chatbot.py`
To run this program make sure you have python 3.12 installed on your system. 
The install uv if it's already not installed i.e.

<pre>
pip install uv
</pre>

After make a folder where you want to clone(download) this project and the open that directory in terminal.
Then run:

<pre>
git clone https://github.com/shery7310/Building-a-Local-Chatbot-using-Ollama-and-Chainlit.git
</pre>

Then after this is cloned in terminal run:

<pre>
cd Building-a-Local-Chatbot-using-Ollama-and-Chainlit
</pre>

Then just run `uv sync` to so that a virtual environment with required packages is created.

Then open terminal to run this chatbot and run:

<pre>
uv run chainlit run chatbot.py -w
</pre>

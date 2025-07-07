import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from pprint import pformat
from assistant_utils import create_dcc_assistant

def setup_llama(base_dir):
    base_dir = Path(base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(base_dir)

    print(f"Setting up Llama in: {base_dir}")

    # install packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])  # , "transformers", "torch", "sentencepiece"])        
    
    model_url = "https://huggingface.co/TheBloke/llama-2-7b-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    # model_path = base_dir / "llama-2-7b-chat.Q4_K_M.gguf"
    model_path = os.path.abspath(os.path.join(str(base_dir), "llama-2-7b-chat.Q4_K_M.gguf"))
    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} to {model_path}")
        subprocess.check_call(["wget", model_url, "-O", str(model_path)])   
    else:
        print(f"Model already exists at {model_path}, skipping download.")

    return model_path

def chat_with_llama(model_path):
    from llama_cpp import Llama

    # Load the model
    llm = Llama(
        model_path=str(model_path), 
        n_ctx=2048, 
        n_batch=512, 
        n_gpu_layers=8, 
        n_threads=8,
        # chat_format='llama'
    )
    print("\nLlama Chat Interface")
    print("Llama model: {}".format(llm))
    print("dir(llm):\n{}".format(pformat(dir(llm), indent=4)))
    print("Type 'exit' or 'quit' to stop chatting.\n")
    print("."*50)

    messages = []

    # Example chat interaction
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        
        messages.append({"role": "user", "content": user_input})
        response = llm.create_chat_completion(  
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            stop=["<|endoftext|>"]
        )
        # print("full response: \n{}".format(pformat(response, indent=4)))
        assistant_message = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("\nLlama: {}\n".format(assistant_message))

        messages.append({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    # if True:
    custom_dir = "llama_model"
    try:
        if len(sys.argv) > 1:
            dcc = sys.argv[1]
        else:
            dcc = "llama_model"


        model_path = setup_llama(custom_dir) 
        print("model_path: ", model_path)
        print("DCC: {}".format(dcc))
        create_dcc_assistant(model_path, dcc)
        # chat_with_llama(model_path)
    except Exception as e:
    # else:
        print(f"An error occurred: {e}")
        sys.exit(1) 
    
    # Optionally, you can import and use the Llama model here
    # from llama_cpp import Llama
    # model = Llama(model_path=str(model_path))
    # print("Llama model loaded successfully.")
import logging
import os, sys
import time
import base64
import argparse
import json
from pathlib import Path
# working directory = /source/autonn_core/tangochat
COMMON_ROOT     = Path("/shared/common")
DATASET_ROOT    = Path("/shared/datasets")
MODEL_ROOT      = Path("/shared/models")

CORE_DIR        = Path(__file__).resolve().parent.parent # /source/autonn_core
sys.path.append(str(CORE_DIR))
CFG_DIR         = CORE_DIR / 'tangochat' / 'common' / 'cfg'
# HF_HOME         = os.environ.get('HF_HOME', '/root/.cache/huggingface')

import torch
import streamlit as st
# import huggingface_hub
# from tangochat.loader.download import (  
#     download_model,
#     list_model,
#     remove_model,
#     _get_diretory_size
# )
# from tangochat.inference.generate import (  
#     Generator,
#     BuilderArgs,
#     TokenizerArgs,
#     GeneratorArgs
# )
from tangochat.tuner.rag import (
    load_and_retrieve_docs, 
    get_rag_formatted_prompt,
)
import ollama

# logging ----------------------------------------------------------------------
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# def run_tangochat():
# page config ------------------------------------------------------------------
st.set_page_config(
    page_title="TANGO+Chat",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a part of TANGO project by ETRI."
    }
)

# init -------------------------------------------------------------------------
st.session_state.uploader_key = 0

def reset_per_message_state():
    _update_uploader_key()

def _update_uploader_key():
    st.session_state.uploader_key = int(time.time())

start_state = [
    {
        "role": "system",
        "content": "**Select an AI model you want** [:red[DOWNLOAD] > :blue[RUN] > :green[APPLY]]",
    },
    {
        "role": "assistant",
        "content": "Welcome! How can I help you?"
    },
]

if "messages" not in st.session_state:
    st.session_state['messages'] = start_state

if "token" not in st.session_state:
    st.session_state.token = {'hf_token': ""}

if "brain" not in st.session_state:
    args = argparse.Namespace()
    args_path = CFG_DIR / 'args_chat.json'
    if os.path.isfile(args_path):
        with open(args_path, 'r') as f:
            args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
    st.session_state.brain = {
        "local_gen_name": None,
        "local_gen_obj": None,
        "local_gen_args": args
    }

if "bye" not in st.session_state:
    st.session_state.bye = False

if "img_prompt" not in st.session_state:
    st.session_state.img_prompt = "invisible"

if "rag" not in st.session_state:
    st.session_state.rag = {"active": False,
                            "url": "",
                            "embed": "",
                            "retreiver": None,}

# huggingface ------------------------------------------------------------------
# hf_token_cache = f'{HF_HOME}/token'
# if os.path.isfile(hf_token_cache):
#     with open(hf_token_cache, 'r') as f:
#         cached_token = f.read()
#     st.session_state.token = {"hf_token": cached_token}

# @st.dialog("Hugging Face Token")
# def get_token_and_login():
#     # with st.form(key='hf_token_submit_form'):
#     "If you do not have any, " + \
#     "**[get a new huggingface token](https://huggingface.co/settings/tokens)**"
#     hf_token = st.text_input(
#         label = "Access Token",
#         key = "huggingface_token",
#         type = 'password',
#         value = st.session_state.token['hf_token'],
#         help ='Access tokens authenticate your identity to the Hugging Face Hub ' + \
#             'and allow Tango+Chat to download LLMs based on token permissions.')
#     if st.button("Submit"):
#         st.session_state.token['hf_token'] = hf_token
#         success = login_hf()
#         if success:
#             st.rerun()

# def login_hf():
#     hf_token = st.session_state.token['hf_token']
#     try:
#         logger.info(f"Logging in the Hugging Face Hub with token: {hf_token}")
#         huggingface_hub.login(token=hf_token, write_permission=True, add_to_git_credential=True)
#         logger.info(f"Success logging in")
#         success = True
#     except Exception as e:
#         st.warning(f"***{e} Please input a valid token.***")
#         logger.warning(f"{e} Please input a valid token.")
#         st.session_state.token['hf_token'] = ""
#         success = False
#         logger.warning(f"Fail logging in")
#     finally:
#         return success

# exit -------------------------------------------------------------------------
if st.session_state.bye:
    logger.info("Completed\n")
    st.stop()


def switch_ollama_model():
    model_name = st.session_state.brain['local_gen_name']
    # logger.info(f"{st.session_state.brain}")
    if model_name is None:
        return
    st.session_state.brain['local_gen_obj'] = model_name
    st.balloons()
    
    st.session_state['messages'] = start_state
    for msg in st.session_state.messages:
        if msg['role'] == 'system':
            m_name = lists_for_ollama[model_name].split("***")[1]
            msg['content'] = f"Let's talk to **{m_name.upper()}** !!"
            if m_name == 'LLaVA 1.6':
                st.session_state.img_prompt = 'visible'

# middle top menu --------------------------------------------------------------
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**DOWNLOAD**", "**RUN**", "DELETE", "R-A-G", "FINETUNE", "DEPLOY"])
# lists = [
#             "***Llama2-7B-base***", 
#             "***Llama2-7B-chat***", 
#             "***Llama2-13B-chat***", 
#             "***Llama2-70B-chat***", 
#             "***Llama3-8B-base***", 
#             "***Llama3-8B-instruct***", 
#             "***Llama3-70B-instruct***", 
#             "***Llama3.1-8B-base***", 
#             "***Llama3.1-8B-instruct***", 
#             "***Llama3.1-8B-instruct-tune***", 
#             "***Llama3.1-70B-instruct***",
#             "***Llama3.1-70B-instruct-tune***",
#             "***CodeLlama-7B***", 
#             "***CodeLlama-34B***",
#             "***Mistral-7B-v0.1-base***", 
#             "***Mistral-7B-v0.1-instruct***",
#             "***Mistral-7B-v0.2-instruct***", 
#             "***Open-Llama-7B***",
#             "***Tiny-Llama-stories15M***", 
#             "***Tiny-Llama-stories42M***",
#             "***Tiny-Llama-stories110M***",
#         ]
tab1, tab2, tab3, tab4 = st.tabs(["**DOWNLOAD**", "**RUN**", "**DELETE**", "**RAG**"])
lists_for_ollama = {
            "llama3.2"      : "***Llama 3.2***   (3B)",
            "llama3.1"      : "***Llama 3.1***   (8B)",
            "phi3.5"        : "***Phi 3.5***     (4B)",
            "mistral"       : "***Mistral 0.3*** (7B)",
            "neural-chat"   : "***Neural-Chat*** (7B)",
            "codellama"     : "***CodeLlama***   (7B)",
            "llava"         : "***LLaVA 1.6***   (8B)",
            "gemma2"        : "***Gemma 2***     (9B)",
            "qwen2.5"       : "***Qwen 2.5***    (7B)",
            "bnksys/yanolja-eeve-korean-instruct-10.8b": "***EEVE Korean*** (11B)",
        }
captions_for_ollama = [
            "Meta: small",
            "Meta: lastest",
            "Microsoft",
            "Mistral AI",
            "Intel",
            "Meta: coding",
            "UW-Madison: vision",
            "Google",
            "Alibaba",
            "Yanolja: 한국어",
]
embed_lists = {"mxbai-embed-large"  : "**MXBai-embed-large** (334M)",
                "nomic-embed-text"  : "**Nomic-embed-text** (137M)",
                "all-minilm"        : "**All-miniLM** (23M)",
            }
captions_embed = [
    "MixedBread AI",
    "Nomic AI",
    "SBERT.net",
]

with tab1:
    llm = st.radio(
        "Download a model",
        options=list(lists_for_ollama.keys()),
        format_func=lambda x: lists_for_ollama[x],
        captions=captions_for_ollama,
        index=None,
        horizontal=True,
    )
    loader_btn = st.button("Download", type="primary")
    if loader_btn:
        # model = llm.split("***")[1]
        model = llm
        # if st.session_state.token['hf_token'] == "":
        #     st.warning("No cached token exists!")
        #     get_token_and_login()
        # else:
        #     success = login_hf()
        #     if not success:
        #         st.warning(f"Cached token is not valid anymore!")
        #         st.session_state.token['hf_token'] = ""
        #         if os.path.isfile(hf_token_cache):
        #             os.remove(hf_token_cache)
        #         get_token_and_login()
        #     else:
        #         logger.info(f"\tStart downloading LLM:{model} from HuggingFace")
        #         logger.info(f"\tHuggingFace Home is set to {HF_HOME}")
        #         hf_token = st.session_state.token['hf_token'] 
        #         download_model(str(model), MODEL_ROOT, hf_token)
        with st.status("Downloading... ", expanded=True) as sts:
            start = time.time()
            ollama.pull(model)
            elapsed_time = time.time()-start
            st.write(f"Done({elapsed_time:.2f} sec).")
        sts.update(label=f"{model} is successfully downloaded.", state="complete")

with tab2:
    dn_lists = ollama.list()['models']
    local_lists = []
    for m in dn_lists:
        # local_lists.append(f"***{m['name'].split(':')[0]}***")
        key = f"{m['name'].split(':')[0]}"
        value = lists_for_ollama.get(key, None)
        if value is not None:
            local_lists.append(value)
    selected_list = st.radio("Make TangoChat smart!", local_lists)
    # aliase_lists, local_lists = list_model(MODEL_ROOT)
    # downloaded_dirs = []
    # for basename, d in zip(aliase_lists, local_lists):
    #     # basename = os.path.basename(d)
    #     d = f'{d}/model.pth'
    #     logger.info(d)
    #     if os.path.isfile(d):
    #         logger.info(f"{d} exists")
    #         size = os.path.getsize(d)/(1024**3)
    #         d = f'***{basename}*** | {d} (**{size:.2f} GB**)'
    #     downloaded_dirs.append(d)
    # selected_list = st.radio("Available models in your local inventory", downloaded_dirs)
    # selected_model = selected_list.split("***")[1]
    if len(local_lists) > 0:
        apply_btn = st.button("Apply ", type="primary")
        if apply_btn:
            logger.info(f"Apply this LLM {selected_list} as Tango+Chat brain")
            # selected_model = selected_list.split('***')[1]
            for k,v in lists_for_ollama.items():
                if v == selected_list:
                    selected_model = k
            st.session_state.brain['local_gen_name'] = selected_model
            st.session_state.brain['local_gen_obj'] = None
            st.session_state.rag = {
                "active": False,
                "url": "",
                "embed": "",
                "retreiver": None,}
            st.rerun()

with tab3:
    remove_list = st.radio("Remove some models to spare storage", local_lists)
    if len(local_lists) > 0:
        # remove_model = remove_list.split("***")[1]
        for k,v in lists_for_ollama.items():
            if v == remove_list:
                remove_model = k
        remove_btn = st.button("Remove", type="primary")
        if remove_btn:
            logger.info(f"Remove this LLM:{remove_list} from local storage")
            ollama.delete(remove_model)
            st.rerun()

with tab4:
    model_name = st.session_state.brain['local_gen_name']
    _model = ""
    if model_name is not None:
        _model = lists_for_ollama[model_name]
    left, middle, right = st.columns(3)
    with left:
        st.subheader("Large Language Model")
        st.text(f"TangoChat is now running by")
        st.markdown(f"{_model}")
    with middle:
        st.subheader("Embedding Model")
        embed_model = st.radio(
            label="**:blue[VECTOR] to LLM ⇇ :red[TEXT] from URL**", 
            options=list(embed_lists.keys()),
            format_func=lambda x: embed_lists[x],
            # captions=captions_embed,
        )
        emb_btn = st.button("PULL & APPLY", type="primary")
        if emb_btn:
            with st.status("Downloading... ", expanded=True) as sts1:
                start = time.time()
                ollama.pull(embed_model)
                elapsed_time = time.time()-start
                st.write(f"Done({elapsed_time:.2f} sec).")
            st.session_state.rag['active'] = True
            st.session_state.rag['embed'] = embed_model
            sts1.update(label=f"{embed_model} is successfully applied.", state="complete")
        if st.session_state.rag['active'] == True:
            st.text(f"TangoChat is now using {embed_model}.")
    with right:
        st.subheader("Retrieve Source")
        _url = st.text_input(
            label="**URL**",
            placeholder = "e.g. https://github.com/ML-TANGO/TANGO",
        )
        st.session_state.rag['url'] = _url
        rt_btn = st.button("Retrieve", type='primary')
        if rt_btn:
            with st.status("Retrieving... ", expanded=False) as sts2:
                start = time.time()
                _emb_model = st.session_state.rag['embed']
                _retriever = load_and_retrieve_docs(_url, _emb_model)
                elapsed_time = time.time()-start
                st.write(f"Done({elapsed_time:.2f} sec).")
                st.session_state.rag['retriever'] = _retriever
                # st.session_state['messages'] = start_state
                switch_ollama_model()
            sts2.update(label=f"successfully retrieved.", state="complete")
        st.text(f"TangoChat retrieves from")
        st.markdown(f"***{_url}***")

# with tab5:
#     finetune_list = st.radio("What would you finetune?", local_lists)
#     if len(local_lists) > 0:
#         finetune_model = finetune_list.split("***")[1]
#         finetune_btn = st.button("FineTune", type="primary")
#         if finetune_btn:
#             logger.info(f"Try improve this LLM:{finetune_list} through finetuning")

# with tab6:
#     deploy_list = st.radio("Select a LLM as TANGO+CHAT brain", local_lists)
#     if len(local_lists) > 0:
#         deploy_model = deploy_list.split("***")[1]
#         deploy_btn = st.button("Deploy", type="primary")
#         if deploy_btn:
#             logger.info(f"Try improve this LLM:{deploy_list} as the Tango+Chat brain")

# apply a local model to TangoChat ---------------------------------------------
# def set_generator():
#     model_name = st.session_state.brain['local_gen_name']
#     args = st.session_state.brain['local_gen_args']
#     logger.info(f"{st.session_state.brain}")
#     if model_name is None:
#         return
#     args.model = model_name
#     # args.prompt = prompt
#     with st.status(label=f"Loading {model_name.upper()}... ", expanded=True) as l_sts:
#         start = time.time()
#         builder_args = BuilderArgs.from_args(args)
#         speculative_builder_args = BuilderArgs.from_speculative_args(args)
#         tokenizer_args = TokenizerArgs.from_args(args)
#         generator_args = GeneratorArgs.from_args(args)
#         TANGOCHAT = Generator(
#             builder_args,
#             speculative_builder_args,
#             tokenizer_args,
#             generator_args,
#             args.profile,
#             args.quantize,
#             args.draft_quantize,
#         )
#         st.balloons()
#         l_sts.update(
#             label=f"Done. ({time.time() - start:.2f} sec)",
#             state="complete",
#         )
#     st.session_state.brain['local_gen_obj'] = TANGOCHAT
#     st.session_state.brain['local_gen_args'] = args
#     st.session_state['messages'] = start_state
#     for msg in st.session_state.messages:
#         if msg['role'] == 'system':
#             msg['content'] = f"Let's talk to **{model_name.upper()}** !!"
#     return





if st.session_state.brain['local_gen_obj'] == None:
    # set_generator()
    switch_ollama_model()

# def show_img_loader():
#     with st.sidebar:
#         image_prompts = st.file_uploader(
#             "Image Prompts",
#             type=["jpeg"],
#             accept_multiple_files=True,
#             key=st.session_state.uploader_key,
#         )

#         for image in image_prompts:
#             st.image(image)

# if st.session_state.img_prompt == 'visible':
#     show_img_loader()

if st.session_state.rag == 'active':
    with st.sidebar:
        url_prompt = st.chat_input(
            placeholder = "Let me know a specific URL to retreive",
            key = "url_prompt",
            )
        st.session_state.rag['url'] = url_prompt

st.divider()    

# title ------------------------------------------------------------------------
st.title(":rainbow[TangoChat]🔎 v1.0")

# parsing messages -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if type(msg['content']) is list:
            for content in msg['content']:
                # if content['type'] == 'image_url':
                #     extension = (
                #         content['image_url'].split(';base64')[0].split("image/")[1]
                #     )
                #     base64_repr = content['image_url'].split('base64,')[1]
                #     st.image(base64.b64decode(base64_repr))
                # else:
                #     st.write(content['text'])
                st.write(content)
        elif type(msg['content']) is dict:
            # if msg['content']['type'] == 'image_url':
            #     st.image(msg['content']['image_url'])
            # else:
            #     st.write(msg['content']['text'])
            st.write(msg['content'])
        elif type(msg['content']) is str:
            st.write(msg['content'])
        else:
            st.write(f"Unhandled content type: {type(msg['content'])}")

# user message input -----------------------------------------------------------
if prompt := st.chat_input():
    if st.session_state.rag['active']:
        _retriever = st.session_state.rag['retriever']
        prompt = get_rag_formatted_prompt(_retriever, prompt)

    user_message = {
        "role": "user",
        # "content": [{"type": "text",      # torch-style
        #             "text": prompt}]
        "content": prompt                   # ollama-style
    }

    if user_message["content"].lower() == 'bye':
        logger.info("Completed\n")
        st.session_state.bye == True
        st.rerun()
    
    # if image_prompts:
    #     for image_prompt in image_prompts:
    #         extension = Path(image_prompt.name).suffix.strip(".")
    #         image_bytes = image_prompt.getvalue()
    #         base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    #         user_message['content'].append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": f"data:image/{extension};base64,{base64_encoded}",
    #             }
    #         )
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)
        # for img in image_prompts:
        #     st.image(img)

    # image_prompts = None
    reset_per_message_state()



    # completion generator -----------------------------------------------------
    with st.chat_message("assistant"), st.status(
        "Generating... ", expanded=True
    ) as status:
        # use api from other frameworks ----------------------------------------
        def get_streamed_completion(completion_generator):
            start = time.time()
            tokcount = 0
            for chunk in completion_generator:
                tokcount += 1
                # yield chunk.choices[0].delta.content  # open-ai style
                if chunk['done']:
                    break
                yield chunk['message']['content']       # ollama style
            
            speed = tokcount / (time.time() - start)
            status.update(
                label=f"Done, averaged {speed:.2f} tokens/second",
                state="complete",
            )

        # dumb brain -----------------------------------------------------------
        _LOREM_IPSUM = """
        Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, 
        sed do eiusmod tempor ncididunt ut labore et dolore magna aliqua. 
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi 
        ut aliquip ex ea commodo consequat.
        """
        _HANGUL_TEXT = """
        이 메세지는 아무런 의미가 없습니다. 
        화면을 위로 올린 후, **DOWNLOAD** 탭에서 모델을 다운로드 하세요. 
        **RUN** 탭에서 원하는 모델을 선택하고, **APPLY** 버튼을 눌러서 TangoChat을 깨우세요😀. 
        TangoChat을 종료하시려면 ***BYE*** 라고 쓰세요.
        """
        def temp_local_stream_data():
            start = time.time()
            for w in _LOREM_IPSUM.split(" "):
                yield w + " "
                time.sleep(0.02)
            
            import pandas as pd
            import numpy as np
            yield pd.DataFrame(
                np.random.randn(5,9),
                columns=["j", "k", "l", "m", "n", "o", "p", "q", "r"]
            )

            for w in _HANGUL_TEXT.split(" "):
                yield w + " "
                time.sleep(0.02)

            # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
            # st.bar_chart(chart_data)

            status.update(
                label=f"Done, elapsed time {time.time()-start:.2f} sec",
                state="complete",
            )

        # use local sources ----------------------------------------------------
        # def get_answer():
        #     TANGOCHAT = st.session_state.brain['local_gen_obj']
        #     args = st.session_state.brain['local_gen_args']
        #     if torch.cuda.is_available():
        #         torch.cuda.reset_peak_memory_stats()
        #     start = time.time()
        #     tokcount = 0
        #     generator_args = GeneratorArgs.from_args(args)
        #     generator_args.prompt = prompt
        #     for chunk in TANGOCHAT.chat(generator_args):
        #         logger.info(f"TANGOCHAT.chat().type = {type(chunk)}")
        #         logger.info(chunk)
        #         tokcount += 1
        #         yield chunk[0]
        #     status.update(
        #         label="Done, averaged {:.2f} tockens/second".format(
        #             tokcount / {time.time() - start}
        #         ),
        #         state="complete",
        #     )

        try:
            # response = st.write_stream(
            #     get_streamed_completion(
            #         client.chat.completions.create(
            #             model="llama3",
            #             messages=st.session_state.messages,
            #             max_tokens=response_max_tokens,
            #             temperature=temperature,
            #             stream=True,
            #         )
            #     )
            # )[0]  # open-ai style
            if st.session_state.brain['local_gen_obj'] is not None:
                logger.info(f"Local brain = {st.session_state.brain['local_gen_name']}")
                response = st.write_stream(
                    get_streamed_completion(
                        ollama.chat(
                            model=st.session_state.brain['local_gen_name'],
                            messages=st.session_state.messages,
                            # format='json',
                            stream=True,
                        )
                    )
                )   # ollama style
            else:
                logger.info("Local brain is not loaded...")
                response = st.write_stream(
                    temp_local_stream_data()
                )[0]    # dumb messages

        except Exception as e:
            response = st.error(f"Exception: {e}")
            logger.warning(f"Exception: {e}")
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
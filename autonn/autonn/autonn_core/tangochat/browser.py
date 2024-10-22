import logging
import os, sys
import time
# import base64
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
import ollama

# logging ----------------------------------------------------------------------
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tangochat():
    # init ---------------------------------------------------------------------
    st.session_state.uploader_key = 0

    def reset_per_message_state():
        _update_uploader_key()

    def _update_uploader_key():
        st.session_state.uploader_key = int(time.time())

    start_state = [
        {
            "role": "system",
            "content": "***WELCOME TO TANGO+CHAT v1.0*** integrated by ETRI",
        },
        {
            "role": "assistant",
            "content": "Hello! Tanguera and Tanguero. How can I help you?"
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

    # huggingface --------------------------------------------------------------
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
    

    # exit ---------------------------------------------------------------------
    if st.session_state.bye:
        logger.info("Completed\n")
        return


    # middle top menu ----------------------------------------------------------
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**DOWNLOAD**", "**RUN**", "DELETE", "R-A-G", "FINETUNE", "DEPLOY"])
    tab1, tab2 = st.tabs(["**DOWNLOAD**", "**RUN**"])
    lists = [
                # "***Llama2-7B-base***", 
                # "***Llama2-7B-chat***", 
                # "***Llama2-13B-chat***", 
                # "***Llama2-70B-chat***", 
                # "***Llama3-8B-base***", 
                # "***Llama3-8B-instruct***", 
                # "***Llama3-70B-instruct***", 
                # "***Llama3.1-8B-base***", 
                # "***Llama3.1-8B-instruct***", 
                # "***Llama3.1-8B-instruct-tune***", 
                # "***Llama3.1-70B-instruct***",
                # "***Llama3.1-70B-instruct-tune***",
                # "***CodeLlama-7B***", 
                # "***CodeLlama-34B***",
                # "***Mistral-7B-v0.1-base***", 
                # "***Mistral-7B-v0.1-instruct***",
                # "***Mistral-7B-v0.2-instruct***", 
                # "***Open-Llama-7B***",
                # "***Tiny-Llama-stories15M***", 
                # "***Tiny-Llama-stories42M***",
                # "***Tiny-Llama-stories110M***",
                "***llama3.2***",
                "***phi3***",
                "***mistral***",
                "***neural-chat***",
                "***starling-lm***",
                "***codellama***",
                "***llava***",
                "***gemma2***",
            ]
    with tab1:
        llm = st.radio(
            "Downloadable Models",
            lists
        )
        loader_btn = st.button("Download", type="primary")
        if loader_btn:
            model = llm.split("***")[1]
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
            local_lists.append(f"***{m['name'].split(':')[0]}***")
        selected_list = st.radio("Availabel modes in your local inventory", local_lists)
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
            apply_btn = st.button("Apply", type="primary")
            if apply_btn:
                logger.info(f"Apply this LLM {selected_list} as Tango+Chat brain")
                selected_model = selected_list.split('***')[1]
                st.session_state.brain['local_gen_name'] = selected_model
                st.session_state.brain['local_gen_obj'] = None
                st.rerun()

    # with tab3:
    #     remove_list = st.radio("You can remove some models to spare storage", local_lists)
    #     if len(local_lists) > 0:
    #         remove_model = remove_list.split("***")[1]
    #         remove_btn = st.button("Remove", type="primary")
    #         if remove_btn:
    #             logger.info(f"Remove this LLM:{remove_list} from local storage")
    #             ollama.delete(remove_model)
    #             dn_lists = ollama.list()['models']
    #             local_lists = []
    #             for m in dn_lists:
    #                 local_lists.append(f"***{m['name'].split(':')[0]}***")

    # with tab4:
    #     rag_list = st.radio("What would you do RAG?", local_lists)
    #     if len(local_lists) > 0:
    #         rag_model = rag_list.split("***")[1]
    #         rag_btn = st.button("RAG", type="primary")
    #         if rag_btn:
    #             logger.info(f"Try improve this LLM:{rag_list} through RAG")
    
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

    # apply a local model to TangoChat -----------------------------------------
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

    def switch_ollama_model():
        model_name = st.session_state.brain['local_gen_name']
        logger.info(f"{st.session_state.brain}")
        if model_name is None:
            return
        st.session_state.brain['local_gen_obj'] = model_name
        st.balloons()
        
        st.session_state['messages'] = start_state
        for msg in st.session_state.messages:
            if msg['role'] == 'system':
                msg['content'] = f"Let's talk to **{model_name.upper()}** !!"
        return

    if st.session_state.brain['local_gen_obj'] == None:
        # set_generator()
        switch_ollama_model()

    st.divider()    



    # title --------------------------------------------------------------------
    st.title(":rainbow[TangoChat]🔎 v1.0")


    # parsing messages ---------------------------------------------------------
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
                    st.write(content['text'])
            elif type(msg['content']) is dict:
                # if msg['content']['type'] == 'image_url':
                #     st.image(msg['content']['image_url'])
                # else:
                #     st.write(msg['content']['text'])
                st.write(msg['content']['text'])
            elif type(msg['content']) is str:
                st.write(msg['content'])
            else:
                st.write(f"Unhandled content type: {type(msg['content'])}")


    # user message input -------------------------------------------------------
    if prompt := st.chat_input():
        user_message = {
            "role": "user",
            # "content": [{"type": "text",      # torch-style
            #             "text": prompt}]
            "content": prompt                   # ollama-style
        }
        if str(prompt).lower() == 'bye':
            logger.info("Completed\n")
            st.session_state.bye == True
            st.rerun()
        # st.info("User messages are received!!!")
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


        # completion generator -------------------------------------------------
        with st.chat_message("assistant"), st.status(
            "Generating... ", expanded=True
        ) as status:
            
            # use api from other frameworks ------------------------------------
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

            # dumb brain -------------------------------------------------------
            _LOREM_IPSUM = """
            Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
            """
            _HANGUL_TEXT = """
            이 메세지는 아무런 의미가 없습니다. 
            스크롤 업하신 후 상단 탭 중 ***RUN*** 에서 원하는 모델을 선택하시고,
            "APPLY" 버튼을 눌러서 TangoChat을 깨우세요😀. 
            ***RUN*** 탭에서 원하는 모델이 없으면 ***DOWNLOAD*** 탭에서 모델을 다운로드 하세요.
            그리고 끝내고 싶으시면 *BYE* 라고 쓰세요. 
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

            # use local sources ------------------------------------------------
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

if __name__ == '__main__':
    # browser page config ------------------------------------------------------
    st.set_page_config(
        page_title="TANGO+Chat",
        page_icon="💃",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "This is a part of TANGO project by ETRI."
        }
    )
    run_tangochat()

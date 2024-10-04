import logging
import os

import time
import base64
from pathlib import Path

import streamlit as st
from loader.download import download_model

# directory --------------------------------------------------------------------
# working directory = /source/autonn_core/tangochat
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
MODEL_ROOT = Path("/shared/models")
CORE_DIR = Path(__file__).resolve().parent.parent # /source/autonn_core
CFG_DIR = CORE_DIR / 'tangochat' / 'common' / 'cfg'
HF_HOME = os.environ.get('HF_HOME', '/root/.cache/huggingface')

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

    # left side menu -----------------------------------------------------------
    with st.sidebar:
        api_base_url = st.text_input(
            label = "OpenAPI base URL",
            value = "http://127.0.0.1:8102/v1",
            help = "The base URL for the OpenAPI to connenct to",
        )

        hf_token_cache = f'{HF_HOME}/token'
        if os.path.isfile(hf_token_cache):
            with open(hf_token_cache, 'r') as f:
                cached_token = f.read()
        else:
            cached_token = ""
        hf_token = st.text_input(
            label = "Hugging Face Access Token",
            key = "huggingface_token",
            value = cached_token, 
            # type = 'password',
            help ='The access token for downloading LLM')

        st.divider()
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.01
        )

        response_max_tokens = st.slider(
            "Max Response Tokens", min_value=10, max_value=1000, value=250, step=10
        )
        if st.button("Reset Chat", type="primary"):
            st.session_state['messages'] = start_state
        
        image_prompts = st.file_uploader(
            "Image Prompts",
            type=['jpeg'],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
        )

        for image in image_prompts:
            st.image(image)

        st.divider()
        if st.button("Complete", type="primary"):
            logger.info("Completed")



    # middle top menu ----------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Download", "List", "Remove", "R-A-G", "FineTune"])
    # lists = get_locally_downloaded_model_lists()
    lists = ["***Mistral-7B-v0.1***", "***Meta-Llama-3.1-8B***", 
            "***Phi-3.5-vision-instruct***", "***gemma-2-2b***"]

    with tab1:
        llm = st.radio(
            "Downloadable Models (requires **HuggingFace token**)",
            ["***Mistral-7B-v0.1***", "***Meta-Llama-3.1-8B***", 
             "***Phi-3.5-vision-instruct***", "***gemma-2-2b***"]
        )
        loader_btn = st.button("Download", type="primary")
        if loader_btn:
            model = llm.split("***")[1]
            if "Mistral" in model:
                model = Path('mistralai') / model
            elif 'Meta' or 'Llama' in model:
                model = Path('meta-llama') / model
            elif 'Phi' in model:
                model = Path('microsoft') / model
            elif 'gemma' in model:
                model = Path('goggle') / model
            else:
                logger.warning(f'Unknown LLM')
                st.warninig(f'Unknown LLM, Can not download {model}')
            
            if not hf_token:
                st.warning("Fill valid access token in the left textbox to login HuggingFace")
                "If you forget it or do not have any, **[Get a Hugging Face Token](https://huggingface.co/settings/tokens)**"
            else:
                logger.info(f"Try to log-in HuggingFace")
                import huggingface_hub
                huggingface_hub.login(token=hf_token, write_permission=True, add_to_git_credential=True)
                logger.info(f"Start downloading LLM:{model} from HuggingFace")
                logger.info(f"{HF_HOME}")
                download_model(str(model), MODEL_ROOT, hf_token)
    
    with tab2:
        selected_list = st.radio("Available Models in your local inventory", lists)
        selected_model = selected_list.split("***")[1]
        apply_btn = st.button("Apply", type="primary")
        if apply_btn:
            logger.info(f"Apply this LLM {selected_model} to Tango+Chat brain")
            # apply_model_to_tangochat()
    
    with tab3:
        remove_list = st.radio("You can remove some models to spare storage", lists)
        remove_model = remove_list.split("***")[1]
        remove_btn = st.button("Remove", type="primary")
        if remove_btn:
            logger.info(f"Remove this LLM:{remove_model} from local storage")
            # remove_model()

    with tab4:
        rag_list = st.radio("What would you do RAG?", lists)
        rag_model = rag_list.split("***")[1]
        rag_btn = st.button("RAG", type="primary")
        if rag_btn:
            logger.info(f"Try improve this LLM:{rag_model} through RAG")
            # rag_model()
    
    with tab5:
        finetune_list = st.radio("What would you finetune?", lists)
        finetune_model = finetune_list.split("***")[1]
        finetune_btn = st.button("FineTune", type="primary")
        if finetune_btn:
            logger.info(f"Try improve this LLM:{finetune_model} through finetuning")
            # tune_model()

    # if "log" not in st.session_state:
    #     st.session_state['log'] = ""

    # for msg in st.session_state.log:
    #     st.write(msg)

    st.divider()    






    # title --------------------------------------------------------------------
    st.title(":rainbow[TangoChat]🔎 v1.0")

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




    # from OpenAI import OpenAI
    # client = OpenAI(
    #     base_url=api_base_url,
    #     api_key="813"
    # )


    # no message ---------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state['messages'] = start_state


    # parsing messages ---------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            if type(msg['content']) is list:
                for content in msg['content']:
                    if content['type'] == 'image_url':
                        extension = (
                            content['image_url'].split(';base64')[0].split("image/")[1]
                        )
                        base64_repr = content['image_url'].split('base64,')[1]
                        st.image(base64.b64decode(base64_repr))
                    else:
                        st.write(content['text'])
            elif type(msg['content']) is dict:
                if msg['content']['type'] == 'image_url':
                    st.image(msg['content']['image_url'])
                else:
                    st.write(msg['content']['text'])
            elif type(msg['content']) is str:
                st.write(msg['content'])
            else:
                st.write(f"Unhandled content type: {type(msg['content'])}")


    # user message input -------------------------------------------------------
    if prompt := st.chat_input():
        user_message = {
            "role": "user",
            "content": [{"type": "text",
                        "text": prompt}]
        }
        st.info("User messages are received!!!")
        if image_prompts:
            for image_prompt in image_prompts:
                extension = Path(image_prompt.name).suffix.strip(".")
                image_bytes = image_prompt.getvalue()
                base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                user_message['content'].append(
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{extension};base64,{base64_encoded}",
                    }
                )
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.write(prompt)
            for img in image_prompts:
                st.image(img)
        
        image_prompts = None
        reset_per_message_state()

        # answering ------------------------------------------------------------
        with st.chat_message("assistant"), st.status(
            "Generating... ", expanded=True
        ) as status:
            
            def get_streamed_completion(completion_generator):
                start = time.time()
                tokcount = 0
                for chunk in completion_generator:
                    tokcount += 1
                    yield chunk.choices[0].delta.content
                
                status.update(
                    label="Done, averaged {:.2f} tockens/second".format(
                        tokcount / {time.time() - start}
                    ),
                    state="complete",
                )
            
            _LOREM_IPSUM = """
            Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
            """
            _HANGUL_TEXT = """
            태산같은 자부심을 갖고, **누운 풀**처럼 자기를 낮추어라. 
            임금처럼 위엄을 갖추고, 구름처럼 한가로워라.
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

                status.update(
                    label=f"Done, elapsed time {time.time()-start:.2f} sec",
                    state="complete",
                )

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
                # )[0]


                response = st.write_stream(
                    temp_local_stream_data
                )[0]

                # import pandas as pd
                # import numpy as np
                # import altair as alt
                # df = pd.DataFrame(
                #     np.random.randn(200,3),
                #     columns=["Earth", "Moon", "Sun"]
                # )
                # c = (
                #     alt.Chart(df).mark_circle().encode(x="Earth", y="Moon", size="Sun", tooltip=["Earth", "Moon", "Sun"])
                # )
                # response = st.write(c)
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
 

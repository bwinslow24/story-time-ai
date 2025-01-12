import base64
import os
import re
import uuid

import replicate
import streamlit as st
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from streamlit.components import v1 as components

from langchain_community.chat_models import ChatPerplexity
from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory

from dotenv import load_dotenv
from transformers import AutoTokenizer

from templates import topic_selection_template, title_selection_template, story_builder_template, ResponseWithOptions, Response
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import langchain
langchain.debug = True
# TODO store messages for chat in separate list so i can display specific messages that i want
# TODO build more robust get tag function to handle response issues
# TODO if response cant be parsed retry prompt until response can be parsed
# TODO build handle_response function
# TODO Add text to speech so we can have the story and options read to us

load_dotenv()
st.set_page_config(layout="wide")
st.title("Story Time AI")

# initialize app
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'topic'

if 'selected_topics' not in st.session_state:
    st.session_state['selected_topics'] = []

config = {"configurable": {"session_id": "abc2"}}


# get llm agent
# repo_id = 'meta-llama/llama-3.1-8b-instruct'
# repo_id = 'meta-llama/Llama-3.1-8B-Instruct'

# repo_id = 'meta-llama/Llama-3.2-3B-Instruct'
# # repo_id = 'google/gemma-7b'
# # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
#
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     # model=repo_id,
#     # max_new_tokens=512,
#     temperature=0.9,
#     # huggingfacehub_api_token=hf_key,
#     # stop_sequences=["\n\n", "END"]
#     # model_kwargs={"tokenizer": tokenizer}
#     # model_kwargs={
#     #     # "max_new_tokens": 256,
#     #     "temperature": 0.9,
#     #     "top_p": 0.95,
#     # }
#
# )

# chat = ChatHuggingFace(llm=llm)
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# chat = ChatPerplexity(temperature=0.9, model="llama-3.1-8b-instruct")
# chat = Replicate(
#     model='meta/meta-llama-3-8b-instruct',
#     # model_kwargs={"temperature": 0.9}
# )
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Build Chains
topic_selection_chain = topic_selection_template | chat

title_selection_chain = title_selection_template | chat
story_builder_chain = story_builder_template | chat
parser_with_options = PydanticOutputParser(pydantic_object=ResponseWithOptions)
parser_with_out_options = PydanticOutputParser(pydantic_object=Response)


topic_selection_cwh = RunnableWithMessageHistory(
    topic_selection_chain,
    lambda session_id: msgs,
    input_messages_key="text",
    history_messages_key="history",
)

title_selection_cwh = RunnableWithMessageHistory(
    topic_selection_chain,
    lambda session_id: msgs,
    input_messages_key="text",
    history_messages_key="history",
)

story_builder_cwh = RunnableWithMessageHistory(
    story_builder_chain,
    lambda session_id: msgs,
    input_messages_key="text",
    history_messages_key="history",
)

chain_map = {'topic': topic_selection_cwh, 'title': title_selection_cwh, 'story': story_builder_cwh}
parser_map = {'topic': parser_with_options, 'title': parser_with_options, 'story': parser_with_options}

def get_chain():
    return chain_map[st.session_state.stage]


# Select 5 topics to start building story
# Once 5 topics are selected select title based on selected topics
# Once title is selected build story. based on title and selected topics

def generate_response(prompt) -> ResponseWithOptions:
    # convo = topic_selection_convo[0]
    # msgs.add_user_message(convo['prompt'])
    # msgs.add_ai_message(convo['response'])
    #
    # return convo['response']

    stage = st.session_state.stage
    response = 'Whoops something went wrong...'
    if stage == 'topic':
        response = generate_topic_response(prompt=prompt)
    elif stage == 'title':
        response = generate_title_response(prompt=prompt)
    elif stage == 'story':
        response = generate_story_response(prompt=prompt)
    print(response)
    parser = parser_map[stage]
    parsed_output = parser.parse(response)
    return parsed_output


def generate_topic_response(prompt):
    chain = get_chain()
    return chain.invoke(
        {'text': prompt},
        config=config
    ).content


def generate_title_response(prompt):
    chain = get_chain()
    topics = ', '.join(st.session_state['selected_topics'])

    return chain.invoke(
        {'text': prompt, 'topics': topics},
        config=config
    ).content


def generate_story_response(prompt):
    chain = get_chain()
    topics = ', '.join(st.session_state['selected_topics'])
    title = st.session_state['selected_title']
    return chain.invoke(
        {'text': prompt, 'topics': topics, 'title': title},
        config=config
    ).content


def get_tag(input_string, tag) -> list[str]:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, input_string, re.DOTALL)
    # print(f'{tag}: {matches}')
    return matches


def get_options(response) -> list[str]:
    book_options = get_tag(input_string=response, tag='OPTIONS')[0]
    book_options_array = book_options.split(',')
    return book_options_array


def get_ai_message(response) -> str:
    return get_tag(input_string=response, tag='AI_MESSAGE')[0]


def select_option(option):
    if st.session_state.stage == 'topic':
        # if in stage topic and topic selected add to selected topic list
        # once topic list has a lenghth of 5 change stage to title and promp
        select_topic(option)
    elif st.session_state.stage == 'title':
        select_title(option)
    elif st.session_state.stage == 'story':
        select_story(option)


def select_topic(topic):
    st.session_state['selected_topics'].append(topic)

    if len(st.session_state['selected_topics']) >= 5:
        st.session_state.stage = 'title'
        prompt = 'Please give me 4 options for a title to the book.'
        # message_container.chat_message("human").write(prompt)
        # st.session_state['chat_messages'].append(HumanMessage(prompt))

        # response = generate_response(prompt=title_prompt)
    else:
        prompt = f'{topic}'
        message_for_chat = {'text': HumanMessage(prompt)}
        write_chat_message(container=message_container, message=message_for_chat)

        # message_container.chat_message("human").write(prompt)

        st.session_state['chat_messages'].append(message_for_chat)

        # response = generate_response(prompt=topic_prompt)

    response = generate_response(prompt=prompt)
    print(f'Response: {response}')

    text_for_chat = response.message
    options = response.options

    # text_for_chat = get_ai_message(response=response)
    # audio_for_chat = text_to_speech(text_for_chat)
    # options = get_options(response=response)

    message_for_chat = {'text': AIMessage(text_for_chat)}
    write_chat_message(container=message_container, message=message_for_chat)

    # message_container.chat_message("ai").write(message_for_chat)
    st.session_state['chat_messages'].append(message_for_chat)
    st.session_state['options'] = options


def select_title(title):
    st.session_state['selected_title'] = title
    st.session_state.stage = 'story'

    prompt = 'Please write my complete story.'
    response = generate_response(prompt=prompt)
    print(f'Response: {response}')
    # story_content = get_ai_message(response=response)
    st.session_state['story_content'] = response.message

    # options = get_options(response=response)
    # message_container.chat_message("ai").write(message_for_chat)
    st.session_state['options'] = []



def select_story(story):
    st.session_state['selected_story'].append(story)


st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)

# Custom CSS for the TTS icon
st.markdown("""
    <style>
    .tts-icon {
        cursor: pointer;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)



def tts_button(message, key):
    button_id = f"tts_button_{key}"
    button_html = f"""
    <button id="{button_id}" onclick="speak('{message}')">🔊</button>
    <script>
    function speak(text) {{
        const utterance = new SpeechSynthesisUtterance(text);
        speechSynthesis.speak(utterance);
    }}
    </script>
    """
    components.html(button_html, height=30)


def text_to_speech(text, language="en"):
    target_voice_path = '/media/rowyourboat.mp3'

    output = replicate.run(
        "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
        input={
            "text": text,
            "language": language,
            "speaker": open(target_voice_path, "rb"),
            "speed": 1.0  # Adjust speed if needed
        }
    )
    return output


def write_chat_message(container, message):
    with container.chat_message(message['text'].type):
        st.write(message['text'].content)
        # if message['text'].type == 'ai':
        #     if st.button("🔊", key=f"tts_{hash(message['text'].content)}"):
        #         st.audio(message['audio'], format='audio/mp3')


chat_col, story_col = st.columns(2)
# React to user input
with chat_col:
    with st.container(height=500):
        message_container = st.container(height=400)
        if 'chat_messages' not in st.session_state:
            starting_message = 'What should we write about today?'
            response = generate_response(prompt=starting_message)
            message_for_chat = response.message
            options = response.options

            # message_for_chat = get_ai_message(response=response)
            # # audio_for_chat = text_to_speech(text=message_for_chat)
            # options = get_options(response=response)

            # message_container.chat_message("ai").write(message_for_chat)
            st.session_state['chat_messages'] = [
                {
                    'text': AIMessage(message_for_chat),
                 # 'audio': audio_for_chat
                 }
            ]
            st.session_state['options'] = options

        for msg in st.session_state.chat_messages:
            write_chat_message(container=message_container, message=msg)


        if prompt := st.chat_input():
            # Add user input to chat messages
            user_input_message = {'text': HumanMessage(prompt)}
            write_chat_message(container=message_container, message=user_input_message)

            st.session_state['chat_messages'].append(user_input_message)
            # generate response for user input
            response = generate_response(prompt=prompt)
            print(f'Response: {response}')
            if st.session_state.stage == 'story':
                st.session_state['story_content'] = response.message
            else:
                # message_for_chat = get_ai_message(response=response)
                # audio_for_chat = text_to_speech(text=message_for_chat)
                # options = get_options(response=response)

                message_for_chat = response.message
                # audio_for_chat = text_to_speech(text=message_for_chat)
                options = response.options

                # update current options
                st.session_state.options = options

                ai_chat_message = {'text': AIMessage(message_for_chat)}
                st.session_state['chat_messages'].append(ai_chat_message)





    with st.container():
        if 'options' in st.session_state:
            options = st.session_state.options

            num_rows = (len(options) + 1) // 2

            # Loop through the rows
            for i in range(num_rows):
                # Create two columns for each row
                col1, col2 = st.columns(2)

                # Calculate the index for the options
                index1 = i * 2
                index2 = index1 + 1

                # Add a button to the first column if the index is within the range of options
                if index1 < len(options):
                    with col1:
                        if st.button(options[index1], on_click=select_option, args=(options[index1],), use_container_width=True):
                            pass

                # Add a button to the second column if the index is within the range of options
                if index2 < len(options):
                    with col2:
                        if st.button(options[index2], on_click=select_option, args=(options[index2],),use_container_width=True):
                            pass

with story_col:
    if 'story_content' in st.session_state:
        st.write(st.session_state['story_content'])

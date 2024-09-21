import os
import re

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory

from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv

# from transformers import pipeline
from PIL import Image

from templates import topic_selection_template, title_selection_template, story_builder_template
from test_prompts import topic_selection_convo

# from transformers import pipeline

# TODO store messages for chat in separate list so i can display specific messages that i want
# TODO build more robust get tag function to handle response issues
# TODO if response cant be parsed retry prompt until response can be parsed
# TODO build handle_response function

load_dotenv()
st.set_page_config(layout="wide")
st.title("Story Builder")

# initialize app
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'topic'

if 'selected_topics' not in st.session_state:
    st.session_state['selected_topics'] = []




config = {"configurable": {"session_id": "abc2"}}
chat = ChatPerplexity(temperature=0.9, model="llama-3.1-8b-instruct")
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Build Chains
topic_selection_chain = topic_selection_template | chat
title_selection_chain = title_selection_template | chat
story_builder_chain = story_builder_template | chat

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



def get_chain():
    return chain_map[st.session_state.stage]


# Select 5 topics to start building story
# Once 5 topics are selected select title based on selected topics
# Once title is selected build story. based on title and selected topics

def generate_response(prompt):
    # convo = topic_selection_convo[0]
    # msgs.add_user_message(convo['prompt'])
    # msgs.add_ai_message(convo['response'])
    #
    # return convo['response']

    stage = st.session_state.stage

    if stage == 'topic':
        return generate_topic_response(prompt=prompt)
    elif stage == 'title':
        return generate_title_response(prompt=prompt)
    elif stage == 'story':
        return generate_story_response(prompt=prompt)

    return 'Whoops something went wrong...'


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
        message_container.chat_message("human").write(prompt)
        st.session_state['chat_messages'].append(HumanMessage(prompt))

        # response = generate_response(prompt=topic_prompt)

    response = generate_response(prompt=prompt)
    print(f'Response: {response}')
    message_for_chat = get_ai_message(response=response)
    options = get_options(response=response)
    message_container.chat_message("ai").write(message_for_chat)
    st.session_state['chat_messages'].append(AIMessage(message_for_chat))
    st.session_state['options'] = options


def select_title(title):
    st.session_state['selected_title'] = title
    st.session_state.stage = 'story'

    prompt = 'Please write my complete story.'
    response = generate_response(prompt=prompt)
    print(f'Response: {response}')
    # story_content = get_ai_message(response=response)
    st.session_state['story_content'] = response

    # options = get_options(response=response)
    # message_container.chat_message("ai").write(message_for_chat)
    st.session_state['options'] = []



def select_story(story):
    st.session_state['selected_story'].append(story)





chat_col, story_col = st.columns(2)
# React to user input
with chat_col:
    with st.container(height=500):
        message_container = st.container(height=400)
        if 'chat_messages' not in st.session_state:
            starting_message = 'What should we write about today?'
            response = generate_response(prompt=starting_message)
            message_for_chat = get_ai_message(response=response)
            options = get_options(response=response)
            # message_container.chat_message("ai").write(message_for_chat)
            st.session_state['chat_messages'] = [AIMessage(message_for_chat)]
            st.session_state['options'] = options

        for msg in st.session_state.chat_messages:
            # for msg in msgs.messages:
            message_container.chat_message(msg.type).write(msg.content)

            # if msg.type == 'ai':
            #     # message_container.chat_message(msg.type).write(get_ai_message(msg.content))
            # else:
            #     message_container.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input():
            # Add user input to chat messages
            message_container.chat_message("human").write(prompt)
            st.session_state['chat_messages'].append(HumanMessage(prompt))
            # generate response for user input
            response = generate_response(prompt=prompt)
            print(f'Response: {response}')
            if st.session_state.stage == 'story':
                st.session_state['story_content'] = response

            else:
                message_for_chat = get_ai_message(response=response)
                options = get_options(response=response)
                # update current options
                st.session_state.options = options
                message_container.chat_message("ai").write(message_for_chat)
                st.session_state['chat_messages'].append(AIMessage(message_for_chat))





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
                        if st.button(options[index1], on_click=select_option, args=(options[index1],)):
                            pass

                # Add a button to the second column if the index is within the range of options
                if index2 < len(options):
                    with col2:
                        if st.button(options[index2], on_click=select_option, args=(options[index2],)):
                            pass

with story_col:
    if 'story_content' in st.session_state:
        st.write(st.session_state['story_content'])

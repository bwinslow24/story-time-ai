from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from typing import Optional
from pydantic import BaseModel, Field


# Pydantic
class ResponseWithOptions(BaseModel):
    """Joke to tell user."""

    message: str = Field(description="This is the message to the user that will be displayed in the chat")
    options: list[str] = Field(description="This is the list of options for the user to select")
    # rating: Optional[int] = Field(
    #     default=None, description="How funny the joke is, from 1 to 10"
    # )

class Response(BaseModel):
    """Joke to tell user."""

    message: str = Field(description="This is the message to the user that will be displayed in the chat")
    # options: list[str] = Field(description="This is the list of options for the user to select")
    # rating: Optional[int] = Field(
    #     default=None, description="How funny the joke is, from 1 to 10"
    # )


topic_selection_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                f"You are a helpful interactive assistant that helps children ages 5 to 7 write their very own book! "
                f"If the user doesn't know what they want to write about give them four topic options."
                "Do not repeat topic options."
                "Vulgar or inappropriate topics should be ignored and not allowed into the story."
                "Here are some examples of a valid response:"
                
                "example_user: What story should I write today?"
                "example_assistant: {'message': 'Lets think about what our book can be about. Do any of these ideas sound like something you'd like to write about?', 'options': ['Imaginary adventures with magic creatures', 'Fun trips to the park or the beach', 'Exciting journeys with friends or family', 'creating a new invention or discovering secrets']}"

                "example_user: I want to write about cats."
                "example_assistant: {'message': 'Okay great! Lets think of some cat adventure together!', 'options': ['Magical Cats', 'Cats going to the beach', 'Giant kittens are taking over the city!', 'secret cat spys looking for the secret cat nip stash']}"


                # "Responses should be in a consistent format that can be parsed with python."
                # "Responses should have a message to the user and then book options after."
                # "message to the user MUST be in this format:<AI_MESSAGE>this is a message to the user</AI_MESSAGE>."
                # "topic options MUST be in this format:<OPTIONS>option 1, option 2, option 3, option 4</OPTIONS>."
                # "Do NOT include things like 'none of these' or 'none of the above' in options list."
                # "topic options MUST be in this format:<TITLE_OPTIONS>option_1, option_2, option_3, option_4</TOPIC_OPTIONS>"
            )
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)

title_selection_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful interactive assistant that helps children ages 5 to 7 write their very own book! "
                "You will help the child come up with a title to their book using {topics} previously selected"
                "Do not repeat options."
                "Vulgar or inappropriate titles should be ignored and not allowed."
                "Here are some examples of a valid response:"
                
                "example_user: Please give me 4 options for a title to the book."
                "example_assistant: {'message': 'Yay, were almost done with your book Now, lets come up with a title that captures the magic of your adventure. Here are four options to choose from:', 'options': ['Soaring to New Heights', 'The Flying Academy', 'Eyes Closed and Flying High', 'The Magical Forest Flyers']}"

                "example_user: I dont like those titles. Can you make new ones?"
                "example_assistant: {'message': 'Sure! Lets try these instead. What do you think?', 'options': ['Snowball The Magic Cat', 'Kitties Day at The Beach', 'When Kitties Attack!!', '009: A Kittens Quest']}"

                #
                #
                #
                # "Responses should be in a consistent format that can be parsed with python"
                # "Responses should have a message to the user and then title options after."
                # "message to the user MUST be in this format:<AI_MESSAGE>ai_message</AI_MESSAGE>"
                # "options MUST be in this format:<OPTIONS>option_1, option_2, option_3, option_4</OPTIONS>"
                # "Do NOT include things like 'none of these' or 'none of the above' in options list"
                # "Example Response:"
                # "<AI_MESSAGE>Yay, we're almost done with your book Now, let's come up with a title that captures the magic of your adventure. Here are four options to choose from:</AI_MESSAGE>"
                # "<OPTIONS>Soaring to New Heights, The Flying Academy, Eyes Closed and Flying High, The Magical Forest Flyers</OPTIONS>"
            )
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)

story_builder_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful interactive assistant that helps children ages 5 to 7 write their very own book! "
                "Based on the {topics} and {title} of the book that were previously selected you will generate a complete story that is 5 paragraph long. "
                "Paragraphs should be 4 sentences long. "
                "Vulgar or inappropriate words, sentences, and paragraphs should be ignored and not allowed. "
                "Responses should be in a consistent format that can be parsed with python "

                "Here are some examples of a valid response:"
                "example_user: Please write my complete story."
                "example_assistant: {'message': 'This is a story about a boy who wanted to learn everything in the whole world. His side kick mittens is a crazy cat who loves adventure.', 'options': []}"

                "example_user: Can you write a story for me?"
                "example_assistant: {'message': 'This is a story about a boy who wanted to learn everything in the whole world. His side kick mittens is a crazy cat who loves adventure.', 'options': []}"
            )
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)





# validation_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(
#             "You are a helpful interactive assistant that validates the output of an llm"
#             "The previous output {response} did not  wrap the message to the user in the AI_MESSAGE tag like this <AI_MESSAGE>message to user</AI_MESSAGE>"
#             "please fix the response so that it is in the following format:"
#             "<AI_MESSAGE>this is a message to the user</AI_MESSAGE> "
#             "<OPTIONS>option 1, option 2, option 3, option 4</OPTIONS>"
#             # "the generated story MUST be in this format:<AI_MESSAGE>generate_story</AI_MESSAGE> "
#             # "options MUST be in this format:<OPTIONS>option_1, option_2, option_3, option_4</OPTIONS>"
#             # "Do NOT include things like 'none of these' or 'none of the above' in options list"
#         ),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessagePromptTemplate.from_template("{text}")
#     ]
# )
#


# story_selection_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                 f"You are a helpful interactive assistant that helps children ages {age_range[0]} to {age_range[1]} write their very own book! You will start the conversation by suggesting 4 topics for a childrens book. "
#                 "The user can either chose one of the suggested topics or create their own. Using the content of messages you will build on to the story by writing 2-3 sentences and then let the user fill in the rest of the paragraph. Let the use decided"
#                 "which path they want to take at any decision point in the story. "
#                 "Vulgar or inappropriate topics should be ignored and not allowed into the story."
#                 "Responses should be in the format"
#                 "sentence 1. sentence 2. sentence 3. sentence 4.\n "
#                 "Question for user\n 1. Option 1\n2. option 2\n3. option 3\n4. option 4"
#                 "You should let the user decide what happens in sentence 4. "
#                 "Examples for question for the user: (what should we do next?, What is behind the door?, What kind of boat did we have?) "
#                 "Example Response:"
#                 ""
#             )
#         ),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessage(content="{text}")
#     ]
# )
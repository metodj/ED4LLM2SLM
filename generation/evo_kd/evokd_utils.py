import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.gpt4 import gpt4_call


def evo_kd_prompt(inputs, n_gen: int = 2):
    correct_questions, correct_answers, wrong_questions, wrong_answers = (
        inputs["correct_questions"],
        inputs["correct_answers"],
        inputs["wrong_questions"],
        inputs["wrong_answers"],
    )
    assert len(correct_questions) == len(correct_answers) == 2
    assert len(wrong_questions) == len(wrong_answers) == 2
    prompt = "There is a language model that is solving math problems. I have some problems that the model correctly solved, and some other problems that the model solved wrong. The model solved wrong: \n"
    for i, qa in enumerate(zip(wrong_questions, wrong_answers)):
        q, a = qa
        prompt += f"Example {i + 1}:\nQ: {q}\nA: {a}\n\n"
    prompt += "The model solved correctly: \n"
    for i, qa in enumerate(zip(correct_questions, correct_answers)):
        q, a = qa
        prompt += f"Example {i + 1}:\nQ: {q}\nA: {a}\n\n"

    prompt += f"I want you to summarize the patterns of problems that are prone to be incorrectly solved by the model, based on the examples I provided above. Then I want you to generate {n_gen} problems which you think the model might solve wrong and 2 problems with you think the model might solve correctly.\n"
    prompt += "Provide your answer in the following template (make sure to follow the format precisely):\nSummary of patterns:\nThe model may incorrectly predict:\n"
    for i in range(n_gen):
        prompt += f"Q: ... \n"
    prompt += "The model may correctly predict:\n"
    for i in range(n_gen):
        prompt += f"Q: ... \n"
    return prompt


def evo_kd_extract_questions(text):
    questions = {"hard": [], "easy": []}

    # Split the text into sections
    incorrect_section = re.search(
        r"The model may incorrectly predict:(.*?)(?=\nThe model may correctly predict:|$)",
        text,
        re.DOTALL,
    )
    correct_section = re.search(
        r"The model may correctly predict:(.*)", text, re.DOTALL
    )

    # Function to find questions in a given section
    def find_questions(section):
        if section:
            return re.findall(r"Q: (.*?)(?=\nQ:|$)", section.group(1), re.DOTALL)
        else:
            return []

    # Extract questions for incorrect predictions
    if incorrect_section:
        questions["hard"] = find_questions(incorrect_section)

    # Extract questions for correct predictions
    if correct_section:
        questions["easy"] = find_questions(correct_section)

    return questions


evo_kd_questions = lambda x: gpt4_call(
    x, assistant_message=None, user_func=evo_kd_prompt
)

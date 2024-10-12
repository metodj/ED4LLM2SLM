import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.gpt4 import gpt4_call


def ama_extract_questions(text, n_questions=1):
    text = text.split("<target>")
    questions = []
    for i in range(1, len(text)):
        try:
            questions.append(text[i].split("<question>")[1].strip())
        except IndexError:
            print(f"Error extracting question {i}")
            print(text[i])
            continue
    return questions[:n_questions]


def ama_prompt(row):
    return f"{AMA_PROMPT}\nQ: {row['question']}\nAnswer: {int(row['answer'].split('####')[1])}"


ama_questions = lambda x: gpt4_call(x, assistant_message=None, user_func=ama_prompt)


AMA_PROMPT = """
Your goal is to create multiple word problems from a given word problem and its answer. First convert the question of the word problem into a statement. Then for each number in the converted problem create a new word problem. Here are some examples:

Example 1: 
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: 72
Replacing question with statement: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia sold altogether 72 clips in April and May.
All questions:
<target> 48
<question> Natalia sold clips to some of her friends in April, and then she sold half as many clips in May. Natalia sold altogether 72 clips in April and May. How many clips did she sell in April?
<target> half
<question> Natalia sold clips to 48 of her friends in April, and then she sold some clips in May. Natalia sold altogether 72 clips in April and May. What is the ratio of the number clips sold in April to number clips sold in May?
<target> 72
<question> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Example 2: 
Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: 10
Replacing question with statement: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. She earned $10.
All questions:
<target> 12
<question> Weng earns a certain amount per hour for babysitting. Yesterday, she just did 50 minutes of babysitting and earned 10. How much does she earn per hour?
<target> 50
<question> Weng earns 12 an hour for babysitting. Yesterday, she just did some babysitting and earned 10. How much time did she spend on babysitting?
<target> 10
<question> Weng earns 12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Example 3: 
Q: Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: 5
Replacing question with statement: Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents gave her twice as much as her parents. She needs 5 more to buy the wallet.
All questions:
<target> 100
<question> Betty is saving money for a new wallet. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. She needs 5 more to buy the wallet. What is the cost of the wallet?
<target> half
<question> Betty is saving money for a new wallet which costs 100. She has some money saved, her parents decided to give her 15, and her grandparents gave her twice as much as her parents. Now, Betty needs 5 more to buy the wallet. What is the ratio of the money Betty have saved initially to the cost of wallet?
<target> 15
<question> Betty is saving money for a new wallet which costs 100. She has half of the money she needs, her parents decided to give her some money, and her grandparents gave her twice as much as her parents. Now, Betty needs 5 more to buy the wallet. How much money did her parents give her?
<target> twice
<question> Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents also chipped in. Now, Betty needs 5 more to buy the wallet. What is the ratio of the amount given by her grandparents to the amount given by her parents?
<target> 5
<question> Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

Now solve this:
Example 4:
"""

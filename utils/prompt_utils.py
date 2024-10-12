GSM_SYSTEM_PROMPT = "Here is a question that describes a math problem. Write a response that appropriately and accurately solves the math problem. Provide a single response."
GPT4_ORCA_MATH_SYSTEM_PROMPT = "As an expert Math teacher, your role is to evaluate a student’s answer to a word problem. The problem is accompanied by a correct solution provided by the problem setter. It is important to remember that there may be various methods to solve a word problem, so the student’s steps might not always align with those in the problem setter’s solution. However, the final answer, typically a number, should be unique and match the problem setter’s answer. Your task involves analyzing the student’s solution to identify any mistakes and determine whether the answer can be modified to correct the error. If the student’s answer is unfixable, consider creating practice problems to help improve their understanding. Use the following format: Error Analysis: In one sentence, extract the final answer from the problem setter’s solution and compare it with the student’s answer. Do they match? Final Verdict: Correct/Incorrect"
COT = "Solve the problem step-by-step."
GPT4_MATH_LABEL_SYSTEM_PROMPT = "Here is a question that describes a math problem. Write a response that appropriately and accurately solves the math problem."


def orca_math_label_template(question, answer, generation):
    return f"Question: {question} \n Problem Setter’s answer: {answer} \n Student’s answer: {generation}"

def create_prompt_mistral_harness(row):
  """
  Prompt template for Mistral7B-instruct model: 
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    https://huggingface.co/docs/transformers/main/en/chat_templating
  """

  message = [
      {"role": "user", "content": f"Question: {row['question']}\nAnswer:"},
      {"role": "assistant", "content": row['answer']}
  ]

  return message

def create_prompt_test_mistral_harness(dataset, question_token="question"):
  """
  Prompt template for Mistral7B-instruct model: 
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    https://huggingface.co/docs/transformers/main/en/chat_templating
  """

  messages = []
  for i in range(len(dataset)):
    messages.append([
      {"role": "user", "content": f"Question: {dataset[i][question_token]}\nAnswer:"},
  ])

  return messages

def create_prompt_mistral_harness_math(row):
  """
  Prompt template for Mistral7B-instruct model: 
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    https://huggingface.co/docs/transformers/main/en/chat_templating
  """

  message = [
      {"role": "user", "content": f"Question: {row['problem']}\nAnswer:"},
      {"role": "assistant", "content": row['solution']}
  ]

  return message

def create_prompt_test_mistral_harness_math(dataset):
  """
  Prompt template for Mistral7B-instruct model: 
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    https://huggingface.co/docs/transformers/main/en/chat_templating
  """

  messages = []
  for i in range(len(dataset)):
    messages.append([
      {"role": "user", "content": f"Question: {dataset[i]['problem']}\nAnswer:"},
  ])

  return messages



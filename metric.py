import re
import math
import numpy as np


def extract_answer(response, verbose=False):
    if verbose:
        print("*" * 100)
        print("Origin response: ", response)
    if "```json" in response:
        pred = eval(re.findall(r"```json\s*(.*?)\s*```", response, re.DOTALL)[-1])
    elif "```" in response:
        pred = eval(re.findall(r"```\s*(.*?)\s*```", response, re.DOTALL)[-1])
    elif "`" in response:
        pred = eval(re.findall(r"`\s*(.*?)\s*`", response, re.DOTALL)[-1])
    elif "(" in response.replace("[", "(").replace("]", ")"):
        pred = eval(
            re.findall(
                r"\(\s*(.*?)\s*\)",
                response.replace("[", "(").replace("]", ")"),
                re.DOTALL,
            )[-1]
        )
    elif "</box>" in response:
        pred = [
            int(x)
            for x in re.findall(r"<box>\s*(.*?)\s*</box>", response, re.DOTALL)[
                -1
            ].split(" ")
        ]
    elif ":" in response:
        pred = eval(response.split(":")[1].strip())
    else:
        pred = eval(response)
    if type(pred) in [int, float]:
        pred = [pred]
    if verbose:
        print("Extracted response: ", pred)
    return pred


def click_acc_metric(gt, pred):
    x1, y1, x2, y2 = gt
    if len(pred) != 2:
        pred = [(pred[0]+pred[2])/2, (pred[1]+pred[3])/2]
    x, y = pred
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 1
    else:
        return 0


def iou_metric(gt, pred, thres=0.5):
    # Calculate intersection coordinates
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])

    # Calculate intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate areas of both bounding boxes
    boxAArea = (gt[2] - gt[0]) * (gt[3] - gt[1])
    boxBArea = (pred[2] - pred[0]) * (pred[3] - pred[1])

    # Calculate IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return 1 if IoU exceeds threshold, 0 otherwise
    if iou > thres:
        return 1
    else:
        return 0


def extract_numbers(llm, question, answer):
    try:
        res = [eval(answer)]
    except:
        prompt_template = """- Goal: Extract numerical values from a given text answer in response to a question.
        
- Constrains: 
1. Only return the numerical values that answer the question excluding all the explanation.
2. Convert non-Arabic numerals to Arabic numerals, including k(1e3), thousand(1e3), m/M(1e6), etc.
3. If the answer is a numerical range, return the first and last values. 
3. The answer is either a numerical range or a value. 
4. Handle of exponential notations.

- OutputFormat: The extracted numerical values should be returned in Arabic numerals, and if it is a range, return two values separated by a ~. If no numerical values are found, return None.

- Example 1: 
    Question: What is the y-axis ranges at x=10?
    Answer: The y-axis ranges at x=10 is approximately -2^-3-3 × 10^4, which equals to -2^-3-3e4, or -2^-3~30,000.
    Response: -2^-3~3e4

- Example 2: 
    Question: What is the value of a?
    Answer: 10⁻³ or 0.1% (in units of 10³).
    Response: 0.001

- Example 3: 
    Question: What is the value of b?
    Answer: 3 thousand to 5m.
    Response: 3e3~5e6

- Now start the task:
    Question: {question}
    Answer: {answer}
    Your Response (no explanation!): 
"""
        prompt = prompt_template.format(question=question, answer=answer)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        response = llm(messages)
        response = response.strip().lower().replace("\n", "")
        res = extract_numbers_postprocess(response)
    return res


def extract_numbers_postprocess(response):
    if type(response) != str:
        response = response["choices"][0]["message"]["content"]
    if "response:" in response:
        response = response.split("response:")[1].strip()
    if ":" in response:
        response = response.split(":")[1].strip()
    vals = response.split("~")

    res = []
    for val in vals:
        if "^" in val:
            val = val.replace("^", "**")
        try:
            val = eval(val)
        except:
            pass
        res.append(val)
    print(f"Response: {response}")
    return res


def compute_value_score(answer_numbers, pred_numbers, axis_type, minmax, thres=0.1):
    scores = []
    if len(answer_numbers) == 0:
        raise ValueError("answer_numbers is empty")
    if len(answer_numbers) != len(pred_numbers):
        if len(answer_numbers) == 1:
            if axis_type == "log":
                pred_numbers = [pow(np.prod(pred_numbers), 1 / len(pred_numbers))]
            else:
                pred_numbers = [np.mean(pred_numbers)]
        else:
            return None
    for answer_number, pred_number in zip(answer_numbers, pred_numbers):
        if type(answer_number) == str:
            score = float(answer_number == pred_number)
        else:
            if type(pred_number) == str and pred_number.isdigit():
                pred_number = eval(pred_number)
            if type(pred_number) in [str, list, tuple]:
                return None
            else:
                if axis_type == "log":
                    range = math.log10(minmax[1]) - math.log10(minmax[0])
                    if pred_number <= 0:
                        diff = 1
                    else:
                        answer_number = math.log10(answer_number)
                        pred_number = math.log10(pred_number)
                        diff = abs((answer_number - pred_number) / range)
                else:
                    range = minmax[1] - minmax[0]
                    diff = abs((answer_number - pred_number) / range)
                if diff > thres:
                    diff = 1
                score = 1 - diff
        scores.append(score)
    return max(0, np.mean(scores))


if __name__ == "__main__":
    response = "<box>1 2</box><box>10</box>"
    # Extract all JSON content and clean the strings before evaluation
    pred = eval(re.findall(r"<box>\s*(.*?)\s*</box>", response, re.DOTALL)[-1])
    print(pred)

SCORE_SYSTEM_PROMPT="You function as an insightful assistant whose role is to assist individuals in making decisions that align with their personal preferences. Use your understanding of their likes, dislikes, and inclinations to provide relevant and thoughtful recommendations."
SCORE_PROBLEM_PROMPT_TEMPLATE="""[User Question] You will be presented with several plot summaries, each accompanied by a review from the same critic. Your task is to analyze both the plot summaries and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new plot and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

Please follow the above critic and give a review for the given plot. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Plot]
{plot}
[The End of Plot]
"""

SCORE_CASE_TEMPLATE="""[The Start of Plot {n}]
{plot}
[The End of Plot {n}]
[Review]
```json
{{
  "Review": "{review}",
  "Score": {score}
}}
```
"""



RANK_SYSTEM_PROMPT="You function as an insightful assistant whose role is to assist individuals in making decisions that align with their personal preferences. Use your understanding of their likes, dislikes, and inclinations to provide relevant and thoughtful recommendations."
RANK_PROBLEM_PROMPT_TEMPLATE="""[User Question] You will be presented with two separate plot summaries and the response from one user. Here is an example to describe this user preference:

[The Start of User Preference]
{icl_example}
[The End of User Preference]

[User Question] Based on the above user preference, compare the following two plots:

[The Start of Plot A]
{plan1}
[The End of Plot A]

[The Start of Plot B]
{plan2}
[The End of Plot B]

The response should use this specific format: 
```json
{{
  "Choice": <"Plot A" or "Plot B" or "N/A">,
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.
"""

RANK_CASE_TEMPLATE="""[The Start of Plot A]
{plan1}
[The End of Plot A]
[The Start of Plot B]
{plan2}
[The End of Plot B]
[Response]
{completion}
"""

RANK_COMPLETION = """```json
{{
  "Choice": "{choice}",
}}
```"""
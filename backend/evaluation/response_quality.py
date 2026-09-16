import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "response_quality_results.csv"
)


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

ENV_FILE = os.path.join(
    BACKEND_DIR,
    ".env"
)

load_dotenv(ENV_FILE)


# ============================================================
# 3. GROQ API
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in backend/.env"
    )

client = Groq(
    api_key=api_key
)


# ============================================================
# 4. JUDGE MODEL
# ============================================================

JUDGE_MODEL = "openai/gpt-oss-20b"


# ============================================================
# 5. LOAD INPUT
# ============================================================

print("=" * 70)
print("RESPONSE QUALITY EVALUATION")
print("=" * 70)

print("\nJudge model:")
print(JUDGE_MODEL)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nTotal responses found: {len(df)}"
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "evaluation_id",
    "conversation_id",
    "query",
    "predicted_intent",
    "response"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 7. JUDGE FUNCTION
# ============================================================

def judge_response(
    query,
    predicted_intent,
    response
):

    prompt = f"""
Evaluate this AI customer-support response.

Customer:
{query}

Intent:
{predicted_intent}

AI response:
{response}

Give five scores from 1 to 5:

Correctness
Relevance
Helpfulness
Grounding
Overall

Return ONLY ONE LINE:

4|5|4|4|4|Short reason

Rules:
- First five values must be integers 1 to 5.
- Reason must be less than 15 words.
- Do not use JSON.
- Do not use markdown.
- Do not explain anything.
- Return only the required line.
"""

    for attempt in range(3):

        try:

            result = client.chat.completions.create(

                model=JUDGE_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0,

                max_tokens=300,

                # IMPORTANT FOR GPT-OSS
                reasoning_effort="low"
            )


            # ------------------------------------------------
            # CHECK CHOICES
            # ------------------------------------------------

            if not result.choices:

                print(
                    "    No choices returned"
                )

                time.sleep(2)

                continue


            choice = result.choices[0]


            # ------------------------------------------------
            # PRINT FINISH REASON
            # ------------------------------------------------

            print(
                f"    Finish reason: "
                f"{choice.finish_reason}"
            )


            # ------------------------------------------------
            # GET MESSAGE CONTENT
            # ------------------------------------------------

            text = choice.message.content


            # ------------------------------------------------
            # HANDLE EMPTY RESPONSE
            # ------------------------------------------------

            if text is None or not str(text).strip():

                print(
                    f"    Empty response "
                    f"(attempt {attempt + 1}/3)"
                )

                time.sleep(2)

                continue


            text = str(text).strip()


            # ------------------------------------------------
            # PRINT RAW OUTPUT
            # ------------------------------------------------

            print(
                f"    Judge output: {text}"
            )


            # ------------------------------------------------
            # FIND VALID LINE
            # ------------------------------------------------

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            valid_line = None

            for line in lines:

                parts = line.split("|", 5)

                if len(parts) != 6:
                    continue

                try:

                    scores = [
                        int(parts[0].strip()),
                        int(parts[1].strip()),
                        int(parts[2].strip()),
                        int(parts[3].strip()),
                        int(parts[4].strip())
                    ]

                except ValueError:

                    continue


                if all(
                    1 <= score <= 5
                    for score in scores
                ):

                    valid_line = line

                    break


            # ------------------------------------------------
            # INVALID OUTPUT
            # ------------------------------------------------

            if valid_line is None:

                print(
                    "    Could not parse judge output."
                )

                time.sleep(2)

                continue


            # ------------------------------------------------
            # PARSE
            # ------------------------------------------------

            parts = valid_line.split("|", 5)


            correctness = int(
                parts[0].strip()
            )

            relevance = int(
                parts[1].strip()
            )

            helpfulness = int(
                parts[2].strip()
            )

            grounding = int(
                parts[3].strip()
            )

            overall = int(
                parts[4].strip()
            )

            reason = parts[5].strip()


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            return {
                "correctness": correctness,
                "relevance": relevance,
                "helpfulness": helpfulness,
                "grounding": grounding,
                "overall": overall,
                "reason": reason
            }


        except Exception as e:

            print(
                f"    Judge attempt "
                f"{attempt + 1}/3 failed: {e}"
            )

            time.sleep(2)


    # ========================================================
    # FAILED
    # ========================================================

    return {
        "correctness": None,
        "relevance": None,
        "helpfulness": None,
        "grounding": None,
        "overall": None,
        "reason": "Judge failed after 3 attempts"
    }


# ============================================================
# 8. LOAD EXISTING RESULTS
# ============================================================

existing = pd.DataFrame()

if os.path.exists(OUTPUT_FILE):

    try:

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        print(
            f"\nExisting results found: "
            f"{len(existing)}"
        )

    except Exception as e:

        print(
            f"\nCould not read existing results: {e}"
        )

        existing = pd.DataFrame()


# ============================================================
# 9. FIND SUCCESSFULLY PROCESSED IDS
# ============================================================

processed_ids = set()

if (
    not existing.empty
    and "evaluation_id" in existing.columns
    and "overall" in existing.columns
):

    valid_existing = existing[
        pd.to_numeric(
            existing["overall"],
            errors="coerce"
        ).notna()
    ]

    processed_ids = set(
        valid_existing[
            "evaluation_id"
        ]
        .astype(str)
    )


print(
    f"Successfully processed already: "
    f"{len(processed_ids)}"
)


# ============================================================
# 10. EVALUATE
# ============================================================

new_results = []

total = len(df)

print("\nStarting response-quality evaluation...")
print("-" * 70)


for index, row in df.iterrows():

    evaluation_id = str(
        row["evaluation_id"]
    )


    # --------------------------------------------------------
    # SKIP ONLY SUCCESSFUL RESULTS
    # --------------------------------------------------------

    if evaluation_id in processed_ids:

        print(
            f"[{index + 1}/{total}] "
            f"Skipping {evaluation_id} "
            f"(already evaluated)"
        )

        continue


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    query = str(
        row["query"]
    )

    predicted_intent = str(
        row["predicted_intent"]
    )

    response = str(
        row["response"]
    )


    print(
        f"\n[{index + 1}/{total}] "
        f"Evaluating {evaluation_id}"
    )

    print(
        f"Intent: {predicted_intent}"
    )


    # --------------------------------------------------------
    # JUDGE
    # --------------------------------------------------------

    scores = judge_response(
        query,
        predicted_intent,
        response
    )


    # --------------------------------------------------------
    # RESULT ROW
    # --------------------------------------------------------

    result_row = {

        "evaluation_id":
            evaluation_id,

        "conversation_id":
            row["conversation_id"],

        "query":
            query,

        "predicted_intent":
            predicted_intent,

        "response":
            response,

        "correctness":
            scores["correctness"],

        "relevance":
            scores["relevance"],

        "helpfulness":
            scores["helpfulness"],

        "grounding":
            scores["grounding"],

        "overall":
            scores["overall"],

        "judge_reason":
            scores["reason"]
    }


    new_results.append(
        result_row
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"    Correctness : "
        f"{scores['correctness']}"
    )

    print(
        f"    Relevance   : "
        f"{scores['relevance']}"
    )

    print(
        f"    Helpfulness : "
        f"{scores['helpfulness']}"
    )

    print(
        f"    Grounding   : "
        f"{scores['grounding']}"
    )

    print(
        f"    Overall     : "
        f"{scores['overall']}"
    )

    print(
        f"    Reason      : "
        f"{scores['reason']}"
    )


    # --------------------------------------------------------
    # SAVE PROGRESS
    # --------------------------------------------------------

    current_new = pd.DataFrame(
        new_results
    )

    if not existing.empty:

        combined = pd.concat(
            [
                existing,
                current_new
            ],
            ignore_index=True
        )

    else:

        combined = current_new


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=["evaluation_id"],
        keep="last"
    )


    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        "    Progress saved."
    )


    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    time.sleep(1)


# ============================================================
# 11. FINAL RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    final_df = pd.read_csv(
        OUTPUT_FILE
    )

else:

    final_df = pd.DataFrame(
        new_results
    )


print("\n")
print("=" * 70)
print("FINAL RESPONSE QUALITY RESULTS")
print("=" * 70)


print(
    f"\nTotal rows: {len(final_df)}"
)


# ============================================================
# 12. AVERAGES
# ============================================================

score_columns = [
    "correctness",
    "relevance",
    "helpfulness",
    "grounding",
    "overall"
]


print("\nAverage Scores")
print("-" * 40)


for column in score_columns:

    values = pd.to_numeric(
        final_df[column],
        errors="coerce"
    ).dropna()


    if len(values) == 0:

        print(
            f"{column.capitalize():15s}: "
            f"No valid scores"
        )

    else:

        average = values.mean()

        print(
            f"{column.capitalize():15s}: "
            f"{average:.2f} / 5"
        )


# ============================================================
# 13. OVERALL PERCENTAGE
# ============================================================

overall_values = pd.to_numeric(
    final_df["overall"],
    errors="coerce"
).dropna()


if len(overall_values) > 0:

    overall_average = (
        overall_values.mean()
    )

    percentage = (
        overall_average / 5
    ) * 100

    print(
        f"\nOverall quality: "
        f"{overall_average:.2f} / 5"
    )

    print(
        f"Overall quality percentage: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 14. SCORE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL SCORE DISTRIBUTION")
print("=" * 70)


distribution = (
    pd.to_numeric(
        final_df["overall"],
        errors="coerce"
    )
    .dropna()
    .value_counts()
    .sort_index()
)


for score, count in distribution.items():

    print(
        f"Score {int(score)}: "
        f"{count} responses"
    )


# ============================================================
# 15. FAILED JUDGMENTS
# ============================================================

failed_count = (
    pd.to_numeric(
        final_df["overall"],
        errors="coerce"
    )
    .isna()
    .sum()
)


print("\n")
print("=" * 70)
print("JUDGE STATUS")
print("=" * 70)


print(
    f"Successful judgments: "
    f"{len(final_df) - failed_count}"
)

print(
    f"Failed judgments: "
    f"{failed_count}"
)


# ============================================================
# 16. LOWEST QUALITY RESPONSES
# ============================================================

print("\n")
print("=" * 70)
print("5 LOWEST-QUALITY RESPONSES")
print("=" * 70)


temp = final_df.copy()

temp["overall_numeric"] = pd.to_numeric(
    temp["overall"],
    errors="coerce"
)


lowest = (
    temp
    .dropna(
        subset=["overall_numeric"]
    )
    .sort_values(
        "overall_numeric"
    )
    .head(5)
)


if len(lowest) == 0:

    print(
        "No valid scores available."
    )

else:

    for number, (_, row) in enumerate(
        lowest.iterrows(),
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"#{number}"
        )

        print(
            f"Overall: {row['overall']}"
        )

        print(
            f"Intent: {row['predicted_intent']}"
        )

        print(
            f"Query: {row['query']}"
        )

        print(
            f"Response: {row['response']}"
        )

        print(
            f"Reason: {row['judge_reason']}"
        )


# ============================================================
# 17. DONE
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)
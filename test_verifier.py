from agents.verifier import verify_answer

QUESTION = (
    "What drove Intelligent Cloud revenue growth "
    "in fiscal year 2025?"
)

GOOD_ANSWER = """
Intelligent Cloud revenue increased $18.8 billion or 21%.
Server products and cloud services revenue increased $18.6 billion
or 23%, driven by Azure and other cloud services. Azure and other
cloud services revenue grew 34% driven by demand for Microsoft's
portfolio of services. Server products revenue separately decreased
3% due to lower transactional purchasing and continued customer
shift to cloud offerings.
""".strip()

BAD_ANSWER = """
The overall change in Intelligent Cloud revenue was an increase of
$18.8 billion or 21%. Azure and other cloud services revenue grew 34%.
Server products revenue also increased significantly.
No supporting figures are available.
""".strip()

CONTEXT = """
Intelligent Cloud
Revenue increased $18.8 billion or 21%.

Server products and cloud services revenue increased $18.6 billion
or 23% driven by Azure and other cloud services.

Azure and other cloud services revenue grew 34% driven by demand
for our portfolio of services.

Server products revenue decreased 3% driven by a decrease in
transactional purchasing with continued customer shift to cloud offerings.
""".strip()


def run_test(label: str, answer: str) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    result = verify_answer(
        question=QUESTION,
        answer=answer,
        context=CONTEXT,
    )

    print("Verdict:", result["verdict"])
    print("Supported:", result["supported"])
    print("Issues:", result["issues"])
    print("Feedback:", result["feedback"])


if __name__ == "__main__":
    run_test("TEST 1 - GOOD ANSWER", GOOD_ANSWER)
    run_test("TEST 2 - BAD ANSWER", BAD_ANSWER)

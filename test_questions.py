
import requests
import time
import json
from statistics import mean

# =========================================================
# CONFIG
# =========================================================

API_URL = "http://localhost:8000/ask"

CONFIDENCE_THRESHOLD = 0.40
MAX_LATENCY_SECONDS = 15

# =========================================================
# TEST CASES
# =========================================================

test_cases = [

    {
        "name": "control system definition",

        "question": "What is control system ?",

        "expected_keywords": [
            "input",
            "output",
            "control loop",
            "desired response",
        ],

        "expect_images": False,
        "expect_sources": False,
    },

    {
        "name": "Closed Loop Definition",

        "question": "What is a closed loop control system?",

        "expected_keywords": [
            "feedback",
            "error",
            "output"
        ],

        "expect_images": False,
        "expect_sources": False,
    },

    {
        "name": "Diagram Retrieval",

        "question": "Show me the block diagram of a control system",

        "expected_keywords": [
            "diagram",
            "system"
        ],

        "expect_images": True,
        "expect_sources": False,
    },

    {
        "name": "Source Retrieval",

        "question": "Explain transfer function with sources",

        "expected_keywords": [
            "transfer",
            "system"
        ],

        "expect_images": False,
        "expect_sources": True,
    },

    {
        "name": "Greeting Test",

        "question": "hi how are you",

        "expected_keywords": [],

        "expect_images": False,
        "expect_sources": False,
    },

    {
        "name": "5 Marks Structured Answer",

        "question": "Explain control system for 5 marks",

        "expected_keywords": [
            "definition",
            "key points",
            "conclusion"
        ],

        "expect_images": False,
        "expect_sources": False,
    },
]

# =========================================================
# METRICS STORAGE
# =========================================================

results = []

total_passed = 0
total_failed = 0

latencies = []
confidence_scores = []

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_separator():
    print("\n" + "=" * 70)


def pass_test(message):
    global total_passed
    total_passed += 1
    print(f"✅ PASS: {message}")


def fail_test(message):
    global total_failed
    total_failed += 1
    print(f"❌ FAIL: {message}")


def keyword_check(answer, keywords):

    if not keywords:
        return

    lower_answer = answer.lower()

    for kw in keywords:

        if kw.lower() in lower_answer:
            pass_test(f"Keyword found → '{kw}'")
        else:
            fail_test(f"Keyword missing → '{kw}'")


def image_check(images, expected):

    if expected and images:
        pass_test("Images correctly returned")

    elif expected and not images:
        fail_test("Expected images but none returned")

    elif not expected and not images:
        pass_test("No images expected and none returned")

    elif not expected and images:
        fail_test("Unexpected images returned")


def source_check(sources, expected):

    if expected and sources:
        pass_test("Sources correctly returned")

        for s in sources:

            print(
                f"   ↳ {s['file']} | "
                f"Page {s['page']} | "
                f"Score: {s.get('score', 0)}"
            )

    elif expected and not sources:
        fail_test("Expected sources but none returned")

    elif not expected and not sources:
        pass_test("No sources expected")

    elif not expected and sources:
        fail_test("Unexpected sources returned")


def confidence_check(sources):

    if not sources:
        print("ℹ No sources available for confidence check")
        return

    scores = []

    for s in sources:
        if "score" in s:
            scores.append(float(s["score"]))

    if not scores:
        print("ℹ No reranker scores available")
        return

    max_score = max(scores)

    confidence_scores.append(max_score)

    print(f"Top reranker confidence: {max_score:.3f}")

    if max_score >= CONFIDENCE_THRESHOLD:
        pass_test("Confidence above threshold")
    else:
        fail_test("Low confidence retrieval")


def latency_check(latency):

    latencies.append(latency)

    print(f"Latency: {latency:.2f}s")

    if latency <= MAX_LATENCY_SECONDS:
        pass_test("Latency acceptable")
    else:
        fail_test("Latency too high")


def hallucination_check(answer):

    suspicious_phrases = [
        "i don't know",
        "not sure",
        "maybe",
        "possibly",
        "i think",
        "cannot determine",
    ]

    lower_answer = answer.lower()

    found = False

    for phrase in suspicious_phrases:

        if phrase in lower_answer:
            found = True
            fail_test(f"Possible hallucination phrase detected → '{phrase}'")

    if not found:
        pass_test("No obvious hallucination phrases")


# =========================================================
# MAIN TEST LOOP
# =========================================================

print_separator()
print("RUNNING FULL RAG QUALITY CHECK")
print_separator()

for idx, test in enumerate(test_cases):

    print_separator()

    print(f"TEST {idx + 1}: {test['name']}")
    print(f"Question: {test['question']}")

    try:

        start = time.time()

        response = requests.post(
            API_URL,
            json={
                "query": test["question"],
                "session_id": "qc_session"
            },
            timeout=60
        )

        latency = time.time() - start

        # -------------------------------------------------
        # HTTP CHECK
        # -------------------------------------------------

        if response.status_code != 200:
            fail_test(f"HTTP Error → {response.status_code}")
            continue

        pass_test("API request successful")

        # -------------------------------------------------
        # PARSE RESPONSE
        # -------------------------------------------------

        data = response.json()

        answer = data.get("answer", "")
        images = data.get("images", [])
        sources = data.get("sources", [])
        answer_type = data.get("answer_type", "")

        # -------------------------------------------------
        # PRINT RESPONSE
        # -------------------------------------------------

        print("\nANSWER:")
        print(answer[:1000])

        print("\nANSWER TYPE:")
        print(answer_type)

        print("\nIMAGES:")
        print(images if images else "None")

        print("\nSOURCES:")
        print(json.dumps(sources, indent=2) if sources else "None")

        # -------------------------------------------------
        # RUN CHECKS
        # -------------------------------------------------

        keyword_check(answer, test["expected_keywords"])

        image_check(
            images,
            test["expect_images"]
        )

        source_check(
            sources,
            test["expect_sources"]
        )

        confidence_check(sources)

        latency_check(latency)

        hallucination_check(answer)

        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        results.append({
            "test_name": test["name"],
            "question": test["question"],
            "answer_type": answer_type,
            "latency": latency,
            "image_count": len(images),
            "source_count": len(sources),
        })

    except Exception as e:

        fail_test(f"Exception occurred → {str(e)}")

# =========================================================
# FINAL REPORT
# =========================================================

print_separator()
print("FINAL QC REPORT")
print_separator()

print(f"TOTAL PASSED: {total_passed}")
print(f"TOTAL FAILED: {total_failed}")

if latencies:
    print(f"\nAverage Latency: {mean(latencies):.2f}s")
    print(f"Max Latency: {max(latencies):.2f}s")

if confidence_scores:
    print(f"\nAverage Confidence: {mean(confidence_scores):.3f}")
    print(f"Max Confidence: {max(confidence_scores):.3f}")

success_rate = (
    (total_passed / (total_passed + total_failed)) * 100
    if (total_passed + total_failed) > 0 else 0
)

print(f"\nSUCCESS RATE: {success_rate:.2f}%")

print_separator()

if success_rate >= 90:
    print("🏆 EXCELLENT RAG SYSTEM")

elif success_rate >= 75:
    print("✅ GOOD RAG SYSTEM")

elif success_rate >= 50:
    print("⚠ NEEDS IMPROVEMENT")

else:
    print("❌ MAJOR ISSUES DETECTED")

print_separator()
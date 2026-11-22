"""Masking of query results and redaction of free text."""

import time

import pandas as pd
import pytest

from agent.guardrails.pii import mask_dataframe, scan_text

# What a caller passes for a final answer: the query results and the user's own messages.
KNOWN = [
    "brand,revenue\nMichael Kors,48210.5\nJessica Simpson,31977.0\nMiss Me,20114.25\nDr. Martens,9120.0\n",
    "How did our designer brands do in Q3 2026?",
]

# The ways scan_text is called: without sources, for logs, and with the sources of an answer.
in_every_mode = pytest.mark.parametrize(
    "known", [None, (), KNOWN], ids=["without known texts", "strict", "with known texts"]
)

REPORT_PARAGRAPH = (
    "Womenswear delivered robust top-line growth in Q3 2026, with revenue climbing 18.1% quarter-over-quarter "
    "driven by stronger customer acquisition and higher order volume. Intimates ($38,074), Dresses ($35,958), "
    "and Jeans ($35,182) generated the highest revenue. Blazers & Jackets achieved the highest margin rate at "
    "62.2% ($19,869 revenue, $12,361 gross margin), whereas Clothing Sets trailed at 37.2%. Action item 1: lift "
    "Average Order Value back above the Q2 level of $81.01 from the current $79.80."
)

BUSINESS_TEXT = [
    "Revenue: $398,861 (up 18.1% from $337,792 in Q2)",
    "Orders: 6,756 (up 23.8% from 5,458 in Q2); customers: 6,016 active buyers",
    "From 2026-07-01 to 2026-09-30 (Q3 2026) the return rate was 13.0%.",
    "User ID 37775: $1,494 total spend across 3 orders (Average Order Value: $498.00)",
    "New York, North Carolina and San Antonio lead; Texas follows with 1,204 orders.",
    "## Key metrics",
    "## Action items",
    "| Texas | 2,104 orders | $81.01 | 13.0% |",
    "Outerwear & Coats (55.4% margin rate, $34,918 revenue) and Blazers & Jackets",
    "The category name is Outerwear & Coats",
    "Brand name: Calvin Klein",
    "Report name: Q3 Review",
    "Customers are listed by user id only; names and contact details are never part of an analysis.",
    REPORT_PARAGRAPH,
]


def test_scan_text_redacts_contact_details_only():
    text, found = scan_text(
        "Mail jane.doe@example.com or call (415) 555-0199, 12 Baker Street. Revenue 2024-01-01: 1,234,567.89 for user 48213."
    )
    assert "jane.doe" not in text and "555-0199" not in text and "Baker" not in text
    assert sorted(found) == ["address", "email", "phone"]
    assert "2024-01-01" in text and "1,234,567.89" in text and "48213" in text


def test_scan_text_leaves_business_text_alone():
    text = "Q1 revenue was $415,222 across 3,112 orders (2025-01-01 to 2025-03-31); return rate 10.2%."
    assert scan_text(text) == (text, [])


@in_every_mode
@pytest.mark.parametrize("text", BUSINESS_TEXT)
def test_business_text_comes_back_unchanged_in_every_mode(text, known):
    assert scan_text(text, known) == (text, [])


def test_customer_details_are_fully_redacted_in_strict_mode():
    text, found = scan_text("Customer Jane Doe (jane.doe@example.com), postal code 94105", known=())
    assert text == "Customer [REDACTED NAME] ([REDACTED EMAIL]), postal code [REDACTED POSTAL CODE]"
    assert sorted(found) == ["email", "name", "postal_code"]


def test_known_texts_do_not_excuse_contact_details_or_labelled_values():
    text = "Customer name: Jane Doe (jane.doe@example.com), ZIP 94105, lat 37.7749"
    clean, found = scan_text(text, known=[text])
    assert clean == (
        "Customer name: [REDACTED NAME] ([REDACTED EMAIL]), ZIP [REDACTED POSTAL CODE], lat [REDACTED COORDINATES]"
    )
    assert sorted(found) == ["coordinates", "email", "name", "postal_code"]


# ------------------------------------------------------- postal codes and coordinates


@in_every_mode
@pytest.mark.parametrize(
    "text, expected",
    [
        ("postal code 94105", "postal code [REDACTED POSTAL CODE]"),
        ("Postcode: SW1A 1AA", "Postcode: [REDACTED POSTAL CODE]"),
        ("the zip code is 94105-1234.", "the zip code is [REDACTED POSTAL CODE]."),
        ("The ZIP is: 94105", "The ZIP is: [REDACTED POSTAL CODE]"),
        ("ZIP 94105", "ZIP [REDACTED POSTAL CODE]"),
        ("- **Postal code:** 02675-031", "- **Postal code:** [REDACTED POSTAL CODE]"),
        ("Postal code: 01310100", "Postal code: [REDACTED POSTAL CODE]"),
        ("San Francisco, CA 94105", "San Francisco, CA [REDACTED POSTAL CODE]"),
        ("Austin, TX 78701-1234, USA", "Austin, TX [REDACTED POSTAL CODE], USA"),
    ],
)
def test_postal_code_is_redacted_after_a_label_or_a_state(text, expected, known):
    assert scan_text(text, known) == (expected, ["postal_code"])


@in_every_mode
@pytest.mark.parametrize(
    "text",
    [
        "User ID 37775 spent $1,494",
        "Intimates made $38,074 from 38074 items",
        "20,412 items in 12345 orders",
        "Columbia Women's Full Zip 2024 Fleece Jacket",
        "Postal codes are personal data and are not available.",
        "Revenue by state: TX 10451.75, CA 1204512",
        "Top state: TX\n12045 orders",
    ],
)
def test_number_that_is_not_introduced_as_a_postal_code_is_left_alone(text, known):
    assert scan_text(text, known) == (text, [])


def test_number_too_long_for_a_postal_code_or_a_coordinate_is_not_redacted_in_part():
    text = "Postal code 12345678901 and LAT 2024 are style numbers."
    assert scan_text(text) == (text, [])


@in_every_mode
@pytest.mark.parametrize(
    "text, expected, hits",
    [
        ("Ships to 37.7749, -122.4194.", "Ships to [REDACTED COORDINATES].", 1),
        ("(37.7749295,-122.4194155)", "([REDACTED COORDINATES])", 1),
        (
            "latitude 37.7749 and longitude -122.4194",
            "latitude [REDACTED COORDINATES] and longitude [REDACTED COORDINATES]",
            2,
        ),
        ("lat: 51.5074, lng: -0.1278", "lat: [REDACTED COORDINATES], lng: [REDACTED COORDINATES]", 2),
        ("lon=-74.0060", "lon=[REDACTED COORDINATES]", 1),
    ],
)
def test_coordinates_are_redacted_as_a_pair_or_after_a_label(text, expected, hits, known):
    assert scan_text(text, known) == (expected, ["coordinates"] * hits)


@in_every_mode
@pytest.mark.parametrize(
    "text",
    [
        "Return rates were 0.1304, 0.1161 and 0.1502.",
        "Texas,2104,81.0123,0.1304",
        "Average order values: 81.0123, 79.8004, 80.1150",
        "Average order value: $79.80, $81.01",
        "Items per order: 1.45, 1.52",
        "Margin rates of 62.2%, 55.4%",
    ],
)
def test_figures_side_by_side_are_not_coordinates(text, known):
    assert scan_text(text, known) == (text, [])


# ---------------------------------------------------------------- names behind a label


@in_every_mode
@pytest.mark.parametrize(
    "text, expected",
    [
        ("First name: Jane", "First name: [REDACTED NAME]"),
        ("The last name is Doe.", "The last name is [REDACTED NAME]."),
        ("Full name: Jane Q. Doe", "Full name: [REDACTED NAME]"),
        ("Customer name: Zofia Kowalska-Nowak, 3 orders", "Customer name: [REDACTED NAME], 3 orders"),
        ("The customer's name is José García.", "The customer's name is [REDACTED NAME]."),
        ("The customer's name is: Jane Doe", "The customer's name is: [REDACTED NAME]"),
        ("Buyer's name: Liam O'Neil", "Buyer's name: [REDACTED NAME]"),
        ("Buyer's name: Élodie Østergaard", "Buyer's name: [REDACTED NAME]"),
        ("The user's name is Priya", "The user's name is [REDACTED NAME]"),
        ("- **Customer name:** Jane Doe\nRevenue: $500", "- **Customer name:** [REDACTED NAME]\nRevenue: $500"),
        ("first_name=Jane", "first_name=[REDACTED NAME]"),
        ("Customer name: Mr. Smith", "Customer name: [REDACTED NAME]"),
        ("Customer name: J. K. Rowling, 2 orders", "Customer name: [REDACTED NAME], 2 orders"),
        ("Full name: Maria de la Cruz (user 37775)", "Full name: [REDACTED NAME] (user 37775)"),
    ],
)
def test_labelled_name_is_redacted_whole_and_its_label_kept(text, expected, known):
    assert scan_text(text, known) == (expected, ["name"])


@in_every_mode
@pytest.mark.parametrize(
    "stand_in",
    [
        "Not available (personal data is never shown)",
        "No name on file",
        "None",
        "Unknown",
        "Unavailable",
        "Redacted",
        "Hidden",
        "Withheld",
        "Private",
        "Personal data, not shown",
        "User 37775",
        "Customer 37775",
        "ID 37775",
    ],
)
def test_what_stands_in_for_a_name_is_not_taken_for_one(stand_in, known):
    text = f"Customer name: {stand_in}. The customer's name is not stored."
    assert scan_text(text, known) == (text, [])


@in_every_mode
def test_name_label_without_a_colon_or_is_introduces_nothing(known):
    text = "First Name and Last Name Are Personal Data"
    assert scan_text(text, known) == (text, [])


# ---------------------------------------------------------------- names behind a title


@in_every_mode
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Mr. Smith placed 3 orders", "Mr. [REDACTED NAME] placed 3 orders"),
        ("Mrs Jones returned 2 items", "Mrs [REDACTED NAME] returned 2 items"),
        ("Ms. Nguyen is the top buyer", "Ms. [REDACTED NAME] is the top buyer"),
        ("Miss Brown and 4 others", "Miss [REDACTED NAME] and 4 others"),
        ("Dr. Patel spent $1,494", "Dr. [REDACTED NAME] spent $1,494"),
        ("Dr. Priya K. Patel spent $1,494", "Dr. [REDACTED NAME] spent $1,494"),
        ("Mrs. Mary Jane Watson returned 2 items", "Mrs. [REDACTED NAME] returned 2 items"),
        ("Mr. J. van der Berg is second", "Mr. [REDACTED NAME] is second"),
    ],
)
def test_name_behind_a_title_is_redacted_in_every_mode(text, expected, known):
    assert scan_text(text, known) == (expected, ["name"])


def test_title_of_a_next_name_ends_the_name_before_it():
    text, found = scan_text("Mr. Smith Mrs. Jones and Customer name: Jane Doe Dr. Patel")
    assert text == "Mr. [REDACTED NAME] Mrs. [REDACTED NAME] and Customer name: [REDACTED NAME] Dr. [REDACTED NAME]"
    assert found == ["name", "name", "name", "name"]


def test_brand_with_a_title_stays_only_when_a_known_text_has_it():
    text = "Miss Me led the Jeans category; Dr. Martens followed."
    redacted = "Miss [REDACTED NAME] led the Jeans category; Dr. [REDACTED NAME] followed."
    assert scan_text(text, known=KNOWN) == (text, [])
    assert scan_text(text) == (redacted, ["name", "name"])
    assert scan_text(text, known=()) == (redacted, ["name", "name"])


def test_full_stop_and_spacing_do_not_decide_whether_a_name_is_known():
    known = ["brand\nDr.  Martens\nMr Coffee\nJessica  Simpson\n"]
    text, found = scan_text("Dr Martens, Mr. Coffee and Jessica Simpson grew while Mr Smith placed 3 orders.", known)
    assert text == "Dr Martens, Mr. Coffee and Jessica Simpson grew while Mr [REDACTED NAME] placed 3 orders."
    assert found == ["name"]


# -------------------------------------------------------- full names and their source


@pytest.mark.parametrize(
    "name",
    ["Jane Doe", "Mary Jane Watson", "John Q. Public", "Sarah O'Connor", "David Smith-Jones", "Maria de la Cruz"],
)
def test_full_name_without_a_source_is_redacted_whole(name):
    text, found = scan_text(f"Your top customer is {name}, with 4 orders.", known=KNOWN)
    assert text == "Your top customer is [REDACTED NAME], with 4 orders."
    assert found == ["name"]


def test_brand_named_after_a_person_stays_when_it_comes_from_the_data():
    text = "Michael Kors led with $48,211; Jessica Simpson's dresses, Miss Me and Dr. Martens followed."
    assert scan_text(text, known=KNOWN) == (text, [])


def test_capitalised_word_before_a_known_brand_does_not_make_it_a_persons_name():
    text = "In Texas Michael Kors led again."
    assert scan_text(text, known=KNOWN) == (text, [])


def test_given_name_that_ends_a_known_brand_or_city_does_not_start_a_persons_name():
    known = ["category,brand\nJeans,Sean John\n", "state,city\nCalifornia,Santa Barbara\nMinnesota,St. Paul\n"]
    text = "Sean John Jeans led; Santa Barbara California and St. Paul Minnesota followed."
    assert scan_text(text, known) == (text, [])
    assert scan_text(text, known=()) == (
        "Sean [REDACTED NAME] led; Santa [REDACTED NAME] and St. [REDACTED NAME] followed.",
        ["name", "name", "name"],
    )
    assert scan_text("Customer John Smith placed 3 orders.", known) == (
        "Customer [REDACTED NAME] placed 3 orders.",
        ["name"],
    )


def test_name_is_looked_up_by_its_first_two_words():
    text = "Ralph Lauren Polo shirts and Miss Me Jeans grew."
    assert scan_text(text, known=["brand\nPolo Ralph Lauren\nMiss Me\n"]) == (text, [])


def test_name_is_known_only_as_whole_words():
    assert scan_text("Jane Doe", known=["Jane Doe's Outlet"]) == ("Jane Doe", [])
    assert scan_text("Jane Doe", known=["Jane Doering Outlet"]) == ("[REDACTED NAME]", ["name"])
    assert scan_text("Jane Doe", known=["Maryjane Doe Outlet"]) == ("[REDACTED NAME]", ["name"])
    assert scan_text("Mr. J. Smith", known=["Mr. Jones Menswear"]) == ("Mr. [REDACTED NAME]", ["name"])


def test_known_texts_are_matched_case_insensitively_and_can_be_any_iterable():
    shouted = (source.upper() for source in KNOWN)
    text = "Michael Kors and Jessica Simpson"
    assert scan_text(text, known=shouted) == (text, [])


def test_invented_name_is_redacted_when_no_known_text_has_it():
    text, found = scan_text("Your top customer is Jane Doe, who mostly buys Michael Kors.", known=KNOWN)
    assert text == "Your top customer is [REDACTED NAME], who mostly buys Michael Kors."
    assert found == ["name"]


def test_strict_mode_redacts_every_person_looking_name():
    text, found = scan_text("Michael Kors, Miss Me and Jane Doe", known=())
    assert text == "[REDACTED NAME], Miss [REDACTED NAME] and [REDACTED NAME]"
    assert found == ["name", "name", "name"]


def test_full_names_are_left_alone_without_known_texts():
    text = "Your top customer is Jane Doe; John Q. Public is second and mostly buys Michael Kors."
    assert scan_text(text) == (text, [])
    assert scan_text(text, known=None) == (text, [])


@in_every_mode
def test_redacted_text_is_stable_when_scanned_again(known):
    once, found = scan_text("Customer name: Jane Doe, ZIP 94105, lat 37.7749, Mr. Smith and Mary Major", known)
    assert set(found) == {"coordinates", "name", "postal_code"}
    assert scan_text(once, known) == (once, [])


def test_long_padding_after_a_label_does_not_slow_the_scan_down():
    padding = " " * 40_000
    texts = [f"zip is{padding}x", f"Customer name:{padding}x", f"first name{padding}:{padding}x", f"lat{padding}x"]
    started = time.perf_counter()
    for text in texts:
        assert scan_text(text, known=()) == (text, [])
    assert time.perf_counter() - started < 3  # a fraction of a second, unless a pattern re-reads the padding


# ----------------------------------------------------------------------- query results


def test_mask_dataframe_drops_pii_columns_and_redacts_values():
    df = pd.DataFrame(
        {
            "user_id": [1],
            "contact_email": ["a@b.com"],
            "Email": ["c@d.com"],
            "note": ["reach me at x@y.org"],
            "spend": [10.5],
        }
    )
    masked, events = mask_dataframe(df)
    assert list(masked.columns) == ["user_id", "note", "spend"]
    assert masked["note"][0] == "reach me at [REDACTED EMAIL]"
    assert len(events) == 2


def test_mask_dataframe_redacts_labelled_values_and_leaves_brands_named_after_people_alone():
    df = pd.DataFrame(
        {
            "brand": ["Michael Kors", "Miss Me", "Dr. Martens"],
            "note": ["ship to Austin, TX 78701", "customer name: Jane Doe", "gift wrap"],
        }
    )
    masked, events = mask_dataframe(df)
    assert list(masked["brand"]) == ["Michael Kors", "Miss Me", "Dr. Martens"]
    assert list(masked["note"]) == [
        "ship to Austin, TX [REDACTED POSTAL CODE]",
        "customer name: [REDACTED NAME]",
        "gift wrap",
    ]
    assert events == ["redacted 2 value(s) in column note"]


def test_name_is_recognised_by_its_role_in_the_sentence_not_only_by_a_common_given_name():
    assert scan_text("Dmitry Ivanov is your top customer.", known=()) == (
        "[REDACTED NAME] is your top customer.",
        ["name"],
    )
    assert scan_text("The top buyer, Zoltan Kovacs, spent most.", known=()) == (
        "The top buyer, [REDACTED NAME], spent most.",
        ["name"],
    )
    assert scan_text("customer named Aiko Tanaka", known=()) == ("customer named [REDACTED NAME]", ["name"])
    # A brand from the data in the same position stays.
    known = ["brand,revenue\nCalvin Klein,100\n"]
    assert scan_text("Calvin Klein is the top brand; customer Calvin Klein fans buy more.", known=known)[1] == []


@pytest.mark.parametrize(
    "text",
    [
        "Customer Lifetime Value rose in Q3.",
        "## Customer Acquisition Trends",
        "Outerwear & Coats is the top category for your customers.",
        "The top customer is User 37775 with 3 orders.",
        "New York is the largest market by customer count.",
        "Key Metrics: revenue per customer and orders per customer.",
    ],
)
def test_role_words_do_not_turn_business_terms_into_names(text):
    assert scan_text(text, known=()) == (text, [])

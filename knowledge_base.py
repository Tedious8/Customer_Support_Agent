"""
knowledge_base.py — Static FAQ store with keyword-indexed retrieval.
In production, swap with a vector store (e.g. Pinecone, ChromaDB).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FAQEntry:
    id: str
    category: str
    question: str
    answer: str
    keywords: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Knowledge Base Entries
# ─────────────────────────────────────────────

FAQ_ENTRIES: list[FAQEntry] = [
    # Shipping
    FAQEntry(
        id="SHP-001",
        category="shipping",
        question="How long does standard shipping take?",
        answer=(
            "Standard shipping takes 5–7 business days within the continental US. "
            "Express shipping (2–3 business days) and overnight options are also available at checkout. "
            "International orders typically arrive within 10–14 business days depending on destination."
        ),
        keywords=["shipping", "delivery", "how long", "arrive", "days", "transit"],
    ),
    FAQEntry(
        id="SHP-002",
        category="shipping",
        question="Do you offer free shipping?",
        answer=(
            "Yes! Orders over $50 qualify for free standard shipping within the US. "
            "Premium and VIP members receive free standard shipping on all orders regardless of cart value."
        ),
        keywords=["free shipping", "shipping cost", "free delivery", "$50"],
        related_ids=["SHP-001"],
    ),
    FAQEntry(
        id="SHP-003",
        category="shipping",
        question="Can I track my order?",
        answer=(
            "Absolutely. Once your order ships, you'll receive a confirmation email with a tracking link. "
            "You can also track orders in your account dashboard under 'My Orders'. "
            "Tracking updates are refreshed every 4 hours."
        ),
        keywords=["track", "tracking", "where is my order", "status", "shipment"],
    ),
    # Returns & Refunds
    FAQEntry(
        id="RET-001",
        category="returns",
        question="What is your return policy?",
        answer=(
            "We accept returns within 30 days of delivery for most items in original, unused condition. "
            "Digital products and perishable goods are non-returnable. "
            "To initiate a return, go to 'My Orders', select the item, and click 'Start Return'. "
            "Refunds are processed within 5–10 business days after we receive the item."
        ),
        keywords=["return", "refund", "send back", "policy", "30 days", "money back"],
    ),
    FAQEntry(
        id="RET-002",
        category="returns",
        question="My item arrived damaged. What do I do?",
        answer=(
            "We're sorry about that! Please keep the item and all packaging materials. "
            "Take photos of the damage and contact support within 7 days of delivery. "
            "We'll send a prepaid return label and ship a replacement immediately — no need to wait for the return to arrive."
        ),
        keywords=["damaged", "broken", "defective", "arrived damaged", "wrong item", "faulty"],
        related_ids=["RET-001"],
    ),
    # Account
    FAQEntry(
        id="ACC-001",
        category="account",
        question="How do I reset my password?",
        answer=(
            "Click 'Forgot Password' on the login page and enter your email address. "
            "You'll receive a reset link within 2 minutes. "
            "If you don't see it, check your spam folder. "
            "Links expire after 30 minutes for security."
        ),
        keywords=["password", "reset", "forgot password", "login", "sign in", "can't log in"],
    ),
    FAQEntry(
        id="ACC-002",
        category="account",
        question="How do I update my payment method?",
        answer=(
            "Go to Account Settings → Payment Methods. "
            "You can add, remove, or set a default payment method there. "
            "We accept Visa, Mastercard, Amex, PayPal, and Apple Pay. "
            "Payment changes apply to future orders only."
        ),
        keywords=["payment", "credit card", "billing", "update payment", "change card", "paypal"],
    ),
    # Subscription / Membership
    FAQEntry(
        id="SUB-001",
        category="subscription",
        question="How do I cancel my subscription?",
        answer=(
            "You can cancel anytime from Account Settings → Subscription → Cancel Plan. "
            "Cancellations take effect at the end of your current billing period — you'll retain access until then. "
            "No cancellation fees apply. After cancellation, you'll revert to a free account."
        ),
        keywords=["cancel", "subscription", "cancel plan", "unsubscribe", "stop billing"],
    ),
    FAQEntry(
        id="SUB-002",
        category="subscription",
        question="What are the benefits of a Premium membership?",
        answer=(
            "Premium members enjoy: free shipping on all orders, priority customer support (< 2hr response), "
            "10% off every order, early access to sales and new products, and a dedicated account manager for VIP tier. "
            "Premium is $9.99/month or $89.99/year (save 25%)."
        ),
        keywords=["premium", "membership", "benefits", "vip", "perks", "upgrade"],
    ),
    # Billing
    FAQEntry(
        id="BIL-001",
        category="billing",
        question="I was charged twice for the same order.",
        answer=(
            "Double charges are usually a temporary authorization hold that drops off within 3–5 business days. "
            "If both charges appear as 'completed' on your statement after 5 days, please contact support "
            "with your order number and bank statement screenshot. We'll investigate and refund the duplicate immediately."
        ),
        keywords=["charged twice", "double charge", "duplicate charge", "overcharged", "extra charge"],
    ),
    # Product
    FAQEntry(
        id="PRD-001",
        category="product",
        question="Do you offer warranty on your products?",
        answer=(
            "Yes. Most products come with a 1-year manufacturer warranty covering defects in materials and workmanship. "
            "Extended warranties (2–3 years) are available for purchase. "
            "Warranty does not cover accidental damage, misuse, or normal wear and tear. "
            "To file a claim, contact support with your order number and proof of purchase."
        ),
        keywords=["warranty", "guarantee", "defect", "broken", "coverage", "claim"],
    ),
]


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

def search_faq(query: str, top_k: int = 3) -> list[dict]:
    """
    Keyword-based FAQ search. Returns top_k most relevant entries.
    Production upgrade path: replace with semantic embedding search.
    """
    query_tokens = set(query.lower().split())
    scored: list[tuple[int, FAQEntry]] = []

    for entry in FAQ_ENTRIES:
        # Score by keyword overlap + question similarity
        kw_score = sum(
            1 for kw in entry.keywords
            if kw.lower() in query.lower()
        )
        q_tokens = set(entry.question.lower().split())
        overlap_score = len(query_tokens & q_tokens)
        total_score = kw_score * 2 + overlap_score

        if total_score > 0:
            scored.append((total_score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, entry in scored[:top_k]:
        results.append({
            "id": entry.id,
            "category": entry.category,
            "question": entry.question,
            "answer": entry.answer,
            "score": score,
        })

    return results


def get_faq_by_id(faq_id: str) -> dict | None:
    for entry in FAQ_ENTRIES:
        if entry.id == faq_id:
            return {
                "id": entry.id,
                "category": entry.category,
                "question": entry.question,
                "answer": entry.answer,
            }
    return None
"""Demo tenant: customers, orders, tickets and the sample knowledge base in sample_data/."""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Customer, Order, Ticket

logger = logging.getLogger(__name__)
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
TITLES = {"faq.json": "Frequently Asked Questions", "resolved_tickets.csv": "Resolved Support Tickets"}

CUSTOMERS = [
    ("cus_1001", "Jane Cooper", "jane@example.com", "pro", "active", [
        ("ORD-50021", "delivered", 24900, [{"sku": "THERMO-2", "name": "Acme Thermostat Gen 2", "qty": 1}], 40),
        ("ORD-50388", "shipped", 5900, [{"sku": "SENSOR-R", "name": "Room Sensor", "qty": 2}], 3),
    ]),
    ("cus_1002", "Marcus Lee", "marcus@example.com", "starter", "past_due", [
        ("ORD-49877", "delivered", 12900, [{"sku": "PLUG-4", "name": "Smart Plug 4-pack", "qty": 1}], 95),
    ]),
    ("cus_1003", "Priya Shah", "priya@example.com", "enterprise", "active", [
        ("ORD-50410", "processing", 189000, [{"sku": "HUB-PRO", "name": "Acme Hub Pro", "qty": 10}], 1),
    ]),
]


def seed_demo_data(session: Session, ingestion) -> bool:
    if session.scalar(select(func.count()).select_from(Customer)):
        return False
    now = datetime.now(UTC)
    for ext_id, name, email, plan, status, orders in CUSTOMERS:
        c = Customer(external_id=ext_id, name=name, email=email, plan=plan, status=status,
                     created_at=now - timedelta(days=400))
        for number, ostatus, cents, items, days_ago in orders:
            c.orders.append(Order(order_number=number, status=ostatus, total_cents=cents, items=items,
                                  created_at=now - timedelta(days=days_ago)))
        session.add(c)
    session.flush()
    marcus = session.scalar(select(Customer).where(Customer.external_id == "cus_1002"))
    session.add(Ticket(ticket_number="TCK-DEMO0001", customer_id=marcus.id, subject="Card declined on renewal",
                       description="Customer reports card declined during plan renewal.", priority="normal",
                       status="open", channel="web"))
    session.commit()

    for path in sorted((SAMPLE_DIR / "docs").glob("*")):
        if path.is_file():
            visibility = "internal" if path.name.startswith("internal_") else "public"
            ingestion.ingest_file(session, path.name, path.read_bytes(), visibility, TITLES.get(path.name))
    logger.info("demo data seeded")
    return True

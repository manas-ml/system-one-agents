"""A mock shop backend: typed records, typed tool arguments, typed tool results.

In a real deployment these tools would sit behind an MCP server or internal APIs.
The agent only sees `TOOLS`: name, description, argument model, risk.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

TODAY = date(2026, 11, 24)

REFUND_POLICY = """\
Refunds go to the original payment method.
Lost in transit: if the carrier marks the parcel lost, offer the customer either a full \
refund of the order or a free replacement, whichever they prefer. Express shipping on a \
replacement for a lost parcel is free.
Late delivery: if the order arrived more than 5 days after the promised date, refund the \
shipping fee.
Delivered but not received: if the carrier marks the parcel delivered and the customer says \
it never arrived, do not refund or replace it. Escalate to a human agent, who opens a \
carrier claim.
Still in transit and not late: no refund. Tell the customer the expected delivery date.
Damaged or wrong items: full refund of the item price.\
"""


# --------------------------------------------------------------------------- records


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Customer(Record):
    id: str
    name: str
    email: str
    tier: Literal["standard", "gold"]


class Item(Record):
    name: str
    price_cents: int


class Order(Record):
    id: str
    customer_id: str
    placed: date
    items: list[Item]
    shipping_fee_cents: int
    promised_delivery: date
    refunded_cents: int = 0

    @property
    def total_cents(self) -> int:
        return sum(i.price_cents for i in self.items) + self.shipping_fee_cents

    def summary(self) -> str:
        names = ", ".join(i.name for i in self.items)
        return f"{self.id}: {names}; placed {self.placed}; total €{self.total_cents / 100:.2f}"


class Shipment(Record):
    order_id: str
    carrier: str
    tracking: str
    status: Literal["in_transit", "delivered", "lost"]
    eta: date | None
    last_scan: str


class OrderList(Record):
    orders: list[Order]


class Refund(Record):
    id: str
    order_id: str
    amount_cents: int
    reason: RefundReason


class Reshipment(Record):
    id: str
    order_id: str
    express: bool
    eta: date


class Escalation(Record):
    id: str
    reason: EscalationReason
    customer_id: str | None
    order_id: str | None


class EmailReceipt(Record):
    id: str
    to: str
    template: EmailTemplate
    ref_id: str


# --------------------------------------------------------------------------- enums


class RefundReason(StrEnum):
    LOST_IN_TRANSIT = "lost_in_transit"
    LATE_DELIVERY = "late_delivery"
    DAMAGED_ITEM = "damaged_item"
    WRONG_ITEM = "wrong_item"


class EscalationReason(StrEnum):
    CARRIER_SAYS_DELIVERED = "carrier_says_delivered"
    POLICY_CHECK_FAILED = "policy_check_failed"
    AGENT_UNSURE = "agent_unsure"
    CUSTOMER_UNREACHABLE = "customer_unreachable"
    OTHER = "other"


class EmailTemplate(StrEnum):
    REFUND_ISSUED = "refund_issued"
    REPLACEMENT_SHIPPED = "replacement_shipped"
    DELIVERY_UPDATE = "delivery_update"
    ESCALATED = "escalated"


Refund.model_rebuild()
Escalation.model_rebuild()
EmailReceipt.model_rebuild()


# --------------------------------------------------------------------------- tool args


class FindCustomerArgs(Record):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ListOrdersArgs(Record):
    customer_id: str
    since_days: int = 30


class TrackShipmentArgs(Record):
    order_id: str


class IssueRefundArgs(Record):
    order_id: str
    amount_cents: int = Field(gt=0)
    reason: RefundReason


class ReshipOrderArgs(Record):
    order_id: str
    express: bool = False


class EscalateArgs(Record):
    reason: EscalationReason
    customer_id: str | None = None
    order_id: str | None = None


class SendEmailArgs(Record):
    customer_id: str
    template: EmailTemplate
    ref_id: str


# --------------------------------------------------------------------------- backend


class ToolError(RuntimeError):
    pass


@dataclass
class ShopDB:
    customers: dict[str, Customer]
    orders: dict[str, Order]
    shipments: dict[str, Shipment]
    refunds: list[Refund] = field(default_factory=list)
    reshipments: list[Reshipment] = field(default_factory=list)
    escalations: list[Escalation] = field(default_factory=list)
    outbox: list[EmailReceipt] = field(default_factory=list)
    next_id: int = 901

    def _id(self, prefix: str) -> str:
        self.next_id += 1
        return f"{prefix}-{self.next_id}"

    def find_customer(self, a: FindCustomerArgs) -> Customer:
        for c in self.customers.values():
            if c.email.lower() == a.email.lower():
                return c
        raise ToolError(f"no customer with email {a.email}")

    def list_orders(self, a: ListOrdersArgs) -> OrderList:
        if a.customer_id not in self.customers:
            raise ToolError(f"unknown customer {a.customer_id}")
        since = TODAY - timedelta(days=a.since_days)
        found = [o for o in self.orders.values()
                 if o.customer_id == a.customer_id and o.placed >= since]
        return OrderList(orders=sorted(found, key=lambda o: o.placed, reverse=True))

    def track_shipment(self, a: TrackShipmentArgs) -> Shipment:
        if a.order_id not in self.shipments:
            raise ToolError(f"no shipment for {a.order_id}")
        return self.shipments[a.order_id]

    def issue_refund(self, a: IssueRefundArgs) -> Refund:
        order = self.orders.get(a.order_id)
        if order is None:
            raise ToolError(f"unknown order {a.order_id}")
        balance = order.total_cents - order.refunded_cents
        if a.amount_cents > balance:
            raise ToolError(f"refund {a.amount_cents} exceeds refundable balance {balance}")
        self.orders[a.order_id] = order.model_copy(
            update={"refunded_cents": order.refunded_cents + a.amount_cents})
        refund = Refund(id=self._id("RF"), order_id=a.order_id,
                        amount_cents=a.amount_cents, reason=a.reason)
        self.refunds.append(refund)
        return refund

    def reship_order(self, a: ReshipOrderArgs) -> Reshipment:
        if a.order_id not in self.orders:
            raise ToolError(f"unknown order {a.order_id}")
        r = Reshipment(id=self._id("RS"), order_id=a.order_id, express=a.express,
                       eta=TODAY + timedelta(days=1 if a.express else 4))
        self.reshipments.append(r)
        return r

    def escalate_to_human(self, a: EscalateArgs) -> Escalation:
        e = Escalation(id=self._id("ESC"), reason=a.reason,
                       customer_id=a.customer_id, order_id=a.order_id)
        self.escalations.append(e)
        return e

    def send_email(self, a: SendEmailArgs) -> EmailReceipt:
        customer = self.customers.get(a.customer_id)
        if customer is None:
            raise ToolError(f"unknown customer {a.customer_id}")
        m = EmailReceipt(id=self._id("EM"), to=customer.email, template=a.template,
                         ref_id=a.ref_id)
        self.outbox.append(m)
        return m


# --------------------------------------------------------------------------- registry


class ToolName(StrEnum):
    FIND_CUSTOMER = "find_customer"
    LIST_ORDERS = "list_orders"
    TRACK_SHIPMENT = "track_shipment"
    ISSUE_REFUND = "issue_refund"
    RESHIP_ORDER = "reship_order"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    SEND_EMAIL = "send_email"


A = TypeVar("A", bound=Record)
R = TypeVar("R", bound=Record)


@dataclass(frozen=True)
class Tool(Generic[A, R]):
    name: ToolName
    description: str
    args: type[A]
    result: type[R]
    risk: Literal["read", "write"]
    run: Callable[[ShopDB, A], R]


_TOOL_LIST: list[Tool[Any, Any]] = [
        Tool(ToolName.FIND_CUSTOMER, "Look up the customer account by email address.",
             FindCustomerArgs, Customer, "read", ShopDB.find_customer),
        Tool(ToolName.LIST_ORDERS, "List the customer's recent orders.",
             ListOrdersArgs, OrderList, "read", ShopDB.list_orders),
        Tool(ToolName.TRACK_SHIPMENT,
             "Get the carrier status (in transit, delivered or lost) of one order.",
             TrackShipmentArgs, Shipment, "read", ShopDB.track_shipment),
        Tool(ToolName.ISSUE_REFUND, "Refund money for an order to the original payment method.",
             IssueRefundArgs, Refund, "write", ShopDB.issue_refund),
        Tool(ToolName.RESHIP_ORDER, "Send a free replacement of an order.",
             ReshipOrderArgs, Reshipment, "write", ShopDB.reship_order),
        Tool(ToolName.ESCALATE_TO_HUMAN,
             "Hand the case to a human agent, e.g. for a carrier claim or anything the "
             "policy does not let the agent resolve.",
             EscalateArgs, Escalation, "write", ShopDB.escalate_to_human),
        Tool(ToolName.SEND_EMAIL, "Email the customer about what was done.",
             SendEmailArgs, EmailReceipt, "write", ShopDB.send_email),
]
TOOLS: dict[ToolName, Tool[Any, Any]] = {t.name: t for t in _TOOL_LIST}


# --------------------------------------------------------------------------- seed data


def _d(days_ago: int) -> date:
    return TODAY - timedelta(days=days_ago)


_SEED = ShopDB(
    customers={c.id: c for c in [
        Customer(id="C-001", name="Jane Doe", email="jane@acme.com", tier="gold"),
        Customer(id="C-002", name="Omar Haddad", email="omar@example.com", tier="standard"),
        Customer(id="C-003", name="Li Wei", email="li@example.com", tier="standard"),
        Customer(id="C-004", name="Sam Rivera", email="sam@example.com", tier="standard"),
        Customer(id="C-005", name="Ana Silva", email="ana@example.com", tier="gold"),
    ]},
    orders={o.id: o for o in [
        Order(id="ORD-1042", customer_id="C-001", placed=_d(8),
              items=[Item(name="Wireless headphones", price_cents=5400)],
              shipping_fee_cents=499, promised_delivery=_d(3)),
        Order(id="ORD-0977", customer_id="C-001", placed=_d(90),
              items=[Item(name="USB-C cable", price_cents=1200)],
              shipping_fee_cents=0, promised_delivery=_d(85)),
        Order(id="ORD-2001", customer_id="C-002", placed=_d(4),
              items=[Item(name="Running shoes", price_cents=8900)],
              shipping_fee_cents=0, promised_delivery=TODAY + timedelta(days=2)),
        Order(id="ORD-3001", customer_id="C-003", placed=_d(9),
              items=[Item(name="Coffee grinder", price_cents=6500)],
              shipping_fee_cents=599, promised_delivery=_d(4)),
        Order(id="ORD-4001", customer_id="C-004", placed=_d(6),
              items=[Item(name="Noise-cancelling headphones", price_cents=12900)],
              shipping_fee_cents=0, promised_delivery=_d(2)),
        Order(id="ORD-4002", customer_id="C-004", placed=_d(5),
              items=[Item(name="Phone case", price_cents=1900)],
              shipping_fee_cents=399, promised_delivery=_d(1)),
        Order(id="ORD-5001", customer_id="C-005", placed=_d(7),
              items=[Item(name="Desk lamp", price_cents=4500)],
              shipping_fee_cents=499, promised_delivery=_d(2)),
    ]},
    shipments={s.order_id: s for s in [
        Shipment(order_id="ORD-1042", carrier="ParcelCo", tracking="PC123456", status="lost",
                 eta=None, last_scan="Sorting hub, 6 days ago; marked lost yesterday"),
        Shipment(order_id="ORD-0977", carrier="ParcelCo", tracking="PC100001",
                 status="delivered", eta=None, last_scan="Delivered 85 days ago"),
        Shipment(order_id="ORD-2001", carrier="FastShip", tracking="FS777", status="in_transit",
                 eta=TODAY + timedelta(days=2), last_scan="Regional depot, today"),
        Shipment(order_id="ORD-3001", carrier="ParcelCo", tracking="PC300300", status="lost",
                 eta=None, last_scan="Marked lost 2 days ago"),
        Shipment(order_id="ORD-4001", carrier="FastShip", tracking="FS4001", status="lost",
                 eta=None, last_scan="Marked lost yesterday"),
        Shipment(order_id="ORD-4002", carrier="FastShip", tracking="FS4002",
                 status="in_transit", eta=TODAY + timedelta(days=1),
                 last_scan="Out for delivery tomorrow"),
        Shipment(order_id="ORD-5001", carrier="ParcelCo", tracking="PC5001", status="delivered",
                 eta=None, last_scan="Delivered to front door, 2 days ago"),
    ]},
)


def seed_db() -> ShopDB:
    """A fresh copy of the demo data (each scenario starts from it)."""
    return copy.deepcopy(_SEED)

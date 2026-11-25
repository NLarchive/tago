"""Ring 0 - domain: pure data contracts with zero infrastructure dependencies.

Contains canonical hashing helpers, error types, the agent step contract and the ten
typed synthetic "worlds" (the private facts of one fictional enterprise case).
"""

# %% cell: utilities
import ast, copy, hashlib, json, math, re, time, uuid, unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Union, Optional
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

def canonical(value):
    """Deterministic JSON text: sorted keys, no whitespace. Used for hashing and exact citations."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

def norm(text):
    """Case/accent-insensitive normalisation for lexical retrieval."""
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text).casefold()) if not unicodedata.combining(c))

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class BudgetError(RuntimeError): pass      # call/time budget exhausted -> FAILED, never retried
class ContextError(RuntimeError): pass     # prompt would not fit the context window -> FAILED before any call
class CaseError(ValueError): pass          # generated case is inconsistent -> dataset error, not a workflow result
class ToolError(ValueError): pass          # tool call refused -> repair loop

def load_public_model(load):
    """Run `load()` (a Hugging Face download of a *public* model). If it fails because a stale or expired token is
    attached (Colab injects the user's `HF_TOKEN` secret automatically; an expired one turns a public download into
    `401 Unauthorized`), retry once without any token. Other errors are re-raised unchanged."""
    import os
    try: return load()
    except Exception as error:
        text = f'{type(error).__name__} {error}'.lower()
        if not any(k in text for k in ('401', 'unauthorized', 'token', 'repositorynotfound')): raise
        for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN'): os.environ.pop(name, None)
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        try: return load()
        except Exception as second:
            raise RuntimeError('Hugging Face download failed even without a token (%s). Check network access, or the model id.' % second) from second

# %% cell: step-contract
class AuditVerdict(Strict):
    valid: bool
    request_matches_world: bool
    style_matches: bool
    reason: str = Field(min_length=1, max_length=600)

class Citation(Strict):
    source_id: str
    quote: str = Field(min_length=2, max_length=800)

class Finding(Strict):
    key: str
    value: Any
    source_ids: list[str] = Field(min_length=1)

class ToolRequest(Strict):
    """Transport-neutral tool call passed to the tool registry."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class AgentStepBase(Strict):
    """Shared validation for the generic and the per-pack constrained step schemas."""
    @model_validator(mode='after')
    def consistent(self):
        if self.kind == 'tool' and (self.tool is None or self.status is not None or self.action is not None):
            raise ValueError('Tool steps require a tool and no final status/action')
        if self.kind == 'final' and (self.tool is not None or self.status is None or not self.action or not self.findings):
            raise ValueError('Final steps require status, action and findings, with tool=null')
        return self

class AgentStep(AgentStepBase):
    """v37-compatible generic step. v38 workers use build_step_schema() instead (section 8.1)."""
    kind: Literal['tool', 'final']
    tool: Optional[ToolRequest] = None
    status: Optional[Literal['ANSWERED', 'PROPOSED', 'PENDING_APPROVAL']] = None
    action: Optional[str] = None
    findings: list[Finding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    summary: str = Field(default='', max_length=1500)

def tool_request_of(step):
    """Convert a generic or constrained tool step into a ToolRequest."""
    tool = step.tool
    arguments = tool.arguments if isinstance(tool.arguments, dict) else tool.arguments.model_dump()
    return ToolRequest(name=tool.name, arguments=arguments)

# %% cell: quote-fragments
def quote_fragments(content, limit=300, maximum=40):
    """Verbatim fragments of an observation that may be cited: the whole text, its lines, and every top-level
    `"key":value` pair of a JSON object (also one embedded after a text prefix). Every fragment is a literal substring of
    `content`, so a citation chosen from this list always satisfies the evidence guard.

    Why a list at all: a 7B model asked to *copy* JSON into a free string re-formats it (`"refund_cents": 9000`) and
    repeats the mistake after feedback (v3.8.2 Colab run, 6 of 12 cases). Choosing among exact fragments keeps the
    provenance rule literal and moves the model's job from typing to selecting.
    """
    content = str(content); found = []
    def add(text):
        text = text.strip()
        if text and len(text) <= limit and text in content and text not in found: found.append(text)
    add(content)
    start = content.find('{')
    for candidate in ([0] if content.lstrip().startswith('{') else []) + ([start] if start > 0 else []):
        try: data = json.loads(content[candidate:])
        except ValueError: continue
        if isinstance(data, dict):
            for key, value in data.items(): add(json.dumps({key: value}, separators=(',', ':'), ensure_ascii=False)[1:-1])
            if candidate: add(content[candidate:])
    for line in content.splitlines(): add(line)
    return found[:maximum]

# %% cell: arithmetic-rule
def arithmetic_value(expression, n):
    """E06 domain rule. Evaluate n, small int constants, +, -, *, // only. Never calls eval()."""
    if len(expression) > 120: raise ToolError('Expression too long')
    try: tree = ast.parse(expression, mode='eval')
    except SyntaxError as error: raise ToolError(f'Expression is not valid arithmetic: {error.msg}')
    if len(list(ast.walk(tree))) > 35: raise ToolError('Expression too complex')
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Name) and node.id == 'n': return n
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 100: return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            else:
                if right == 0: raise ToolError('Division by zero')
                value = left // right
            if abs(value) > 10**9: raise ToolError('Arithmetic magnitude limit')
            return value
        raise ToolError('Expression contains disallowed syntax. Allowed: the variable n, integer constants with absolute value <= 100, and the operators + - * // with parentheses')
    return visit(tree)

# %% cell: worlds-support-sales
class SupportWorld(Strict):
    ticket_id: str = Field(min_length=3, max_length=40)
    customer_id: str = Field(min_length=3, max_length=40)
    refund_cents: int = Field(gt=0, le=1000000)
    self_service_limit_cents: int = Field(ge=0, le=1000000)
    @model_validator(mode='after')
    def escalation_case(self):
        if self.refund_cents <= self.self_service_limit_cents: raise ValueError('Refund must exceed self-service limit')
        return self

class SalesWorld(Strict):
    account_id: str = Field(min_length=3, max_length=40)
    company_name: str = Field(min_length=3, max_length=80)
    deal_cents: int = Field(gt=0, le=100000000)
    next_step: Literal['schedule_demo']

# %% cell: worlds-finance-it
class FinanceWorld(Strict):
    invoice_id: str = Field(min_length=3, max_length=40)
    billed_units: int = Field(gt=0, le=10000)
    received_units: int = Field(ge=0, le=10000)
    unit_price_cents: int = Field(gt=0, le=100000)
    @model_validator(mode='after')
    def discrepancy(self):
        if self.billed_units <= self.received_units: raise ValueError('Billed units must exceed received units')
        return self

class ITWorld(Strict):
    employee_id: str = Field(min_length=3, max_length=40)
    application: str = Field(min_length=3, max_length=60)
    privileged: Literal[True]
    manager_approved: Literal[False]

# %% cell: worlds-knowledge-engineering
class KnowledgeWorld(Strict):
    project_id: str = Field(min_length=3, max_length=40)
    chosen_database: Literal['PostgreSQL', 'SQLite']
    rationale: Literal['transaction_integrity', 'embedded_deployment']
    @model_validator(mode='after')
    def decision_pair(self):
        pairs = {'PostgreSQL': 'transaction_integrity', 'SQLite': 'embedded_deployment'}
        if pairs[self.chosen_database] != self.rationale: raise ValueError('Decision/rationale mismatch')
        return self

class EngineeringWorld(Strict):
    issue_id: str = Field(min_length=3, max_length=40)
    operation: Literal['inclusive_sum']
    buggy_expression: Literal['n * (n - 1) // 2']
    maximum_n: int = Field(ge=5, le=100)

# %% cell: worlds-procurement-legal
class Vendor(Strict):
    vendor_id: str = Field(min_length=3, max_length=40)
    unit_price_cents: int = Field(gt=0, le=100000)
    lead_days: int = Field(ge=1, le=90)

class ProcurementWorld(Strict):
    sku: str = Field(min_length=3, max_length=40)
    on_hand: int = Field(ge=0, le=1000)
    target_stock: int = Field(gt=0, le=1000)
    maximum_lead_days: int = Field(ge=1, le=90)
    vendors: list[Vendor] = Field(min_length=2, max_length=4)
    @model_validator(mode='after')
    def feasible(self):
        if self.on_hand >= self.target_stock: raise ValueError('Stock must be below target')
        if len({v.vendor_id for v in self.vendors}) != len(self.vendors): raise ValueError('Duplicate vendor')
        if not any(v.lead_days <= self.maximum_lead_days for v in self.vendors): raise ValueError('No feasible vendor')
        return self

class LegalWorld(Strict):
    entity_id: str = Field(min_length=3, max_length=40)
    jurisdiction: Literal['SYNTH_A', 'SYNTH_B']
    employee_count: int = Field(ge=1, le=100000)
    rule_threshold: int = Field(ge=1, le=100000)
    @model_validator(mode='after')
    def applicable_case(self):
        if self.employee_count < self.rule_threshold: raise ValueError('This canonical rule must apply')
        return self

# %% cell: worlds-security-assistant
class SecurityWorld(Strict):
    alert_id: str = Field(min_length=3, max_length=40)
    source_host: str = Field(min_length=3, max_length=40)
    sensitive_asset: str = Field(min_length=3, max_length=40)
    failed_logins: int = Field(ge=1, le=10000)
    threshold: int = Field(ge=1, le=10000)
    @model_validator(mode='after')
    def actionable(self):
        if self.failed_logins < self.threshold or self.source_host == self.sensitive_asset:
            raise ValueError('Require threshold breach and distinct connected assets')
        return self

class Offer(Strict):
    offer_id: str = Field(min_length=3, max_length=40)
    hour: int = Field(ge=0, le=23)
    price_cents: int = Field(gt=0, le=1000000)

class AssistantWorld(Strict):
    trip_id: str = Field(min_length=3, max_length=40)
    busy_hours: list[int] = Field(min_length=1, max_length=8)
    budget_cents: int = Field(gt=0, le=1000000)
    offers: list[Offer] = Field(min_length=2, max_length=4)
    @model_validator(mode='after')
    def feasible(self):
        if any(type(h) is not int or not 0 <= h <= 23 for h in self.busy_hours): raise ValueError('Invalid busy hour')
        if len({o.offer_id for o in self.offers}) != len(self.offers): raise ValueError('Duplicate offer')
        if not any(o.hour not in self.busy_hours and o.price_cents <= self.budget_cents for o in self.offers):
            raise ValueError('No feasible offer')
        return self

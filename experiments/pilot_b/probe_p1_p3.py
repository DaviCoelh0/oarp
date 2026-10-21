"""Component B pilot, probes P1-P3 (feasibility check, not part of the oarp package).

Usage (from the repo root):
  venv/bin/python experiments/pilot_b/probe_p1_p3.py --dry-run     # no API calls
  venv/bin/python experiments/pilot_b/probe_p1_p3.py --probes P1   # real, paid calls

The API key is read from oarp/.env and never printed or logged.
"""

import argparse
import json
import pathlib
import re
import sys
import time
from types import SimpleNamespace

OARP = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OARP / "src"))
from oarp.core.tools import Tool, ToolError, ToolRegistry, validate_arguments  # noqa: E402
from oarp.core.types import ToolCall, ToolSpec  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"

# --- fixed generation settings (recorded in every log line) -------------------
MODEL = "claude-haiku-5-5"
SETTINGS = {"max_tokens": 4096, "output_config": {"effort": "medium"}}
TOOL_CHOICE = {"type": "auto", "disable_parallel_tool_use": True}
# Haiku 5.5 prices for prompts <= 100K tokens (USD per token), verified 2026-10-19.
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6
PROMPT_TIER_LIMIT = 100_000

SYSTEM = (
    "You are an agent that solves arithmetic tasks using tools. Use the add tool for "
    "every addition; do not compute sums yourself. When you have the final result, "
    "call finish with the answer."
)

ADD = ToolSpec(
    "add",
    "Add two integers and return their sum.",
    {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)
FINISH = ToolSpec(
    "finish",
    "Call this when you are done. Give your final answer.",
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)


def api_tool(spec: ToolSpec) -> dict:
    return {"name": spec.name, "description": spec.description, "input_schema": spec.parameters}


# --- budget guard --------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Client-side guard. It refuses a call whose WORST-CASE cost could cross the
    cap, but it can only estimate input size before the call; the real charge is
    known only from `usage` afterwards."""

    def __init__(self, cap_usd: float) -> None:
        self.cap, self.spent, self.calls = cap_usd, 0.0, 0

    def check(self, est_input_tokens: int, max_tokens: int) -> None:
        if est_input_tokens > PROMPT_TIER_LIMIT * 0.8:
            raise BudgetExceeded("prompt too close to the 100K price tier")
        worst = est_input_tokens * PRICE_IN + max_tokens * PRICE_OUT
        if self.spent + worst > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} + worst case ${worst:.4f} > cap ${self.cap}")

    def add(self, usage) -> float:
        cost = usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT
        self.spent += cost
        self.calls += 1
        return cost


# --- model access ---------------------------------------------------------------
def load_key() -> str:
    text = (OARP / ".env").read_text()
    m = re.search(r'^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*["\']?([^"\'\s]+)', text, re.M)
    if not m:
        raise SystemExit("ANTHROPIC_API_KEY not found in .env")
    return m.group(1)


class FakeClient:
    """Dry-run stand-in: always calls add(2, 3) once, then finish('5')."""

    class _Messages:
        def create(self, **kwargs):
            has_result = any(
                isinstance(m["content"], list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
                for m in kwargs["messages"]
            )
            tool_name, args = ("finish", {"answer": "5"}) if has_result else ("add", {"a": 2, "b": 3})
            block = SimpleNamespace(type="tool_use", id=f"fake_{len(kwargs['messages'])}", name=tool_name, input=args)
            return SimpleNamespace(
                content=[block], stop_reason="tool_use", model="fake",
                usage=SimpleNamespace(input_tokens=500, output_tokens=50), _request_id="dry-run",
            )

    messages = _Messages()


def estimate_tokens(request: dict) -> int:
    # Conservative heuristic (~3 chars/token) + tool-use system prompt overhead.
    return len(json.dumps(request, default=str)) // 3 + 286


def call_model(client, budget: Budget, log, *, probe: str, run: str, step: int, **request) -> object:
    request = {"model": MODEL, **SETTINGS, **request}
    est = estimate_tokens(request)
    budget.check(est, request["max_tokens"])
    started = time.perf_counter()
    response = client.messages.create(**request)
    latency = time.perf_counter() - started
    cost = budget.add(response.usage)
    log({
        "probe": probe, "run": run, "step": step, "model": response.model,
        "settings": SETTINGS, "tool_choice": request.get("tool_choice"),
        "request_id": getattr(response, "_request_id", None),
        "stop_reason": response.stop_reason, "latency_s": round(latency, 3),
        "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
        "est_input_tokens": est, "cost_usd": round(cost, 6),
        "blocks": [b.type for b in response.content],
        "tool_calls": [{"name": b.name, "input": b.input} for b in response.content if b.type == "tool_use"],
        "text": " ".join(b.text for b in response.content if b.type == "text")[:300],
    })
    return response


# --- P1: basic communication -----------------------------------------------------
def p1(client, budget, log) -> dict:
    r = call_model(client, budget, log, probe="P1", run="pong", step=1,
                   messages=[{"role": "user", "content": "Reply with exactly the word: pong"}])
    text = " ".join(b.text for b in r.content if b.type == "text").strip()
    return {"probe": "P1", "stop_reason": r.stop_reason, "text": text, "pass": text.lower().strip(".") == "pong"}


# --- P2/P3: mini tool loop --------------------------------------------------------
def make_add_handler(scenario: str):
    """Returns a handler for `add` implementing the probe scenario."""
    state = {"calls": 0}

    def handler(a: int, b: int) -> str:
        state["calls"] += 1
        if scenario == "transient" and state["calls"] == 1:
            raise ToolError("service temporarily unavailable, please retry")
        if scenario == "range" and (abs(a) > 100 or abs(b) > 100):
            # v1 (2026-10-21 04:06 run): unsolvable for 150 + 275, since chained
            # sums exceed 100 and can't be fed back in. Kept for the logged run.
            raise ToolError("each argument must be between -100 and 100")
        if scenario == "range_b" and abs(b) > 100:
            # v2: solvable, e.g. add(150, 100) -> add(250, 100) -> add(350, 75).
            raise ToolError("argument 'b' must be between -100 and 100")
        if scenario == "down":
            raise ToolError("the add service is down")
        return str(a + b)

    return handler


def tool_loop(client, budget, log, *, probe: str, run: str, prompt: str, scenario: str, max_calls: int) -> dict:
    registry = ToolRegistry([Tool(ADD, make_add_handler(scenario))])
    messages: list[dict] = [{"role": "user", "content": prompt}]
    record = {"probe": probe, "run": run, "scenario": scenario, "prompt": prompt,
              "model_calls": 0, "tool_calls": 0, "invalid_calls": 0, "tool_errors": 0,
              "multi_call_responses": 0, "stop": "max_calls", "answer": None, "tool_outputs": []}
    for step in range(1, max_calls + 1):
        r = call_model(client, budget, log, probe=probe, run=run, step=step, system=SYSTEM,
                       tools=[api_tool(ADD), api_tool(FINISH)], tool_choice=TOOL_CHOICE, messages=messages)
        record["model_calls"] += 1
        # Append the full content unchanged: thinking blocks must be passed back as-is.
        messages.append({"role": "assistant", "content": r.content})
        uses = [b for b in r.content if b.type == "tool_use"]
        if len(uses) > 1:
            record["multi_call_responses"] += 1
        if not uses:
            record["stop"] = f"no_tool_call ({r.stop_reason})"
            break
        results = []
        for use in uses:
            record["tool_calls"] += 1
            if use.name == "finish":
                answer = use.input.get("answer") if isinstance(use.input, dict) else None
                if isinstance(answer, str):
                    record["stop"], record["answer"] = "finished", answer
                    return record
                record["invalid_calls"] += 1
                results.append({"type": "tool_result", "tool_use_id": use.id,
                                "content": "Error: missing required argument(s): answer", "is_error": True})
                continue
            # "Invalid" = the model's call itself was malformed (unknown tool or bad
            # arguments), as opposed to a valid call that the tool then rejected.
            try:
                if use.name not in registry:
                    raise ToolError("unknown tool")
                validate_arguments(ADD.parameters, use.input)
            except ToolError:
                record["invalid_calls"] += 1
            res = registry.execute(ToolCall(use.id, use.name, use.input))
            if res.is_error:
                record["tool_errors"] += 1
            else:
                record["tool_outputs"].append(res.content)
            results.append({"type": "tool_result", "tool_use_id": use.id,
                            "content": res.content, "is_error": res.is_error})
        messages.append({"role": "user", "content": results})
    return record


P2_CASES = [  # (run id, prompt, expected answer)
    ("sum2", "What is 17 + 25?", "42"),
    ("sum3", "Compute 123 + 456 + 789.", "1368"),
    ("negative", "What is negative 40 plus 15?", "-25"),
    ("words", "Add three hundred and twelve to 1,000.", "1312"),
    ("float", "What is 2.5 + 2.5?", None),  # schema allows only integers: observe behaviour
]
P3_CASES = [  # (run id, scenario, prompt, expected answer)
    ("transient", "transient", "What is 30 + 12?", "42"),
    ("range", "range", "What is 150 + 275?", "425"),
    ("down", "down", "What is 8 + 9?", None),  # tool always fails: does it report failure?
    ("range_b", "range_b", "What is 150 + 275?", "425"),  # v2 of "range", solvable
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probes", default="P1,P2,P3")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cap", type=float, default=0.50)
    ap.add_argument("--only", default="", help="comma-separated P2/P3 run ids to run, e.g. range_b")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tag = "dryrun" if args.dry_run else time.strftime("%Y%m%dT%H%M%S")
    calls_path, runs_path = OUT / f"{tag}_calls.jsonl", OUT / f"{tag}_runs.jsonl"
    calls_f, runs_f = calls_path.open("w"), runs_path.open("w")

    def log(entry: dict) -> None:
        calls_f.write(json.dumps(entry) + "\n")
        calls_f.flush()

    def save(rec: dict, expected=None) -> None:
        rec["expected"] = expected
        # Heuristic grade: last integer in the answer (commas removed, unicode minus
        # normalised) must equal the expected value. Answers are also reviewed by hand.
        numbers = re.findall(r"-?\d+", (rec.get("answer") or "").replace(",", "").replace("−", "-"))
        rec["correct"] = None if expected is None else bool(numbers) and numbers[-1] == expected
        # Stricter check (added after a false positive in range#0): the expected
        # value must also have been produced by a successful tool call.
        if "tool_outputs" in rec:
            rec["correct_via_tool"] = None if expected is None else bool(rec["correct"]) and expected in rec["tool_outputs"]
        runs_f.write(json.dumps(rec) + "\n")
        runs_f.flush()
        print(json.dumps(rec))

    if args.dry_run:
        client = FakeClient()
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=load_key(), timeout=60.0, max_retries=1)

    budget = Budget(args.cap)
    probes = args.probes.split(",")
    only = set(filter(None, args.only.split(",")))
    p2_cases = [c for c in P2_CASES if not only or c[0] in only]
    p3_cases = [c for c in P3_CASES if (not only and c[0] != "range_b") or c[0] in only]
    try:
        if "P1" in probes:
            save(p1(client, budget, log))
        for rep in range(args.repeats):
            if "P2" in probes:
                for run, prompt, expected in p2_cases:
                    save(tool_loop(client, budget, log, probe="P2", run=f"{run}#{rep}", prompt=prompt,
                                   scenario="normal", max_calls=6), expected)
            if "P3" in probes:
                for run, scenario, prompt, expected in p3_cases:
                    save(tool_loop(client, budget, log, probe="P3", run=f"{run}#{rep}", prompt=prompt,
                                   scenario=scenario, max_calls=8), expected)
    except BudgetExceeded as exc:
        print(f"STOPPED by budget guard: {exc}")
    finally:
        calls_f.close()
        runs_f.close()
        print(f"model calls: {budget.calls}, estimated spend: ${budget.spent:.4f}")
        print(f"logs: {calls_path}, {runs_path}")


if __name__ == "__main__":
    main()

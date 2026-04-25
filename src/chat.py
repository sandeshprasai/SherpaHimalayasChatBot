"""
chat.py
────────
Interactive terminal interface for the Sherpa Himalaya chatbot.
Run from project root:

    python src/chat.py
"""

import os
# Early telemetry/analytics suppression to avoid noisy errors from deps
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    import analytics
    analytics.capture = lambda *args, **kwargs: None
except Exception:
    pass
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import sys

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.pipeline import Pipeline
from config.settings import settings

# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BLUE   = "\033[94m"


def banner():
    print(f"""
{CYAN}{BOLD}
  ███████╗██╗  ██╗███████╗██████╗ ██████╗  █████╗
  ██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗
  ███████╗███████║█████╗  ██████╔╝██████╔╝███████║
  ╚════██║██╔══██║██╔══╝  ██╔══██╗██╔═══╝ ██╔══██║
  ███████║██║  ██║███████╗██║  ██║██║     ██║  ██║
  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
{RESET}
{BOLD}  🏔️  Himalaya Travel Assistant{RESET}
{DIM}  Powered by {settings.llm_model}{RESET}
{DIM}  Type  'help'  for commands  |  'quit' to exit{RESET}
""")


def help_text():
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}help{RESET}          Show this help
  {CYAN}debug on/off{RESET}  Toggle routing debug info
  {CYAN}stats{RESET}         Show session token usage
  {CYAN}clear{RESET}         Clear the screen
  {CYAN}quit / exit{RESET}   Exit the chatbot

{BOLD}Example queries:{RESET}
  {DIM}What is the duration of Ghorepani Circuit Trek?
  Which trek has the highest altitude?
  Which trek is the easiest?
  Which trek has the longest duration?
  Suggest a 7-day trek in Annapurna region
  Which trek is best for beginners?
  Compare Ghorepani Circuit Trek vs Everest Base Camp Trek
  Describe the full itinerary of Langtang Valley Trek
  What is included in the EBC trek package?{RESET}
""")


def print_response(resp, debug: bool):
    print()

    # Answer
    print(f"{GREEN}{BOLD}Assistant:{RESET}")
    print(f"{resp.answer}")

    # Debug info
    if debug:
        source = "metadata" if not resp.used_llm else "vector/llm"
        print(f"\n{DIM}  ┌─ debug ──────────────────────────────────────────")
        print(f"  │ intent    : {resp.intent.value}")
        print(f"  │ strategy  : {resp.strategy.value}")
        print(f"  │ llm used  : {resp.used_llm}")
        print(f"  │ fallback  : {resp.is_fallback}")
        print(f"  │ source    : {source}")
        if resp.tokens_used:
            print(f"  │ tokens    : {resp.tokens_used}")
            print(f"  │ cost      : ${resp.cost_usd:.6f}")
        print(f"  └──────────────────────────────────────────────{RESET}")

    print()


def print_stats(pipeline: Pipeline):
    tokens = pipeline.tokens_used
    cost   = tokens * 0.59 / 1_000_000
    budget = settings.daily_token_limit
    pct    = (tokens / budget * 100) if budget else 0
    bar_w  = 30
    filled = int(bar_w * pct / 100)
    bar    = "█" * filled + "░" * (bar_w - filled)

    print(f"""
{BOLD}Session Stats:{RESET}
  Tokens used : {tokens:,} / {budget:,}
  Cost        : ${cost:.4f}
  Budget used : [{CYAN}{bar}{RESET}] {pct:.1f}%
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    banner()

    debug    = False
    pipeline = Pipeline(debug=False)   # pipeline has its own internal debug flag

    print(f"{DIM}Initializing... (first query will load the embedding model){RESET}\n")

    while True:
        try:
            raw = input(f"{CYAN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye! 🏔️{RESET}\n")
            break

        if not raw:
            continue

        cmd = raw.lower()

        # ── Commands ──────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print(f"\n{DIM}Goodbye! 🏔️{RESET}\n")
            break

        if cmd == "help":
            help_text()
            continue

        if cmd == "stats":
            print_stats(pipeline)
            continue

        if cmd == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            banner()
            continue

        if cmd == "debug on":
            debug = True
            pipeline.debug = True
            print(f"{DIM}Debug mode ON{RESET}\n")
            continue

        if cmd == "debug off":
            debug = False
            pipeline.debug = False
            print(f"{DIM}Debug mode OFF{RESET}\n")
            continue

        # ── Query ─────────────────────────────────────────────────────────
        print(f"{DIM}Thinking...{RESET}", end="\r")

        try:
            resp = pipeline.run(raw)
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}\n")
            continue

        print(" " * 15, end="\r")   # clear "Thinking..."
        print_response(resp, debug)


if __name__ == "__main__":
    main()
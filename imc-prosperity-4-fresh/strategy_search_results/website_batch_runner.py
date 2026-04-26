from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path

import websocket


ROOT = Path("/Users/lakshaykumar/Documents/Playground/imc-prosperity-4-fresh")
OUT = ROOT / "strategy_search_results" / "website_candidate_scores.csv"


def page_websocket_url() -> str:
    tabs = json.load(urllib.request.urlopen("http://localhost:9222/json/list", timeout=5))
    for tab in tabs:
        if tab.get("type") == "page" and "prosperity.imc.com/game" in tab.get("url", ""):
            return tab["webSocketDebuggerUrl"]
    raise RuntimeError("Prosperity tab not found. Keep the logged-in Chrome tab open.")


class ChromePage:
    def __init__(self) -> None:
        self.ws = websocket.create_connection(
            page_websocket_url(),
            timeout=10,
            suppress_origin=True,
        )
        self.counter = 0
        self.cdp("Runtime.enable")
        self.cdp("DOM.enable")

    def close(self) -> None:
        self.ws.close()

    def cdp(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        self.ws.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.counter:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def evaluate(self, expression: str):
        result = self.cdp(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        return result.get("result", {}).get("value")

    def body_text(self) -> str:
        return self.evaluate("document.body.innerText") or ""

    def click_tab(self, label: str) -> None:
        upper = label.upper()
        self.evaluate(
            f"""
(() => {{
  const target = [...document.querySelectorAll('button,[role=button],a')]
    .find((el) => (el.innerText || '').trim().toUpperCase() === {upper!r});
  if (target) target.click();
  return Boolean(target);
}})()
"""
        )

    def upload_file(self, path: Path) -> None:
        self.click_tab("UPLOAD & LOG")
        time.sleep(1)
        root = self.cdp("DOM.getDocument", {"depth": 1, "pierce": False})["root"]["nodeId"]
        query = self.cdp(
            "DOM.querySelector",
            {"nodeId": root, "selector": 'input[type=file][accept=".py"]'},
        )
        node_id = query.get("nodeId")
        if not node_id:
            raise RuntimeError("Upload file input not found")
        self.cdp("DOM.setFileInputFiles", {"nodeId": node_id, "files": [str(path)]})

    def wait_for_completion(self, filename: str, timeout_s: int = 900) -> str:
        deadline = time.time() + timeout_s
        prefixed_name = ""
        last_status = ""
        while time.time() < deadline:
            time.sleep(10)
            text = self.body_text()
            matches = re.findall(rf"\b(\d+-{re.escape(filename)})\b", text)
            if matches:
                prefixed_name = matches[0]
            interesting = [
                line
                for line in text.splitlines()
                if filename in line
                or "Simulation complete" in line
                or "Processing" in line
                or "Active algorithm" in line
                or "Error" in line
            ]
            status = " | ".join(interesting[:10])
            if status != last_status:
                print("status:", status[:800], flush=True)
                last_status = status
            if prefixed_name and f"Simulation complete for {prefixed_name}" in text:
                return prefixed_name
            if prefixed_name and "Error" in status and "Processing" not in status:
                raise RuntimeError(status)
        raise TimeoutError(f"Timed out waiting for {filename}")

    def select_performance_algorithm(self, prefixed_name: str) -> None:
        self.click_tab("PERFORMANCE")
        time.sleep(1)
        clicked = self.evaluate(
            f"""
(() => {{
  const exact = {prefixed_name!r};
  const trigger = [...document.querySelectorAll('button,[role=button]')]
    .find((el) => (el.innerText || '').includes('Choose algorithm')
      || (el.innerText || '').includes('mainbh')
      || (el.innerText || '').includes('cand_')
      || (el.innerText || '').includes('s9700_')
      || (el.innerText || '').includes('w9700_')
      || (el.innerText || '').includes('main_final'));
  if (trigger) trigger.click();
  let option = [...document.querySelectorAll('li[role=option], [role=option]')]
    .find((el) => (el.innerText || el.textContent || '').trim() === exact);
  if (!option) {{
    const select = [...document.querySelectorAll('select')]
      .find((el) => [...el.options].some((opt) => opt.text.trim() === exact));
    if (select) {{
      const opt = [...select.options].find((candidate) => candidate.text.trim() === exact);
      select.value = opt.value;
      select.dispatchEvent(new Event('input', {{ bubbles: true }}));
      select.dispatchEvent(new Event('change', {{ bubbles: true }}));
      return 'select';
    }}
  }}
  if (option) {{
    option.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true, cancelable: true, view: window }}));
    option.click();
    option.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true, cancelable: true, view: window }}));
    return 'option';
  }}
  return null;
}})()
"""
        )
        if not clicked:
            raise RuntimeError(f"Could not select performance algorithm {prefixed_name}")
        time.sleep(4)

    def chart_score(self) -> dict | None:
        return self.evaluate(
            r"""
(() => {
  function findPoints(node) {
    const seen = new Set();
    const stack = [node];
    while (stack.length) {
      const obj = stack.pop();
      if (!obj || typeof obj !== 'object' || seen.has(obj)) continue;
      seen.add(obj);
      if (Array.isArray(obj.points) && obj.points.length && obj.points[0].payload && 'value' in obj.points[0].payload) {
        const pts = obj.points;
        return {
          len: pts.length,
          first: pts[0].payload,
          last: pts[pts.length - 1].payload,
        };
      }
      for (const key of Object.keys(obj).slice(0, 100)) {
        try {
          const value = obj[key];
          if (value && typeof value === 'object') stack.push(value);
        } catch (e) {}
      }
    }
    return null;
  }
  const curves = [...document.querySelectorAll('.recharts-line-curve,.recharts-dot,.recharts-bar-rectangle')];
  for (const el of curves) {
    for (const key of Object.keys(el)) {
      if (key.startsWith('__reactFiber$') || key.startsWith('__reactProps$')) {
        const found = findPoints(el[key]);
        if (found) return found;
      }
    }
  }
  return null;
})()
"""
        )


def append_result(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT.exists()
    with OUT.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "file",
                "uploaded_name",
                "final_pnl",
                "first_pnl",
                "points",
                "note",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def score_file(filename: str, note: str = "") -> dict:
    page = ChromePage()
    try:
        path = ROOT / filename
        print(f"uploading: {filename}", flush=True)
        page.upload_file(path)
        uploaded_name = page.wait_for_completion(filename)
        print(f"complete: {uploaded_name}", flush=True)
        page.select_performance_algorithm(uploaded_name)
        score = page.chart_score()
        if not score:
            raise RuntimeError(f"No chart score found for {uploaded_name}")
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file": filename,
            "uploaded_name": uploaded_name,
            "final_pnl": score["last"]["value"],
            "first_pnl": score["first"]["value"],
            "points": score["len"],
            "note": note,
        }
        append_result(row)
        print("RESULT", json.dumps(row), flush=True)
        return row
    finally:
        page.close()


def main() -> None:
    candidates = [
        ("cand_ret1_010.py", "reduce recent-move mean reversion"),
        ("cand_ret1_000.py", "remove recent-move mean reversion"),
        ("cand_ret5_004.py", "weaker ret5 mean reversion"),
        ("cand_ret5_012.py", "stronger ret5 mean reversion"),
        ("cand_no_micro_depth.py", "remove unstable micro/depth trend terms"),
        ("cand_pressure_stronger.py", "stronger top-level liquidity pressure"),
        ("cand_offset_tighter.py", "quote slightly tighter"),
        ("cand_qsize25.py", "quote larger size"),
    ]
    for filename, note in candidates:
        score_file(filename, note)


if __name__ == "__main__":
    main()

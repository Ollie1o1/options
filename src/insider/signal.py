"""Pure cluster-buy scoring over parsed Form 4 transactions."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

NOTABLE_VALUE = 100_000.0     # single insider open-market buy worth flagging


def cluster_score(transactions: List[Dict[str, Any]],
                  today: Optional[str] = None,
                  window_days: int = 90) -> Dict[str, Any]:
    """Score insider activity in the trailing window.

    Only open-market purchases (code P) score — sells are reported but carry
    no signal (Cohen-Malloy-Pomorski: routine sells are noise). Labels:
      CLUSTER BUY  (≥2 distinct buyers)            score ≥ 0.8
      NOTABLE BUY  (1 officer/director, ≥ $100k)   0.5–0.79
      WEAK BUY     (any other purchase)            0.3
      NONE                                          0.0
    """
    today_dt = (datetime.strptime(today, "%Y-%m-%d") if today
                else datetime.now())
    cutoff = today_dt - timedelta(days=window_days)

    buys, sell_value = [], 0.0
    for t in transactions:
        try:
            tx_dt = datetime.strptime(str(t.get("date", ""))[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if tx_dt < cutoff or tx_dt > today_dt:
            continue
        if t.get("code") == "P":
            buys.append(t)
        elif t.get("code") == "S":
            sell_value += float(t.get("value") or 0)

    buyers = {b["owner"] for b in buys if b.get("owner")}
    buy_value = sum(float(b.get("value") or 0) for b in buys)

    if len(buyers) >= 2:
        score = min(1.0, 0.8 + 0.05 * (len(buyers) - 2))
        label = "CLUSTER BUY"
    elif len(buyers) == 1:
        b = buys[0]
        if (b.get("is_officer") or b.get("is_director")) and buy_value >= NOTABLE_VALUE:
            score = min(0.79, 0.5 + 0.2 * min(buy_value / 1_000_000.0, 1.0))
            label = "NOTABLE BUY"
        else:
            score, label = 0.3, "WEAK BUY"
    else:
        score, label = 0.0, "NONE"

    return {
        "n_buyers": len(buyers),
        "n_buys": len(buys),
        "buy_value": buy_value,
        "sell_value": sell_value,
        "score": score,
        "label": label,
        "window_days": window_days,
    }

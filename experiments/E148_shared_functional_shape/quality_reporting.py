"""Stable final-Q line, emitted by evaluation and copied into queue logs."""
def quality_line(result,path):
    return (f"[Q] arm={result['arm']} count={result['count']} "
        f"mixed_Q={result['mixed_Q']:.9f} weighted_Q={result['weighted_Q']:.9f} mean_Q={result['mean_Q']:.9f} "
        f"hard_legal={result['hard_legal']}/{result['count']} best_epoch={result['best_epoch']} "
        f"mean_ms={1000*result['mean_complete_seconds']:.3f} p95_ms={1000*result['p95_complete_seconds']:.3f} source={path}")

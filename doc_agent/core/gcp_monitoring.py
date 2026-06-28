"""
Optional "verified vs Google" cross-check for the token-usage panel.

The numbers shown in the UI come from each Gemini response's usage_details (the
authoritative count -- see core/llm.py). This module adds a trust signal by
reading Google's own server-side **request count** for the Generative Language
API straight from Cloud Monitoring, so the UI can show "calls: N (matches Google)".

Only the call count is available this way: an AI-Studio key exposes
`serviceruntime.googleapis.com/api/request_count` (resource type `consumed_api`)
as a free, native metric, but NOT a server-side token metric -- token totals can
only be eyeballed on the AI Studio dashboard. So this module reconciles calls
only; tokens are reconciled manually via the dashboard link in summarize_run_metrics().

Everything here degrades gracefully: if google-cloud-monitoring isn't installed,
or no GCP project / credentials are configured, every function returns None and the
UI simply omits the auto-reconciliation line. Requires (all free tier):
  - GCP_PROJECT_ID env var (the project behind the AI-Studio key)
  - Application Default Credentials, e.g. GOOGLE_APPLICATION_CREDENTIALS pointing
    at a service-account JSON with roles/monitoring.viewer
"""

import os
import logging

log = logging.getLogger(__name__)

_GEMINI_SERVICE = "generativelanguage.googleapis.com"
_REQUEST_COUNT_METRIC = "serviceruntime.googleapis.com/api/request_count"


def is_configured() -> bool:
    """True if a project id is set; cheap guard the caller can use before querying."""
    return bool(os.environ.get("GCP_PROJECT_ID"))


def get_google_call_count(window_start, window_end) -> int | None:
    """Sum Gemini API request_count over [window_start, window_end] from Cloud Monitoring.

    `window_start`/`window_end` are epoch seconds (floats/ints). Returns the total
    number of requests Google recorded for the Generative Language API in that
    window, or None if monitoring isn't available/configured or the query fails.
    Note: Cloud Monitoring ingestion lags ~1-4 min, so call this only after a short
    delay (the UI labels the figure accordingly).
    """
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        return None
    try:
        from google.cloud import monitoring_v3
    except Exception as e:  # library not installed
        log.info("Cloud Monitoring cross-check unavailable (import failed): %s", e)
        return None

    try:
        client = monitoring_v3.MetricServiceClient()
        interval = monitoring_v3.TimeInterval(
            {
                "start_time": {"seconds": int(window_start)},
                "end_time": {"seconds": int(window_end)},
            }
        )
        # Restrict to the Gemini service so we don't count unrelated API traffic.
        flt = (
            f'metric.type = "{_REQUEST_COUNT_METRIC}" '
            f'AND resource.label.service = "{_GEMINI_SERVICE}"'
        )
        results = client.list_time_series(
            request={
                "name": f"projects/{project_id}",
                "filter": flt,
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        total = 0
        for series in results:
            for point in series.points:
                val = point.value
                # request_count is a DELTA int64; sum the points across the window.
                total += int(getattr(val, "int64_value", 0) or 0)
        return total
    except Exception as e:
        log.info("Cloud Monitoring cross-check query failed: %s", e)
        return None

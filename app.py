import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple

import jwt
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("meta-attestation")

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

META_APP_ACCESS_TOKEN = os.getenv("META_APP_ACCESS_TOKEN", "").strip()
PLAYFAB_TITLE_ID = os.getenv("PLAYFAB_TITLE_ID", "").strip()
VALID_PACKAGE = os.getenv("VALID_PACKAGE", "").strip()
VALID_CERTS = {
    value.replace(":", "").strip().lower()
    for value in os.getenv("VALID_CERT_SHA256", "").split(",")
    if value.strip()
}

# enforce: require StoreRecognized + the selected device integrity policy.
# report_only: still require a real Meta-signed token, matching nonce, package,
#              and certificate, but log app/device state mismatches while testing.
ATTESTATION_MODE = os.getenv("ATTESTATION_MODE", "enforce").strip().lower()
ALLOW_BASIC_DEVICE = os.getenv("ALLOW_BASIC_DEVICE", "false").lower() == "true"

PHOTON_JWT_SECRET = os.getenv("PHOTON_JWT_SECRET", "").strip()
PHOTON_PROVIDER_SECRET = os.getenv("PHOTON_PROVIDER_SECRET", "").strip()
PHOTON_TOKEN_LIFETIME_SECONDS = int(os.getenv("PHOTON_TOKEN_LIFETIME_SECONDS", "180"))

CHALLENGE_TTL_SECONDS = int(os.getenv("CHALLENGE_TTL_SECONDS", "300"))
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "12"))
CLOCK_SKEW_SECONDS = int(os.getenv("CLOCK_SKEW_SECONDS", "60"))

META_VERIFY_URL = "https://graph.oculus.com/platform_integrity/verify"
META_USER_NONCE_URL = "https://graph.oculus.com/user_nonce_validate"

# One free Render instance + one Gunicorn worker uses this in-memory challenge store.
# A restart only makes the client request a new challenge; it does not ban anyone.
_challenges: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now() -> int:
    return int(time.time())


def clean_old_challenges() -> None:
    cutoff = now()
    with _lock:
        expired = [
            key for key, value in _challenges.items()
            if int(value.get("expires_at", 0)) < cutoff
        ]
        for key in expired:
            _challenges.pop(key, None)


def json_error(
    status: int,
    message: str,
    reason_code: str,
    retryable: bool,
    security_violation: bool = False,
):
    return jsonify({
        "ok": False,
        "allowed": False,
        "retryable": retryable,
        "security_violation": security_violation,
        "message": message,
        "reason_code": reason_code,
    }), status


def required_config_present() -> Tuple[bool, str]:
    required = {
        "META_APP_ACCESS_TOKEN": META_APP_ACCESS_TOKEN,
        "PLAYFAB_TITLE_ID": PLAYFAB_TITLE_ID,
        "VALID_PACKAGE": VALID_PACKAGE,
        "VALID_CERT_SHA256": ",".join(VALID_CERTS),
        "PHOTON_JWT_SECRET": PHOTON_JWT_SECRET,
        "PHOTON_PROVIDER_SECRET": PHOTON_PROVIDER_SECRET,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        return False, "Missing environment variables: " + ", ".join(missing)
    if ATTESTATION_MODE not in {"enforce", "report_only"}:
        return False, "ATTESTATION_MODE must be enforce or report_only."
    return True, "OK"


def verify_playfab_session(session_ticket: str) -> Tuple[bool, Optional[str], str]:
    """Validate the player's existing PlayFab client session without a title secret."""
    if not session_ticket:
        return False, None, "Missing PlayFab session ticket."

    url = f"https://{PLAYFAB_TITLE_ID}.playfabapi.com/Client/GetAccountInfo"
    try:
        response = requests.post(
            url,
            headers={
                "X-Authorization": session_ticket,
                "Content-Type": "application/json",
            },
            json={},
            timeout=UPSTREAM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"PlayFab request failed: {exc}") from exc

    if response.status_code != 200:
        return False, None, "PlayFab session is invalid or expired."

    try:
        playfab_id = response.json()["data"]["AccountInfo"]["PlayFabId"]
    except (KeyError, TypeError, ValueError):
        return False, None, "PlayFab returned an unreadable account response."

    return bool(playfab_id), str(playfab_id), "OK"


def verify_meta_user_proof(meta_user_id: str, user_proof: str) -> Tuple[bool, str]:
    """Verify that the Meta user ID and GetUserProof nonce belong together."""
    try:
        response = requests.post(
            META_USER_NONCE_URL,
            data={
                "access_token": META_APP_ACCESS_TOKEN,
                "user_id": meta_user_id,
                "nonce": user_proof,
            },
            timeout=UPSTREAM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Meta user verification request failed: {exc}") from exc

    if response.status_code != 200:
        return False, "Meta account proof was rejected."

    try:
        valid = response.json().get("is_valid") is True
    except ValueError:
        return False, "Meta account verification returned unreadable JSON."

    return valid, "OK" if valid else "Meta account proof was invalid."


def request_meta_attestation_verdict(token: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            META_VERIFY_URL,
            params={
                "token": token,
                "access_token": META_APP_ACCESS_TOKEN,
            },
            timeout=UPSTREAM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Meta attestation verification request failed: {exc}") from exc

    # 5xx and rate limiting are temporary. They must never become bans.
    if response.status_code >= 500 or response.status_code == 429:
        raise RuntimeError(f"Meta attestation service returned HTTP {response.status_code}.")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Meta attestation service returned unreadable JSON.") from exc


def decode_base64url_json(value: str) -> Dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing claims.")
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Claims were not a JSON object.")
    return decoded


def normalize_cert_list(value: Any) -> set[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []

    return {
        str(item).replace(":", "").strip().lower()
        for item in values
        if str(item).strip()
    }


def issue_photon_token(playfab_id: str, meta_user_id: str, unique_id: str) -> str:
    issued = now()
    claims = {
        "iss": "meta-attestation-backend",
        "aud": "photon",
        "sub": playfab_id,
        "meta_user_id": meta_user_id,
        "device_id_hash": hashlib.sha256(unique_id.encode("utf-8")).hexdigest()[:24]
            if unique_id else "",
        "iat": issued,
        "nbf": issued - 5,
        "exp": issued + max(60, PHOTON_TOKEN_LIFETIME_SECONDS),
        "jti": secrets.token_urlsafe(18),
    }
    return jwt.encode(claims, PHOTON_JWT_SECRET, algorithm="HS256")


def policy_failure(
    message: str,
    reason_code: str,
    security_violation: bool,
):
    if ATTESTATION_MODE == "report_only":
        log.warning("REPORT-ONLY attestation policy mismatch: %s (%s)", message, reason_code)
        return None
    return json_error(
        403,
        message,
        reason_code,
        retryable=False,
        security_violation=security_violation,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    configured, message = required_config_present()
    return jsonify({
        "ok": configured,
        "service": "meta-attestation-backend",
        "mode": ATTESTATION_MODE,
        "message": message,
    }), 200 if configured else 503


@app.post("/v1/attestation/challenge")
def create_challenge():
    configured, message = required_config_present()
    if not configured:
        log.error(message)
        return json_error(503, "Backend is not configured.", "backend_not_configured", True)

    clean_old_challenges()
    data = request.get_json(silent=True) or {}

    meta_user_id = str(data.get("meta_user_id", "")).strip()
    user_proof = str(data.get("user_proof", "")).strip()
    playfab_ticket = str(data.get("playfab_session_ticket", "")).strip()

    if not meta_user_id or not user_proof or not playfab_ticket:
        return json_error(400, "Missing authentication data.", "missing_fields", False)

    try:
        playfab_ok, playfab_id, playfab_message = verify_playfab_session(playfab_ticket)
    except RuntimeError as exc:
        log.warning("%s", exc)
        return json_error(503, "PlayFab is temporarily unavailable.", "playfab_unavailable", True)

    if not playfab_ok or not playfab_id:
        return json_error(401, playfab_message, "invalid_playfab_session", False)

    try:
        meta_ok, meta_message = verify_meta_user_proof(meta_user_id, user_proof)
    except RuntimeError as exc:
        log.warning("%s", exc)
        return json_error(503, "Meta account verification is temporarily unavailable.", "meta_user_service_unavailable", True)

    if not meta_ok:
        return json_error(401, meta_message, "invalid_meta_user_proof", False)

    challenge_id = secrets.token_urlsafe(24)
    # 32 bytes -> 43 Base64URL characters without padding, within Meta's required range.
    challenge_nonce = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")

    with _lock:
        _challenges[challenge_id] = {
            "nonce": challenge_nonce,
            "meta_user_id": meta_user_id,
            "playfab_id": playfab_id,
            "expires_at": now() + CHALLENGE_TTL_SECONDS,
        }

    return jsonify({
        "ok": True,
        "retryable": False,
        "challenge_id": challenge_id,
        "challenge_nonce": challenge_nonce,
        "message": "Challenge created.",
        "reason_code": "ok",
    }), 200


@app.post("/v1/attestation/verify")
def verify_attestation():
    configured, message = required_config_present()
    if not configured:
        log.error(message)
        return json_error(503, "Backend is not configured.", "backend_not_configured", True)

    clean_old_challenges()
    data = request.get_json(silent=True) or {}
    challenge_id = str(data.get("challenge_id", "")).strip()
    token = str(data.get("attestation_token", "")).strip()

    if not challenge_id or not token:
        return json_error(400, "Missing attestation data.", "missing_fields", False)

    with _lock:
        challenge = _challenges.get(challenge_id)

    if not challenge:
        return json_error(
            409,
            "The security challenge expired. A new challenge is required.",
            "challenge_missing_or_expired",
            True,
        )

    try:
        verdict = request_meta_attestation_verdict(token)
    except RuntimeError as exc:
        log.warning("%s", exc)
        # Keep the challenge so the client can retry after a temporary outage.
        return json_error(
            503,
            "Meta Attestation is temporarily unavailable.",
            "meta_attestation_unavailable",
            True,
        )

    data_entries = verdict.get("data")
    if not isinstance(data_entries, list) or not data_entries:
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(
            403,
            "Meta did not return a valid attestation verdict.",
            "invalid_meta_verdict",
            False,
        )

    entry = data_entries[0] if isinstance(data_entries[0], dict) else {}
    meta_message = str(entry.get("message", "")).strip().lower()
    if meta_message != "success":
        with _lock:
            _challenges.pop(challenge_id, None)

        # Deny the session, but do not automatically ban or accuse the player.
        return json_error(
            403,
            f"Meta rejected the attestation token: {meta_message or 'unknown error'}.",
            "meta_token_rejected",
            False,
            security_violation=False,
        )

    try:
        claims = decode_base64url_json(entry.get("claims", ""))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        with _lock:
            _challenges.pop(challenge_id, None)
        log.warning("Could not decode verified claims: %s", exc)
        return json_error(403, "Verified claims could not be decoded.", "claims_decode_failed", False)

    request_details = claims.get("request_details") or {}
    app_state = claims.get("app_state") or {}
    device_state = claims.get("device_state") or {}
    device_ban = claims.get("device_ban") or {}

    # Nonce and time checks prevent replay.
    claim_nonce = str(request_details.get("nonce", ""))
    if not hmac.compare_digest(claim_nonce, str(challenge["nonce"])):
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(
            403,
            "Attestation nonce did not match the server challenge.",
            "nonce_mismatch",
            False,
            security_violation=True,
        )

    current_time = now()
    try:
        issued_at = int(request_details.get("timestamp", 0))
        expires_at = int(request_details.get("exp", 0))
    except (TypeError, ValueError):
        issued_at = 0
        expires_at = 0

    if issued_at <= 0 or expires_at <= 0:
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(403, "Attestation timestamps were missing.", "invalid_timestamps", False)

    if issued_at > current_time + CLOCK_SKEW_SECONDS or expires_at < current_time - CLOCK_SKEW_SECONDS:
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(409, "Attestation token expired. Try again.", "token_expired", True)

    package_id = str(app_state.get("package_id", ""))
    if package_id != VALID_PACKAGE:
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(
            403,
            "The app package did not match this game.",
            "package_mismatch",
            False,
            security_violation=False,
        )

    token_certs = normalize_cert_list(app_state.get("package_cert_sha256_digest"))
    if not token_certs.intersection(VALID_CERTS):
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(
            403,
            "The app signing certificate did not match the configured certificate.",
            "certificate_mismatch",
            False,
            security_violation=False,
        )

    app_integrity = str(app_state.get("app_integrity_state", "NotEvaluated"))
    if app_integrity != "StoreRecognized":
        result = policy_failure(
            f"App integrity state was {app_integrity}. Install the build through a Meta release channel.",
            "app_not_store_recognized",
            security_violation=(app_integrity == "NotRecognized"),
        )
        if result is not None:
            with _lock:
                _challenges.pop(challenge_id, None)
            return result

    device_integrity = str(device_state.get("device_integrity_state", "NotTrusted"))
    allowed_device_states = {"Advanced"}
    if ALLOW_BASIC_DEVICE:
        allowed_device_states.add("Basic")

    if device_integrity not in allowed_device_states:
        result = policy_failure(
            f"Device integrity state was {device_integrity}.",
            "device_integrity_rejected",
            security_violation=(device_integrity == "NotTrusted"),
        )
        if result is not None:
            with _lock:
                _challenges.pop(challenge_id, None)
            return result

    if device_ban.get("is_banned") is True:
        with _lock:
            _challenges.pop(challenge_id, None)
        return json_error(
            403,
            "This device is blocked from this application.",
            "device_banned",
            False,
            security_violation=True,
        )

    # Consume the challenge only after successful verification.
    with _lock:
        _challenges.pop(challenge_id, None)

    playfab_id = str(challenge["playfab_id"])
    meta_user_id = str(challenge["meta_user_id"])
    unique_id = str(device_state.get("unique_id", ""))

    photon_token = issue_photon_token(playfab_id, meta_user_id, unique_id)

    return jsonify({
        "allowed": True,
        "retryable": False,
        "security_violation": False,
        "photon_token": photon_token,
        "photon_user_id": playfab_id,
        "message": "Attestation verified.",
        "reason_code": "ok",
    }), 200


@app.route("/v1/photon/auth", methods=["GET", "POST"])
def photon_custom_auth():
    """
    Photon Custom Authentication endpoint.

    Configure Photon to call:
      https://YOUR-SERVICE.onrender.com/v1/photon/auth

    Add a static dashboard parameter:
      provider_secret=<same value as PHOTON_PROVIDER_SECRET>

    The Unity client supplies:
      username=<PlayFabId>
      token=<short-lived JWT>
    """
    supplied_provider_secret = str(request.values.get("provider_secret", ""))
    if not PHOTON_PROVIDER_SECRET or not hmac.compare_digest(
        supplied_provider_secret,
        PHOTON_PROVIDER_SECRET,
    ):
        return jsonify({
            "ResultCode": 3,
            "Message": "Invalid authentication provider.",
        }), 200

    username = str(request.values.get("username", "")).strip()
    token = str(request.values.get("token", "")).strip()

    if not username or not token:
        return jsonify({
            "ResultCode": 3,
            "Message": "Missing username or token.",
        }), 200

    try:
        claims = jwt.decode(
            token,
            PHOTON_JWT_SECRET,
            algorithms=["HS256"],
            audience="photon",
            issuer="meta-attestation-backend",
            leeway=10,
        )
    except jwt.ExpiredSignatureError:
        return jsonify({
            "ResultCode": 2,
            "Message": "Security token expired. Reconnect through the game.",
        }), 200
    except jwt.InvalidTokenError:
        return jsonify({
            "ResultCode": 2,
            "Message": "Invalid security token.",
        }), 200

    subject = str(claims.get("sub", ""))
    if not subject or not hmac.compare_digest(subject, username):
        return jsonify({
            "ResultCode": 2,
            "Message": "Token identity mismatch.",
        }), 200

    return jsonify({
        "ResultCode": 1,
        "UserId": subject,
        "AuthCookie": {
            "Attested": True,
            "MetaUserId": str(claims.get("meta_user_id", "")),
            "DeviceHash": str(claims.get("device_id_hash", "")),
        },
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
